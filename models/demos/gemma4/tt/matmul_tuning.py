# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Explicit program configs for the batch-1 decode linears.

Every Gemma4 weight matmul is a bare ``ttnn.linear(x, w)``, which lets ttnn pick
the program config. MEASURED (`tests/unit/test_matmul_weight_placement.py`): that
automatic choice is **1.1-2.4x off** at the shapes the decode path actually uses,
with the weights in DRAM where they already are:

    shape (M,K,N)          auto      explicit 1D    speedup
    32 x  256 x 1024      5.92 us      4.53 us       1.31x   wqkv / gate / up
    32 x  256 x 1536      6.66 us      5.90 us       1.13x   post_projection
    32 x  512 x  256      7.93 us      4.25 us       1.87x   o_proj
    32 x 1024 x  256     13.79 us      6.64 us       2.08x   down_proj
    32 x 3072 x  256     37.35 us     15.53 us       2.41x   pre_projection
    32 x 1536 x 1536     26.58 us     17.34 us       1.53x   target-scale
    32 x 1536 x 3072     36.99 us     24.73 us       1.50x   target-scale

Why the heuristic misses: `create_simple_matmul_program_config`
(`ttnn/cpp/.../config/matmul_program_config.cpp`) only recomputes the block and
subblock sizes from the real Kt/Mt/Nt when *every* operand is DRAM-interleaved;
otherwise it keeps `in0_block_w = 2` and a generic per-core factor. Even on the
all-DRAM path it prefers a 2D config for shapes where a 1D multicast along N is
much better at M = 1 tile row.

The config below is essentially forced by the shape — grid from Nt, `per_core_N`
from Nt/cores, `per_core_M` = Mt = 1, subblocks 1x1 because each core owns one
output tile. The only free knob is ``in0_block_w``; a sweep over its divisors
picked the largest divisor of Kt <= 8 at every drafter shape.

Default OFF (`enabled=False`), so the target model and existing tests are
unaffected until explicitly opted in.
"""

import os

from loguru import logger

import ttnn


def _env_on(name):
    """``1|true|yes|on`` -> True; unset / anything else -> False.

    A bare ``os.getenv(name)`` truth test makes ``NAME=0`` mean ENABLED, which is
    the opposite of every other knob in this model (``GEMMA4_TUNE_MATMULS``,
    ``GEMMA4_WEIGHTS_IN_L1``, ``GEMMA4_SHARD_ACTIVATIONS``). That silently
    contaminates any A/B run with the variable exported as 0.
    """
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _block_cap():
    """Ceiling on ``in0_block_w``. Default 8; ``GEMMA4_MM_BLOCK_CAP`` overrides.

    The 8 is a heuristic, not a ttnn rule — the only hard constraints are
    ``Kt % in0_block_w == 0`` and, for a width-sharded in0,
    ``in0_shard_width_tiles % in0_block_w == 0``. MEASURED isolated on
    pre_projection 3072x256 (MEASUREMENT_RECORD.md 9d): the cap costs **-31.3%**
    with a pinned weight (6 -> 12) and **-35.2%** on the all-interleaved default
    (8 -> 32). It is left at 8 because raising it is NOT free:

      * the CBs grow linearly (in0 CB = in0_block_w x 2 tiles, and the in1 CB
        matches it when the weight is interleaved), so blk=48 wants 384 KB/core
        against a ~320 KB budget and collides with weight pinning;
      * it is NOT bit-neutral — reblocking changes the K accumulation order, so
        it moves generated tokens and therefore drafter acceptance, exactly like
        the tuner itself (see ``from_env``'s caveat).

    Note the scope: with ``GEMMA4_TUNE_MATMULS`` unset the tuner runs on the
    TARGET only (``DEFAULT_SCOPES``), so changing this knob moves the 35-layer
    target by default and the drafter only when the tuner is force-enabled.
    """
    try:
        return max(1, int(os.getenv("GEMMA4_MM_BLOCK_CAP", "8")))
    except ValueError:
        return 8


def _largest_divisor(n, cap=None):
    cap = _block_cap() if cap is None else cap
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def _pick_grid(n_tiles, max_x, max_y):
    """Largest core rectangle whose core count divides n_tiles."""
    best = (1, 1)
    for gy in range(1, max_y + 1):
        for gx in range(1, max_x + 1):
            c = gx * gy
            if n_tiles % c == 0 and c > best[0] * best[1]:
                best = (gx, gy)
    return best


def derive_decode_1d_config(m, k, n, max_x=8, max_y=8, in0_shard_tiles=None):
    """1D multicast config for a batch-1 decode linear, or None if it doesn't apply.

    Returns None (caller keeps ttnn's automatic choice) unless the shape is a
    single tile row and K/N are tile-aligned — the conditions the measured speedup
    was established under.

    ``m`` is the LOGICAL row count and is normally 1 at decode; it occupies one
    padded tile row. Requiring ``m % TILE_SIZE == 0`` here would reject every real
    decode shape, so round up instead.

    ``in0_shard_tiles`` is the width in tiles of each core's in0 shard when the
    ACTIVATION is width-sharded (see activation_sharding.py). The 1D mcast factory
    requires ``in0_shard_width_tiles % in0_block_w == 0``, so the blocking is
    clamped to a divisor of the shard. This matters a lot: MEASURED at K=256,N=1024,
    a 1-tile shard forces in0_block_w=1 and costs **+108%**, while a 4-tile shard
    (in0_block_w=4) costs **+5.7%**.
    """
    T = ttnn.TILE_SIZE
    if k % T or n % T:
        return None
    mt, kt, nt = -(-m // T), k // T, n // T
    if mt != 1:  # decode only; prefill has its own shapes and is not covered here
        return None
    gx, gy = _pick_grid(nt, max_x, max_y)
    cores = gx * gy
    if cores < 2:
        return None
    blk = _largest_divisor(kt)
    if in0_shard_tiles:
        blk = _largest_divisor(in0_shard_tiles, cap=min(_block_cap(), in0_shard_tiles))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=blk,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=mt,
        per_core_N=nt // cores,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _pick_grid_2(n_tiles, k_tiles, max_x=8, max_y=8):
    """Largest rectangle whose core count divides BOTH Nt and Kt.

    `gather_in0` shards the ACTIVATION over K and the weight over N on the same
    grid, so the core count must divide both. This is what confines the mechanism
    to the N=256 shapes.
    """
    best = (1, 1)
    for gy in range(1, max_y + 1):
        for gx in range(1, max_x + 1):
            c = gx * gy
            if n_tiles % c == 0 and k_tiles % c == 0 and c > best[0] * best[1]:
                best = (gx, gy)
    return best


def derive_decode_1d_gather_config(m, k, n, max_x=8, max_y=8):
    """`gather_in0` ring config, or None when it would cost parallelism.

    Returns None unless the grid that divides both Nt and Kt is the SAME grid the
    mcast config would have used. That single guard is what keeps this off the
    wide-N shapes: at K=256 the ring is capped at Kt=8 cores, so wqkv/gate/up
    (32 cores) and post_projection (48) would lose 4-6x parallelism. Only the
    N=256 shapes — o_proj, down_proj, pre_projection — survive it.

    MEASURED per matmul against the tuned mcast baseline, WITH the in0 reshard
    inside the timed region (`test_matmul_weight_placement.py` arm e+):
        pre_projection 3072x256   15.54 -> 6.48 us   -58.3%
        down_proj      1024x256    6.32 -> 4.30      -32.0%
        o_proj          512x256    4.33 -> 4.18       -3.4%
    """
    T = ttnn.TILE_SIZE
    if k % T or n % T:
        return None
    mt, kt, nt = -(-m // T), k // T, n // T
    if mt != 1:
        return None
    gx, gy = _pick_grid_2(nt, kt, max_x, max_y)
    cores = gx * gy
    if cores < 2:
        return None
    if (gx, gy) != _pick_grid(nt, max_x, max_y):  # would lose N-parallelism
        return None
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=kt // cores,  # the factory overwrites this with the shard width
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=mt,
        per_core_N=nt // cores,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
    )


def gather_shard_specs(k, n, max_x=8, max_y=8):
    """(in0 spec, in1 spec, out spec) for the gather config, or None.

    All three must live on the SAME grid: the validator requires
    `in0.shard_spec().grid == in1.shard_spec().grid` when in1 is in L1
    (`matmul_device_operation.cpp:1741-1749`), and the output must be sharded.
    """
    T = ttnn.TILE_SIZE
    if derive_decode_1d_gather_config(1, k, n, max_x, max_y) is None:
        return None
    gx, gy = _pick_grid_2(n // T, k // T, max_x, max_y)
    cores = gx * gy
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})

    def _mc(shape):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, shape, ttnn.ShardOrientation.ROW_MAJOR),
        )

    return _mc([T, k // cores]), _mc([k, n // cores]), _mc([T, n // cores])


class DecodeMatmulTuner:
    """Derives and caches decode matmul configs; a no-op passthrough when disabled.

    Shapes are only known at call time (TP splits them), so configs are built
    lazily on first use and cached by (in0 shape, in1 shape) — the same
    lazy-cache pattern as ``RMSNorm._build_sharded_cfg``.
    """

    #: Scopes tuned when GEMMA4_TUNE_MATMULS is unset. `target` only, on measurement:
    #: end to end it is +7.62% (48.38 -> 52.07 tok/s/u on the spec-decode path,
    #: 3 runs/arm against a 0.11% noise floor), because the 35-layer target is ~92%
    #: of a speculative-decoding iteration. The drafter is the other 7.6%, where the
    #: same change is worth ~0 and costs acceptance (it reblocks the K accumulation,
    #: so proposed draft tokens change), so it stays opt-in.
    DEFAULT_SCOPES = frozenset({"target"})

    @classmethod
    def from_env(cls, mesh_device=None, scope="draft"):
        """Build from ``GEMMA4_TUNE_MATMULS``.

        ``1``/``true``/``all`` enables everywhere; ``0``/``off`` disables everywhere;
        a comma list selects scopes (``draft``, ``target``, ``draft,target``) so the
        two can be A/B'd independently. **Unset now means `DEFAULT_SCOPES`, i.e. the
        target only** — it used to mean "off everywhere".

        Caveat, and it is a product decision rather than a perf one: the tuned config
        is not bit-neutral. It reblocks the K accumulation (per-op 0.99988 vs 0.99997
        PCC against fp32 — both correct), and on the target that changes the
        generated text from about the third character. `GEMMA4_TUNE_MATMULS=0`
        restores the previous numerics exactly.
        """
        raw = (os.getenv("GEMMA4_TUNE_MATMULS") or "").strip().lower()
        if raw in ("1", "true", "yes", "on", "all"):
            enabled = True
        elif raw in ("0", "false", "no", "off"):
            enabled = False
        elif raw == "":
            enabled = scope in cls.DEFAULT_SCOPES
        else:
            enabled = scope in {s.strip() for s in raw.split(",")}
        return cls(mesh_device, enabled=enabled, label=scope)

    def __init__(self, mesh_device=None, enabled=False, label="mm"):
        self.enabled = bool(enabled)
        self.label = label
        self._cache = {}
        self._max_x, self._max_y = 8, 8
        if self.enabled and mesh_device is not None:
            grid = mesh_device.compute_with_storage_grid_size()
            self._max_x, self._max_y = min(8, grid.x), min(8, grid.y)

    @staticmethod
    def _in0_shard_tiles(x):
        """Width in tiles of x's per-core shard, or None when x is interleaved."""
        try:
            if not x.is_sharded():
                return None
            spec = x.memory_config().shard_spec
            return int(spec.shape[1]) // ttnn.TILE_SIZE if spec is not None else None
        except Exception:  # noqa: BLE001
            return None

    def config_for(self, x, w):
        if not self.enabled:
            return None
        shard_tiles = self._in0_shard_tiles(x)
        key = (tuple(x.shape), tuple(w.shape), shard_tiles)
        if key not in self._cache:
            pc = derive_decode_1d_config(
                int(x.shape[-2]),
                int(x.shape[-1]),
                int(w.shape[-1]),
                self._max_x,
                self._max_y,
                in0_shard_tiles=shard_tiles,
            )
            self._cache[key] = pc
            # One line per DISTINCT shape (the cache makes this fire once each), so
            # this stays ~10 lines for a 35-layer model and gives the attribution
            # for any measured speedup rather than just a count.
            grid = f" grid={pc.compute_with_storage_grid_size.x}x{pc.compute_with_storage_grid_size.y}" if pc else ""
            blk = f" in0_block_w={pc.in0_block_w} per_core_N={pc.per_core_N}" if pc else ""
            logger.info(
                f"[mm-tune:{self.label}] M={int(x.shape[-2])} K={int(x.shape[-1])} N={int(w.shape[-1])}"
                f" -> {'TUNED' if pc else 'auto'}{grid}{blk}"
            )
        return self._cache[key]

    def stats(self):
        """(tuned, total) shape count — lets a test assert the tuner actually fired."""
        vals = list(self._cache.values())
        return sum(v is not None for v in vals), len(vals)

    @staticmethod
    def _l1_width_sharded(t):
        try:
            mc = t.memory_config()
            return (
                t.is_sharded()
                and mc.buffer_type == ttnn.BufferType.L1
                and mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
            )
        except Exception:  # noqa: BLE001
            return False

    def _gather_plan(self, x, w):
        """(config, in0_spec, out_spec) when this call can use the ring, else None.

        Gathering is opted into implicitly: it fires only when the WEIGHT is
        already L1 WIDTH_SHARDED, which `WeightPlacement` decides. So the
        allow-list is `GEMMA4_L1_ONLY`, and no separate knob is needed.
        """
        dbg = _env_on("GEMMA4_GATHER_DEBUG")

        def _no(why):
            if dbg:
                logger.info(f"[gather] SKIP {tuple(x.shape)[-1]}x{tuple(w.shape)[-1]}: {why}")
            return None

        if not self.enabled:
            return _no("tuner disabled")
        if not _env_on("GEMMA4_GATHER_IN0"):
            return _no("GEMMA4_GATHER_IN0 not set")
        # ttnn permits gather_in0 with in1 width-sharded, DRAM-INTERLEAVED, or fed via a
        # global CB (matmul_device_operation.cpp:1730-1741). Requiring L1 WIDTH_SHARDED here
        # is THIS codebase's policy, not a ttnn constraint: it encodes MEASUREMENT_RECORD.md
        # §9b, where on E2B the ring LOST with a DRAM weight (5.00 us vs 4.32 mcast) and won
        # only when pinned (2.68). GEMMA4_GATHER_DRAM_WEIGHT=1 lifts the policy so that
        # contrast can be re-measured on other models -- 12B has already reversed the ring's
        # sign once (§6.1), so the E2B result is not assumed to carry. L1 INTERLEAVED stays
        # refused because ttnn itself TT_FATALs on it.
        _allow_dram = _env_on("GEMMA4_GATHER_DRAM_WEIGHT")
        _w_mc = w.memory_config()
        _w_interleaved = _w_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        _w_dram_interleaved = _w_mc.buffer_type == ttnn.BufferType.DRAM and _w_interleaved
        # GEMMA4_GATHER_L1_INTERLEAVED admits an L1-INTERLEAVED weight and NOTHING else, so the
        # ring fires on exactly the relocated layers and the per-layer expected_ring arithmetic
        # still holds. That keeps "ring on a remote-SRAM weight" a clean arm rather than the
        # mixture ANY_WEIGHT gives (it also rings every DRAM weight, priced at +4.43 us each in
        # §6.5). Requires the loosened ttnn check.
        _allow_l1_il = _env_on("GEMMA4_GATHER_L1_INTERLEAVED")
        _w_l1_interleaved = _w_mc.buffer_type == ttnn.BufferType.L1 and _w_interleaved
        # GEMMA4_GATHER_ANY_WEIGHT=1 drops the model-level layout gate entirely and lets
        # ttnn's own validation adjudicate (matmul_device_operation.cpp:1728-1740). Exists so
        # "L1 INTERLEAVED + ring is impossible" can be DEMONSTRATED rather than asserted: the
        # model gate would otherwise refuse first and the TT_FATAL would never fire.
        if _env_on("GEMMA4_GATHER_ANY_WEIGHT"):
            pass
        elif (
            not self._l1_width_sharded(w)
            and not (_allow_dram and _w_dram_interleaved)
            and not (_allow_l1_il and _w_l1_interleaved)
        ):
            return _no(f"weight not L1 WIDTH_SHARDED ({_w_mc.buffer_type.name}/" f"{_w_mc.memory_layout.name})")
        k, n = int(x.shape[-1]), int(w.shape[-1])
        pc = derive_decode_1d_gather_config(1, k, n, self._max_x, self._max_y)
        if pc is None:
            return _no(f"no gather config for K={k} N={n} grid<={self._max_x}x{self._max_y}")
        specs = gather_shard_specs(k, n, self._max_x, self._max_y)
        if specs is None:
            return _no("gather_shard_specs returned None")
        in0_spec, in1_spec, out_spec = specs
        # The weight must sit on exactly the grid the ring expects -- but only when it IS
        # sharded. A DRAM-interleaved weight has no shard_spec; ttnn streams it per block.
        if w.memory_config().shard_spec is not None:
            have, want = list(w.memory_config().shard_spec.shape), list(in1_spec.shard_spec.shape)
            if have != want:
                return _no(f"weight shard shape {have} != ring's {want}")
        if dbg:
            logger.info(f"[gather] TAKE {k}x{n}")
        return pc, in0_spec, out_spec

    def linear(self, x, w, **kwargs):
        """``ttnn.linear`` with a tuned program config when one applies.

        When the weight is L1 width-sharded on the ring grid and the shape is
        grid-neutral, this takes the `gather_in0` path instead: reshard the
        activation over K, run the ring, and hand back an interleaved output so
        call sites are unchanged. MEASURED with both reshards included, this is
        -58.3% on pre_projection and -32.0% on down_proj (arm e+).
        """
        plan = self._gather_plan(x, w) if kwargs.get("program_config") is None else None
        if plan is not None:
            pc, in0_spec, out_spec = plan
            want_mc = kwargs.pop("memory_config", None) or ttnn.DRAM_MEMORY_CONFIG
            xs = x if x.memory_config() == in0_spec else ttnn.to_memory_config(x, in0_spec)
            out = ttnn.linear(xs, w, program_config=pc, memory_config=out_spec, **kwargs)
            if xs is not x:
                xs.deallocate(True)
            back = ttnn.sharded_to_interleaved(out, want_mc)
            out.deallocate(True)
            return back
        if kwargs.get("program_config") is None:
            pc = self.config_for(x, w)
            if pc is not None:
                kwargs["program_config"] = pc
        return ttnn.linear(x, w, **kwargs)


#: Shared disabled instance so call sites can do ``(tuner or DISABLED).linear(...)``.
DISABLED = DecodeMatmulTuner(enabled=False)


def resolve(tuner):
    return tuner if tuner is not None else DISABLED

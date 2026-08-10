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

from loguru import logger

import ttnn


def _largest_divisor(n, cap=8):
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


def derive_decode_1d_config(m, k, n, max_x=8, max_y=8):
    """1D multicast config for a batch-1 decode linear, or None if it doesn't apply.

    Returns None (caller keeps ttnn's automatic choice) unless the shape is a
    single tile row and K/N are tile-aligned — the conditions the measured speedup
    was established under.

    ``m`` is the LOGICAL row count and is normally 1 at decode; it occupies one
    padded tile row. Requiring ``m % TILE_SIZE == 0`` here would reject every real
    decode shape, so round up instead.
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
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=_largest_divisor(kt),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=mt,
        per_core_N=nt // cores,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


class DecodeMatmulTuner:
    """Derives and caches decode matmul configs; a no-op passthrough when disabled.

    Shapes are only known at call time (TP splits them), so configs are built
    lazily on first use and cached by (in0 shape, in1 shape) — the same
    lazy-cache pattern as ``RMSNorm._build_sharded_cfg``.
    """

    def __init__(self, mesh_device=None, enabled=False):
        self.enabled = bool(enabled)
        self._cache = {}
        self._max_x, self._max_y = 8, 8
        if self.enabled and mesh_device is not None:
            grid = mesh_device.compute_with_storage_grid_size()
            self._max_x, self._max_y = min(8, grid.x), min(8, grid.y)

    def config_for(self, x, w):
        if not self.enabled:
            return None
        key = (tuple(x.shape), tuple(w.shape))
        if key not in self._cache:
            pc = derive_decode_1d_config(int(x.shape[-2]), int(x.shape[-1]), int(w.shape[-1]), self._max_x, self._max_y)
            self._cache[key] = pc
            logger.debug(f"[mm-tune] {tuple(x.shape)} x {tuple(w.shape)} -> {'tuned' if pc else 'auto'}")
        return self._cache[key]

    def stats(self):
        """(tuned, total) shape count — lets a test assert the tuner actually fired."""
        vals = list(self._cache.values())
        return sum(v is not None for v in vals), len(vals)

    def linear(self, x, w, **kwargs):
        """``ttnn.linear`` with a tuned program config when one applies."""
        if kwargs.get("program_config") is None:
            pc = self.config_for(x, w)
            if pc is not None:
                kwargs["program_config"] = pc
        return ttnn.linear(x, w, **kwargs)


#: Shared disabled instance so call sites can do ``(tuner or DISABLED).linear(...)``.
DISABLED = DecodeMatmulTuner(enabled=False)


def resolve(tuner):
    return tuner if tuner is not None else DISABLED

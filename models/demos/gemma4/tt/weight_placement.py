# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where a Gemma4 weight tensor lives: DRAM (default) or L1/SRAM.

Every Gemma4 weight is loaded with a bare ``ttnn.as_tensor(...,
memory_config=ttnn.DRAM_MEMORY_CONFIG)`` and every weight matmul is a bare
``ttnn.linear(x, w)`` with no matmul program config. That combination means L1
residency for a weight is a *memory-config swap* — no height-sharding, no split
matmul, no concat. This module is the policy object that decides the swap.

Motivation: the it-assistant drafter is small enough that its transformer
weights fit in L1 (~14 MB per device at tp=2 for E2B), so the question "does
pinning draft weights in SRAM speed up drafting?" can be answered directly.
The output embedding does not fit (128 MiB per device, replicated) and falls
back to DRAM by budget rather than by a hardcoded exclusion, so the accounting
in :meth:`WeightPlacement.report` doubles as the capacity measurement.

Default is ``mode="dram"``, and every consumer defaults its ``weight_placement``
kwarg to ``None``, so the target model and existing tests are unaffected.

Sizing caveat: the budget is an explicit knob, not something derived from
``ttnn.get_max_worker_l1_unreserved_size()``. The hardware number (~1.46 MB per
core) is the size of the L1 *bank*, but statically-allocated circular buffers
and L1 buffers share one bank address space, and the CB high-water mark reaches
~1.25 MB per core on the SDPA decode cores. The real ceiling is therefore the
leftover, which is program-dependent. We log the hardware number for reference
and let the operator tune ``GEMMA4_L1_WEIGHT_BUDGET_MB`` against the loud
failure ("Statically allocated circular buffers ... clash with L1 buffers").
"""

import math
import os
import re
from dataclasses import dataclass, field

from loguru import logger

import ttnn

# Bytes per element. bfloat8_b is block-float: 1 mantissa byte per element plus
# one shared exponent byte per 16-element face row.
_BYTES_PER_ELEM = {
    ttnn.bfloat16: 2.0,
    ttnn.bfloat8_b: 1.0625,
    ttnn.bfloat4_b: 0.5625,
    ttnn.float32: 4.0,
    ttnn.uint32: 4.0,
    ttnn.int32: 4.0,
    ttnn.uint16: 2.0,
    ttnn.uint8: 1.0,
}

DEFAULT_BUDGET_MB = 32

_LAYER_RE = re.compile(r"layer_(\d+)")


def _parse_layer_spec(spec):
    """ "0,1,2" or "0-3" or "0-3,7" -> frozenset of ints; empty -> ()."""
    spec = (spec or "").strip()
    if not spec:
        return ()
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return frozenset(out)


def _pick_grid(n_tiles, max_x=8, max_y=8):
    """Largest core rectangle whose core count divides n_tiles.

    MUST match DecodeMatmulTuner's grid choice — the matmul validator requires
    per_core_N == in1 shard width in tiles, so a mismatch is a hard TT_FATAL.
    """
    best = (1, 1)
    for gy in range(1, max_y + 1):
        for gx in range(1, max_x + 1):
            c = gx * gy
            if n_tiles % c == 0 and c > best[0] * best[1]:
                best = (gx, gy)
    return best


def _is_replicating(mesh_mapper) -> bool:
    """True when the mapper puts a full copy of the tensor on every device.

    ``None`` means single-device or an explicit replicate; ``ShardTensor2dMesh``
    (what ``MeshConfig.column_parallel`` / ``row_parallel`` return) splits it.
    """
    if mesh_mapper is None:
        return True
    return "Replicate" in type(mesh_mapper).__name__


def per_device_bytes(shape, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None, num_devices=1) -> int:
    """Bytes this weight occupies on ONE device, after tile padding and sharding."""
    dims = list(shape)
    if layout == ttnn.TILE_LAYOUT and len(dims) >= 2:
        dims[-1] = math.ceil(dims[-1] / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        dims[-2] = math.ceil(dims[-2] / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    elems = 1
    for d in dims:
        elems *= int(d)
    if not _is_replicating(mesh_mapper) and num_devices > 1:
        elems = math.ceil(elems / num_devices)
    return int(math.ceil(elems * _BYTES_PER_ELEM.get(dtype, 2.0)))


@dataclass
class _Entry:
    name: str
    nbytes: int
    placed_l1: bool
    reason: str


@dataclass
class WeightPlacement:
    """Decides DRAM vs L1 per weight tensor, with a per-device byte budget.

    Args:
        mode: ``"dram"`` (default, no behavior change) or ``"l1"``.
        budget_bytes: per-device L1 budget for weights. Defaults to
            ``GEMMA4_L1_WEIGHT_BUDGET_MB`` (default 32 MB).
        label: shows up in the log lines, so a DRAM arm and an L1 arm built in
            the same process stay distinguishable.
    """

    #: ``"dram"`` | ``"l1"`` (interleaved) | ``"l1_sharded"``.
    #: MEASURED: interleaved L1 is worth ~0 (it stripes the weight across all 110
    #: cores, so a compute core still fetches over the NoC). ``l1_sharded`` puts
    #: each core's own N-slice in its own L1 and is worth 42-85% per matmul at the
    #: target's shapes. It REQUIRES the matching explicit program config, so it
    #: only pays off with DecodeMatmulTuner enabled.
    mode: str = "dram"
    budget_bytes: int = None
    label: str = "weights"
    #: Substring allow-list. When non-empty, only weights whose name contains one
    #: of these substrings are eligible for L1. Doubles as the tiering knob (pin
    #: just the MLPs, say) and as the bisect tool when an op misbehaves on an
    #: L1 operand. Env: GEMMA4_L1_ONLY="gate_proj,up_proj".
    only: tuple = ()
    #: Layer allow-list for selective pinning. Empty = every layer is eligible.
    #: Names carry ``layer_<i>/``, so this filters on the parsed index.
    #: Env: GEMMA4_L1_LAYERS="0,1,2" or "0-3".
    layers: tuple = ()
    used_bytes: int = 0
    entries: list = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in ("dram", "l1", "l1_sharded"):
            raise ValueError(f"WeightPlacement.mode must be dram|l1|l1_sharded, got {self.mode!r}")
        if self.budget_bytes is None:
            self.budget_bytes = int(float(os.getenv("GEMMA4_L1_WEIGHT_BUDGET_MB", DEFAULT_BUDGET_MB)) * (1 << 20))
        if not self.only:
            raw = os.getenv("GEMMA4_L1_ONLY", "").strip()
            self.only = tuple(s.strip() for s in raw.split(",") if s.strip()) if raw else ()
        if not self.layers:
            self.layers = _parse_layer_spec(os.getenv("GEMMA4_L1_LAYERS", ""))

    @classmethod
    def from_env(cls, default_mode="dram", **kwargs):
        """Build from ``GEMMA4_WEIGHTS_IN_L1``.

        ``sharded`` (recommended) -> WIDTH_SHARDED on the matmul grid;
        ``1``/``l1`` -> interleaved (measured worthless, kept for A/B);
        ``0``/unset -> DRAM.
        """
        raw = (os.getenv("GEMMA4_WEIGHTS_IN_L1") or "").strip().lower()
        mode = default_mode
        if raw in ("sharded", "l1_sharded", "shard"):
            mode = "l1_sharded"
        elif raw in ("1", "true", "yes", "on", "l1"):
            mode = "l1"
        elif raw in ("0", "false", "no", "off"):
            mode = "dram"
        return cls(mode=mode, **kwargs)

    @property
    def enabled(self) -> bool:
        return self.mode in ("l1", "l1_sharded")

    @property
    def sharded(self) -> bool:
        return self.mode == "l1_sharded"

    def _layer_allowed(self, name):
        if not self.layers:
            return True
        m = _LAYER_RE.search(name or "")
        # Weights outside any layer (projections, CME tables) stay eligible.
        return True if m is None else int(m.group(1)) in self.layers

    def log_hardware_ceiling(self, mesh_device):
        """Log the raw L1 bank capacity. Reference only — see the module docstring."""
        try:
            per_core = ttnn.get_max_worker_l1_unreserved_size()
            grid = mesh_device.compute_with_storage_grid_size()
            cores = grid.x * grid.y
        except Exception as e:  # noqa: BLE001 - diagnostics only, never fatal
            logger.info(f"[placement:{self.label}] could not query L1 capacity: {e}")
            return
        logger.info(
            f"[placement:{self.label}] mode={self.mode} budget={self.budget_bytes/(1<<20):.1f} MB/device; "
            f"hardware L1 = {per_core} B/core x {cores} cores ({grid.x}x{grid.y}) = "
            f"{per_core*cores/(1<<20):.1f} MB/device RAW (CBs share this space; budget is the real limit)"
        )

    def memory_config(
        self, name, shape=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=None, mesh_device=None
    ):
        """Memory config for one weight, recording the decision.

        Returns ``ttnn.DRAM_MEMORY_CONFIG`` unless L1 mode is on AND the tensor
        fits in the remaining per-device budget.
        """
        if not self.enabled:
            return ttnn.DRAM_MEMORY_CONFIG
        if shape is None:
            # Dummy-weight paths pass a None tensor; nothing to account for.
            self.entries.append(_Entry(name, 0, False, "unknown shape"))
            return ttnn.DRAM_MEMORY_CONFIG
        if self.only and not any(s in name for s in self.only):
            self.entries.append(_Entry(name, 0, False, "not in GEMMA4_L1_ONLY"))
            return ttnn.DRAM_MEMORY_CONFIG
        if not self._layer_allowed(name):
            self.entries.append(_Entry(name, 0, False, "layer not in GEMMA4_L1_LAYERS"))
            return ttnn.DRAM_MEMORY_CONFIG

        num_devices = 1
        if mesh_device is not None and hasattr(mesh_device, "get_num_devices"):
            num_devices = mesh_device.get_num_devices()
        nbytes = per_device_bytes(shape, dtype, layout, mesh_mapper, num_devices)

        if self.used_bytes + nbytes <= self.budget_bytes:
            self.used_bytes += nbytes
            self.entries.append(_Entry(name, nbytes, True, "fits"))
            # l1_sharded needs the PER-DEVICE shape to build its shard spec, which
            # is only knowable after the mesh mapper has run. Load to DRAM here and
            # let place_as_tensor reshard; it records the outcome either way.
            return ttnn.DRAM_MEMORY_CONFIG if self.sharded else ttnn.L1_MEMORY_CONFIG

        self.entries.append(
            _Entry(
                name,
                nbytes,
                False,
                f"over budget ({(self.used_bytes+nbytes)/(1<<20):.1f} MB > " f"{self.budget_bytes/(1<<20):.1f} MB)",
            )
        )
        return ttnn.DRAM_MEMORY_CONFIG

    # ── reporting ────────────────────────────────────────────────────────────
    def summary(self):
        l1 = [e for e in self.entries if e.placed_l1]
        dram = [e for e in self.entries if not e.placed_l1]
        return {
            "mode": self.mode,
            "budget_bytes": self.budget_bytes,
            "l1_bytes": sum(e.nbytes for e in l1),
            "l1_count": len(l1),
            "dram_bytes": sum(e.nbytes for e in dram),
            "dram_count": len(dram),
        }

    def report(self, top_n=12) -> str:
        s = self.summary()
        lines = [
            f"[placement:{self.label}] mode={s['mode']} "
            f"L1={s['l1_bytes']/(1<<20):.2f} MB/device over {s['l1_count']} tensors, "
            f"DRAM={s['dram_bytes']/(1<<20):.2f} MB/device over {s['dram_count']} tensors, "
            f"budget={s['budget_bytes']/(1<<20):.1f} MB"
        ]
        rejected = [e for e in self.entries if not e.placed_l1 and e.nbytes > 0]
        if rejected:
            lines.append(f"[placement:{self.label}] fell back to DRAM:")
            for e in sorted(rejected, key=lambda e: -e.nbytes)[:top_n]:
                lines.append(f"    {e.name:<48} {e.nbytes/(1<<20):8.2f} MB  ({e.reason})")
        biggest = sorted([e for e in self.entries if e.placed_l1], key=lambda e: -e.nbytes)[:top_n]
        if biggest:
            lines.append(f"[placement:{self.label}] largest tensors pinned to L1:")
            for e in biggest:
                lines.append(f"    {e.name:<48} {e.nbytes/(1<<20):8.2f} MB")
        return "\n".join(lines)


#: Shared no-op instance so call sites can write ``(placement or DRAM_ONLY)``
#: without allocating a policy object per weight.
DRAM_ONLY = WeightPlacement(mode="dram", label="default")


def resolve(placement):
    """``None`` -> the shared DRAM-only policy."""
    return placement if placement is not None else DRAM_ONLY


def shard_l1_width(tensor, mesh_device):
    """Reshard a loaded weight to L1 WIDTH_SHARDED on the matmul's compute grid.

    Each core ends up holding exactly the N-slice it multiplies, which is what
    makes L1 residency worth 42-85% per matmul (interleaved L1 buys ~0).

    The grid comes from ``derive_decode_1d_config`` — the SAME function the matmul
    tuner uses — because the validator requires
    ``per_core_N == in1_shard_width_tiles`` and a duplicated grid heuristic drifts
    out of sync (it did: "shard width in tiles (2) must equal per_core_N (1)").

    Returns the resharded tensor, or None when the weight does not qualify
    (ROW_MAJOR norms, shapes with no valid grid).
    """
    from models.demos.gemma4.tt.matmul_tuning import derive_decode_1d_config

    T = ttnn.TILE_SIZE
    if tensor.layout != ttnn.TILE_LAYOUT:
        return None  # norms / embedding tables are not matmul in1
    try:
        local = ttnn.get_device_tensors(tensor)[0].shape if mesh_device.get_num_devices() > 1 else tensor.shape
    except Exception:  # noqa: BLE001
        local = tensor.shape
    if len(local) < 2:
        return None
    k, n = int(local[-2]), int(local[-1])

    pc = derive_decode_1d_config(1, k, n)
    if pc is None:
        return None
    gx, gy = pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y
    cores = gx * gy
    shard_w = n // cores
    if shard_w // T != pc.per_core_N:  # would FATAL in the matmul validator
        return None
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [k, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.to_memory_config(tensor, mc)


def place_as_tensor(
    placement,
    name,
    torch_tensor,
    *,
    device,
    dtype,
    layout,
    mesh_mapper=None,
    cache_file_name=None,
    shape=None,
):
    """``ttnn.as_tensor`` that actually honours the placement decision.

    This wrapper exists because of a sharp edge in ``ttnn.as_tensor``: on a
    tensor-cache **hit** it returns
    ``ttnn._ttnn.tensor.load_tensor_flatbuffer(cache_file_name, device=device)``
    (``ttnn/ttnn/operations/core.py``) and **never applies memory_config**. Only
    the cache-miss path calls ``tensor.to(device, memory_config)``. So the first
    run of an L1 arm would place weights in L1 and every subsequent run — with
    the cache warm — would silently put them back in DRAM, turning the A/B into
    a no-op that still looks like it worked.

    We therefore check the buffer type we actually got and relocate if needed.
    """
    placement = resolve(placement)
    if shape is None and torch_tensor is not None:
        shape = torch_tensor.shape
    mem = placement.memory_config(
        name, shape=shape, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper, mesh_device=device
    )
    tensor = ttnn.as_tensor(
        torch_tensor,
        device=device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=mesh_mapper,
        cache_file_name=cache_file_name,
        memory_config=mem,
    )
    if placement.sharded and mem.buffer_type == ttnn.BufferType.DRAM:
        # memory_config() returns DRAM for sharded mode; a recorded L1 entry means
        # "pin this one". Anything else genuinely stays in DRAM.
        wanted = placement.entries and placement.entries[-1].name == name and placement.entries[-1].placed_l1
        if not wanted:
            return tensor
        try:
            moved = shard_l1_width(tensor, device)
            if moved is None:
                # Normal for norms / non-matmul weights: give the budget back and
                # record it, but do not shout about it.
                for entry in reversed(placement.entries):
                    if entry.name == name and entry.placed_l1:
                        entry.placed_l1 = False
                        entry.reason = "not a shardable matmul weight"
                        placement.used_bytes -= entry.nbytes
                        break
                return tensor
            tensor.deallocate(True)
            return moved
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[placement:{placement.label}] could not L1-shard {name}: {str(e)[:110]}")
            for entry in reversed(placement.entries):
                if entry.name == name and entry.placed_l1:
                    entry.placed_l1 = False
                    entry.reason = f"shard failed: {type(e).__name__}"
                    placement.used_bytes -= entry.nbytes
                    break
            return tensor

    if device is None or mem.buffer_type == ttnn.BufferType.DRAM:
        return tensor
    if tensor.memory_config().buffer_type == mem.buffer_type:
        return tensor
    try:
        moved = ttnn.to_memory_config(tensor, mem)
        tensor.deallocate(True)
        return moved
    except Exception as e:  # noqa: BLE001
        # Keep the accounting honest rather than reporting a pin that never happened.
        logger.warning(f"[placement:{placement.label}] could not relocate {name} to L1: {e}")
        for entry in reversed(placement.entries):
            if entry.name == name and entry.placed_l1:
                entry.placed_l1 = False
                entry.reason = f"relocate failed: {type(e).__name__}"
                placement.used_bytes -= entry.nbytes
                break
        return tensor

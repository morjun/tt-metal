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

    mode: str = "dram"
    budget_bytes: int = None
    label: str = "weights"
    #: Substring allow-list. When non-empty, only weights whose name contains one
    #: of these substrings are eligible for L1. Doubles as the tiering knob (pin
    #: just the MLPs, say) and as the bisect tool when an op misbehaves on an
    #: L1 operand. Env: GEMMA4_L1_ONLY="gate_proj,up_proj".
    only: tuple = ()
    used_bytes: int = 0
    entries: list = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in ("dram", "l1"):
            raise ValueError(f"WeightPlacement.mode must be 'dram' or 'l1', got {self.mode!r}")
        if self.budget_bytes is None:
            self.budget_bytes = int(float(os.getenv("GEMMA4_L1_WEIGHT_BUDGET_MB", DEFAULT_BUDGET_MB)) * (1 << 20))
        if not self.only:
            raw = os.getenv("GEMMA4_L1_ONLY", "").strip()
            self.only = tuple(s.strip() for s in raw.split(",") if s.strip()) if raw else ()

    @classmethod
    def from_env(cls, default_mode="dram", **kwargs):
        """Build from ``GEMMA4_WEIGHTS_IN_L1`` (1/true/l1 enables L1 mode)."""
        raw = os.getenv("GEMMA4_WEIGHTS_IN_L1")
        mode = default_mode
        if raw is not None:
            mode = "l1" if raw.strip().lower() in ("1", "true", "yes", "on", "l1") else "dram"
        return cls(mode=mode, **kwargs)

    @property
    def enabled(self) -> bool:
        return self.mode == "l1"

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

        num_devices = 1
        if mesh_device is not None and hasattr(mesh_device, "get_num_devices"):
            num_devices = mesh_device.get_num_devices()
        nbytes = per_device_bytes(shape, dtype, layout, mesh_mapper, num_devices)

        if self.used_bytes + nbytes <= self.budget_bytes:
            self.used_bytes += nbytes
            self.entries.append(_Entry(name, nbytes, True, "fits"))
            return ttnn.L1_MEMORY_CONFIG

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

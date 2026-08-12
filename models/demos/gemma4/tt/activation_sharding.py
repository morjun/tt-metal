# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Keep the decode residual stream WIDTH_SHARDED instead of rebuilding it per norm.

``RMSNorm._forward_sharded`` does ``interleaved_to_sharded -> sharded rms_norm ->
sharded_to_interleaved`` on EVERY call, and the ``not x.is_sharded()`` guard in
``RMSNorm.forward`` then makes the next norm rebuild the identical layout.
MEASURED on the E2B drafter (``test_rms_norm_layout_churn``): **21 I2S + 21 S2I =
42 conversions per backbone step**, ~382 us of a ~671 us backbone.

The conversions, not the kernel, are the problem. At the drafter's dim=256
(``test_rms_norm_path_costs`` / ``test_rms_norm_core_sweep``, us):

    plain interleaved rms_norm               6.41
    sharded kernel  (8 cores, 1 tile each)   4.45     <- always faster than plain
    sharded kernel  (2 cores, 4 tiles each)  4.33
    I2S + S2I overhead                       4.2      <- this is what sinks it

So the sharded kernel is a 30% win that the round-trip more than gives back.

**Why 2 cores and not 8.** Chaining only pays if the CONSUMER takes the sharded
tensor. The consumers are the K=256 matmuls, and ``mcast_in0`` requires
``in0_shard_width_tiles % in0_block_w == 0``. At 8 cores the shard is ONE tile
wide, forcing ``in0_block_w=1`` — 8 multicast/semaphore round-trips instead of 1,
measured at **+108%** on the matmul. At 2 cores (4 tiles) the matmul takes the
sharded in0 for **+0.3% to +5.7%**, and the norm is indifferent to the core count
(4.33 vs 4.45 us). Hence ``MIN_SHARD_TILES``.

Net per norm+matmul pair at dim=256: 12.92 us today, 10.68 unsharded, **8.85
chained**.

Default OFF. Every consumer takes ``activation_sharding=None``, so the target
model and existing tests are untouched until explicitly opted in via
``GEMMA4_SHARD_ACTIVATIONS=1``.
"""

import os

import ttnn

#: Minimum tiles per core in the residual-stream shard. Below this the matmul's
#: `in0_block_w` is forced down and the matmul loses far more than the norm gains.
MIN_SHARD_TILES = 4


class ActivationSharding:
    """Decides the residual stream's shard spec, and converts to/from it."""

    def __init__(self, mesh_device=None, enabled=False, min_shard_tiles=MIN_SHARD_TILES, max_cores=8):
        self.enabled = bool(enabled)
        self.min_shard_tiles = min_shard_tiles
        self.max_cores = max_cores
        self.mesh_device = mesh_device
        self._specs = {}

    @classmethod
    def from_env(cls, mesh_device=None, **kw):
        raw = (os.getenv("GEMMA4_SHARD_ACTIVATIONS") or "").strip().lower()
        enabled = raw in ("1", "true", "yes", "on")
        tiles = os.getenv("GEMMA4_SHARD_MIN_TILES")
        if tiles:
            kw["min_shard_tiles"] = int(tiles)
        return cls(mesh_device, enabled=enabled, **kw)

    def cores_for(self, dim):
        """Most cores that still leave >= min_shard_tiles per core, or None."""
        if not self.enabled or dim % ttnn.TILE_SIZE:
            return None
        tiles = dim // ttnn.TILE_SIZE
        best = None
        for c in range(1, self.max_cores + 1):
            if tiles % c == 0 and tiles // c >= self.min_shard_tiles:
                best = c
        return best

    def spec(self, dim):
        """WIDTH_SHARDED memory config for a [*, 32, dim] decode activation."""
        if dim in self._specs:
            return self._specs[dim]
        cores = self.cores_for(dim)
        spec = None
        if cores:
            spec = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, dim // cores),
                core_grid=ttnn.CoreGrid(x=cores, y=1),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        self._specs[dim] = spec
        return spec

    def shard_tiles(self, dim):
        """Tiles per core for `dim`, or None. The matmul's in0_block_w must divide this."""
        cores = self.cores_for(dim)
        return (dim // ttnn.TILE_SIZE) // cores if cores else None

    def applies(self, x):
        """Is `x` a decode-shaped activation we have a spec for?"""
        return (
            self.enabled
            and hasattr(x, "shape")
            and len(x.shape) == 4
            and 1 <= int(x.shape[-2]) <= ttnn.TILE_SIZE
            and self.spec(int(x.shape[-1])) is not None
        )

    def matches(self, x):
        """Is `x` ALREADY in our spec? Then neither reshard nor conversion is needed."""
        if not self.applies(x) or not x.is_sharded():
            return False
        want = self.spec(int(x.shape[-1])).shard_spec
        got = x.memory_config().shard_spec
        return got is not None and list(got.shape) == list(want.shape) and got.grid == want.grid

    def to_stream(self, x):
        """Into the shared shard spec. No-op if already there or not applicable."""
        if not self.applies(x) or self.matches(x):
            return x
        return ttnn.to_memory_config(x, self.spec(int(x.shape[-1])))

    def to_stream_like(self, x, other):
        """Give `x` the same layout as `other`, for a binary op.

        Binary ops need identical shard specs on both operands. `other` is the
        norm output: sharded when chaining is on, interleaved otherwise. Returns
        `x` untouched whenever the layouts already agree, which is the common
        case once the residual stream is sharded end-to-end — only the first
        layer, whose residual comes from the entry projection, pays a conversion.
        """
        if not self.enabled or not hasattr(other, "is_sharded"):
            return x
        if other.is_sharded():
            if x.is_sharded() and x.memory_config() == other.memory_config():
                return x
            return ttnn.to_memory_config(x, other.memory_config())
        return self.from_stream(x) if x.is_sharded() else x

    def from_stream(self, x, memory_config=None):
        """Back to interleaved, for ops that cannot take a sharded operand."""
        if not hasattr(x, "is_sharded") or not x.is_sharded():
            return x
        return ttnn.sharded_to_interleaved(x, memory_config or ttnn.DRAM_MEMORY_CONFIG)


#: Shared disabled instance so call sites can write ``(policy or DISABLED)``.
DISABLED = ActivationSharding(enabled=False)


def resolve(policy):
    return policy if policy is not None else DISABLED

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch import nn

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.activation_sharding import resolve as resolve_activation_sharding
from models.demos.gemma4.tt.weight_placement import place_as_tensor
from models.demos.gemma4.tt.weight_placement import resolve as resolve_placement
from models.demos.gemma4.utils.general_utils import get_cache_file_name


class RMSNorm(nn.Module):
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        tensor_cache_path=None,
        mesh_config=None,
        with_scale=True,
        weight_placement=None,
        activation_sharding=None,
    ):
        super().__init__()
        self.with_scale = with_scale
        placement = resolve_placement(weight_placement)
        # When enabled, the norm consumes and produces the SHARED residual-stream
        # shard spec instead of rebuilding it: no I2S in, no S2I out. See
        # activation_sharding.py for why the grid is deliberately narrow.
        self.act_shard = resolve_activation_sharding(activation_sharding)

        if with_scale and state_dict and "weight" in state_dict:
            torch_weight = state_dict["weight"].reshape((1, 1, -1, ttnn.TILE_SIZE))
        else:
            torch_weight = None

        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))
        self.is_distributed = False

        if with_scale:
            norm_mapper = (
                self.mesh_config.shard_mapper(mesh_device, mesh_dims=(None, -2)) if self.is_distributed else None
            )
            self.tt_weight = place_as_tensor(
                placement,
                f"{tensor_cache_path or 'rms_norm'}/weight",
                torch_weight,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=norm_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            )
        else:
            self.tt_weight = None

        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device

        # Decode width-sharded fast path. The plain (interleaved) rms_norm runs
        # the RMS reduction over the full hidden width on few cores — ~76 us for
        # a single-token [1,1,32,hidden] norm on Gemma4-31B (hidden=5376). Width-
        # sharding the activation across a core grid parallelizes the reduction
        # (LayerNormShardedMultiCoreProgramConfig handles the cross-core gather),
        # cutting it to <10 us. Built lazily on first decode-shaped call so we
        # can read the activation's true (padded) hidden width, then cached.
        self._sharded_cfg = None  # (input_memcfg, program_config) or None if unavailable
        self._sharded_dim = None

    def _build_sharded_cfg(self, dim):
        """Pick the largest core grid whose core count divides dim/32 and build
        the width-sharded input memcfg + LayerNorm program config. Returns None
        if no usable grid divides the tile-width evenly (falls back to plain)."""
        if dim % ttnn.TILE_SIZE != 0:
            return None
        tiles = dim // ttnn.TILE_SIZE
        grid = self.mesh_device.compute_with_storage_grid_size()
        best = None  # (num_cores, gx, gy)
        for gy in range(1, grid.y + 1):
            for gx in range(1, grid.x + 1):
                n = gx * gy
                if tiles % n == 0 and (best is None or n > best[0]):
                    best = (n, gx, gy)
        if best is None or best[0] == 1:
            return None
        num_cores, gx, gy = best
        block_w = tiles // num_cores
        subblock_w = 4
        while subblock_w > 1 and block_w % subblock_w != 0:
            subblock_w -= 1
        input_memcfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, dim // num_cores),
            core_grid=ttnn.CoreGrid(x=gx, y=gy),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[gx, gy],
            subblock_w=subblock_w,
            block_h=1,
            block_w=block_w,
            inplace=False,
        )
        return (input_memcfg, program_config)

    def _chained_cfg(self, dim):
        """Program config matching ActivationSharding's spec for `dim`. Cached."""
        if getattr(self, "_chain_cfg_dim", None) == dim:
            return self._chain_cfg
        cores = self.act_shard.cores_for(dim)
        block_w = (dim // ttnn.TILE_SIZE) // cores
        subblock_w = 4
        while subblock_w > 1 and block_w % subblock_w != 0:
            subblock_w -= 1
        self._chain_cfg_dim = dim
        self._chain_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[cores, 1],
            subblock_w=subblock_w,
            block_h=1,
            block_w=block_w,
            inplace=False,
        )
        return self._chain_cfg

    def _forward_chained(self, x):
        """Sharded rms_norm in and out — no layout conversion in the common case."""
        dim = int(x.shape[-1])
        x_sh = self.act_shard.to_stream(x)  # no-op when the producer already sharded
        out = ttnn.rms_norm(
            x_sh,
            weight=self.tt_weight,
            epsilon=self.eps,
            program_config=self._chained_cfg(dim),
            memory_config=self.act_shard.spec(dim),
        )
        if x_sh is not x:
            x_sh.deallocate(True)
        return out

    def _forward_sharded(self, x):
        """Width-sharded decode RMSNorm: I2S -> sharded rms_norm -> S2I."""
        x_sh = ttnn.to_memory_config(x, self._sharded_cfg[0])
        out = ttnn.rms_norm(
            x_sh,
            weight=self.tt_weight,
            epsilon=self.eps,
            program_config=self._sharded_cfg[1],
        )
        x_sh.deallocate(True)
        out_interleaved = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        return out_interleaved

    def forward(self, x):
        if self.is_distributed:
            activation_grid_bounding_box_size = x.memory_config().shard_spec.grid.bounding_box().grid_size()
            shard_height, shard_width = x.memory_config().shard_spec.shape
            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=activation_grid_bounding_box_size,
                subblock_w=1,
                block_h=ttnn.core.divup(shard_height, ttnn.TILE_SIZE),
                block_w=ttnn.core.divup(shard_width, ttnn.TILE_SIZE),
                inplace=False,
            )

            tt_gathered_stats_memory_config = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * self.mesh_shape[1]],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            tt_stats = ttnn.rms_norm_pre_all_gather(x, program_config=program_config, dtype=ttnn.bfloat16)

            tt_gathered_stats = ttnn.all_gather(
                tt_stats,
                dim=3,
                num_links=1,
                cluster_axis=1,
                mesh_device=self.mesh_device,
                memory_config=tt_gathered_stats_memory_config,
                topology=ttnn.Topology.Ring,
            )
            ttnn.deallocate(tt_stats)

            tt_output = ttnn.rms_norm_post_all_gather(
                x,
                tt_gathered_stats,
                program_config=program_config,
                epsilon=self.eps,
                weight=self.tt_weight,
                dtype=ttnn.bfloat16,
                stats=tt_gathered_stats,
            )
            ttnn.deallocate(tt_gathered_stats)
            return tt_output
        else:
            decode_shaped = (
                self.with_scale
                and self.tt_weight is not None
                and len(x.shape) == 4
                and 1 <= x.shape[-2] <= ttnn.TILE_SIZE
            )

            # CHAINED path: consume and produce the shared residual-stream shard
            # spec, so neither this norm nor the next one pays a layout
            # conversion. The kernel is 30% faster than plain interleaved at
            # every width; it was only ever the I2S/S2I round-trip that made the
            # sharded path a net loss (42 conversions per drafter step).
            if decode_shaped and self.act_shard.applies(x):
                return self._forward_chained(x)

            # Unchained sharded fast path: I2S -> sharded rms_norm -> S2I. Still a
            # win at large dim (dim>=1024 measured), a loss below that; kept as
            # the default because the chained path needs every consumer to accept
            # a sharded operand. Prefill (height > 32) and the no-weight per-head
            # norms keep the plain path.
            if decode_shaped and not x.is_sharded():
                dim = x.shape[-1]
                if self._sharded_cfg is None or self._sharded_dim != dim:
                    self._sharded_dim = dim
                    self._sharded_cfg = self._build_sharded_cfg(dim)
                if self._sharded_cfg:
                    return self._forward_sharded(x)

            # A sharded input with no chaining policy would FATAL in ttnn
            # ("Sharded inputs require sharded outputs"), so fall back explicitly.
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

            if self.with_scale:
                tt_output = ttnn.rms_norm(
                    x,
                    weight=self.tt_weight,
                    epsilon=self.eps,
                )
            else:
                tt_output = ttnn.rms_norm(
                    x,
                    epsilon=self.eps,
                )
            return tt_output

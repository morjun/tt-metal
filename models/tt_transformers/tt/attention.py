# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt import l1_kv_perf
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup


class Attention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.num_devices = configuration.num_devices
        self.TG = self.num_devices == 32
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.paged_attention_config = paged_attention_config
        self.min_kv_prefill_shard_seqlen = configuration.min_kv_prefill_shard_seqlen
        self.ccl_dtype = configuration.ccl_dtype
        self.num_reduce_scatter_links = configuration.num_reduce_scatter_links
        self.num_all_gather_links = configuration.num_all_gather_links
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN
        self.tile_size = configuration.tile_size
        self.rms_norm_add_unit_offset = configuration.rms_norm_add_unit_offset
        self.num_device_groups = self.num_devices // self.n_kv_heads
        self.num_devices_per_group = self.n_kv_heads if self.TG else self.num_devices
        self.batch_size_per_device_group = (
            max(self.max_batch_size // self.num_device_groups, 1) if self.TG else self.max_batch_size
        )

        self.n_local_heads = self.n_heads // self.num_devices_per_group
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group

        self.arch_name = configuration.arch_name
        # TODO: Fix this once all-gather supports < tile_size
        if self.TG:
            weight = torch.zeros(1, 32, 8, 32)
            for i in range(32):
                col = i % 4  # This determines which group of 8 to select
                weight[:, i, :, col * 8 : (col + 1) * 8] = torch.eye(8)

            self.slice_mat = ttnn.from_torch(
                weight,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            )
            user_selection_matrix = torch.eye(8, 8)
            user_selection_matrix = torch.nn.functional.pad(user_selection_matrix, (0, 24), "constant", 0)  # (8, 32)
            user_selection_matrix = [user_selection_matrix] * 4
            user_selection_matrix = torch.block_diag(*user_selection_matrix)  # (32, 128)
            self.user_selection_matrix = ttnn.from_torch(
                user_selection_matrix,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        self.dtype = dtype

        self.max_seq_len = configuration.max_seq_len
        self.grid_size = configuration.max_grid_size
        self.l1_kv_window_size = getattr(configuration, "l1_kv_window_size", 0)
        self.l1_kv_sink_size = getattr(configuration, "l1_kv_sink_size", 0)
        self.l1_kv_use_sharded = getattr(configuration, "l1_kv_use_sharded", False)
        self.l1_kv_min_expected_hit_ratio = getattr(configuration, "l1_kv_min_expected_hit_ratio", 0.0)
        self.use_adaptive_l1_kv_cache = getattr(configuration, "use_adaptive_l1_kv_cache", False)
        # StreamingLLM-style attention sink default for the adaptive path: keep
        # the first tile (32 tokens) of prefill K/V permanently in L1. Outside the
        # ring's reach, so they stay valid for the entire decode session and let
        # the model retain its "attention sink" tokens that anchor the softmax.
        # User can override via configuration.l1_kv_sink_size.
        if self.use_adaptive_l1_kv_cache and self.l1_kv_sink_size == 0:
            self.l1_kv_sink_size = 32
        if self.l1_kv_sink_size > 0 and not self.use_adaptive_l1_kv_cache:
            assert self.l1_kv_window_size > 0, "Pinned L1 sink tokens require a non-zero recent ring window"
        # Adaptive L1 KV cores to AVOID — typically the same grid that hosts the
        # 63-core "L-shape" model intermediate buffer (QKV / embedding path), so
        # the KV-tier allocation doesn't push the L-shape's bottom_so_far up into
        # program 49's CB region (causing the CB-clash we mitigate with
        # runtime_pad_bytes). For Llama-8B on Blackhole p150, the L-shape spans
        # cores (0..10, 0..4) plus (0..7, 5). Override via
        # configuration.l1_kv_avoid_cores = [(x, y), ...] or set to [] to disable.
        self.l1_kv_avoid_cores = getattr(
            configuration,
            "l1_kv_avoid_cores",
            None,
        )
        # StreamingLLM L1-only inference mode. When True, SDPA decode is told
        # to iterate ONLY the L1 cache contents (sink + ring), skipping DRAM
        # reads. Achieved by passing a clamped cur_pos (= L1_total - 1) and
        # decode_start_pos = 0 to the SDPA op. The DRAM/ring write paths keep
        # the REAL current_pos unchanged so caches remain coherent.
        # Quality tradeoff: attention is restricted to sink ∪ ring; middle
        # prefill positions are dropped. Per StreamingLLM, with attention sinks
        # the quality is close to full attention.
        self.l1_kv_only_mode = getattr(configuration, "l1_kv_only_mode", False)
        if self.l1_kv_avoid_cores is None:
            # Default: filter out the L-shape's footprint for the adaptive path.
            # Avoiding overlap cores also REDUCES the L1 write dispatch overhead
            # (fewer cores per tier → less coordination work), which is the
            # dominant cost in the adaptive write path. A/B in /tmp/phase5_*
            # showed disjoint (11 cores y=9, 32 tokens) at 9.19 tok/s vs overlap
            # (27 cores y=0..8 x=8..10, 96 tokens) at 5.23 tok/s for the same
            # runtime_pad — narrower cores beats more cache.
            # In L1-only mode we need enough capacity to host a substantial
            # ring on top of the attention sink; the disjoint subset (~32
            # tokens at viable runtime_pad) is too small. Fall back to the
            # full core set so 192-token caches become reachable.
            if self.use_adaptive_l1_kv_cache and not self.l1_kv_only_mode:
                self.l1_kv_avoid_cores = [(x, y) for x in range(11) for y in range(5)] + [(x, 5) for x in range(8)]
            else:
                self.l1_kv_avoid_cores = []
        self.l1_kv_total_size = self.l1_kv_sink_size + self.l1_kv_window_size
        self.l1_kv_sink_size_tiles = math.ceil(self.l1_kv_sink_size / self.tile_size) if self.l1_kv_sink_size > 0 else 0
        self.l1_kv_window_size_tiles = (
            math.ceil(self.l1_kv_window_size / self.tile_size) if self.l1_kv_window_size > 0 else 0
        )
        # Adaptive tier state (populated by allocate_l1_kv_cache when use_adaptive_l1_kv_cache=True)
        # l1_kv_tiers: list of (k_tensor, v_tensor, token_start, tok_count)
        #   t[0] k_tensor    — HEIGHT_SHARDED K cache tensor in L1
        #   t[1] v_tensor    — HEIGHT_SHARDED V cache tensor in L1
        #   t[2] token_start — first global token index this tier covers
        #   t[3] tok_count   — number of tokens this tier holds
        self.l1_kv_tiers: list = []
        self.l1_kv_adaptive_total_capacity: int = 0  # sum of all tier tok_counts; set after allocation
        # decode_start_pos: cur_pos at the first decode step that writes to L1 tiers.
        # Used as the lower bound by the SDPA decode kernel when deciding whether a
        # sequence position is in L1. Positions < decode_start_pos came from prefill
        # and are never written to the L1 ring. Captured on first ring write.
        self._l1_kv_decode_start_pos: int = 0
        self._l1_kv_decode_start_pos_captured: bool = False

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi2_fp16 = configuration.compute_kernel_config_hifi2_fp16

        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

        self.transformation_mats = transformation_mats
        self.is_sliding = (
            configuration.layer_types[layer_num] == "sliding_attention" if configuration.layer_types else False
        )
        self.sliding_window = configuration.sliding_window if self.is_sliding else None

        self.model_config = configuration.get_model_config()
        self.ccl_topology = configuration.ccl_topology()
        self.is_multichip = configuration.is_multichip
        self.activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        self.wqkv_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.WQKV
        )
        self.wo_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.WO
        )
        self.kv_cache_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.KV_CACHE
        )
        self.li_qkv_decode_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_DECODE, configuration=configuration
        )
        self.sdpa_decode_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_DECODE, configuration=configuration
        )
        self.li_o_decode_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_DECODE, configuration=configuration
        )
        self.sdpa_prefill_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_PREFILL, configuration=configuration
        )
        self.li_qkv_prefill_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_PREFILL, configuration=configuration
        )
        self.li_o_prefill_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_PREFILL, configuration=configuration
        )

        layer_name = configuration.get_state_dict_prefix(self.__class__.__name__, layer_num)
        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.wq"
        wk_str = f"{layer_name}.wk"
        wv_str = f"{layer_name}.wv"
        wo_str = f"{layer_name}.wo"
        q_norm_str = f"{layer_name}.q_norm"
        k_norm_str = f"{layer_name}.k_norm"

        # Initialize bias tensors as None
        self.wqkv_bias_decode = None
        self.wqkv_bias_prefill = None

        # Create combined QKV bias if present in state dict
        if f"{wq_str}.bias" in state_dict:
            qkv_bias = torch.concat(
                [
                    torch.concat(
                        [
                            torch.chunk(state_dict[f"{wq_str}.bias"], configuration.num_devices)[i],
                            torch.chunk(state_dict[f"{wk_str}.bias"], configuration.num_devices)[i],
                            torch.chunk(state_dict[f"{wv_str}.bias"], configuration.num_devices)[i],
                        ],
                        dim=-1,
                    )
                    for i in range(configuration.num_devices)
                ],
                dim=-1,
            )
            # Prefill can use broadcasting on the bias add so wants a 1d tensor
            self.wqkv_bias_prefill = ttnn.as_tensor(
                qkv_bias,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name("wqkv_bias_prefill_sharded"),
            )
            # as_tensor returns (32, dim) which is incorrect, this reshape updates the padded size to the correct size
            self.wqkv_bias_prefill = ttnn.reshape(
                self.wqkv_bias_prefill,
                (1, 1, 1, self.wqkv_bias_prefill.shape[-1]),
                (1, 1, self.wqkv_bias_prefill.shape[-2], self.wqkv_bias_prefill.shape[-1]),
            )

            # Broadcasting does not seem to be supported inside execute_trace so expand to the whole batch size
            # Create a list of bias tensors for each multiple of tile_size up to max_batch_size
            self.wqkv_bias_decode = []
            for batch_size in range(
                configuration.tile_size,
                configuration.tile_padded_batch_rows + configuration.tile_size,
                configuration.tile_size,
            ):
                qkv_bias_decode = qkv_bias.unsqueeze(0).expand(batch_size, -1)
                bias_tensor = ttnn.as_tensor(
                    qkv_bias_decode,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                    cache_file_name=cache_name(f"wqkv_bias_decode_sharded_{batch_size}"),
                )
                self.wqkv_bias_decode.append(bias_tensor)

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices_per_group == 0
        assert self.n_kv_heads % self.num_devices_per_group == 0
        assert configuration.qkv_size % self.num_devices_per_group == 0
        assert configuration.dim % self.num_devices_per_group == 0

        # wqkv: 4096 x 3072 (2 devices): width-sharded on 12 banks, 3072 over 12 banks.
        wqkv_mem_config = configuration.create_dram_sharded_mem_config(
            configuration.dim, configuration.qkv_size // configuration.num_devices
        )

        qkv_list = []
        for i in range(self.num_devices_per_group):
            # Chunk weights
            wq_selected = torch.chunk(state_dict[f"{wq_str}.weight"], self.num_devices_per_group, dim=0)[i]
            wk_selected = torch.chunk(state_dict[f"{wk_str}.weight"], self.num_devices_per_group, dim=0)[i]
            wv_selected = torch.chunk(state_dict[f"{wv_str}.weight"], self.num_devices_per_group, dim=0)[i]

            # Transpose the selected chunks
            wq = torch.transpose(wq_selected, -2, -1)
            wk = torch.transpose(wk_selected, -2, -1)
            wv = torch.transpose(wv_selected, -2, -1)

            qkv = torch.cat([wq, wk, wv], dim=-1)
            qkv_list.append(qkv)

        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        self.wqkv = ttnn.as_tensor(
            qkv_cat,
            dtype=self.wqkv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if self.TG else wqkv_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(3, 2) if self.TG else (2, 3), mesh_shape=configuration.cluster_shape
            ),
            cache_file_name=cache_name("wqkv_sharded_2d"),
        )

        def norm_reshard(x, norm, mode):
            """Hack until RMSNorm supports height-sharded output config"""
            if mode == "decode":
                mem_cfg = x.memory_config()
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=x.dtype)
            x = norm(x, mode)
            if mode == "decode":
                x = ttnn.to_memory_config(x, mem_cfg, dtype=x.dtype)
            return x

        if f"{q_norm_str}.weight" in state_dict:
            fn_q_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                eps=configuration.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,  # we already prefix q_norm_str
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=q_norm_str,
                add_unit_offset=self.rms_norm_add_unit_offset,
                is_distributed=False,
                sharded_program_config=None,  # FIXME: add height-sharded support. self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=None,  # FIXME: add height-sharded support. self.model_config["CREATE_QKV_DECODE_SHARD"]
                tt_ccl=self.tt_ccl,
            )
            self.q_norm = lambda x, mode: norm_reshard(x, fn_q_norm, mode)
        else:
            self.q_norm = lambda x, mode: x

        if f"{k_norm_str}.weight" in state_dict:
            fn_k_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                eps=configuration.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,  # we already prefix k_norm_str
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=k_norm_str,
                add_unit_offset=self.rms_norm_add_unit_offset,
                is_distributed=False,
                sharded_program_config=None,  # FIXME: add height-sharded support. self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=None,  # FIXME: add height-sharded support. self.model_config["CREATE_QKV_DECODE_SHARD"],
                tt_ccl=self.tt_ccl,
            )
            self.k_norm = lambda x, mode: norm_reshard(x, fn_k_norm, mode)
        else:
            self.k_norm = lambda x, mode: x
        # For ring topology we can use all gather matmul for wo
        self.use_fused_all_gather_matmul = self.model_config["USE_FUSED_ALL_GATHER_MATMUL"]
        pt_wo = state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        wo_mem_config = configuration.create_dram_sharded_mem_config(
            (configuration.n_heads * configuration.head_dim) // configuration.num_devices, configuration.dim
        )

        # Create wo tensor before L1 partitioning logic
        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=self.wo_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if (self.use_fused_all_gather_matmul or self.TG) else wo_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(2, 3) if (self.use_fused_all_gather_matmul or self.TG) else (3, 2),
                mesh_shape=configuration.cluster_shape,
            ),
            cache_file_name=(
                cache_name("wo_width_sharded_2d") if (self.use_fused_all_gather_matmul or self.TG) else cache_name("wo")
            ),
        )

        if not use_paged_kv_cache:
            # vLLM provides its own kv cache
            self.init_kv_cache(configuration, weight_cache_path)

        if configuration.query_pre_attn_scalar is not None:
            self.scale = configuration.query_pre_attn_scalar**-0.5
        else:
            self.scale = self.head_dim**-0.5

    def init_kv_cache(self, configuration, weight_cache_path):
        """
        Generates empty KV cache and pushed to device memory
        """

        if self.paged_attention_config:
            cache_k = torch.zeros(
                (
                    self.paged_attention_config.max_num_blocks,
                    self.n_local_kv_heads,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.paged_attention_config.max_num_blocks,
                    self.n_local_kv_heads,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            )
        else:
            cache_k = torch.zeros(
                (
                    self.batch_size_per_device_group,
                    self.n_local_kv_heads,
                    self.max_seq_len,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.batch_size_per_device_group,
                    self.n_local_kv_heads,
                    self.max_seq_len,
                    self.head_dim,
                )
            )

        self.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                dtype=self.kv_cache_dtype,
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=(
                    f"{weight_cache_path}/kvcache_{k_or_v.shape}"
                    if weight_cache_path and not configuration.dummy_weights
                    else None
                ),
            )
            for k_or_v in [cache_k, cache_v]
        ]

        if self.l1_kv_total_size > 0 and not self.paged_attention_config and not self.use_adaptive_l1_kv_cache:
            # Legacy fixed-window path: allocate exactly l1_kv_total_size tokens in L1 right now.
            # Skipped when use_adaptive_l1_kv_cache is on — the adaptive path manages its own
            # L1 layout via l1_kv_tiers + a sink prefix seeded from the DRAM cache after
            # prefill, independently of l1_kv_total_size.
            # Layout is chosen once here — no later to_memory_config() copy needed.
            shape = (
                self.batch_size_per_device_group,
                self.n_local_kv_heads,
                self.l1_kv_total_size,
                self.head_dim,
            )
            if self.l1_kv_use_sharded:
                l1_memcfg = self._create_l1_kv_sharded_memcfg(shape)  # HEIGHT_SHARDED
                if l1_memcfg is None:
                    logger.warning(
                        "[L1 KV] Could not build HEIGHT_SHARDED config; "
                        "falling back to L1_MEMORY_CONFIG (interleaved)."
                    )
                    l1_memcfg = ttnn.L1_MEMORY_CONFIG
            else:
                l1_memcfg = ttnn.L1_MEMORY_CONFIG  # plain interleaved

            l1_cache_k = torch.zeros(shape)
            l1_cache_v = torch.zeros_like(l1_cache_k)
            self.l1_kv_cache = [
                ttnn.as_tensor(
                    k_or_v,
                    dtype=self.kv_cache_dtype,
                    layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                    device=self.mesh_device,
                    memory_config=l1_memcfg,  # chosen once; no re-sharding at read time
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                for k_or_v in [l1_cache_k, l1_cache_v]
            ]
            self.l1_kv_sharded_memcfg = l1_memcfg if self.l1_kv_use_sharded else None
            layout_str = "HEIGHT_SHARDED" if self.l1_kv_use_sharded else "L1_MEMORY_CONFIG"
            logger.info(
                f"[L1 KV] Fixed-window cache allocated (layer head_dim={self.head_dim}): "
                f"{self.l1_kv_total_size} tokens, {layout_str}."
            )
        elif self.use_adaptive_l1_kv_cache and not self.paged_attention_config:
            # Adaptive N-tier path: defer until after decode compile so we can
            # measure per-core headroom. allocate_l1_kv_cache() will be called by the generator.
            self.l1_kv_cache = None
            self.l1_kv_sharded_memcfg = None
        else:
            self.l1_kv_cache = None
            self.l1_kv_sharded_memcfg = None

    def allocate_l1_kv_cache(
        self,
        headroom_map: dict,
        safety_margin_bytes: int = 64 * 1024,
        min_viable_tokens: int = 64,
    ):
        """
        Post-compile deferred allocation of the adaptive N-tier L1 KV cache.
        Called once by the generator after the decode compile step.

        Only handles use_adaptive_l1_kv_cache=True.
        The legacy fixed-window path (l1_kv_window_size > 0) is allocated
        eagerly in init_kv_cache() and never reaches this function.
        """
        if self.paged_attention_config:
            return
        if not self.use_adaptive_l1_kv_cache:
            return  # fixed-window path already allocated in init_kv_cache
        if self.l1_kv_cache is not None:
            return  # already allocated
        self._allocate_adaptive_l1_kv_tiers(headroom_map, safety_margin_bytes)

    def _allocate_adaptive_l1_kv_tiers(
        self,
        headroom_map: dict,
        safety_margin_bytes: int,
    ):
        """
        Non-uniform bucketed HEIGHT_SHARDED allocation.
        Each core class (by headroom) becomes an independent tier tensor.
        Tier i covers token range [token_start_i, token_start_i + token_count_i).
        All 130 cores participate; high-headroom cores store more tile-rows.
        """
        if self.l1_kv_tiers:
            return  # already allocated

        tiers = self._build_adaptive_l1_memcfg_tiers(headroom_map, safety_margin_bytes)
        if not tiers:
            logger.warning("[L1 KV adaptive] No viable tiers; L1 KV cache disabled for this layer.")
            return

        token_cursor = 0
        for memcfg, tile_rows_per_core, cores, tok_count in tiers:
            shape = (
                self.batch_size_per_device_group,
                self.n_local_kv_heads,
                tok_count,
                self.head_dim,
            )
            zeros = torch.zeros(shape)
            # The headroom measurement can over-report available L1 due to
            # fragmentation from existing L1 buffers.  Catch OOM here and
            # skip the tier rather than crashing.
            try:
                k_tensor = ttnn.as_tensor(
                    zeros,
                    dtype=self.kv_cache_dtype,
                    layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                    device=self.mesh_device,
                    memory_config=memcfg,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            except RuntimeError as e:
                logger.warning(
                    f"[L1 KV adaptive] OOM allocating K tier "
                    f"({tile_rows_per_core} tile-rows × {len(cores)} cores, {tok_count} tokens) — skipping. {e}"
                )
                continue
            try:
                v_tensor = ttnn.as_tensor(
                    zeros,
                    dtype=self.kv_cache_dtype,
                    layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                    device=self.mesh_device,
                    memory_config=memcfg,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            except RuntimeError as e:
                ttnn.deallocate(k_tensor)
                logger.warning(
                    f"[L1 KV adaptive] OOM allocating V tier "
                    f"({tile_rows_per_core} tile-rows × {len(cores)} cores, {tok_count} tokens) — skipping. {e}"
                )
                continue
            self.l1_kv_tiers.append((k_tensor, v_tensor, token_cursor, tok_count))
            logger.info(
                f"[L1 KV adaptive] Tier: {len(cores)} cores × {tile_rows_per_core} tile-rows "
                f"= {tok_count} tokens [tokens {token_cursor}..{token_cursor + tok_count - 1}], "
                f"{tile_rows_per_core * self.tile_size * self.head_dim * 2 // 1024} KiB/core."
            )
            token_cursor += tok_count

        # l1_kv_tiers: list of (k_tensor, v_tensor, token_start, tok_count)
        #   t[0] k_tensor    — HEIGHT_SHARDED K cache tensor in L1
        #   t[1] v_tensor    — HEIGHT_SHARDED V cache tensor in L1
        #   t[2] token_start — first global token index this tier covers
        #   t[3] tok_count   — number of tokens this tier holds
        total = sum(t[3] for t in self.l1_kv_tiers)  # t[3] = tok_count
        self.l1_kv_adaptive_total_capacity = total
        logger.info(f"[L1 KV adaptive] Total L1 KV tokens across {len(self.l1_kv_tiers)} tiers: {total}")

        # Sanity: sink_size must fit entirely in Tier 0 (the first tier in flat L1
        # layout). The ring math relies on slots [0, sink_size) being untouched by
        # decode-time ring writes; that requires Tier 0's tok_count >= sink_size.
        if self.l1_kv_sink_size > 0 and self.l1_kv_tiers:
            tier0_tok_count = self.l1_kv_tiers[0][3]
            if tier0_tok_count < self.l1_kv_sink_size:
                logger.warning(
                    f"[L1 KV adaptive] sink_size={self.l1_kv_sink_size} > Tier 0 tok_count="
                    f"{tier0_tok_count}; disabling sink for this run."
                )
                self.l1_kv_sink_size = 0

    def seed_adaptive_l1_sinks(self, dram_k_cache, dram_v_cache):
        """
        Seed the L1 adaptive tier's first sink_size slots with prefill K/V[0..sink_size).

        Approach: `ttnn.fill_cache` requires INTERLEAVED cache, but our adaptive
        tier is HEIGHT_SHARDED. So we instead RE-CREATE Tier 0 with the sink
        K/V baked into the host-side initial torch tensor:
          1. Slice DRAM K/V[..., 0:sink, ...] on device, lift to host via to_torch.
          2. Deallocate the existing Tier 0 K and V tensors.
          3. Build new torch tensors that are zeros everywhere except the first
             sink_size rows along the seq dim, which hold the prefill K/V.
          4. Re-allocate Tier 0 K and V using the same memory_config / dtype /
             layout as the original, with the seeded torch tensor as source.
          5. Update self.l1_kv_tiers[0] with the new tensors.

        The ring write path in `_build_adaptive_l1_write_pos` already skips slots
        [0, sink_size) when sink_size > 0, so the seeded sink K/V remains pinned
        for the duration of decode.

        One host sync per layer at warmup (acceptable one-time cost).
        """
        if self.l1_kv_sink_size <= 0 or not self.l1_kv_tiers:
            return
        k_tier0, v_tier0, token_start, tok_count = self.l1_kv_tiers[0]
        if token_start != 0:
            logger.warning(
                f"[L1 KV adaptive] sink seed: Tier 0 token_start={token_start} (expected 0); "
                f"sink would land in the wrong tier — skipping seed."
            )
            return
        sink = self.l1_kv_sink_size
        try:
            # Slice DRAM K/V[..., 0:sink, ...] on device. The slice is small
            # (one tile-row deep) so the host transfer is cheap.
            k_starts = [0] * len(dram_k_cache.shape)
            v_starts = [0] * len(dram_v_cache.shape)
            k_ends = list(dram_k_cache.shape)
            v_ends = list(dram_v_cache.shape)
            k_ends[2] = sink
            v_ends[2] = sink
            k_slice = ttnn.slice(dram_k_cache, k_starts, k_ends)
            v_slice = ttnn.slice(dram_v_cache, v_starts, v_ends)
            k_sink_torch = ttnn.to_torch(k_slice)  # shape [B*mesh?, K, sink, D]
            v_sink_torch = ttnn.to_torch(v_slice)
            ttnn.deallocate(k_slice)
            ttnn.deallocate(v_slice)

            # Capture the original memory_config and dtype before deallocation.
            tier_memcfg_k = k_tier0.memory_config()
            tier_memcfg_v = v_tier0.memory_config()
            tier_dtype = self.kv_cache_dtype
            tier_layout = self.model_config["ATTN_W_LAYOUT_TILE"]

            # Drop the existing zero-initialised tier tensors so we can recreate
            # them at the same allocator address (top-down allocator places this
            # tier deterministically once others have settled — but for safety
            # we DON'T rely on identical addresses; the L1 layout is recomputed
            # whenever a buffer is allocated).
            ttnn.deallocate(k_tier0)
            ttnn.deallocate(v_tier0)

            # Build the seeded torch tensors. The ReplicateTensorToMesh mapper
            # broadcasts a single batch tensor across all mesh devices, so we
            # build a [B_per_device, K, tok_count, D]-shaped torch tensor and
            # let the mapper handle replication.
            shape = (
                self.batch_size_per_device_group,
                self.n_local_kv_heads,
                tok_count,
                self.head_dim,
            )
            zeros_k = torch.zeros(shape)
            zeros_v = torch.zeros(shape)
            # k_sink_torch shape may include a mesh dim (e.g. [B*mesh, K, sink, D]);
            # since ReplicateTensorToMesh writes the same tensor on every device,
            # all mesh slices are identical and we can just use the first.
            if k_sink_torch.dim() == 4 and k_sink_torch.shape[0] == shape[0]:
                zeros_k[:, :, :sink, :] = k_sink_torch
                zeros_v[:, :, :sink, :] = v_sink_torch
            elif k_sink_torch.dim() == 4:
                # The mesh dim got concatenated into batch dim; take the first slice.
                zeros_k[:, :, :sink, :] = k_sink_torch[: shape[0]]
                zeros_v[:, :, :sink, :] = v_sink_torch[: shape[0]]
            else:
                logger.warning(
                    f"[L1 KV adaptive] Unexpected DRAM cache shape {k_sink_torch.shape} for sink seed; skipping."
                )
                return

            new_k = ttnn.as_tensor(
                zeros_k,
                dtype=tier_dtype,
                layout=tier_layout,
                device=self.mesh_device,
                memory_config=tier_memcfg_k,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            new_v = ttnn.as_tensor(
                zeros_v,
                dtype=tier_dtype,
                layout=tier_layout,
                device=self.mesh_device,
                memory_config=tier_memcfg_v,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            self.l1_kv_tiers[0] = (new_k, new_v, token_start, tok_count)
            logger.info(f"[L1 KV adaptive] Seeded {sink} sink slots from DRAM K/V into Tier 0.")
        except RuntimeError as e:
            logger.warning(f"[L1 KV adaptive] sink seed failed: {e}. Continuing without sinks.")
            self.l1_kv_sink_size = 0

    def _build_adaptive_l1_memcfg_tiers(
        self,
        headroom_map: dict,
        safety_margin_bytes: int,
        bank_allocatable_bytes: int = 1_470_080,  # Blackhole P150 main L1 region
        # 1024 KiB pad caps cumulative KV at ~1 tile-row, so model-side sharded
        # intermediates (e.g., the 63-core L-shape buffer the QKV/embedding path
        # produces) land high enough to stay above every program's CB top.
        # Reducing this lets KV grow but pushes those intermediates lower, where
        # they collide with whichever CB top is next-highest. Diagnostic
        # `[L1 per-core query]` log lines identify the offending buffer.
        runtime_pad_bytes: int = 32768,
    ) -> list:
        """
        Build one HEIGHT_SHARDED MemoryConfig per headroom tier.
        Returns list of (memcfg, tile_rows_per_core, sorted_cores, token_count) tuples,
        ordered from lowest to highest tile-rows.

        Two gates limit how much KV the tier set may claim:

        1. Per-core gate (physical headroom): for each core, the per-tile-row
           cost across all layers must fit in
           ``headroom_map[core] - safety_margin_bytes``. This determines the
           maximum ``tile_rows`` for that core's tier.

        2. Cumulative-depth gate (allocator algorithm-space): the tt-metal L1
           BankManager tracks a single shared address space per AllocatorID.
           Every sharded buffer — across ALL tiers and physically disjoint
           cores — consumes ``size_per_bank`` slots from that one space, so the
           sum across tiers must fit in
           ``bank_allocatable_bytes - runtime_pad_bytes - safety_margin_bytes``.
           Without this gate, per-core budgeting passes but cumulative depth
           overflows the algorithm, OOM cascades on later layers, and runtime
           intermediates land at addresses low enough to clash with CB regions
           of unrelated programs.

        Cumulative gate is applied greedily by token-efficiency
        (tokens-per-algo-byte): the most efficient tier (typically the column
        tier — most cores × moderate tile-rows) is kept first, then lower-rank
        tiers absorb whatever budget remains, shrinking their ``tile_rows`` as
        needed; a tier shrunk to zero is dropped.

        See research_codes/documents/l1_kv_cache_cache/per_core_validate_walkthrough.md §7
        for the failure mode that motivated the second gate.

        Cost formula (all 32 layers, K+V, bfloat8_b):
            bytes_per_tile_row = per_tile_bytes * DHt * num_layers * 2
            tile_rows_per_core = floor((H - safety_margin) / bytes_per_tile_row)
        """
        from collections import defaultdict

        # Per-tile bytes including dtype-specific storage overhead. For bfloat8_b
        # tt-metal stores 1024 mantissa bytes + 64 shared-exponent bytes per 32×32 tile,
        # i.e. +6.25% over the naive mantissa-only count. Ignoring this overhead causes
        # the budget to under-count by ~16 KiB/layer on the highest tier and triggers
        # an OOM cascade ~25 layers into a 32-layer model.
        if self.kv_cache_dtype == ttnn.bfloat8_b:
            per_tile_bytes = 1088
        elif self.kv_cache_dtype == ttnn.bfloat16:
            per_tile_bytes = self.tile_size * self.tile_size * 2
        else:
            per_tile_bytes = self.tile_size * self.tile_size * 2

        num_layers = getattr(self, "num_layers", 32)  # default 32 for Llama 3.1 8B
        DHt = self.head_dim // self.tile_size
        bytes_per_tile_row = per_tile_bytes * DHt * num_layers * 2

        # Build per-core tile-row capacity (floor, ignore safety margin).
        # Cores listed in self.l1_kv_avoid_cores are excluded entirely — these
        # are typically the model's "L-shape" intermediate buffer footprint
        # (QKV / embedding path), where placing KV tiers would push the
        # L-shape's allocator address up into program 49's CB region.
        avoid = set(self.l1_kv_avoid_cores) if self.l1_kv_avoid_cores else set()
        tier_cores: dict = defaultdict(list)  # tile_rows -> [(x,y), ...]
        skipped_for_avoidance = 0
        for (x, y), usable in headroom_map.items():
            if (x, y) in avoid:
                skipped_for_avoidance += 1
                continue
            net = usable - safety_margin_bytes
            if net <= 0:
                continue
            tile_rows = net // bytes_per_tile_row
            if tile_rows < 1:
                continue  # less than 1 tile-row: skip
            tier_cores[tile_rows].append((x, y))
        if skipped_for_avoidance > 0:
            logger.info(
                f"[L1 KV adaptive] Skipped {skipped_for_avoidance} cores in l1_kv_avoid_cores "
                f"(model-intermediate footprint)."
            )

        if not tier_cores:
            return []

        # Helper: compute tok_count for a tier given tile_rows and n_cores.
        # Returns 0 if no aligned tile-group fits (caller drops the tier).
        B_H = self.batch_size_per_device_group * self.n_local_kv_heads

        def _tok_count_for(tile_rows: int, n_cores: int) -> int:
            if tile_rows == 0 or n_cores == 0:
                return 0
            step = tile_rows // math.gcd(tile_rows, B_H)
            max_tok_tiles = (tile_rows * n_cores) // B_H
            tok_count_tiles = (max_tok_tiles // step) * step
            return tok_count_tiles * self.tile_size

        # ── Cumulative-depth gate ─────────────────────────────────────────────
        # The tt-metal L1 BankManager tracks a single shared address space per
        # AllocatorID. Each sharded allocation consumes one ``size_per_bank``
        # slot regardless of which physical cores it lives on, so the sum
        # across all tiers must stay under
        #     bank_allocatable - runtime_pad - safety_margin
        # Per tier, cumulative algo-space cost = bytes_per_tile_row × tile_rows
        # (definition: bytes_per_tile_row already includes the 64x for K+V × all
        # layers; multiplying by tile_rows gives the algo-space "depth" the tier
        # occupies across the whole 32-layer run).
        algo_budget = max(0, bank_allocatable_bytes - runtime_pad_bytes - safety_margin_bytes)

        # Initial candidate tiers: (tile_rows, sorted cores, tok_count, cumulative).
        initial = []
        for tile_rows, cores in sorted(tier_cores.items()):
            cores_sorted = sorted(cores)
            tok_count = _tok_count_for(tile_rows, len(cores_sorted))
            if tok_count == 0:
                continue
            cumulative = bytes_per_tile_row * tile_rows
            initial.append((tile_rows, cores_sorted, tok_count, cumulative))

        if not initial:
            return []

        # Shrink-all policy: rather than dropping tiers when budget is tight,
        # give every tier at least 1 tile-row, then grow each tier toward its
        # original size in token-efficiency order (most efficient first). This
        # preserves coverage on every core class — dropping a tier loses more
        # capacity than the algo-space slack it saves, especially when tier
        # core grids are disjoint.
        #
        # Precondition: algo_budget must accommodate N_tiers × bytes_per_tile_row
        # (one tile-row per tier). If not, runtime_pad_bytes is still too high —
        # surface explicitly so the user can tune it.

        # Filter tiers whose minimum (T=1) yields 0 tokens — those are unviable
        # at any size and skipped entirely.
        viable = [
            (tile_rows, cores_sorted, tok_count, cumulative)
            for (tile_rows, cores_sorted, tok_count, cumulative) in initial
            if _tok_count_for(1, len(cores_sorted)) > 0
        ]
        if not viable:
            return []

        min_total = bytes_per_tile_row * len(viable)
        if algo_budget < min_total:
            logger.warning(
                f"[L1 KV adaptive] algo_budget={algo_budget} B is below the minimum "
                f"needed for shrink-all ({len(viable)} tiers × {bytes_per_tile_row} B = "
                f"{min_total} B). Lower runtime_pad_bytes by ≥ {min_total - algo_budget} B "
                f"or accept that some tiers will be dropped."
            )
            # Fall back to greedy drop policy when shrink-all isn't feasible:
            # accept tiers in efficiency order until budget exhausted.
            ranked = sorted(viable, key=lambda t: -(t[2] / t[3]))
            accepted_fallback = []
            used = 0
            for tile_rows, cores_sorted, tok_count, cumulative in ranked:
                remaining = algo_budget - used
                if cumulative <= remaining:
                    accepted_fallback.append((tile_rows, cores_sorted, tok_count, cumulative))
                    used += cumulative
                elif bytes_per_tile_row <= remaining:
                    shrunk_T = remaining // bytes_per_tile_row
                    shrunk_T = min(tile_rows, shrunk_T)
                    shrunk_tok = _tok_count_for(shrunk_T, len(cores_sorted))
                    while shrunk_T > 0 and shrunk_tok == 0:
                        shrunk_T -= 1
                        shrunk_tok = _tok_count_for(shrunk_T, len(cores_sorted))
                    if shrunk_T > 0:
                        shrunk_cum = bytes_per_tile_row * shrunk_T
                        accepted_fallback.append((shrunk_T, cores_sorted, shrunk_tok, shrunk_cum))
                        used += shrunk_cum
                        logger.info(
                            f"[L1 KV adaptive] Shrink-all fallback: tier "
                            f"({len(cores_sorted)} cores) {tile_rows} → {shrunk_T} tile-rows."
                        )
                else:
                    logger.info(
                        f"[L1 KV adaptive] Shrink-all fallback: dropped tier "
                        f"({len(cores_sorted)} cores × {tile_rows} tile-rows) — "
                        f"no room in {remaining} B."
                    )
            logger.info(
                f"[L1 KV adaptive] Cumulative algo-space (fallback): {used}/{algo_budget} B "
                f"({100*used/algo_budget:.1f}%)."
            )
            accepted = accepted_fallback
        else:
            # Seed: every viable tier starts at T=1.
            current = {}  # cores_sorted (immutable key) -> [tile_rows, tok_count, cumulative]
            ranked = sorted(viable, key=lambda t: -(t[2] / t[3]))
            order = []  # preserve efficiency ranking for the grow loop
            used = 0
            for tile_rows, cores_sorted, _full_tok, _full_cum in ranked:
                cores_key = tuple(cores_sorted)
                t0_tok = _tok_count_for(1, len(cores_sorted))
                t0_cum = bytes_per_tile_row * 1
                current[cores_key] = {
                    "tile_rows": 1,
                    "tok_count": t0_tok,
                    "cumulative": t0_cum,
                    "ceiling_tile_rows": tile_rows,
                    "cores": cores_sorted,
                }
                used += t0_cum
                order.append(cores_key)

            # Grow loop: each pass tries to bump one tile-row to the most-efficient
            # tier that has not yet hit its ceiling. Stop when no further growth fits.
            grown = True
            while grown:
                grown = False
                for cores_key in order:
                    entry = current[cores_key]
                    if entry["tile_rows"] >= entry["ceiling_tile_rows"]:
                        continue
                    new_T = entry["tile_rows"] + 1
                    new_tok = _tok_count_for(new_T, len(entry["cores"]))
                    if new_tok == 0 or new_tok <= entry["tok_count"]:
                        # Alignment step prevents this size — skip
                        continue
                    delta_cum = bytes_per_tile_row * (new_T - entry["tile_rows"])
                    if used + delta_cum > algo_budget:
                        continue
                    entry["tile_rows"] = new_T
                    entry["tok_count"] = new_tok
                    entry["cumulative"] = bytes_per_tile_row * new_T
                    used += delta_cum
                    grown = True
                    # Restart the inner loop so the next bump again prefers the
                    # highest-efficiency tier that's still under ceiling.
                    break

            accepted = [
                (entry["tile_rows"], entry["cores"], entry["tok_count"], entry["cumulative"])
                for entry in current.values()
            ]
            logger.info(
                f"[L1 KV adaptive] Shrink-all: {len(accepted)} tiers seeded at T=1, "
                f"used {used}/{algo_budget} B ({100*used/algo_budget:.1f}%); "
                f"bank={bank_allocatable_bytes}, runtime_pad={runtime_pad_bytes}, "
                f"safety={safety_margin_bytes}."
            )

        # Re-sort accepted tiers from smallest tile_rows to largest, so the token
        # cursor in _allocate_adaptive_l1_kv_tiers grows monotonically (preserves
        # the existing contract that earlier tiers cover lower token indices).
        accepted.sort(key=lambda t: t[0])

        # Each tier gets its OWN L1 allocator (allocator_id = tier_index + 1) so it
        # stacks within its own cores' physical gap, independent of the other tiers.
        # A single shared KV allocator stacked all tiers' cumulative depth onto one
        # address line, dropping disjoint-core tier buffers below their cores' CB floor.
        # MUST equal kNumL1KvTierAllocators provisioned in
        # tt_metal/impl/allocator/l1_banking_allocator.cpp (allocator ids 1..K exist).
        MAX_KV_TIER_ALLOCATORS = 8
        result = []
        for tier_index, (tile_rows, cores_sorted, tok_count, _) in enumerate(accepted):
            # shard_shape: each core holds tile_rows tile-rows × head_dim columns
            shard_shape = [
                tile_rows * self.tile_size,  # shard height in elements
                self.head_dim,  # shard width (full head)
            ]
            core_range_set = self._cores_to_core_range_set(cores_sorted)
            shard_spec = ttnn.ShardSpec(
                core_range_set,
                shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            alloc_id = tier_index + 1
            if alloc_id > MAX_KV_TIER_ALLOCATORS:
                # Clamp to a provisioned id — get_allocator_from_id FATALs on id >=
                # num_allocators. Sharing an address line reintroduces clash risk for
                # this overflow tier; bump kNumL1KvTierAllocators if this ever fires.
                logger.warning(
                    f"[L1 KV adaptive] tier_index={tier_index} exceeds "
                    f"MAX_KV_TIER_ALLOCATORS={MAX_KV_TIER_ALLOCATORS}; reusing "
                    f"allocator_id={MAX_KV_TIER_ALLOCATORS} (shares an address line — clash risk)."
                )
                alloc_id = MAX_KV_TIER_ALLOCATORS
            memcfg = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                shard_spec,
                allocator_id=alloc_id,
            )
            result.append((memcfg, tile_rows, cores_sorted, tok_count))

        return result

    @staticmethod
    def _cores_to_core_range_set(cores: list) -> ttnn.CoreRangeSet:
        """
        Pack a sorted list of (x, y) logical core coordinates into a CoreRangeSet
        by merging horizontally contiguous runs within the same row.
        """
        from collections import defaultdict

        rows = defaultdict(list)
        for x, y in cores:
            rows[y].append(x)

        ranges = []
        for y, xs in sorted(rows.items()):
            xs = sorted(xs)
            run_start = xs[0]
            prev = xs[0]
            for x in xs[1:]:
                if x == prev + 1:
                    prev = x
                else:
                    ranges.append(
                        ttnn.CoreRange(
                            ttnn.CoreCoord(run_start, y),
                            ttnn.CoreCoord(prev, y),
                        )
                    )
                    run_start = x
                    prev = x
            ranges.append(
                ttnn.CoreRange(
                    ttnn.CoreCoord(run_start, y),
                    ttnn.CoreCoord(prev, y),
                )
            )
        return ttnn.CoreRangeSet(ranges)

    def _create_l1_kv_sharded_memcfg(self, shape):
        if self.l1_kv_total_size <= 0:
            return None
        total_rows = max(1, shape[0] * shape[1] * math.ceil(shape[2] / self.tile_size))
        max_cores = max(1, min(self.grid_size.x * self.grid_size.y, total_rows))
        grid_x = min(self.grid_size.x, max_cores)
        grid_y = max(1, min(self.grid_size.y, math.ceil(max_cores / grid_x)))
        return ttnn.create_sharded_memory_config_(
            shape,
            ttnn.CoreGrid(x=grid_x, y=grid_y),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
        )

    def _get_sdpa_l1_cache_tensors(self):
        # l1_kv_tiers: list of (k_tensor, v_tensor, token_start, tok_count)
        #   t[0]: K tensor  t[1]: V tensor  t[2]: token_start  t[3]: tok_count
        if self.l1_kv_tiers:
            ks = [t[0] for t in self.l1_kv_tiers]  # t[0]: K tensor
            vs = [t[1] for t in self.l1_kv_tiers]  # t[1]: V tensor
            meta = [(t[2], t[3]) for t in self.l1_kv_tiers]  # t[2]: token_start, t[3]: tok_count
            return ks, vs, meta, ()
        # Fixed-window path
        if self.l1_kv_cache is None:
            return None, None, None, ()
        # Tensor is already in the chosen layout (L1_MEMORY_CONFIG or HEIGHT_SHARDED),
        # decided at allocation time — no to_memory_config() copy needed here.
        return self.l1_kv_cache[0], self.l1_kv_cache[1], None, ()

    def _build_l1_update_pos(self, current_pos):
        """Returns current_pos if any L1 KV write is needed this step, else None."""
        if self.l1_kv_window_size > 0:
            return current_pos  # fixed-window: ring write active
        if self.l1_kv_tiers:
            return current_pos  # adaptive: ring write active
        return None

    def _build_adaptive_l1_write_pos(self, current_pos):
        """
        Compute the flat L1 write position for the adaptive N-tier ring-buffer.
        Returns a ttnn int32 tensor with the flat index in [0, T) where
        T = l1_kv_adaptive_total_capacity.

        Ring layout:
          [0, sink_size)            — attention sink (stable, wrapped last)
          [sink_size, T)            — recency ring of capacity (T - sink_size)

        NOTE: disabled when T == 0 (no tiers allocated yet).
        NOTE: not trace-compatible (returns None when called inside a trace);
              the caller must guard with l1_write_enabled.
        """
        T = self.l1_kv_adaptive_total_capacity
        if T == 0:
            return None
        ring_cap = T - self.l1_kv_sink_size
        if ring_cap <= 0:
            return None
        # On the first decode step that writes to the L1 ring, capture cur_pos as
        # decode_start_pos. The SDPA decode kernel uses this lower bound to avoid
        # reading L1 for positions that were generated during prefill (and thus
        # never written to L1). One host sync per attention layer at warmup —
        # acceptable cost, runs exactly once.
        if not self._l1_kv_decode_start_pos_captured:
            self._l1_kv_decode_start_pos = int(ttnn.to_torch(current_pos).view(-1)[0])
            self._l1_kv_decode_start_pos_captured = True
        orig_shape = current_pos.shape
        l1_pos = ttnn.to_layout(current_pos, ttnn.TILE_LAYOUT)
        l1_pos = ttnn.typecast(l1_pos, ttnn.float32)
        # ring: sink_size + (pos - sink_size) % ring_cap
        if self.l1_kv_sink_size > 0:
            shifted = ttnn.subtract(l1_pos, float(self.l1_kv_sink_size))
            ringed = ttnn.remainder(shifted, float(ring_cap))
            l1_pos = ttnn.add(ringed, float(self.l1_kv_sink_size))
        else:
            l1_pos = ttnn.remainder(l1_pos, float(T))
        l1_pos = ttnn.typecast(l1_pos, ttnn.int32)
        l1_pos = ttnn.to_layout(l1_pos, ttnn.ROW_MAJOR_LAYOUT)
        # Strip tile padding back to original shape
        padded_shape = l1_pos.shape
        slice_starts = [0] * len(padded_shape)
        slice_ends = list(padded_shape)
        for i in range(len(orig_shape)):
            slice_ends[-(i + 1)] = orig_shape[-(i + 1)]
        l1_pos = ttnn.slice(l1_pos, slice_starts, slice_ends)
        return l1_pos

    def _write_adaptive_l1_tiers(self, k_heads_l1, v_heads_l1, flat_l1_pos_tensor):
        """
        Dispatch the ring-buffer write to every tier WITHOUT a host sync.

        For each tier i with (token_start, tok_count), we compute an update_idxs
        tensor on device that holds either:
          * (flat_pos - token_start) when flat_pos is inside this tier's range, or
          * -1 (= UINT32_MAX when reinterpreted as uint32_t) otherwise.

        The paged_update_cache writer/reader kernels honor the -1 sentinel as
        "skip this write" — see
        ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/
        dataflow/{writer,reader}_update_cache_interleaved_start_id.cpp.
        So only the tier whose range contains flat_pos actually writes; the
        others issue a no-op. This removes the per-layer host sync that was the
        dominant per-token overhead.

        flat_l1_pos_tensor is a scalar int32 ttnn tensor (row-major) with value
        in [0, T_total). Trace-compatible (no host transfer).

        l1_kv_tiers: list of (k_tensor, v_tensor, token_start, tok_count)
          t[0]: K tensor  t[1]: V tensor  t[2]: token_start  t[3]: tok_count
        """
        if not self.l1_kv_tiers:
            return

        # Single-tier fast path: with one tier (token_start=0, tok_count=T_total),
        # flat_l1_pos_tensor IS the in-tier offset and is already in [0, T_total).
        # No per-tier compute required — pass it directly to paged_update_cache.
        # Eliminates 8 ttnn ops per layer (subtract/ge/lt/and/where/typecast/
        # to_layout/slice), which is the dominant per-step overhead in the
        # adaptive write path.
        if len(self.l1_kv_tiers) == 1:
            k_tensor, v_tensor, token_start, _ = self.l1_kv_tiers[0]
            if token_start == 0:
                ttnn.experimental.paged_update_cache(k_tensor, k_heads_l1, update_idxs_tensor=flat_l1_pos_tensor)
                ttnn.experimental.paged_update_cache(v_tensor, v_heads_l1, update_idxs_tensor=flat_l1_pos_tensor)
                return

        # Promote flat_pos to TILE/FP32 once — ttnn elementwise ops require
        # TILE_LAYOUT and floating-point math. The per-tier computation reuses
        # this single promoted tensor.
        pos_tile_fp32 = ttnn.to_layout(flat_l1_pos_tensor, ttnn.TILE_LAYOUT)
        pos_tile_fp32 = ttnn.typecast(pos_tile_fp32, ttnn.float32)

        orig_shape = flat_l1_pos_tensor.shape
        for k_tensor, v_tensor, token_start, tok_count in self.l1_kv_tiers:
            # offset_i = flat_pos - token_start (may be negative or out-of-range)
            off = ttnn.subtract(pos_tile_fp32, float(token_start))
            # in_range = (off >= 0) && (off < tok_count)
            ge = ttnn.ge(off, 0.0)
            lt = ttnn.lt(off, float(tok_count))
            in_range = ttnn.logical_and(ge, lt)
            # offset_or_skip: real offset when in range, sentinel -1 otherwise.
            # The kernel reads update_idxs as uint32_t and checks against
            # (uint32_t)-1 == 0xFFFFFFFF, which an int32 -1 satisfies.
            offset_or_skip_fp = ttnn.where(in_range, off, -1.0)
            # Convert back to int32 ROW_MAJOR — paged_update_cache validate() requires it.
            offset_int = ttnn.typecast(offset_or_skip_fp, ttnn.int32)
            offset_int_rm = ttnn.to_layout(offset_int, ttnn.ROW_MAJOR_LAYOUT)
            # Strip tile padding back to the original (post-slice) flat_pos shape.
            padded_shape = offset_int_rm.shape
            slice_ends = list(padded_shape)
            for i in range(len(orig_shape)):
                slice_ends[-(i + 1)] = orig_shape[-(i + 1)]
            slice_starts = [0] * len(padded_shape)
            offset_final = ttnn.slice(offset_int_rm, slice_starts, slice_ends)

            ttnn.experimental.paged_update_cache(k_tensor, k_heads_l1, update_idxs_tensor=offset_final)
            ttnn.experimental.paged_update_cache(v_tensor, v_heads_l1, update_idxs_tensor=offset_final)

            ttnn.deallocate(off)
            ttnn.deallocate(ge)
            ttnn.deallocate(lt)
            ttnn.deallocate(in_range)
            ttnn.deallocate(offset_or_skip_fp)
            ttnn.deallocate(offset_int)
            ttnn.deallocate(offset_int_rm)
            ttnn.deallocate(offset_final)

        ttnn.deallocate(pos_tile_fp32)

    def _make_l1_index_tensor(self, positions: torch.Tensor):
        return ttnn.from_torch(
            positions,
            device=self.mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _prefill_slice(self, tensor, start_idx, end_idx):
        if start_idx == 0 and end_idx == tensor.shape[2]:
            return tensor
        starts = [0] * len(tensor.shape)
        ends = list(tensor.shape)
        starts[2] = start_idx
        ends[2] = end_idx
        return ttnn.slice(tensor, starts, ends)

    def _prefill_write_l1_cache(self, k_fill_l1, v_fill_l1, seq_len, batch_idx):
        if self.l1_kv_cache is None or self.l1_kv_total_size <= 0:
            return

        if k_fill_l1.is_sharded():
            k_fill_interleaved = ttnn.sharded_to_interleaved(k_fill_l1, ttnn.DRAM_MEMORY_CONFIG)
        else:
            k_fill_interleaved = k_fill_l1
        if v_fill_l1.is_sharded():
            v_fill_interleaved = ttnn.sharded_to_interleaved(v_fill_l1, ttnn.DRAM_MEMORY_CONFIG)
        else:
            v_fill_interleaved = v_fill_l1

        k_fill_cache = ttnn.typecast(k_fill_interleaved, dtype=self.kv_cache_dtype)
        v_fill_cache = ttnn.typecast(v_fill_interleaved, dtype=self.kv_cache_dtype)

        if seq_len <= self.l1_kv_total_size:
            fill_k = self._prefill_slice(k_fill_cache, 0, seq_len)
            fill_v = self._prefill_slice(v_fill_cache, 0, seq_len)
            ttnn.fill_cache(self.l1_kv_cache[0], fill_k, batch_idx)
            ttnn.fill_cache(self.l1_kv_cache[1], fill_v, batch_idx)
            if fill_k is not k_fill_cache:
                ttnn.deallocate(fill_k)
            if fill_v is not v_fill_cache:
                ttnn.deallocate(fill_v)
            ttnn.deallocate(k_fill_cache)
            ttnn.deallocate(v_fill_cache)
            if k_fill_interleaved is not k_fill_l1:
                ttnn.deallocate(k_fill_interleaved)
            if v_fill_interleaved is not v_fill_l1:
                ttnn.deallocate(v_fill_interleaved)
            return

        # For long prompts, the existing fill op cannot directly populate a wrapped ring layout.
        # We still initialize the pinned sink rows so decode can immediately use them.
        if self.l1_kv_sink_size > 0:
            sink_k = self._prefill_slice(k_fill_cache, 0, self.l1_kv_sink_size)
            sink_v = self._prefill_slice(v_fill_cache, 0, self.l1_kv_sink_size)
            if sink_k.dtype != self.kv_cache_dtype:
                sink_k_cache = ttnn.typecast(sink_k, dtype=self.kv_cache_dtype)
                if sink_k is not k_fill_cache:
                    ttnn.deallocate(sink_k)
                sink_k = sink_k_cache
            if sink_v.dtype != self.kv_cache_dtype:
                sink_v_cache = ttnn.typecast(sink_v, dtype=self.kv_cache_dtype)
                if sink_v is not v_fill_cache:
                    ttnn.deallocate(sink_v)
                sink_v = sink_v_cache
            ttnn.fill_cache(self.l1_kv_cache[0], sink_k, batch_idx)
            ttnn.fill_cache(self.l1_kv_cache[1], sink_v, batch_idx)
            if sink_k is not k_fill_cache:
                ttnn.deallocate(sink_k)
            if sink_v is not v_fill_cache:
                ttnn.deallocate(sink_v)
        ttnn.deallocate(k_fill_cache)
        ttnn.deallocate(v_fill_cache)
        if k_fill_interleaved is not k_fill_l1:
            ttnn.deallocate(k_fill_interleaved)
        if v_fill_interleaved is not v_fill_l1:
            ttnn.deallocate(v_fill_interleaved)

    def forward_decode(
        self, x, current_pos, rot_mats=None, page_table=None, l1_update_pos=None, l1_write_enabled=True, kv_cache=None
    ):
        """
        x: (seq_len, 1, batch, dim)
        current_pos: (batch_size), current token position in the sequence for each user
        """

        ###
        # QKV matmuls
        # Use HiFi2 for DRAM-sharded matmuls as they are otherwise flop-bound on 12 cores with HiFi4
        ###

        xqkv_fused_sharded = ttnn.linear(
            x,
            self.wqkv,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.model_config["XQKV_DECODE_PROGCFG"],
            compute_kernel_config=self.li_qkv_decode_compute_kernel_cfg,
            dtype=self.ccl_dtype if self.TG else self.activation_dtype or ttnn.bfloat16,
        )

        # FIXME: File bug against dram-sharded matmuls with bias
        if self.wqkv_bias_decode:
            # select the bias tensor based on the number of tiles in the rows
            # WARNING: must not change the batch size between compiling and executing a trace
            num_tiles = int(math.ceil(xqkv_fused_sharded.shape[-2] / self.tile_size))
            xqkv_fused_sharded = xqkv_fused_sharded + self.wqkv_bias_decode[num_tiles - 1]

        ttnn.deallocate(x)
        xqkv_fused = tt_all_reduce(
            xqkv_fused_sharded,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            num_reduce_scatter_links=self.num_reduce_scatter_links,
            num_all_gather_links=self.num_all_gather_links,
            memory_config=self.model_config["QKV_OUT_GATHERED_MEMCFG"](list(self.mesh_device.shape)[1]),
            sharded=True,
            dtype=self.ccl_dtype,
            topology=self.ccl_topology,
        )

        if self.TG:
            # TODO: Slice the fused_query_key_value tensor get batch=8
            xqkv_fused = ttnn.matmul(
                self.slice_mat,
                xqkv_fused,
                dtype=ttnn.bfloat16,
                memory_config=self.model_config["CREATE_HEAD_INPUT_MEMCFG"],
            )
        else:
            # bfloat16 is required by nlp_create_qkv_heads_decode
            xqkv_fused = ttnn.sharded_to_interleaved(xqkv_fused_sharded, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)

        ttnn.deallocate(xqkv_fused_sharded)

        # Reshape such that true unpadded batch is tracked in shape
        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(
            xqkv_fused, (1, 1, self.batch_size_per_device_group, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3])
        )

        ###
        # Reshape and rotary embeddings
        ###
        (
            q_heads_pre_rot_1BQD,
            k_heads_pre_rot_1BKD,
            v_heads_1BKD,
        ) = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=self.model_config["CREATE_QKV_DECODE_SHARD"],
        )

        q_heads_pre_rot_1BQD = self.q_norm(q_heads_pre_rot_1BQD, mode="decode")
        k_heads_pre_rot_1BKD = self.k_norm(k_heads_pre_rot_1BKD, mode="decode")

        ttnn.deallocate(xqkv_fused)

        # Q Rotary Embeddings
        q_heads_1BQD = ttnn.experimental.rotary_embedding_llama(
            q_heads_pre_rot_1BQD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
        )

        # K Rotary Embeddings
        k_heads_1BKD = ttnn.experimental.rotary_embedding_llama(
            k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
        )

        ttnn.deallocate(q_heads_pre_rot_1BQD)
        ttnn.deallocate(k_heads_pre_rot_1BKD)

        ###
        # KV update
        ###
        if kv_cache:
            keys = kv_cache[0]
            values = kv_cache[1]
        else:
            keys = self.layer_past[0]
            values = self.layer_past[1]

        if self.l1_kv_cache is not None and not page_table and l1_write_enabled:
            with l1_kv_perf.timed("decode.l1_clone_path"):
                k_heads_l1 = ttnn.mul(k_heads_1BKD, 1.0)
                v_heads_l1 = ttnn.mul(v_heads_1BKD, 1.0)
        elif self.l1_kv_tiers and not page_table and l1_write_enabled:
            if self.l1_kv_only_mode:
                # L1-only mode skips DRAM K/V write (below), so k_heads_1BKD has
                # no other consumer. Pass it directly to the L1 ring write and
                # let the post-SDPA deallocate at the same site as before. Saves
                # 2 ttnn.mul ops per layer.
                k_heads_l1 = k_heads_1BKD
                v_heads_l1 = v_heads_1BKD
            else:
                with l1_kv_perf.timed("decode.l1_clone_path"):
                    k_heads_l1 = ttnn.mul(k_heads_1BKD, 1.0)
                    v_heads_l1 = ttnn.mul(v_heads_1BKD, 1.0)
        else:
            k_heads_l1 = None
            v_heads_l1 = None
        # k_heads, [seqlen, n_kv_heads, bsz, head_dim]
        # v_heads [seqlen, n_kv_heads, bsz, head_dim]
        # keys, [max_batch_size, n_kv_heads // configuration.num_devices, max_seq_len, head_dim]
        # L1-only inference mode: when the SDPA decode kernel only reads from the
        # L1 adaptive tiers (clamped cur_pos, no DRAM fallback), there's no
        # benefit to maintaining the DRAM K/V cache. Skipping the two
        # paged_update_cache writes per layer removes ~32 ms/token of pure
        # overhead. The DRAM tensors stay populated from prefill but go stale —
        # safe because the kernel never reads them in this mode.
        skip_dram_kv_write = self.l1_kv_only_mode and self.l1_kv_tiers and not page_table and l1_write_enabled
        if not skip_dram_kv_write:
            with l1_kv_perf.timed("decode.dram_kv_write"):
                ttnn.experimental.paged_update_cache(
                    keys, k_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
                )
                ttnn.experimental.paged_update_cache(
                    values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
                )

        if self.l1_kv_tiers and not page_table and l1_write_enabled:
            # Adaptive N-tier ring-buffer write. The model's prepare_inputs_decode
            # now pre-computes the flat ring write position for every step (once,
            # not 32×) and passes it in via l1_update_pos. Use it when available
            # to skip 9 ttnn ops/layer of redundant compute. Fallback to the
            # per-layer compute path for callers that don't pass it.
            with l1_kv_perf.timed("decode.adaptive_l1_kv_write"):
                if l1_update_pos is not None:
                    flat_pos = l1_update_pos
                    locally_built_flat_pos = False
                else:
                    flat_pos = self._build_adaptive_l1_write_pos(current_pos)
                    locally_built_flat_pos = True
                if flat_pos is not None:
                    self._write_adaptive_l1_tiers(k_heads_l1, v_heads_l1, flat_pos)
                    if locally_built_flat_pos:
                        ttnn.deallocate(flat_pos)
        elif self.l1_kv_cache is not None and not page_table and l1_write_enabled:
            l1_pos = l1_update_pos
            if l1_pos is None and self.l1_kv_window_size > 0:
                with l1_kv_perf.timed("decode.l1_index_path"):
                    orig_shape = current_pos.shape
                    l1_pos = ttnn.to_layout(current_pos, ttnn.TILE_LAYOUT)
                    l1_pos = ttnn.typecast(l1_pos, ttnn.float32)
                    l1_pos = ttnn.remainder(l1_pos, float(self.l1_kv_window_size))
                    if self.l1_kv_sink_size > 0:
                        l1_pos = ttnn.add(l1_pos, self.l1_kv_sink_size)
                    l1_pos = ttnn.typecast(l1_pos, ttnn.int32)
                    l1_pos = ttnn.to_layout(l1_pos, ttnn.ROW_MAJOR_LAYOUT)
                    padded_shape = l1_pos.shape
                    slice_starts = [0] * len(padded_shape)
                    slice_ends = list(padded_shape)
                    for i in range(len(orig_shape)):
                        slice_ends[-(i + 1)] = orig_shape[-(i + 1)]
                    l1_pos = ttnn.slice(l1_pos, slice_starts, slice_ends)
            if l1_pos is not None:
                with l1_kv_perf.timed("decode.l1_kv_write"):
                    ttnn.experimental.paged_update_cache(self.l1_kv_cache[0], k_heads_l1, update_idxs_tensor=l1_pos)
                    ttnn.experimental.paged_update_cache(self.l1_kv_cache[1], v_heads_l1, update_idxs_tensor=l1_pos)
                if l1_pos is not l1_update_pos:
                    ttnn.deallocate(l1_pos)

        sdpa_kwargs = {
            "cur_pos_tensor": current_pos,
            "scale": self.scale,
            "sliding_window_size": self.sliding_window,
            "program_config": self.model_config["SDPA_DECODE_PROGCFG"],
            "compute_kernel_config": self.sdpa_decode_compute_kernel_cfg,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "l1_sink_size": self.l1_kv_sink_size,
            "l1_min_expected_hit_ratio": self.l1_kv_min_expected_hit_ratio,
        }
        sharded_l1_tensors = ()
        # Stash any device-side intermediate we create for L1-only cur_pos
        # clamping so we can deallocate it after the SDPA call.
        clamped_cur_pos_to_free = None
        if self.l1_kv_tiers:
            # Adaptive N-tier path: pass all tier tensors + metadata to the extended SDPA op.
            ks, vs, tier_meta, sharded_l1_tensors = self._get_sdpa_l1_cache_tensors()
            if ks:
                sdpa_kwargs["l1_k_tensors"] = ks
                sdpa_kwargs["l1_v_tensors"] = vs
                sdpa_kwargs["l1_tier_token_starts"] = [m[0] for m in tier_meta]
                sdpa_kwargs["l1_tier_token_counts"] = [m[1] for m in tier_meta]
                # decode_start_pos = cur_pos at the first decode step. Captured once
                # in _build_adaptive_l1_write_pos before this kwargs block runs (the
                # write happens earlier in forward_decode). If still uncaptured here
                # we pass 0 — the kernel then treats every ring slot as fresh, which
                # only matches reality when decode_start_pos truly was 0 (no prefill).
                sdpa_kwargs["l1_decode_start_pos"] = self._l1_kv_decode_start_pos

                # L1-only inference mode: keep the real cur_pos_tensor (same
                # fast program-cache path as the hybrid mode), but flip a
                # kernel-side flag so the SDPA decode kernel internally clamps
                # cur_pos to (total_l1_tokens - 1) for its iteration logic.
                # That way the kernel iterates only L1 chunks (no DRAM reads)
                # without any Python clamping overhead. Also set
                # decode_start_pos = 0 so the fresh-window covers the full ring
                # (sinks were seeded, ring is decode-written).
                if self.l1_kv_only_mode and self.l1_kv_adaptive_total_capacity > 0:
                    sdpa_kwargs["l1_decode_start_pos"] = 0
                    sdpa_kwargs["l1_only_mode"] = True
        elif self.l1_kv_cache is not None:
            sdpa_l1_k, sdpa_l1_v, _, sharded_l1_tensors = self._get_sdpa_l1_cache_tensors()
            sdpa_kwargs["l1_k_tensor"] = sdpa_l1_k
            sdpa_kwargs["l1_v_tensor"] = sdpa_l1_v

        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)
        # In L1-only mode we aliased k_heads_l1 = k_heads_1BKD (no clone), so
        # the previous deallocate above already freed them — don't double-free.
        if k_heads_l1 is not None and k_heads_l1 is not k_heads_1BKD:
            ttnn.deallocate(k_heads_l1)
        if v_heads_l1 is not None and v_heads_l1 is not v_heads_1BKD:
            ttnn.deallocate(v_heads_l1)

        # NOTE: Varying the batch size will result in slightly different outputs.
        # For example, a prompt w/ 1 user vs, the same prompt repeated N times for N users, will produce different outputs
        # This is because the SDPA op in decode mode has different number of reductions depending on batch size
        # Which leads to slightly different outputs from attention (due to accumulated errors)
        if page_table:
            sdpa_kwargs["page_table_tensor"] = page_table
            with l1_kv_perf.timed("decode.sdpa_call"):
                attn_output_1G4D = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q_heads_1BQD, keys, values, **sdpa_kwargs
                )
        else:
            with l1_kv_perf.timed("decode.sdpa_call"):
                attn_output_1G4D = ttnn.transformer.scaled_dot_product_attention_decode(
                    q_heads_1BQD, keys, values, **sdpa_kwargs
                )

        ttnn.deallocate(q_heads_1BQD)
        for sharded_tensor in sharded_l1_tensors:
            ttnn.deallocate(sharded_tensor)
        if clamped_cur_pos_to_free is not None:
            for t in clamped_cur_pos_to_free:
                ttnn.deallocate(t)
        # (l1_kv_only_mode static-cur_pos path doesn't create intermediates.)

        attn_output_11BH = ttnn.to_memory_config(
            attn_output_1G4D,
            memory_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"](self.batch_size_per_device_group),
        )
        attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(
            attn_output_11BH,
            num_heads=self.n_local_heads,
        )
        ttnn.deallocate(attn_output_11BH)
        ttnn.deallocate(attn_output_1G4D)

        if self.use_fused_all_gather_matmul:
            attn_output_cat = ttnn.to_memory_config(
                attn_output_cat, self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"]
            )

            # Fused AGMM only valid for ring topology
            if self.ccl_topology == ttnn.Topology.Ring:
                _, dense_out_sharded = ttnn.experimental.all_gather_matmul_async(
                    attn_output_cat,
                    self.wo,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                    all_gather_core_grid_offset=(0, 4),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                    num_links=1,
                    memory_config_ag=self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"],
                    memory_config_mm=self.model_config["DECODE_RESIDUAL_MEMCFG"],
                    program_config=self.model_config["ATTN_ALL_GATHER_MATMUL_PROGCFG"],
                    compute_kernel_config=self.compute_kernel_config_hifi2,
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
            else:
                all_gather_output = ttnn.experimental.all_gather_async(
                    attn_output_cat,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                    num_links=1,
                    topology=self.ccl_topology,
                    memory_config=self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"],
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )

                dense_out_sharded = ttnn.linear(
                    all_gather_output,
                    self.wo,
                    memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
                    program_config=self.model_config["ATTN_ALL_GATHER_MATMUL_PROGCFG"],
                    compute_kernel_config=self.li_o_decode_compute_kernel_cfg,
                )

                ttnn.deallocate(all_gather_output)
            ttnn.deallocate(attn_output_cat)
            dense_out_sharded = ttnn.to_memory_config(dense_out_sharded, self.model_config["DECODE_RESIDUAL_MEMCFG"])
            return dense_out_sharded

        else:
            attn_output = tt_all_gather(
                attn_output_cat,
                self.mesh_device,
                self.tt_ccl,
                dim=2,
                cluster_axis=1,
                num_links=2,
                memory_config=self.model_config["GATHER_USERS_MEMCFG"](list(self.mesh_device.shape)[1]),
                sharded=True,
                # dtype=self.ccl_dtype,  # Running bf16 until we have SDPA output bfp8 df; otherwise we have two sharded to interleaved/interleaved to sharded conversions
            )
            if self.TG:
                attn_output = ttnn.to_memory_config(attn_output, ttnn.L1_MEMORY_CONFIG)
                # user_selection_matrix = [1, 1, 32, 128]
                # user_selection_matrix @ activation -> [1, 1, 32, 128] * [1, 1, 128, 2048] -> [1, 1, 32, 2048]
                attn_output = ttnn.matmul(
                    self.user_selection_matrix,
                    attn_output,
                    core_grid=ttnn.CoreGrid(y=4, x=8),
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )

            # TODO: Fix this once self.TG supports dram-sharded matmuls

            dense_out_sharded = ttnn.matmul(
                attn_output,
                self.wo,
                core_grid=ttnn.CoreGrid(y=4, x=8) if self.TG else None,
                program_config=self.model_config["ATTN_OUTPUT_PROGCFG"] if not self.TG else None,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b if self.TG else None,
                compute_kernel_config=self.li_o_decode_compute_kernel_cfg,
            )

            ttnn.deallocate(attn_output_cat)

            # All reduce
            dense_out_reduced = tt_all_reduce(
                dense_out_sharded,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=0,
                num_reduce_scatter_links=self.num_reduce_scatter_links,
                num_all_gather_links=self.num_all_gather_links,
                dim=0 if (self.TG and self.hidden_size < 8192) else 3,
                topology=self.ccl_topology,
                memory_config=(
                    (
                        self.model_config["SELF_OUT_REDUCE_SCATTER_MEMCFG"]
                        if self.hidden_size == 8192
                        else self.model_config["SELF_OUT_GATHERED_MEMCFG"](list(self.mesh_device.shape)[0])
                    )
                    if self.TG
                    else self.model_config["DECODE_RESIDUAL_MEMCFG"]
                ),
                sharded=True,
                dtype=self.ccl_dtype,
                use_composite=True if self.hidden_size == 8192 else False,
            )

            if not self.TG:
                dense_out_reduced = ttnn.to_memory_config(
                    dense_out_reduced, self.model_config["DECODE_RESIDUAL_MEMCFG"]
                )

            return dense_out_reduced

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        valid_seq_len=None,
        kv_cache=None,
    ):
        seq_len = x_11SH.shape[-2]
        valid_seq_len = seq_len if valid_seq_len is None else valid_seq_len
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            if seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {self.MAX_QKV_MM_SEQ_LEN}")
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

        xqkv_fused = ttnn.linear(
            x_11SH,
            self.wqkv,
            dtype=self.ccl_dtype if self.TG else self.activation_dtype or ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
            program_config=self.model_config["XQKV_PREFILL_PROGCFG"](seq_len),
        )

        # FIXME: surely ttnn.linear bias should work?
        if self.wqkv_bias_prefill is not None:
            xqkv_fused = xqkv_fused + self.wqkv_bias_prefill

        xqkv_fused = tt_all_reduce(
            xqkv_fused,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            num_reduce_scatter_links=self.num_reduce_scatter_links,
            num_all_gather_links=self.num_all_gather_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.ccl_dtype,
        )

        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        ttnn.deallocate(x_11SH)

        # split qkv into heads
        (
            q_heads_1QSD_pre_rot,
            k_heads_1KSD_pre_rot,
            v_heads_1VSD,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        q_heads_1QSD_pre_rot = self.q_norm(q_heads_1QSD_pre_rot, mode="prefill")
        k_heads_1KSD_pre_rot = self.k_norm(k_heads_1KSD_pre_rot, mode="prefill")

        ttnn.deallocate(xqkv_fused)

        ###
        # Rotary embeddings
        ###

        if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:  # Rotary embeddings require bfloat16 inputs
            q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(q_heads_1QSD_pre_rot)

        if k_heads_1KSD_pre_rot.dtype != ttnn.bfloat16:  # Rotary embeddings require bfloat16 inputs
            k_heads_1KSD_pre_rot = ttnn.typecast(k_heads_1KSD_pre_rot, dtype=ttnn.bfloat16)

        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(k_heads_1KSD_pre_rot)

        # Fill KV-Cache
        if kv_cache:
            keys_BKSD, values_BKSD = kv_cache[0], kv_cache[1]
        else:
            keys_BKSD, values_BKSD = self.layer_past[0], self.layer_past[1]
        k_heads_1KSD_8b = ttnn.typecast(k_heads_1KSD, dtype=keys_BKSD.dtype)
        ttnn.deallocate(k_heads_1KSD)

        # sharding k_fill to deal with update_cache memory limitation
        if seq_len >= self.min_kv_prefill_shard_seqlen and not self.TG and not page_table:
            k_fill = ttnn.interleaved_to_sharded(k_heads_1KSD_8b, self.model_config["KV_PREFILL_MEM_CFG"](seq_len))
        else:
            k_fill = k_heads_1KSD_8b

        v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=values_BKSD.dtype)

        ttnn.deallocate(v_heads_1VSD)

        # sharding v_fill to deal with update_cache memory limitation
        if seq_len >= self.min_kv_prefill_shard_seqlen and not self.TG and not page_table:
            v_fill = ttnn.interleaved_to_sharded(v_heads_1VSD_8b, self.model_config["KV_PREFILL_MEM_CFG"](seq_len))
        else:
            v_fill = v_heads_1VSD_8b

        if self.TG:
            k_fill = self.prefill_prepare_tensor_for_kv_cache(k_fill, user_id)
            v_fill = self.prefill_prepare_tensor_for_kv_cache(v_fill, user_id)
        if page_table:
            # In the case that the tokens have been padded along the seq len dimension, we need to fill the cache with the unpadded k/v values.
            # Assume that the page table does not have padding, so we can use it to get the unpadded page len.
            block_size = keys_BKSD.shape[2]
            # If chunked prefill, use chunk_page_table if given, otherwise use page_table.
            fill_page_table = chunk_page_table if chunk_page_table is not None else page_table

            page_len = fill_page_table.shape[1] * block_size
            k_fill_sliced = k_fill[:, :, :page_len, :] if page_len < k_fill.shape[2] else k_fill
            v_fill_sliced = v_fill[:, :, :page_len, :] if page_len < v_fill.shape[2] else v_fill
            ttnn.experimental.paged_fill_cache(keys_BKSD, k_fill_sliced, fill_page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(values_BKSD, v_fill_sliced, fill_page_table, batch_idx=user_id)
        else:
            with l1_kv_perf.timed("prefill.l1_clone_path"):
                k_fill_l1 = ttnn.mul(k_fill, 1.0)
                v_fill_l1 = ttnn.mul(v_fill, 1.0)
            ttnn.fill_cache(
                keys_BKSD,
                k_fill,
                user_id % self.batch_size_per_device_group,
            )
            ttnn.fill_cache(
                values_BKSD,
                v_fill,
                user_id % self.batch_size_per_device_group,
            )
            if self.l1_kv_cache is not None and (chunk_start_idx is None or chunk_start_idx == 0):
                with l1_kv_perf.timed("prefill.l1_fill_cache"):
                    self._prefill_write_l1_cache(
                        k_fill_l1, v_fill_l1, valid_seq_len, user_id % self.batch_size_per_device_group
                    )
            ttnn.deallocate(k_fill_l1)
            ttnn.deallocate(v_fill_l1)
        if seq_len >= self.min_kv_prefill_shard_seqlen and not self.TG and not page_table:
            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)

        # SDPA
        q_heads_1QSD_8b = ttnn.typecast(q_heads_1QSD, dtype=self.activation_dtype or ttnn.bfloat8_b)
        ttnn.deallocate(q_heads_1QSD)

        if chunk_start_idx is not None:
            if self.sliding_window is not None:
                raise NotImplementedError("Sliding window not supported for chunked prefill SDPA")
            attn_output_84SD = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q_heads_1QSD_8b,
                input_tensor_k=keys_BKSD,
                input_tensor_v=values_BKSD,
                page_table_tensor=page_table,
                chunk_start_idx=chunk_start_idx,
                compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
                program_config=self.model_config["SDPA_PROGCFG"](seq_len),
            )
        else:
            attn_output_84SD = ttnn.transformer.scaled_dot_product_attention(
                q_heads_1QSD_8b,
                k_heads_1KSD_8b,
                v_heads_1VSD_8b,
                is_causal=True,
                sliding_window_size=self.sliding_window,
                scale=self.scale,
                compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
                program_config=self.model_config["SDPA_PROGCFG"](seq_len),
            )

        # deallocate keys and values
        ttnn.deallocate(q_heads_1QSD_8b)
        ttnn.deallocate(k_heads_1KSD_8b)
        ttnn.deallocate(v_heads_1VSD_8b)

        attn_output_1QSD = ttnn.reshape(attn_output_84SD, [1, self.n_local_heads, -1, self.head_dim])

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_1QSD)
        # reshaping long sequence to matmul fit on device
        if seq_len > 1024:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // 1024, 1024, -1])

        # Non fused All Gather Matmul
        if self.use_fused_all_gather_matmul:  # is true for Ring topology
            attn_output_11SH = ttnn.experimental.all_gather_async(
                attn_output_11SH,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                topology=self.ccl_topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            compute_kernel_config=self.li_o_prefill_compute_kernel_cfg,
            dtype=self.activation_dtype or ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["WO_PREFILL_PROGCFG"](seq_len),
        )

        if seq_len > 1024:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        # Reduce-scatter
        if not self.use_fused_all_gather_matmul:
            output_11SH = tt_all_reduce(
                output_11SH,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=0,
                dim=0 if self.TG else 3,
                num_reduce_scatter_links=self.num_reduce_scatter_links,
                num_all_gather_links=self.num_all_gather_links,
                topology=self.ccl_topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.ccl_dtype,
            )

        return output_11SH

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        l1_update_pos=None,
        l1_write_enabled=True,
        chunk_page_table=None,
        chunk_start_idx=None,
        valid_seq_len=None,
        kv_cache=None,
    ):
        if mode == "prefill":
            return self.forward_prefill(
                x,
                rot_mats,
                user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                valid_seq_len=valid_seq_len,
                kv_cache=kv_cache,
            )
        else:
            return self.forward_decode(
                x,
                current_pos,
                rot_mats,
                page_table=page_table,
                l1_update_pos=l1_update_pos,
                l1_write_enabled=l1_write_enabled,
                kv_cache=kv_cache,
            )

    def prefill_prepare_tensor_for_kv_cache(self, key_or_value_layer, user_id):
        tensor_copy = ttnn.clone(key_or_value_layer)
        # key_or_value_layer.deallocate(True)
        # Get all tensors from multi-device tensor
        tensors = ttnn.get_device_tensors(tensor_copy)
        # Get only tensors from specific column chips
        # Get every 4th tensor starting from user_id // 8
        single_column_tensors = tensors[user_id // self.batch_size_per_device_group :: 4]
        # Create multi-device tensor
        multi_device_tensor = ttnn.combine_device_tensors(single_column_tensors)

        return multi_device_tensor

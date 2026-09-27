# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 it-assistant drafter model (EAGLE / Multi-Token-Prediction).

The drafter is a tiny Gemma4 text model (4 layers, hidden 1024) that proposes K
candidate tokens from a single backbone position. Each step:

    inputs_embeds = cat(target_embed(last_token), last_hidden)       # [.., 2*backbone] (scaled embed)
    h = pre_projection(inputs_embeds)                                # [.., hidden]
    for layer in 4 decoder layers:                                   # cross-attend
        h = layer(h, kv = target's last {sliding,full} layer KV)     #   into target KV
    h = norm(h)
    logits      = lm_head(h)                                         # next draft token (argmax)
                  # or, when use_ordered_embeddings (E2B): a Centroid Masked
                  # Embedding head over ~4096 candidates — see masked_embedding.py
    next_hidden = post_projection(h)                                 # recurrent hidden

The decoder layers are ordinary ``Gemma4DecoderLayer``s (MoE disabled) run in
decode mode with ``is_kv_shared=True``: they compute only Q (the K/V weights are
synthesized as zeros and discarded) and the SDPA attends into the *target's* KV
cache for that layer type. ``position_ids`` and the target KV are held fixed
across the K drafter steps — matching HF's
``SinglePositionMultiTokenCandidateGenerator``.

Reference: transformers ``Gemma4AssistantForCausalLM.forward`` and
``generation/candidate_generator.py:SinglePositionMultiTokenCandidateGenerator``.

Constraints (first cut):
  * batch = 1
  * the target must use UNBOUNDED sliding KV caches (``bounded_sliding_kv_cache``
    off) so the drafter's cross-attention reads absolute cache positions without
    a circular-buffer modulo (the assistant attention config doesn't carry one).
"""

import os
from typing import NamedTuple

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.activation_sharding import ActivationSharding
from models.demos.gemma4.tt.assistant.masked_embedding import Gemma4TTMaskedEmbedder
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.ccl import ccl_allgather
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.matmul_tuning import DecodeMatmulTuner
from models.demos.gemma4.tt.matmul_tuning import resolve as resolve_tuner
from models.demos.gemma4.tt.rms_norm import RMSNorm
from models.demos.gemma4.tt.weight_placement import WeightPlacement, place_as_tensor
from models.demos.gemma4.utils.general_utils import get_cache_file_name
from models.demos.gemma4.utils.substate import substate


class SplitLogits(NamedTuple):
    """Dense drafter logits produced as column-ordered parts, not one row.

    ``parts``: tuple of [1,1,rows,Ni] TILE tensors whose widths sum to the vocab,
    in vocab-column order. Produced when ``GEMMA4_LMHEAD_L1_COLS`` pins a leading
    slice of ``lm_head`` into L1 and leaves the rest in DRAM, so the step runs two
    matmuls instead of one.

    Consumers dispatch on this type exactly as they do on ``CmeLogits``
    (``spec_decode._logits_to_host`` / ``_argmax_last``), so the drafter-logits
    call sites stay untouched.

    **Combine by concatenating ROW_MAJOR, then one argmax** -- not argmax-per-part.
    Picking between per-part argmaxes needs each part's max VALUE, and ttnn.max
    measured 68.02 us at N=4096 (PERFORMANCE_TRAJECTORY §2.10), slower than the
    argmax itself; two of those would eat the saving. Untilizing each part drops
    the physical 32-row pad, so the concat moves one real row (512 KiB at the full
    vocab) -- see gemma4-12b/MEASUREMENT_RECORD.md §5.3.
    """

    parts: tuple

    def deallocate(self, force=True):
        # Same contract as CmeLogits.deallocate: ~10 spec_decode call sites and the
        # harness's fused body free their logits unconditionally after the argmax.
        for p in self.parts:
            p.deallocate(force)


def _inject_zero_kv_weights(state_dict, text_args):
    """Add zero k_proj/v_proj and a unit k_norm for every assistant layer.

    The assistant checkpoint stores no K/V projections (its layers are all
    KV-shared), but ``Gemma4DecoderLayer``'s attention loader expects a full
    fused QKV. We inject zeros for the K/V columns (the split discards them under
    ``is_kv_shared=True``) and a unit k_norm so the loader's unconditional
    ``k_norm.weight`` read succeeds. Mutates and returns ``state_dict``.
    """
    hidden = text_args.hidden_size
    for i in range(text_args.num_hidden_layers):
        cfg = Gemma4AttentionConfig(text_args, i)
        kv_size = cfg.num_key_value_heads * cfg.head_dim
        prefix = f"model.layers.{i}.self_attn"
        if f"{prefix}.k_proj.weight" not in state_dict:
            state_dict[f"{prefix}.k_proj.weight"] = torch.zeros((kv_size, hidden), dtype=torch.bfloat16)
        # Sliding (non-global) layers load a separate v_proj; global layers tie V=K.
        if not cfg.use_kv_tying and f"{prefix}.v_proj.weight" not in state_dict:
            state_dict[f"{prefix}.v_proj.weight"] = torch.zeros((kv_size, hidden), dtype=torch.bfloat16)
        if f"{prefix}.k_norm.weight" not in state_dict:
            state_dict[f"{prefix}.k_norm.weight"] = torch.ones((cfg.head_dim,), dtype=torch.bfloat16)
    return state_dict


class Gemma4AssistantModel:
    def __init__(
        self,
        mesh_device,
        assistant_args,
        target_model,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
        max_local_batch_size=1,
        weight_placement=None,
        matmul_tuner=None,
        activation_sharding=None,
    ):
        # Explicit decode matmul program configs (default off). See tt/matmul_tuning.py.
        self.mm = resolve_tuner(matmul_tuner)
        self.act_shard = (
            activation_sharding if activation_sharding is not None else ActivationSharding.from_env(mesh_device)
        )
        # A sharded in0 under ttnn's AUTOMATIC matmul config is a REGRESSION
        # (backbone 723 -> 731 us measured); with the tuned config, which clamps
        # in0_block_w to a divisor of the shard width, it is a 9% win (674 -> 613).
        # So chaining is only ever enabled together with the tuner — the same rule
        # WeightPlacement.sharded already follows.
        if self.act_shard.enabled and not self.mm.enabled:
            logger.info("[assistant] activation sharding needs the tuned matmul config — enabling it")
            self.mm = DecodeMatmulTuner(mesh_device, enabled=True, label="draft")
        # Where this drafter's weights live (DRAM by default). The policy object
        # accumulates a per-device byte total across every tensor below, so the
        # construction order below is also the priority order for the L1 budget:
        # decoder layers first, then the projections, then the CME head (whose
        # 128 MiB replicated embed_table is expected to fall back to DRAM).
        self.weight_placement = placement = (
            weight_placement if weight_placement is not None else WeightPlacement.from_env(label="draft")
        )
        if placement.sharded and not self.mm.enabled:
            logger.info("[placement:draft] l1_sharded requires the tuned matmul config; enabling it")
            self.mm = DecodeMatmulTuner(mesh_device, enabled=True, label="draft")
        if placement.enabled:
            placement.log_hardware_ceiling(mesh_device)
        self.mesh_device = mesh_device
        self.max_local_batch_size = max_local_batch_size
        self.args = assistant_args
        self.text_args = assistant_args.text_args
        self.target = target_model
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config
        self.backbone_hidden_size = assistant_args.backbone_hidden_size
        self.hidden_size = self.text_args.hidden_size
        self.vocab_size = self.text_args.vocab_size
        self.layer_types = list(self.text_args.layer_types)

        # E2B's assistant replaces the dense lm_head with a Centroid Masked
        # Embedding head (31B/12B assistants set this False and use lm_head).
        self.use_cme = bool(assistant_args.use_ordered_embeddings)

        tp = mesh_config.tp if mesh_config else 1
        is_mesh = hasattr(mesh_device, "shape")
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        # The drafter shares the target's per-layer-type RoPE caches (identical
        # head_dim / theta), so its Q is RoPE'd consistently with the cached K.
        self.rope_caches_2d = target_model.rope_caches_2d

        state_dict = _inject_zero_kv_weights(dict(state_dict), self.text_args)

        # Decoder layers (reuse the target's layer, MoE disabled, KV-shared).
        self.layers = []
        for i in range(self.text_args.num_hidden_layers):
            layer = Gemma4DecoderLayer(
                mesh_device=mesh_device,
                hf_config=self.text_args,
                state_dict=state_dict,
                layer_idx=i,
                ccl_manager=ccl_manager,
                dtype=dtype,
                tensor_cache_path=f"{tensor_cache_path}/layer_{i}" if tensor_cache_path else None,
                mesh_config=mesh_config,
                max_seq_len=self.text_args.max_seq_len,
                max_local_batch_size=max_local_batch_size,
                weight_placement=placement,
                matmul_tuner=self.mm,
                activation_sharding=self.act_shard,
            )
            self.layers.append(layer)

        # Final norm (model.norm)
        self.norm = RMSNorm(
            activation_sharding=self.act_shard,
            mesh_device=mesh_device,
            hf_config=self.text_args,
            state_dict=substate(state_dict, "model.norm"),
            tensor_cache_path=f"{tensor_cache_path}/final_norm" if tensor_cache_path else None,
            mesh_config=mesh_config,
            weight_placement=placement,
        )

        # pre_projection (2*backbone -> hidden) and post_projection (hidden ->
        # backbone) are small and kept replicated so hidden stays full-width
        # across TP (matching the layer norms / attention which expect full
        # hidden). lm_head (hidden -> vocab) is column-parallel on vocab and
        # all-gathered, mirroring the target.
        col_mapper = mesh_config.column_parallel(mesh_device) if tp > 1 else None

        def _linear(key, mapper, transpose=True):
            w = state_dict.get(key)
            if w is None:
                return None
            wt = w.transpose(-2, -1) if transpose else w
            wt = wt.unsqueeze(0).unsqueeze(0)
            eff_mapper = mapper if mapper is not None else (replicate if is_mesh else None)
            return place_as_tensor(
                placement,
                f"assistant/{key}",
                wt,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=eff_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, key.replace(".", "_")),
            )

        self.pre_projection = _linear("pre_projection.weight", None)
        self.post_projection = _linear("post_projection.weight", None)
        # CME mode computes only ~4096 of the 262144 logits, so it needs a
        # row-gatherable [V, H] copy of the output embedding instead of the dense
        # [H, V] lm_head — building both would waste 134 MB of DRAM.
        self.masked_embedding = None
        self.lm_head = None
        self.lm_head_l1 = None
        self.lm_head_l1_pc = None
        self.lm_head_l1_ckc = None
        self.lm_head_l1_ring = None  # (in0 spec, out spec) when the slice runs as a gather_in0 ring
        if self.use_cme:
            self.masked_embedding = Gemma4TTMaskedEmbedder(
                mesh_device=mesh_device,
                assistant_args=assistant_args,
                state_dict=state_dict,
                dtype=dtype,
                tensor_cache_path=tensor_cache_path,
                mesh_config=mesh_config,
                weight_placement=placement,
                matmul_tuner=self.mm,
            )
        else:
            # lm_head tied to the assistant's own embed_tokens when a separate
            # lm_head.weight isn't stored.
            lm_key = "lm_head.weight" if "lm_head.weight" in state_dict else "model.embed_tokens.weight"
            self.lm_head = _linear(lm_key, col_mapper)
            if self.lm_head is None:
                raise ValueError("Assistant checkpoint missing lm_head weights")
            self._split_lm_head_to_l1(mesh_device)
        if self.pre_projection is None or self.post_projection is None:
            raise ValueError("Assistant checkpoint missing pre_projection / post_projection weights")

        if placement.enabled:
            logger.info("\n" + placement.report())

    def _raw_token_embed(self, token_tt):
        """Target token embedding of a single token id -> [1,1,1,backbone] TILE.

        Uses the *scaled* embedding (``embed_tokens`` = raw table * sqrt(hidden)).
        HF's ``embed_tokens`` is a ``Gemma4TextScaledWordEmbedding`` that applies
        the ``sqrt(hidden)`` normalizer inside its forward, so the drafter input
        ``cat(get_input_embeddings()(token), hidden)`` carries the *scaled*
        embedding. Feeding the unscaled table (~62x too small) starves the
        ``pre_projection`` token branch and collapses drafter acceptance
        (measured 0.19 unscaled -> 1.44 scaled, matching the HF reference).
        """
        emb = self.target.embed_tokens(token_tt)
        if len(emb.shape) == 3:
            emb = ttnn.unsqueeze_to_4D(emb)
        return ttnn.to_layout(emb, ttnn.TILE_LAYOUT)

    def _split_lm_head_to_l1(self, mesh_device):
        """Pin a leading slice of ``lm_head`` into L1, leaving the rest in DRAM.

        ``GEMMA4_LMHEAD_L1_COLS`` = how many vocab columns to pin (0 = off, the
        default, which keeps the single-matmul path byte-for-byte as before).

        The head is the drafter's largest single op -- ``[1024, 262144]``, 512 MiB,
        1.411 ms, 47.6% of the step (MEASUREMENT_RECORD.md §7.0) -- but it cannot be
        pinned whole: at ``per_core_N = 75`` on 110 cores it charges 4800 KiB/bank
        against ~772 KiB available (§5.1.1, §5.2.1). Pinning a COLUMN SLICE is the
        one way to put any of it in L1, because width sharding splits exactly that
        axis (§5.4).

        **The grid is the device's full 11x10 = 110 cores, not the tuner's 8x8.** The
        per-bank charge is ``K x 32 x per_core_N x elem``, so the slice only fits if
        ``per_core_N`` is small, which needs many cores. Every grid helper here caps
        at ``max_x = max_y = 8`` (and ``DecodeMatmulTuner`` clamps the device grid to
        it, ``matmul_tuning.py:274``); ``_pick_grid(1210, 8, 8)`` returns 5x2 = 10
        cores and ``per_core_N = 121`` -- 7.9 MB/bank. With the cap lifted to the
        device grid the same helper returns 11x10 and ``per_core_N = 11``. The cap is
        lifted **for this call only**: raising it in the tuner would also regrid
        ``wqkv``-full (48 -> 72 cores) and ``post_projection`` (40 -> 60), shifting
        every other drafter matmul. ``derive_decode_1d_config`` already takes the
        caps as arguments, so the config it returns matches the shard exactly --
        which is what the validator's ``per_core_N == in1_shard_width_tiles`` needs.
        """
        cols = int(os.getenv("GEMMA4_LMHEAD_L1_COLS", "0"))
        if cols <= 0:
            return
        from models.demos.gemma4.tt.matmul_tuning import derive_decode_1d_config

        T = 32
        grid = mesh_device.compute_with_storage_grid_size()
        k = int(self.lm_head.shape[-2])
        vocab = int(self.lm_head.shape[-1])
        if cols % T or cols >= vocab:
            raise ValueError(
                f"GEMMA4_LMHEAD_L1_COLS={cols}: must be a multiple of {T} and leave a DRAM remainder (<{vocab})"
            )
        pc = derive_decode_1d_config(1, k, cols, max_x=grid.x, max_y=grid.y)
        cores = grid.x * grid.y
        if pc is None or pc.compute_with_storage_grid_size.x * pc.compute_with_storage_grid_size.y != cores:
            # _pick_grid takes the largest rectangle whose core count divides Nt; if
            # that is smaller than the full grid, per_core_N -- and the charge --
            # grow by the same factor, so refuse rather than silently overflow.
            raise ValueError(
                f"GEMMA4_LMHEAD_L1_COLS={cols}: Nt={cols // T} is not divisible by the {cores}-core "
                f"grid; use a multiple of {T * cores} (e.g. {T * cores * 11} or {T * cores * 3})"
            )
        # PRECISION-NEUTRAL, not just grid-correct. matmul_device_operation.cpp:2688 raises
        # the default math fidelity to HiFi2 ONLY when no program_config is passed; with one,
        # it silently drops to LoFi. The unsplit head passes none, so it runs HiFi2 with the
        # automatic in0_block_w -- and MEASURED on this slice, the explicit config at LoFi /
        # in0_block_w=8 is 2.9x noisier against an fp32 reference (1.90e-2 vs 6.53e-3).
        # (in0_block_w=2, HiFi2, packer_l1_acc, no fp32 dest) reproduces the automatic path
        # BIT-FOR-BIT on the same slice (test_lm_head_split asserts it), so the pinned
        # columns compute exactly what the unsplit head did and the split changes placement
        # and nothing else. in0_block_w matters because it decides which K-sums the FPU adds in dest
        # and which the packer adds into a bf16 L1 partial (tt-metal-concepts.md §1.10).
        kt = k // T
        blk = 2 if kt % 2 == 0 else 1
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=pc.compute_with_storage_grid_size,
            in0_block_w=blk,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=pc.per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        per_core_n = pc.per_core_N
        charge = k * T * per_core_n * self.lm_head.element_size()
        logger.info(
            f"[lm_head-split] pinning {cols}/{vocab} cols ({cols/vocab*100:.1f}%) on "
            f"{grid.x}x{grid.y}={cores} cores, per_core_N={per_core_n}, in0_block_w={pc.in0_block_w}, "
            f"charge={charge} B/bank ({charge/1024:.0f} KiB)"
        )
        head = ttnn.slice(self.lm_head, [0, 0, 0, 0], [1, 1, k, cols])
        tail = ttnn.slice(self.lm_head, [0, 0, 0, cols], [1, 1, k, vocab])
        self.lm_head.deallocate(True)
        self.lm_head = tail  # the DRAM remainder keeps the original attribute
        shard = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}),
            [k, cols // cores],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        pinned = ttnn.to_memory_config(
            head, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard)
        )
        head.deallocate(True)
        if os.getenv("GEMMA4_LMHEAD_L1_GATHER", "0") == "1":
            # gather_in0 ring on the same 110 cores, as down_proj's ring. Kt = 32 < 110, so K is
            # sharded UNEVENLY: 1 tile on the first 32 cores, none on the rest (the factory carries
            # per-core unpadded_in0_shard_widths for this). The factory forces in0_block_w to the
            # shard width (1), so this arm is NOT bit-exact with the unsplit head.
            in0_w = -(-kt // cores) * T
            crs = shard.grid

            def _wsh(shape):
                return ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(crs, shape, ttnn.ShardOrientation.ROW_MAJOR),
                )

            pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=pc.compute_with_storage_grid_size,
                in0_block_w=in0_w // T,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
                gather_in0=True,
            )
            self.lm_head_l1_ring = (_wsh([T, in0_w]), _wsh([T, cols // cores]))
            logger.info(f"[lm_head-split] gather_in0 ring: in0 shard {in0_w // T} tile(s)/core on {cores} cores")
        self.lm_head_l1, self.lm_head_l1_pc, self.lm_head_l1_ckc = pinned, pc, ckc

    def step(self, token_tt, target_hidden, shared_kv, page_tables, pos_uint32, pos_int32, return_logits=True):
        """One drafter step.

        Args:
            token_tt: [1,1] uint32 last token id.
            target_hidden: [1,1,1,backbone] TILE — the recurrent hidden (target's
                last-token hidden on the first step, then this method's previous
                ``next_hidden``).
            shared_kv: {layer_type: [k_cache, v_cache]} target caches.
            page_tables: {layer_type: page_table} (or a single page_table reused
                for both types in the simple unbounded case).
            pos_uint32: [1,32] uint32 fixed position for RoPE lookup.
            pos_int32: [1] int32 fixed position for SDPA cur_pos.
            return_logits: when False, skip the output head + its TP all-gather
                and return ``(None, next_hidden)`` (used to isolate the
                lm_head/CCL cost in timing harnesses).

        Returns:
            (logits, next_hidden [1,1,1,backbone]). ``logits`` is a dense
            [1,1,rows,vocab] tensor normally, or a compact ``CmeLogits`` pair
            under Centroid Masked Embedding (E2B). ``spec_decode`` dispatches on
            the type in ``_logits_to_host`` / ``_argmax_last``.
        """
        tok_embed = self._raw_token_embed(token_tt)
        inp = ttnn.concat([tok_embed, target_hidden], dim=-1)
        tok_embed.deallocate(True)

        h = self.mm.linear(inp, self.pre_projection)
        inp.deallocate(True)

        for i, layer in enumerate(self.layers):
            lt = self.layer_types[i]
            pt = page_tables[lt] if isinstance(page_tables, dict) else page_tables
            h = layer(
                h,
                rope_mats=self.rope_caches_2d[lt],
                position_idx=pos_uint32,
                page_table=pt,
                kv_cache=shared_kv[lt],
                is_decode=True,
                token_index=None,
                is_kv_shared=True,
                position_idx_cache=pos_int32,
            )

        normed = self.norm.forward(h)
        # The CME head (topk / transpose / embedding) and post_projection both
        # want an interleaved operand; one conversion serves both, and it
        # replaces the S2I the norm used to do unconditionally.
        normed = self.act_shard.from_stream(normed)  # from_stream = back to interleaved
        h.deallocate(True)

        logits = None
        if return_logits:
            if self.use_cme:
                # Replicated weights, ~4096-wide output: no all-gather needed.
                logits = self.masked_embedding.forward(normed)
            elif self.lm_head_l1 is not None:
                # Split head: the pinned L1 columns and the DRAM remainder, in vocab
                # order. Kept as parts rather than concatenated here -- see SplitLogits.
                if self.lm_head_l1_ring is not None:
                    in0_spec, out_spec = self.lm_head_l1_ring
                    xs = ttnn.to_memory_config(normed, in0_spec)
                    ring_out = ttnn.linear(
                        xs,
                        self.lm_head_l1,
                        program_config=self.lm_head_l1_pc,
                        memory_config=out_spec,
                        compute_kernel_config=self.lm_head_l1_ckc,
                    )
                    xs.deallocate(True)
                    head = ttnn.sharded_to_interleaved(ring_out, ttnn.DRAM_MEMORY_CONFIG)
                    ring_out.deallocate(True)
                else:
                    head = ttnn.linear(
                        normed,
                        self.lm_head_l1,
                        program_config=self.lm_head_l1_pc,
                        compute_kernel_config=self.lm_head_l1_ckc,
                    )
                tail = ttnn.linear(normed, self.lm_head)
                if self.mesh_config is not None and self.mesh_config.tp > 1:
                    head = ccl_allgather(head, self.mesh_config, self.ccl_manager)
                    tail = ccl_allgather(tail, self.mesh_config, self.ccl_manager)
                logits = SplitLogits(parts=(head, tail))
            else:
                logits = ttnn.linear(normed, self.lm_head)
                if self.mesh_config is not None and self.mesh_config.tp > 1:
                    logits = ccl_allgather(logits, self.mesh_config, self.ccl_manager)

        next_hidden = self.mm.linear(normed, self.post_projection)
        normed.deallocate(True)
        return logits, next_hidden

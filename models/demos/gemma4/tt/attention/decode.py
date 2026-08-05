# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Decode-mode attention forward pass for Gemma4.

Uses HF-style ttnn.experimental.rotary_embedding (no transformation matrices).
"""

import os

from loguru import logger

import ttnn

from .operations import (
    apply_allreduce,
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    apply_rope_decode_peruser,
    concat_heads,
    effective_block_size,
    split_qkv_heads_decode,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights

_PV_SDPA_FP = os.environ.get("GEMMA4_PV_SDPA_FP") == "1"
# Shape/config keys already warmed up this process (see the WARM-UP note in
# _packed_verify_sdpa). GEMMA4_PV_WARMUP=0 disables it for A/B.
_PV_WARMED = set()
# Default 0: the warm-up was MEASURED not to fix the cross-process fork (still 2
# variants over 6 processes at GEMMA4_PV_WARMUP=3), so it is off by default and
# kept only as an A/B knob. See the WARM-UP note in _packed_verify_sdpa.
_PV_WARMUP_ITERS = int(os.environ.get("GEMMA4_PV_WARMUP", "0"))

# Cache for the L1 height-sharded ``MemoryConfig`` that
# ``nlp_create_qkv_heads_decode`` produces. The spec only depends on shape
# constants (B, num_heads_local, head_dim, qkv_dim, kv_replicated/global flags)
# — all fixed for a given model+batch. By caching across calls we avoid the
# per-layer probe decode-split that exists ONLY to discover this spec.
# Populated on the first (un-traced compile) call; inside trace capture the
# probe is skipped entirely.
_Q_SHARDED_MEM_CACHE: dict = {}


def _q_sharded_mem_key(B, qkv_dim, config, weights, tp):
    """Hashable key for the q_sharded_mem cache. Captures every input that
    affects ``nlp_create_qkv_heads_decode``'s output shard spec."""
    return (
        int(B),
        int(qkv_dim),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(config.head_dim),
        bool(weights.is_global),
        bool(weights.kv_replicated),
        int(tp),
    )


def decode_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    kv_cache,
    config,
    mesh_config,
    mesh_device,
    position_idx,
    token_index,
    page_table=None,
    ccl_manager=None,
    is_kv_shared=False,
    position_idx_cache=None,
    sequential_kv_write=False,
    rope_presliced=False,
):
    """
    Single-token decode attention, fully on device.

    Args:
        hidden_states: [1, 1, batch, hidden_size] on device
        cos_cache: [max_seq_len, head_dim] 2D cache for embedding lookup, or [1,1,max_seq_len,head_dim] 4D
        sin_cache: same format as cos_cache
        rope_presliced: if True, cos_cache/sin_cache are already position-gathered
            [1, 1, batch_pad, head_dim] tensors (one row per user), shared across all
            layers of this layer_type by the model. The per-layer ttnn.embedding slice
            is skipped and Q+K are RoPE'd in a single fused call. (Consumed in the
            fused-RoPE branch.)
        weights: AttentionWeights container
        kv_cache: [k_cache, v_cache] TT tensors (for shared layers, this is the source layer's cache)
        config: Gemma4AttentionConfig
        mesh_config: MeshConfig
        mesh_device: TT device
        position_idx: [batch] tensor of current positions for KV cache update + RoPE embedding lookup
        token_index: int position for legacy RoPE slicing (unused when cos_cache is 2D)
        page_table: optional paged attention table
        ccl_manager: optional CCL manager for TP > 1
        is_kv_shared: if True, skip K/V projection and cache update (use source layer's KV cache)
    """
    tp = mesh_config.tp if mesh_config else 1

    # 1. Fused QKV projection
    xqkv = apply_qkv_projection(hidden_states, weights)

    # 2. Split into Q, K, V heads
    tt_q, tt_k, tt_v = split_qkv_heads_decode(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
    )

    # 3. Per-head norms (move to DRAM for rms_norm, restore sharded for RoPE)
    q_sharded_mem = tt_q.memory_config()
    tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)

    if is_kv_shared:
        # KV-shared layer: discard own K/V, use source layer's KV cache directly
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    else:
        tt_k = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
        tt_v = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)

    # 4. RoPE — use on-device embedding lookup for trace compatibility
    # use_embedding_rope: cos/sin are per-position [1,1,batch_pad,head_dim] tensors.
    #   - rope_presliced: gathered once per layer_type by the model and shared across
    #     all layers (no per-layer ttnn.embedding here).
    #   - 2D cache: gather the position rows with ttnn.embedding right here (legacy
    #     per-layer path, kept for the it-assistant drafter / direct callers).
    use_embedding_rope = rope_presliced or len(cos_cache.shape) == 2
    if use_embedding_rope:
        if rope_presliced:
            cos_pos, sin_pos = cos_cache, sin_cache  # [1, 1, batch_pad, head_dim], shared
        else:
            # Gather position-specific cos/sin via ttnn.embedding (fully on-device, trace-safe)
            # position_idx: [1, 32] uint32 padded tensor for embedding lookup
            cos_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT))
            sin_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT))
        # RoPE. batch=1 uses the fused single-position rotary_embedding (one core
        # but cheap, no slice/tilize churn). batch>1 needs per-user positions,
        # which that op can't express, so fall back to the manual elementwise
        # q*cos + rotate_half(q)*sin (numerically equivalent — isolation PCC
        # ~0.99999 vs the fused op and the HF reference — but a few ops costlier).
        batch = tt_q.shape[1]
        if batch > 1:
            cos_b = ttnn.transpose(cos_pos, 1, 2)[:, :batch, :, :]  # [1, batch, 1, head_dim]
            sin_b = ttnn.transpose(sin_pos, 1, 2)[:, :batch, :, :]

        def _rope(t):
            if batch == 1:
                return apply_rope(t, cos_pos, sin_pos, token_index=0)
            return apply_rope_decode_peruser(t, cos_b, sin_b)

        # Rotate Q (and K, unless this is a KV-shared layer) with the shared
        # cos/sin. A concat(Q,K)->rope->split "fusion" was tried to collapse the
        # two rotary_embedding calls into one, but at decode batch=1 under metal
        # trace replay it regressed throughput (~3%): host dispatch is already
        # free under replay, so it only added concat+split device kernels while
        # removing one tiny rope kernel. Keep separate rotations.
        tt_q = _rope(tt_q)
        if not is_kv_shared:
            tt_k = _rope(tt_k)
    else:
        # Legacy path: full 4D cache with Python int token_index
        tt_q = apply_rope(tt_q, cos_cache, sin_cache, token_index=token_index)
        if not is_kv_shared:
            tt_k = apply_rope(tt_k, cos_cache, sin_cache, token_index=token_index)

    # 5. KV cache update — skip for KV-shared layers (source layer already updated the cache)
    # Use position_idx_cache (int32) for cache ops when position_idx is uint32 (embedding lookup format)
    cache_pos = position_idx_cache if position_idx_cache is not None else position_idx
    # Bounded sliding-window cache: when set, the op wraps absolute positions into a
    # circular buffer of ``cache_position_modulo`` tokens before the page_table lookup.
    # ``None`` = legacy unbounded behavior (set on full-attention layers or when the
    # bounded-cache mode isn't wired up). Empty kwargs dict on the legacy path keeps
    # kernel signatures and program-cache keys for existing callers byte-identical.
    paged_modulo_kwargs = (
        {"cache_position_modulo": config.cache_position_modulo} if config.cache_position_modulo is not None else {}
    )
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        if not is_kv_shared:
            # After HF-style RoPE, tensors may be in DRAM. Move to HEIGHT_SHARDED for cache update.
            tt_k = ttnn.to_memory_config(tt_k, q_sharded_mem)
            tt_v = ttnn.to_memory_config(tt_v, q_sharded_mem)

            if page_table is not None:
                # Per-device kv-head count of the layer's input view. When the cache
                # was allocated for a different layer type under HMA cross-group
                # sharing (Gemma4-26B-A4B sliding kv=8 / full kv=2 on multi-device
                # TP) cache.padded_shape[1] disagrees with what the kernel needs to
                # write — see paged_update_cache num_kv_heads kwarg. Mirrors
                # split_qkv_heads_decode's local head count.
                num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
                eff_bs = effective_block_size(k_cache, config.head_dim, num_local_kv_heads)
                batch = tt_k.shape[1]
                if sequential_kv_write and batch > 1:
                    # Speculative VERIFY: the B candidates sit at consecutive
                    # positions that share ONE paged block (all batch rows of the
                    # page table point to the same physical blocks). A single
                    # batched paged_update_cache then has multiple batch entries
                    # read-modify-write the SAME block tile concurrently and race
                    # (only some writes survive -> corrupt KV). Writing each
                    # position with its own ordered op serializes the block RMW.
                    # Matmuls + SDPA still run once on the full batch, so the
                    # verify speedup is preserved (only these tiny writes loop).
                    # paged_update_cache wants a sharded update tensor with
                    # num_shards == num_users. Slice each candidate to a batch=1
                    # [1,1,nkv,hd] tensor (from DRAM, where slicing is clean) and
                    # reshard onto a single core. q_sharded_mem shards one user per
                    # core (shard shape = [32, head_dim], head_dim shared by Q/K),
                    # so a 1-core config with that same shard shape is exactly the
                    # per-user layout the op expects.
                    _shard_shape = list(q_sharded_mem.shard_spec.shape)
                    _one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
                    single_user_mem = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(_one_core, _shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
                    )
                    k_seq = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
                    v_seq = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
                    nkv, hd = k_seq.shape[2], k_seq.shape[3]
                    for b in range(batch):
                        kb = ttnn.slice(k_seq, [0, b, 0, 0], [1, b + 1, nkv, hd])
                        vb = ttnn.slice(v_seq, [0, b, 0, 0], [1, b + 1, nkv, hd])
                        kb = ttnn.to_memory_config(kb, single_user_mem)
                        vb = ttnn.to_memory_config(vb, single_user_mem)
                        pos_b = ttnn.slice(cache_pos, [b], [b + 1])
                        pt_b = ttnn.slice(page_table, [b, 0], [b + 1, page_table.shape[1]])
                        ttnn.experimental.paged_update_cache(
                            k_cache,
                            kb,
                            update_idxs_tensor=pos_b,
                            page_table=pt_b,
                            block_size=eff_bs,
                            num_kv_heads=num_local_kv_heads,
                            **paged_modulo_kwargs,
                        )
                        ttnn.experimental.paged_update_cache(
                            v_cache,
                            vb,
                            update_idxs_tensor=pos_b,
                            page_table=pt_b,
                            block_size=eff_bs,
                            num_kv_heads=num_local_kv_heads,
                            **paged_modulo_kwargs,
                        )
                        for t in (kb, vb, pos_b, pt_b):
                            t.deallocate(True)
                    k_seq.deallocate(True)
                    v_seq.deallocate(True)
                else:
                    ttnn.experimental.paged_update_cache(
                        k_cache,
                        tt_k,
                        update_idxs_tensor=cache_pos,
                        page_table=page_table,
                        block_size=eff_bs,
                        num_kv_heads=num_local_kv_heads,
                        **paged_modulo_kwargs,
                    )
                    ttnn.experimental.paged_update_cache(
                        v_cache,
                        tt_v,
                        update_idxs_tensor=cache_pos,
                        page_table=page_table,
                        block_size=eff_bs,
                        num_kv_heads=num_local_kv_heads,
                        **paged_modulo_kwargs,
                    )
            else:
                ttnn.experimental.paged_update_cache(k_cache, tt_k, update_idxs_tensor=cache_pos)
                ttnn.experimental.paged_update_cache(v_cache, tt_v, update_idxs_tensor=cache_pos)
    else:
        k_cache = tt_k
        v_cache = tt_v

    # 6. SDPA (scale=1.0)
    sliding_window = config.sliding_window if config.is_sliding else None

    # Always pass an SDPAProgramConfig so num_cores_per_head stays within
    # MAX_TREE_REDUCTION_ROUNDS=6 (=> 64 cores/head). With program_config=None,
    # the SDPA op falls back to the full device grid, which exceeds 64 cores
    # on Blackhole (>=110 cores) when num_kv_heads is small. The struct's
    # default max_cores_per_head_batch=16 caps the per-head reduction tree.
    # Batched Q is height-sharded row-major across the device grid (one user per
    # core), so the SDPA grid must cover those cores — an 8-wide grid would miss
    # users on columns 8..10 of an 11-wide Blackhole grid and corrupt the output.
    batch_size = tt_q.shape[1]
    device_grid = mesh_device.compute_with_storage_grid_size()
    if config.head_dim >= 512 and batch_size == 1:
        # Single-user global layers: smaller grid — head_dim=512 needs more L1 per core.
        sdpa_grid = ttnn.CoreCoord(8, 4)
    else:
        # Sliding layers, and all batched decode: use the full device compute grid.
        sdpa_grid = ttnn.CoreCoord(device_grid.x, device_grid.y)

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_grid,
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
    )

    if page_table is not None:
        sdpa_num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
        tt_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            cur_pos_tensor=cache_pos,
            page_table_tensor=page_table,
            scale=1.0,
            sliding_window_size=sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program_config,
            # Tell SDPA the layer's view of the cache when the buffer was allocated
            # for a different layer type under HMA cross-group sharing — same
            # rationale as the num_kv_heads override on paged_update_cache.
            paged_cache_geometry=ttnn.PagedCacheGeometryOverride(
                block_size=effective_block_size(k_cache, config.head_dim, sdpa_num_local_kv_heads),
                num_kv_heads=sdpa_num_local_kv_heads,
            ),
            **paged_modulo_kwargs,
        )
    else:
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            cur_pos_tensor=cache_pos,
            scale=1.0,
            sliding_window_size=sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program_config,
        )
    tt_q.deallocate(True)

    # 7. Concat heads + output projection + allreduce
    num_local_heads = config.num_attention_heads // tp
    tt_out = concat_heads(
        tt_sdpa, is_decode_mode=True, num_heads=num_local_heads, head_dim=config.head_dim, mesh_device=mesh_device
    )
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)

    return tt_out


# ── Packed-query verify decode ─────────────────────────────────────────────
# Verifies P speculative positions per user in ONE attention pass by packing
# the P positions into the query-heads dim (H_local*P packed heads) and
# running a single non-causal SDPA with an additive mask that bakes in the
# per-position causal upper bound (and the sliding-window lower bound on
# sliding layers). Ported from the gemma4_cody packed-verify path.


def _packed_sdpa_grid(config, mesh_device):
    """SDPA grid for the packed verify — mirrors the single-token decode's
    choice: small grid for global layers (head_dim=512 needs more L1/core),
    full device grid for sliding layers."""
    if config.head_dim >= 512:
        return ttnn.CoreCoord(8, 4)
    device_grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreCoord(device_grid.x, device_grid.y)


def _verify_head_splits(B, H_local, nkv_local, P, head_dim, grid=None):
    """How many QUERY-HEAD-wise sub-ops to split the packed-verify SDPA into.

    The packed verify folds ``H_local*P`` query rows onto ``nkv_local`` KV
    groups; the per-core SDPA-decode CBs (Q, QK scores, and the flash
    cross-core reduction buffer) scale with the packed query-head tile count
    ``PNHt = H_local*P/32``. Global-attention layers (head_dim=512 ⇒ DHt 16,
    single local KV head at high TP) can overflow the 1.5 MB L1 even at the
    default k_chunk; we split the packed-head dim across ``n`` full-grid ops +
    a concat, carrying ``H_local/n`` heads each. Sliding layers (head_dim 256)
    fit and are left as one op.

    Enabled ONLY when ``nkv_local == 1``: then every query head maps to the
    lone KV head, so each sub-op can SHARE the full (paged) K/V cache +
    page_table with NO cache slicing — any head subset is GQA-correct. With >1
    local KV head a head split would need per-op cache slicing; such configs
    are left unsplit.

    ``GEMMA4_PV_SDPA_HEAD_SPLITS`` overrides the count (clamped to the largest
    valid split <= the request)."""
    if nkv_local != 1:
        return 1
    # Valid splits: whole heads per op AND 32-tile-aligned query-row slices.
    valid = [d for d in range(1, H_local + 1) if H_local % d == 0 and ((H_local // d) * P) % 32 == 0]
    if not valid:
        return 1
    env = os.environ.get("GEMMA4_PV_SDPA_HEAD_SPLITS")
    if env:
        want = max(1, min(int(env), H_local))
        return max(d for d in valid if d <= want)
    if (H_local * P) * head_dim <= 8192:  # not heavy (sliding) — one op fits
        return 1
    # Mirror sdpa_decode_program_factory's core allocation (num_kv_heads == 1)
    # and CB tile counts; keep grid / max_cores_per_head_batch / k_chunk in sync
    # with the packed sdpa_program_config. Budget in 2 KB tiles with margin
    # under the 768-tile (1.5 MB) L1 cap.
    if grid is None:
        grid = 32 if head_dim >= 512 else 64
    max_cores_per_head, k_chunk = 16, 64
    cores_per_head = max(1, min(grid, max_cores_per_head * B) // max(1, B))
    DHt, Sk = head_dim // 32, max(1, k_chunk // 32)
    BUDGET_TILES = 720

    def per_core_tiles(n):
        pnht = (H_local * P) // (32 * n)
        return (
            2 * pnht * DHt  # Q-in + tilized-Q
            + 5 * pnht * DHt  # 3 out-im + out-stats + out-final
            + 2 * pnht * Sk  # attn-mask + QK-im
            + 11 * pnht  # softmax stats CBs
            + 4 * Sk * DHt  # K + V (double-buffered)
            + 3  # scale / identity
            + (pnht * DHt + 2 * pnht) * (cores_per_head - 1)  # cross-core flash reduction
        )

    for d in valid:  # ascending ⇒ smallest split that fits the budget
        if per_core_tiles(d) <= BUDGET_TILES:
            return d
    return valid[-1]


def _packed_verify_sdpa(
    q_packed, k, v, layer_page_table, attn_mask, scale, pc, n_splits, B, H_local, P, head_dim, block_size, num_kv_heads
):
    """Run the packed-verify decode-SDPA, optionally head-split into
    ``n_splits`` full-grid ops + a concat (see ``_verify_head_splits``).
    Splitting SHARES the (paged) K/V cache + page_table across sub-ops unsliced
    — valid only for a single local KV head — and slices only Q and the
    additive mask on the packed query-head dim. Does NOT free its inputs.
    Returns [1, B, H_local*P, head_dim]."""

    # fp32 destination accumulation, on EVERY device count.
    #
    # Single-device (TP=1) needs it for a shape reason: the packed layout folds P
    # candidates into the query-heads dim, so with all heads on one device the SDPA
    # decode's padded-heads-per-core (PNHt) can be odd (e.g. 3). ttnn asserts
    # MUL_BCAST_GRANULARITY = min(PNHt * Sk_chunk_t, dst_size) is a power of 2; with
    # the op-default dst_size=8 and Sk_chunk_t=2 that leaves PNHt*2=6 and aborts.
    # fp32_dest_acc_en=True gives dst_size=4 ⇒ min(2*PNHt, 4) is 2 or 4, no k_chunk/L1
    # increase.
    #
    # Multi-device used to fall through to the op default (None) on the reasoning
    # that heads already split to a power-of-2 PNHt, so nothing forced the issue.
    # That left the packed verify accumulating in bf16 at TP>1, and it is measurably
    # not accurate enough. Measured at 1x2 against the batch=1 verify at the same
    # position (test_varying_acceptance_matches_greedy):
    #     top-2 decision margin            0.1875  (3 bf16 ULP at magnitude ~13.9)
    #     packed-vs-batch1, contended tok  0.6875  (11 ULP)
    #     max|packed - batch1| over the row 1.1250 (18 ULP)
    # The error is ~6x the margin, so the packed argmax deviates from greedy at any
    # position whose top-2 gap is under ~1.0 logit. Since `_accept_greedy` compares
    # drafts against the PACKED logits, that silently commits tokens plain greedy
    # would never emit — speculative decoding's core guarantee. Accuracy, not the
    # PNHt shape constraint, is why this is now unconditional.
    #
    # MEASURED AND REVERTED: forcing fp32_dest_acc_en at TP>1 made things WORSE, not
    # better. Against the batch=1 verify over valid-context rows at 1x2:
    #     default (bf16 accum)  p50 disagreement 1.06 logits (17 ULP), 59 rows,
    #                           first divergence at row 1 of iteration 25
    #     fp32_dest_acc_en=True p50 disagreement 27.68 logits (443 ULP),
    #                           first divergence at row 0 of iteration 1
    # A 26x jump is not a precision effect — it is a functional change, most likely
    # the fp32 dst_size=4 interacting with the head-split path or the program config.
    # So the multi-device default stays the op default, and fp32 is now OPT-IN
    # (GEMMA4_PV_SDPA_FP32=1) rather than opt-out, purely for further investigation.
    #
    # The underlying accuracy problem is NOT fixed: even at the default, the packed
    # verify disagrees with batch=1 by ~17 ULP at p50 while the argmax decision
    # margin is ~3 ULP. Whether packed or batch=1 is the WRONG one still needs a
    # correct ground truth (see GEMMA4_PV_HF_REF, which must apply the model's
    # final_logit_softcapping=30 — comparing against raw lm_head output is a ~40
    # logit systematic offset and tells you nothing).
    _dev = k.device()
    _num_dev = getattr(_dev, "get_num_devices", lambda: 1)()
    # GEMMA4_PV_SDPA_FP32: "1" forces the fp32 config on, "0" forces the op DEFAULT
    # on, unset keeps the historical per-device-count behaviour.
    #
    # Why "0" must be reachable at 1x1: the fp32 config is unconditional on single
    # device, and it is MEASURABLY WRONG. At 1x1 the packed verify's row-0 argmax
    # disagrees with the single-token verify from step 0 on a fresh prefill
    # (13513 vs 131416) — the same comparison passes exactly at 1x2, where this
    # config is NOT applied. It also measured harmful when forced at 1x2
    # (packed-vs-single-token p50 0.79 -> 27.68 logits). That broken verify is what
    # drives live acceptance to 0.02 while the drafter alone reaches 0.98.
    #
    # The stated reason for it here is the MUL_BCAST_GRANULARITY power-of-2
    # assertion, but that only bites when PNHt is odd. At H_local=8, P=4:
    # PNHt = H*P/32 = 1, so min(PNHt*Sk_chunk_t, dst_size) = min(2, 8) = 2, already
    # a power of two — the workaround is not needed for this shape.
    # Apply the fp32 workaround ONLY for the shape that actually needs it.
    #
    # It exists to satisfy ttnn's assertion that
    #     MUL_BCAST_GRANULARITY = min(PNHt * Sk_chunk_t, dst_size)
    # is a power of 2. With bf16 dst_size=8 and Sk_chunk_t = k_chunk(64)/32 = 2 that
    # fails at exactly ONE value, PNHt == 3 (min(6,8)=6); every other PNHt gives
    # 2, 4 or the capped 8. fp32_dest_acc_en halves dst_size to 4, making it 2 or 4.
    #
    # It used to be applied to EVERY single-device call, and that is a correctness
    # bug, not just wasted precision. Measured at 1x1, H_local=8, P=4 (PNHt=1, so the
    # assertion was never at risk):
    #     fp32 on   packed row-0 argmax disagrees with the single-token verify at
    #               step 0 on a fresh prefill (13513 vs 131416);
    #               live acceptance 0.02/3,  8.3 tok/s/u
    #     fp32 off  argmax agrees; acceptance 1.16/3, ~17.7 tok/s/u
    # 1.16 exceeds both the export replay (0.98) and the HF ceiling (1.12). The same
    # config measured harmful when forced at 1x2 (packed-vs-single-token p50
    # 0.79 -> 27.68 logits), which was the first hint and was mis-filed at the time
    # as "fp32 is not a win" rather than "fp32 is broken and 1x1 always runs it".
    #
    # GEMMA4_PV_SDPA_FP32 = "1"/"0" forces on/off for A/B.
    # The packed verify folds candidates into the query-head dim as H_local*P rows,
    # which must be tile-aligned: (H_local*P) % 32 == 0, i.e. P divisible by
    # 32 // H_local. E2B has 8 query heads so H_local = 8 // tp and the smallest legal
    # P is 4*tp => draft_len = 4*tp - 1 (3 at tp=1, 7 at tp=2, 15 at tp=4). Without this
    # check the violation surfaces as "Statically allocated circular buffers ... grow to
    # 2005952 B which is beyond max L1 size of 1572864 B" (program.cpp:1717), which
    # names neither draft_len nor the constraint it broke.
    if (H_local * P) % 32 != 0:
        _mult = 32 // H_local if H_local <= 32 else 1
        raise ValueError(
            f"packed verify needs (H_local*P) % 32 == 0, got H_local={H_local}, P={P} "
            f"(={H_local * P} rows). P must be a multiple of {_mult}, so use "
            f"draft_len = {_mult}*n - 1 (e.g. {_mult - 1}, {2 * _mult - 1}, {3 * _mult - 1})."
        )
    _pnht = (H_local * P) // 32
    _needs_fp32 = _pnht == 3  # the only PNHt where the power-of-2 assertion fails
    _fp32_env = os.environ.get("GEMMA4_PV_SDPA_FP32")
    _use_fp32 = _needs_fp32 if _fp32_env is None else (_fp32_env == "1")
    if not _use_fp32:
        compute_kernel_config = None
    else:
        _fid = getattr(ttnn.MathFidelity, os.environ.get("GEMMA4_PV_SDPA_FIDELITY", "HiFi2"))
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            _dev.arch(),
            math_fidelity=_fid,
            # Approximate exp in the softmax is a second precision lever; keep the
            # historical True as the default so this change moves ONE variable, and
            # expose it for the A/B that follows.
            math_approx_mode=os.environ.get("GEMMA4_PV_SDPA_APPROX", "1") == "1",
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _call(q, m):
        if layer_page_table is not None:
            return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                k,
                v,
                page_table_tensor=layer_page_table,
                is_causal=False,
                attn_mask=m,
                scale=scale,
                sliding_window_size=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=pc,
                compute_kernel_config=compute_kernel_config,
                paged_cache_geometry=ttnn.PagedCacheGeometryOverride(
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                ),
            )
        return ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=m,
            scale=scale,
            sliding_window_size=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,
            compute_kernel_config=compute_kernel_config,
        )

    # GEMMA4_PV_SDPA_FP=1: fingerprint the packed SDPA's INPUTS and OUTPUT.
    # The 50/50 fork was bisected to decoder layer 0 (its input is bit-identical
    # across runs, its output takes one of two values). Layer 0 is a SLIDING layer
    # and this op passes sliding_window_size=None, so it reads the whole KV span
    # and leans entirely on the additive mask. Dumping q/k/v/mask separately from
    # the result says whether the fork is a bad KV READ or a bad SDPA COMPUTE.
    def _fp(tag, t):
        if t is None:
            logger.info(f"[pv-sdpa] {tag} = None")
            return
        import hashlib

        import torch as _t

        raw = ttnn.to_torch(ttnn.get_device_tensors(t)[0])
        # md5 of the RAW bytes, not a sum. A sum is permutation-invariant AND loses
        # small per-element deltas in the fp32 accumulation over ~1M elements, so
        # "identical sums" is far too weak to conclude "identical inputs".
        digest = hashlib.md5(raw.contiguous().view(_t.uint8).numpy().tobytes()).hexdigest()[:16]
        h = raw.float()
        fin = _t.isfinite(h)
        logger.info(
            f"[pv-sdpa] {tag} shape={tuple(h.shape)} md5={digest} sum={h[fin].sum().item():+.6f} "
            f"absmax={h[fin].abs().max().item():.6f} nonfinite={int((~fin).sum())}"
        )

    if _PV_SDPA_FP:
        _fp("q", q_packed)
        _fp("k", k)
        _fp("v", v)
        _fp("mask", attn_mask)
        # The page table is the prime suspect. A SUM over the KV cache is
        # permutation-invariant, so identical k/v fingerprints do not rule out a
        # permuted logical->physical block mapping — and a permuted mapping changes
        # the order SDPA accumulates K chunks, hence its bf16 rounding, from
        # bit-identical inputs. Dump the mapping itself, not a sum of it.
        if layer_page_table is not None:
            _pt = ttnn.to_torch(ttnn.get_device_tensors(layer_page_table)[0]).flatten().tolist()
            logger.info(f"[pv-sdpa] page_table[:32]={[int(x) for x in _pt[:32]]}")
        else:
            logger.info("[pv-sdpa] page_table = None (unpaged)")

        # Buffer ADDRESSES are the last per-process variable left. The standalone
        # reproducer allocates in a fixed order, so its addresses match in every
        # process and (once kernels are cached) it is deterministic. The model has a
        # long allocation history that can differ, so if the two output variants track
        # the addresses, the op is address-sensitive.
        def _addr(nm, t):
            try:
                logger.info(f"[pv-sdpa] addr {nm}={hex(ttnn.get_device_tensors(t)[0].buffer_address())}")
            except Exception as e:  # noqa: BLE001
                logger.info(f"[pv-sdpa] addr {nm}=<unavailable: {type(e).__name__}>")

        for _nm, _t in (("q", q_packed), ("k", k), ("v", v), ("mask", attn_mask)):
            if _t is not None:
                _addr(_nm, _t)

    # ── WARM-UP: the packed SDPA reads uninitialised L1 on its first invocations ──
    # paged_scaled_dot_product_attention_decode in this configuration (is_causal=False
    # + explicit additive attn_mask) is NOT a pure function of its inputs on the first
    # few calls after a cold kernel build. Standalone reproducer, 6 processes, inputs
    # byte-identical by md5, three back-to-back calls each:
    #     process 1   9a1cb7ee  e4716437  b6d3d5e8   <- all three differ
    #     process 2   69ad3879  ffff80c7  ffff80c7   <- converges after one call
    #     process 3-6 ffff80c7  ffff80c7  ffff80c7   <- stable from the start
    # It converges to a value that is the SAME in every process, so the residue is
    # what varies, not the arithmetic. Processes 1-2 are the ones that had to JIT-build
    # kernels, which leaves different L1 behind.
    #
    # In the live model the first packed verify of a generation is exactly such a cold
    # invocation, and that is the whole 50/50 output fork: a ~0.6% delta in layer 0's
    # attention output amplifies to ~7% by layer 1 and eventually flips a token.
    # Everything else was eliminated by measurement first: q/k/v/mask byte-identical,
    # page table the identity, forks just as much at S_k=64 (single K chunk, no masked
    # tail, no cross-core reduction) as at S_k=1024, and plain decode -- which passes
    # cur_pos/sliding_window_size instead of an explicit mask -- is deterministic
    # across processes at every max_seq_len.
    #
    # Fix: on the first call for a given shape/config, run the op a few extra times and
    # throw the results away. SDPA only READS k/v, so repeating it cannot disturb the
    # KV cache. This runs on the eager compile call, i.e. before any trace capture, so
    # no extra ops land inside a trace.
    _wkey = (tuple(q_packed.shape), tuple(attn_mask.shape) if attn_mask is not None else None, n_splits, _use_fp32)
    if _wkey not in _PV_WARMED:
        _PV_WARMED.add(_wkey)
        for _ in range(_PV_WARMUP_ITERS):
            _w = _call(q_packed, attn_mask) if n_splits <= 1 else None
            if _w is not None:
                ttnn.deallocate(_w)

    if n_splits <= 1:
        _o = _call(q_packed, attn_mask)
        if _PV_SDPA_FP:
            _fp("out", _o)
            # GEMMA4_PV_SDPA_TWICE=1: run the SAME op again on the SAME inputs in the
            # SAME process. The op was shown to return one of two outputs across
            # processes from bit-identical q/k/v/mask. If out2 == out, the op is
            # deterministic within a process and the fork is a per-process constant
            # (core assignment / allocator address / L1 left over from init). If
            # out2 != out, it is live nondeterminism (arrival-order reduction or
            # uninitialised L1 read by idle cores).
            if os.environ.get("GEMMA4_PV_SDPA_TWICE") == "1":
                _o2 = _call(q_packed, attn_mask)
                _fp("out2", _o2)
                ttnn.deallocate(_o2)
        return _o
    s_k = attn_mask.shape[3] if attn_mask is not None else None
    rows_per = (H_local // n_splits) * P  # packed query rows per sub-op (32-aligned)
    parts = []
    for s in range(n_splits):
        r0, r1 = s * rows_per, (s + 1) * rows_per
        q_i = ttnn.slice(q_packed, [0, 0, r0, 0], [1, B, r1, head_dim])
        m_i = ttnn.slice(attn_mask, [0, 0, r0, 0], [B, 1, r1, s_k]) if attn_mask is not None else None
        out_i = _call(q_i, m_i)
        ttnn.deallocate(q_i)
        if m_i is not None:
            ttnn.deallocate(m_i)
        parts.append(out_i)
    out = ttnn.concat(parts, dim=2)  # [1, B, H_local*P, head_dim]
    for p in parts:
        ttnn.deallocate(p)
    return out


def _packed_fill_kv_loopfree_embed(cache, staging, new_seq, embed_idx, hot_pt):
    """Loop-free packed KV-cache write via PERSISTENT STAGING and a
    transpose-free ``ttnn.embedding`` row-gather merge — replaces the
    per-position ``paged_update_cache`` loop, and never reads the committed
    cache on the hot path.

    ``staging`` is this layer's resident hot-block copy
    ``[1, nkv, S2, head_dim]`` (S2 = max_batch * PV_HOT_BLOCKS * block_size),
    slot-indexed: slot ``s`` owns seq ``[s*BLK*bs, (s+1)*BLK*bs)`` and already
    holds the committed prefix of its hot block(s) — seeded at the first
    verify, maintained step-to-step by the spec-decode driver. ``new_seq`` is
    the freshly projected/normed/RoPE'd K (or V) ``[1, nkv, n_new, head_dim]``
    (user-major / position-minor rows, n_new tile-aligned — caller pads).

    Per step (all fixed-shape ⇒ trace-safe; cost scales with S2, never with
    max_num_blocks):
      1. merge: ``ttnn.embedding`` row-gathers the flattened
         ``concat([staging, new_seq])`` — committed positions copy from staging
         (identity, or shifted one block on a rollover), the P new positions
         pull from ``new_seq``. ``embed_idx`` ([1, nkv*S2] u32) bakes the
         per-head row offset in: ``embed_idx[h*S2 + j] = h*(S2+n_new) +
         merge_idx[j]``.
      2. ``assign`` the merged hot blocks back into ``staging`` (persists for
         the next step — no cache read needed next time).
      3. one ``paged_fill_cache`` writes each slot's hot block(s) to its
         physical page(s) named by ``hot_pt`` (-1 = skip ⇒ idle slots / the
         unused spill block are left untouched).
    """
    dram = ttnn.DRAM_MEMORY_CONFIG
    nkv = staging.shape[1]
    S2 = staging.shape[2]
    head_dim = staging.shape[3]
    src_seq = S2 + new_seq.shape[2]

    # ① merge resident staging with this step's new K/V — no committed-cache read.
    new_dram = ttnn.to_memory_config(new_seq, dram)
    src = ttnn.concat([staging, new_dram], dim=2, memory_config=dram)  # [1, nkv, src_seq, hd] TILE
    ttnn.deallocate(new_dram)
    # Per-head row-gather merge. Folding all nkv heads into one row axis
    # (``reshape(src, (nkv*src_seq, hd))`` + a single ``embedding``) mis-orders
    # the 2nd+ head's tiles in TILE layout, corrupting cache head>=1 whenever a
    # device holds more than one local KV head (``nkv_local >= 2`` — i.e. TP
    # configs with ``num_kv_heads // tp >= 2``). Gathering each head's
    # ``[src_seq, hd]`` block on its own keeps every head tile-contiguous.
    # ``embed_idx`` bakes a per-head row offset ``h*src_seq``; head 0's slice is
    # the bare merge_idx, valid for every head's local row-gather.
    base_idx = ttnn.slice(embed_idx, [0, 0], [1, S2])  # head-0 merge_idx (no head offset)
    parts = []
    for h in range(nkv):
        src_h = ttnn.slice(src, [0, h, 0, 0], [1, h + 1, src_seq, head_dim])
        src_h2d = ttnn.reshape(src_h, (src_seq, head_dim))
        merged_h = ttnn.embedding(base_idx, src_h2d, layout=ttnn.TILE_LAYOUT)  # [1, S2, hd]
        parts.append(ttnn.reshape(merged_h, (1, 1, S2, head_dim)))
    ttnn.deallocate(src)
    merged = ttnn.concat(parts, dim=1) if nkv > 1 else parts[0]
    merged = ttnn.to_memory_config(merged, dram)  # [1, nkv, S2, hd]

    # ② persist updated hot blocks for next step, then ③ one fill launch.
    if os.environ.get("GEMMA4_KV_MERGE_FP") == "1":
        # Is the loop-free merge value-PRESERVING? Every step is nominally data movement
        # (embedding row-gather, assign, paged_fill_cache) and copies do not round. Note
        # ttnn.embedding above is called WITHOUT an explicit dtype=, so its output dtype is
        # whatever the op defaults to -- if that differs from new_seq's dtype, the merge
        # casts and THAT is the 1-ULP source. If instead the new rows survive exactly, the
        # divergence comes from the P-row projection/RoPE upstream, not from the write.
        pass

        _rn = ttnn.to_torch(ttnn.get_device_tensors(new_seq)[0])
        _rm = ttnn.to_torch(ttnn.get_device_tensors(merged)[0])
        _n_new = new_seq.shape[2]
        _tail = _rn.reshape(-1, _rn.shape[-1])[-_n_new:].float()
        _mflat = _rm.reshape(-1, _rm.shape[-1]).float()
        _res = [float((_mflat - _tail[i]).abs().amax(-1).min()) for i in range(_n_new)]
        logger.info(
            f"[kv-merge] dtypes new_seq={new_seq.dtype} merged={merged.dtype} "
            f"staging={staging.dtype} cache={cache.dtype} | "
            f"new-row residuals={['%.8f' % r for r in _res]} "
            f"-> {'EXACT (pure copy)' if max(_res) == 0.0 else 'ROUNDED (cast inside the merge)'}"
        )

    ttnn.assign(merged, staging)
    ttnn.experimental.paged_fill_cache(cache, merged, hot_pt, batch_idx=0)
    ttnn.deallocate(merged)


def packed_decode_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    kv_cache,
    config,
    mesh_config,
    mesh_device,
    position_idx,
    kv_write_idxs,
    attn_mask,
    packed_p,
    page_table=None,
    ccl_manager=None,
    is_kv_shared=False,
    rope_packed=None,
    kv_staging=None,
    embed_idx=None,
    hot_pt=None,
):
    """Packed multi-token decode attention — P query positions/slot in one pass.

    Hoists QKV projection, prefill-style split, per-head norm, and RoPE OUT of
    any per-position loop: those run once on the full B*P tensor. The KV write
    is loop-free when staging is provided (one paged_fill_cache per K/V), else
    a per-position paged_update_cache fallback.

    Args:
        hidden_states: [1, 1, B*P, hidden_size]. Rows user-major / position-
            minor — row u*P+p is slot u's p-th packed token.
        cos_cache, sin_cache: 2D RoPE caches [max_seq_len, head_dim].
        position_idx: [1, B*P] uint32 — RoPE position per row (cur_pos_u + p).
        kv_write_idxs: list of P int32 [B] device tensors — the cache write
            position for each packed position p (fallback per-p loop only).
        attn_mask: [B, 1, H_local*P, S_k] head-major additive mask baking in
            the per-position causal upper bound and (sliding layers) the
            window lower bound. S_k must be a multiple of k_chunk (64).
        packed_p: P, the number of packed positions per slot.
        rope_packed: optional pre-gathered (cos_bp, sin_bp) for this layer
            type — identical for all layers of a type, so the caller gathers
            once per type per step.
        kv_staging: optional [k_staging, v_staging] persistent hot-block
            buffers (loop-free write path; see _packed_fill_kv_loopfree_embed).
        embed_idx: [1, nkv_local*S2] uint32 merge gather index (loop-free path).
        hot_pt: [1, max_batch*PV_HOT_BLOCKS] int32 physical fill pages, -1=skip.

    Returns:
        [1, 1, B*P, hidden_size] — attention output for every packed position.
    """
    P = packed_p
    tp = mesh_config.tp if mesh_config else 1
    B = hidden_states.shape[2] // P
    H_local = config.num_attention_heads // tp
    head_dim = config.head_dim
    nkv_local = 1 if weights.kv_replicated else config.num_key_value_heads // tp
    if config.cache_position_modulo is not None:
        raise NotImplementedError("packed verify does not support bounded sliding KV caches")
    if kv_cache is None:
        raise ValueError("packed_decode_forward requires a KV cache (it attends through the paged cache)")
    l1 = ttnn.L1_MEMORY_CONFIG

    # ── ① QKV projection (one call on the full B*P, output kept on L1) ──────
    xqkv = apply_qkv_projection(hidden_states, weights, memory_config=l1)
    qkv_dim = xqkv.shape[-1]

    # ── ② L1 height-sharded MemoryConfig for the fallback paged_update_cache ─
    # ``paged_update_cache`` needs the layout ``nlp_create_qkv_heads_decode``
    # emits; the spec depends only on shape constants, so probe once and cache.
    cache_key = _q_sharded_mem_key(B, qkv_dim, config, weights, tp)
    q_sharded_mem = _Q_SHARDED_MEM_CACHE.get(cache_key)
    if q_sharded_mem is None and kv_staging is None and not is_kv_shared:
        probe = ttnn.slice(xqkv, [0, 0, 0, 0], [1, 1, B, qkv_dim])
        q_probe, k_probe, v_probe = split_qkv_heads_decode(
            probe, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
        )
        q_sharded_mem = q_probe.memory_config()
        _Q_SHARDED_MEM_CACHE[cache_key] = q_sharded_mem
        for t in (q_probe, k_probe, v_probe, probe):
            ttnn.deallocate(t)

    # ── ②b Prefill-style split → L1 Q/K/V on B*P ────────────────────────────
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated, memory_config=l1
    )
    # Q: [1, H_local, B*P, head_dim], K/V: [1, nkv_local, B*P, head_dim]
    ttnn.deallocate(xqkv)

    # ── ③ Per-head norms (one call each on B*P, kept on L1) ─────────────────
    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True, memory_config=l1)
    if not is_kv_shared:
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True, memory_config=l1)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False, memory_config=l1)

    # ── ④ RoPE on B*P, prefill mode (kernel iterates along S=B*P) ───────────
    if rope_packed is not None:
        cos_bp, sin_bp = rope_packed
        owns_rope = False
    else:
        cos_bp = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT))
        sin_bp = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT))
        owns_rope = True
    tt_q = apply_rope(tt_q, cos_bp, sin_bp, token_index=None, memory_config=l1)
    if not is_kv_shared:
        tt_k = apply_rope(tt_k, cos_bp, sin_bp, token_index=None, memory_config=l1)
    if owns_rope:
        ttnn.deallocate(cos_bp)
        ttnn.deallocate(sin_bp)

    if is_kv_shared:
        ttnn.deallocate(tt_k)
        ttnn.deallocate(tt_v)

    # ── ⑤ KV write — loop-free persistent staging (primary) ────────────────
    if not is_kv_shared and kv_staging is not None and embed_idx is not None:
        # Park Q off L1 (idle until the SDPA pack ⑥); merge intermediates live
        # in DRAM and Q never re-enters L1 before the SDPA call.
        tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
        k_cache_w, v_cache_w = kv_cache
        k_stg, v_stg = kv_staging
        if k_stg.shape[1] != k_cache_w.shape[1]:
            raise ValueError("staging nkv must match cache nkv (HMA-shared caches unsupported)")
        # Pad new rows up to a tile boundary so the merge concat + flatten
        # reshape stay metadata-only; embed_idx is built against the padded
        # src_seq, so padded rows are never gathered.
        n_new = tt_k.shape[2]
        pad = (-n_new) % 32
        if pad:
            k_pad = ttnn.pad(tt_k, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            v_pad = ttnn.pad(tt_v, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            ttnn.deallocate(tt_k)
            ttnn.deallocate(tt_v)
            tt_k, tt_v = k_pad, v_pad
        _packed_fill_kv_loopfree_embed(k_cache_w, k_stg, tt_k, embed_idx, hot_pt)
        _packed_fill_kv_loopfree_embed(v_cache_w, v_stg, tt_v, embed_idx, hot_pt)
        ttnn.deallocate(tt_k)
        ttnn.deallocate(tt_v)

    # ── ⑤ FALLBACK per-p paged_update_cache loop ────────────────────────────
    elif not is_kv_shared:
        tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
        k_cache_w, v_cache_w = kv_cache
        eff_bs = effective_block_size(k_cache_w, head_dim, nkv_local)
        # Convert the prefill-style [1, nkv, B*P, hd] to decode layout
        # [1, B*P, nkv, hd] (DRAM — the full tensors stay resident across the
        # loop; only the per-p reshard occupies L1).
        tt_k_bp = ttnn.permute(tt_k, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_v_bp = ttnn.permute(tt_v, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tt_k)
        ttnn.deallocate(tt_v)
        # The prefill-style split may leave the B*P sequence rows tile-padded to a
        # multiple of 32 (ttnn-version dependent — same guard as the Q pack at ⑥).
        # Slice back to the logical B*P rows so the (1,B,P,nkv,hd) reshape is
        # volume-exact (padding is always at the tail). The slice can alias the
        # parent's DRAM storage, so keep tt_k_bp/tt_v_bp alive until the end-of-loop
        # cleanup rather than freeing them here.
        k_src, v_src = tt_k_bp, tt_v_bp
        if tt_k_bp.shape[1] != B * P:
            k_src = ttnn.slice(tt_k_bp, [0, 0, 0, 0], [1, B * P, nkv_local, head_dim])
            v_src = ttnn.slice(tt_v_bp, [0, 0, 0, 0], [1, B * P, nkv_local, head_dim])
        tt_k_view = ttnn.reshape(k_src, (1, B, P, nkv_local, head_dim))
        tt_v_view = ttnn.reshape(v_src, (1, B, P, nkv_local, head_dim))
        for p in range(P):
            k_p = ttnn.slice(tt_k_view, [0, 0, p, 0, 0], [1, B, p + 1, nkv_local, head_dim])
            k_p = ttnn.reshape(k_p, (1, B, nkv_local, head_dim))
            v_p = ttnn.slice(tt_v_view, [0, 0, p, 0, 0], [1, B, p + 1, nkv_local, head_dim])
            v_p = ttnn.reshape(v_p, (1, B, nkv_local, head_dim))
            k_p = ttnn.to_memory_config(k_p, q_sharded_mem)
            v_p = ttnn.to_memory_config(v_p, q_sharded_mem)
            if page_table is not None:
                ttnn.experimental.paged_update_cache(
                    k_cache_w,
                    k_p,
                    update_idxs_tensor=kv_write_idxs[p],
                    page_table=page_table,
                    block_size=eff_bs,
                    num_kv_heads=nkv_local,
                )
                ttnn.experimental.paged_update_cache(
                    v_cache_w,
                    v_p,
                    update_idxs_tensor=kv_write_idxs[p],
                    page_table=page_table,
                    block_size=eff_bs,
                    num_kv_heads=nkv_local,
                )
            else:
                ttnn.experimental.paged_update_cache(k_cache_w, k_p, update_idxs_tensor=kv_write_idxs[p])
                ttnn.experimental.paged_update_cache(v_cache_w, v_p, update_idxs_tensor=kv_write_idxs[p])
            ttnn.deallocate(k_p)
            ttnn.deallocate(v_p)
        ttnn.deallocate(tt_k_bp)
        ttnn.deallocate(tt_v_bp)

    k_cache_use, v_cache_use = kv_cache

    # ── ⑥ Head-major pack Q → [1, B, H_local*P, head_dim] → SDPA ────────────
    # ROW_MAJOR on purpose: P (< 32) never lands on a tile axis, so the
    # split/merge reshapes are free views and the rank-5 permute is one strided
    # copy; in TILE the same reshapes re-tile AND pad P→32. The head-major
    # order (h*P+p) is load-bearing: SDPA maps packed query head i → KV head
    # i // group, so a KV group must be a contiguous head block.
    tt_q = ttnn.to_layout(tt_q, ttnn.ROW_MAJOR_LAYOUT)
    # The prefill-style split / RoPE may keep the B*P sequence rows tile-padded
    # to 32 in the logical shape (ttnn-version dependent). Slice back to the
    # logical B*P rows so the head-major pack reshape is volume-exact (padding
    # is always at the tail, so the first B*P rows hold the real positions).
    if tt_q.shape[2] != B * P:
        tt_q_unpad = ttnn.slice(tt_q, [0, 0, 0, 0], [1, H_local, B * P, head_dim])
        ttnn.deallocate(tt_q)
        tt_q = tt_q_unpad
    tt_q = ttnn.reshape(tt_q, (1, H_local, B, P, head_dim))
    tt_q = ttnn.permute(tt_q, (0, 2, 1, 3, 4))  # [1, B, H_local, P, hd]
    tt_q = ttnn.reshape(tt_q, (1, B, H_local * P, head_dim))
    q_packed = ttnn.to_layout(tt_q, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(tt_q)

    # GEMMA4_PV_KCHUNK / GEMMA4_PV_MAXCORES: knobs for the cross-core flash
    # reduction. The 50/50 output fork was bisected to THIS op: across processes
    # q/k/v/mask are byte-identical (md5) and the page table is the identity, yet
    # the output takes one of two values — while a back-to-back re-run inside one
    # process always reproduces. num_cores_per_head is 16 here, so a K chunk's
    # partial softmax is merged across 16 cores. Raising k_chunk_size to S_k
    # collapses that to a single chunk and removes the cross-core merge entirely,
    # which is the A/B that isolates the reduction as the source.
    _kchunk = int(os.environ.get("GEMMA4_PV_KCHUNK", "64"))
    _maxcores = int(os.environ.get("GEMMA4_PV_MAXCORES", "16"))
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=_packed_sdpa_grid(config, mesh_device),
        q_chunk_size=32,
        k_chunk_size=_kchunk,
        exp_approx_mode=False,
        max_cores_per_head_batch=_maxcores,
    )
    _grid = sdpa_program_config.compute_with_storage_grid_size
    n_sdpa_splits = _verify_head_splits(B, H_local, nkv_local, P, head_dim, grid=_grid.x * _grid.y)
    eff_bs_sdpa = effective_block_size(k_cache_use, head_dim, nkv_local)
    tt_sdpa = _packed_verify_sdpa(
        q_packed,
        k_cache_use,
        v_cache_use,
        page_table,
        attn_mask,
        1.0,
        sdpa_program_config,
        n_sdpa_splits,
        B,
        H_local,
        P,
        head_dim,
        eff_bs_sdpa,
        nkv_local,
    )
    ttnn.deallocate(q_packed)

    # ── ⑦ Unpack head-major SDPA output → concat heads + o_proj + AR ────────
    tt_sdpa = ttnn.to_layout(tt_sdpa, ttnn.ROW_MAJOR_LAYOUT, memory_config=l1)  # DRAM TILE → L1 RM
    # Same tile-padding guard as the Q pack: the SDPA output's H_local*P packed
    # query rows may come back padded to 32; slice to the logical extent so the
    # unpack reshape is volume-exact.
    if tt_sdpa.shape[2] != H_local * P:
        tt_sdpa_unpad = ttnn.slice(tt_sdpa, [0, 0, 0, 0], [1, B, H_local * P, head_dim])
        ttnn.deallocate(tt_sdpa)
        tt_sdpa = tt_sdpa_unpad
    tt_sdpa = ttnn.reshape(tt_sdpa, (1, B, H_local, P, head_dim))
    tt_sdpa = ttnn.permute(tt_sdpa, (0, 1, 3, 2, 4))  # [1, B, P, H_local, head_dim]
    tt_sdpa = ttnn.reshape(tt_sdpa, (1, B * P, H_local, head_dim))
    tt_sdpa = ttnn.to_layout(tt_sdpa, ttnn.TILE_LAYOUT, memory_config=l1)

    # Decode-layout concat: transpose batch/heads then nlp_concat_heads.
    tt_sdpa = ttnn.transpose(tt_sdpa, 1, 2)
    tt_out = ttnn.experimental.nlp_concat_heads(tt_sdpa, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(tt_sdpa)
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)
    return tt_out

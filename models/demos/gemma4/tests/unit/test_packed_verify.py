# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed-query verify (spec decode) vs the sequential and batch-dim verifies.

The packed verify folds the K+1 candidates into the query-heads dim of ONE
batch=1 forward (head-major packed SDPA with an additive mask) and writes the
new KV loop-free through the persistent staging + paged_fill_cache path. Greedy
argmax must match the sequential single-token verify chain position-for-position
(up to the same near-tie caveat as the batch-dim verify); the loop-free KV write
must also leave the cache state correct across iterations (a chain of packed
verifies, with rollovers across page boundaries, keeps matching sequential).

Requires HF_MODEL (target). The drafter is irrelevant here.
"""

import math
import os

import pytest
import torch
from loguru import logger

import ttnn

from ...tests.test_factory import parametrize_mesh_with_fabric


def _is_moe_model(model_path):
    """True for Mixture-of-Experts checkpoints (e.g. gemma-4-26B-A4B).

    Packed verify folds the K+1 candidates into ONE multi-token forward, so it
    drives the model with seq_len = P = draft_len+1 (e.g. 5) — not a multiple of
    32. On an MoE checkpoint the experts block routes any seq_len != 1 through
    its prefill path, which asserts ``seq_len % 32 == 0`` (see
    gemma4/tt/experts/__init__.py). Dense models (12B, 31B) have no experts block
    and are unaffected. Padding P up to 32 would run the experts on 32 tokens per
    verify and erase the packed-verify win, so packed verify is unsupported on
    MoE. Detect it cheaply from the config (no weights loaded) so we can skip
    before ``from_pretrained`` — which also avoids writing the weight cache on the
    read-only NAS mount used by CI e2e.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tc = getattr(cfg, "text_config", cfg)
        return bool(getattr(tc, "num_experts", 0))
    except Exception:
        return False


_MOE_UNSUPPORTED_REASON = (
    "packed verify drives the model with seq_len=P=draft_len+1 (not a multiple of 32); "
    "MoE experts prefill requires seq_len%32==0, so packed verify is unsupported on MoE "
    "checkpoints (e.g. gemma-4-26B-A4B). Dense targets (12B, 31B) run this test."
)


def _is_pli_model(model_path):
    """True for per-layer-input (MatFormer) checkpoints (e.g. gemma-4-E2B/E4B).

    These models carry a per-layer input embedding (``hidden_size_per_layer_input``)
    that must be fed to every layer.

    PLI IS now threaded through the speculative-verify path — eagerly via
    ``token_ids_host`` and, under trace, via the persistent ``pli_stacked`` buffer
    (see ``test_packed_verify_traced_pli_matches_eager`` below, which requires a PLI
    checkpoint). The remaining tests here still skip PLI targets because they drive
    ``ttnn_packed_verify_forward`` DIRECTLY with neither argument, which lands in
    ``Gemma4Model._compute_per_layer_inputs(None, None)`` and raises. Dense non-PLI
    targets (12B, 31B) are unaffected.

    Detect it cheaply from the config (no weights loaded) so we can skip before
    ``from_pretrained`` — which also avoids writing the weight cache on the
    read-only NAS mount used by CI e2e.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tc = getattr(cfg, "text_config", cfg)
        return bool(getattr(tc, "hidden_size_per_layer_input", 0))
    except Exception:
        return False


_PLI_UNSUPPORTED_REASON = (
    "this test calls ttnn_packed_verify_forward directly, passing neither token_ids_host nor "
    "pli_stacked, so a PLI/MatFormer checkpoint (gemma-4-E2B/E4B) hits "
    "_compute_per_layer_inputs(None, None) and raises. The PLI verify path itself is covered by "
    "test_packed_verify_traced_pli_matches_eager. Dense non-PLI targets (12B, 31B) run this test."
)

_L1_OVERFLOW_REASON = (
    "packed-verify global-layer SDPA (head_dim=512, fp32-accum, P candidates folded into "
    "heads) exceeds this core's L1 at the current TP. The per-core footprint shrinks ~TP×, "
    "so larger targets need a larger TP mesh: e.g. gemma-4-31B overflows L1 at TP=4 "
    "(4-chip box) but fits at TP>=8. 12B fits at TP=4. Skipping on this SKU/mesh."
)


def _is_l1_cb_overflow(exc):
    """True if `exc` is the ttnn L1 circular-buffer capacity throw.

    This is a deterministic, host-side program-validation failure (raised before
    any on-device launch, so the device stays healthy), matching e.g.:
        "Statically allocated circular buffers on core range [...] grow to
         2005824 B which is beyond max L1 size of 1572864 B".
    Keyed off the message text so it stays model/SKU/mesh-agnostic.
    """
    s = str(exc).lower()
    return "circular buffer" in s and "l1 size" in s


@parametrize_mesh_with_fabric()
def test_packed_verify_matches_sequential(mesh_device, reset_seeds):
    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")

    if _is_moe_model(model_path):
        pytest.skip(_MOE_UNSUPPORTED_REASON)

    if _is_pli_model(model_path):
        pytest.skip(_PLI_UNSUPPORTED_REASON)

    # Single-device (TP=1) is unsupported for packed verify on the 12B model: the
    # global layers (global_head_dim=512) run the packed SDPA with fp32_dest_acc_en
    # (the odd-PNHt MUL_BCAST power-of-2 fix) AND cannot head-split (nkv_local>1 ⇒ no
    # single shared KV head), so the per-core flash cross-reduction CBs overflow the
    # 1.5 MB L1 (~2.8 MB). Capping the reduction fan-in fits L1 but deadlocks the SDPA
    # kernel on-device. Supported on multi-device (TP) meshes, where TP shrinks the
    # per-core packed-head footprint ~TP×.
    if mesh_device.get_num_devices() == 1:
        pytest.skip(
            "single-device L1 limit: 12B global-layer packed SDPA exceeds single-device L1 (use a TP mesh, e.g. 1x4)"
        )

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    max_seq_len = 1024
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = "The capital of France is"
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 24, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    # assistant_model unused — only the verify paths run.
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=None,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=4,
    )
    P = spec.draft_len + 1

    def _prefill():
        # warmup_prefill=False skips the prefill-trace warmup (and its on-device
        # sampling/penalty sweep). This correctness test reads logits to host and
        # does argmax itself, so device sampling is irrelevant here; skipping the
        # warmup also avoids an unrelated TP>1 penalty-path shape mismatch that
        # otherwise aborts before the verify code runs.
        generator.prefill_forward_text(
            in_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            warmup_prefill=False,
        )

    # ── Reference: sequential single-token verify chain (batch=1 takes the
    # plain verify path — packing only engages for multi-token calls).
    _prefill()
    seq = []
    tok, pos = anchor_token, anchor_pos
    for _ in range(3 * P):  # several iterations ⇒ exercises staging rollovers
        logits, h = spec._verify([tok], [pos])
        h.deallocate(True)
        tok = int(torch.argmax(logits[0]))
        seq.append(tok)
        pos += 1

    # ── Packed: chain of packed verifies committing the matched chain tokens.
    _prefill()
    spec._pv_a_prev = -1
    packed = []
    pos = anchor_pos
    chain = [anchor_token] + seq
    it = 0
    while len(packed) < 3 * P:
        tokens = chain[it * P : it * P + P]  # the already-verified greedy chain
        positions = [pos + j for j in range(P)]
        try:
            lh, h = spec._verify(tokens, positions)  # first call compiles the packed SDPA
        except RuntimeError as e:
            if _is_l1_cb_overflow(e):
                pytest.skip(_L1_OVERFLOW_REASON)
            raise
        h.deallocate(True)
        packed.extend(int(torch.argmax(lh[j])) for j in range(P))
        pos += P
        it += 1
    packed = packed[: len(seq)]

    logger.info(f"sequential: {seq}")
    logger.info(f"packed:     {packed}")
    n_match = sum(1 for a, b in zip(seq, packed) if a == b)
    logger.info(f"match {n_match}/{len(seq)}")
    assert packed == seq, f"packed verify diverges from sequential: packed={packed} seq={seq}"


# ───────────────────────── batched (B>1) packed-verify bench ─────────────────
#
# The PR's KV-bandwidth win lives entirely in the verify step: a packed verify
# reads each user's KV ONCE (batch dim = B, the P=K+1 candidates folded into the
# query-heads), whereas the batch-alias verify spends one full KV read per
# pseudo-user (batch dim = B*(K+1)). At batch=1 the decode is weight-bound, so
# this never shows; only once weights are amortized across a batch does per-user
# cost become KV-bound and the B-vs-B*(K+1) read ratio matter.
#
# This bench drives ``ttnn_packed_verify_forward`` directly at B>1. The packed
# attention path (``packed_decode_forward`` → ``_packed_verify_sdpa``) is already
# B-aware (``B = rows // P``, page_table ``[B, blocks]``, mask ``[B, 1, H*P,
# S_k]``); the only batch=1 lock-ins are the spec-decode host builders and the
# loop-free *staging* KV write. We sidestep the staging (a write-side optimization
# that is irrelevant to the read amortization and dwarfed by the full-context read
# at long ctx) by using the per-position ``paged_update_cache`` *fallback* write
# (``kv_write_idxs``, no staging), so this isolates exactly the SDPA read claim.
#
# Asymmetry that *is* the batch>1 story: the batch-alias verify rides the ordinary
# decode path, whose per-user position vector is capped at 32 → ``B*(K+1) <= 32``
# ⇒ B<=8 at P=4. The packed verify keeps batch=B and folds P into heads, so it
# scales to B=32 where a single-pass batch-alias verify cannot even run; there the
# honest baseline is P sequential batch-B decodes (the cost of verifying P
# positions without packing).
#
# Env knobs: GEMMA4_BENCH_B="1,8" (comma list), GEMMA4_BENCH_CTX=2048 (prefill
# length per user), GEMMA4_SPEC_DRAFT_LEN=3 (K; P=K+1), GEMMA4_BENCH_ITERS=20,
# GEMMA4_BENCH_CONFIDENT_GAP=5.0, GEMMA4_BENCH_MAX_CONFIDENT_FLIPS=1 (B>1 only).
# SKU-adaptive mesh (CI picks the largest that fits), matching matches_sequential.
# Pinning a fixed TP (e.g. 1x4) breaks on boxes whose NAS weight cache was built at
# a different TP: the demo populates the cache at the largest mesh's TP, so a TP4
# pin on an 8-chip box misses the cache and tries to *write* it into the read-only
# mount. trace_region_size is needed for this test's trace capture.
@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 256_000_000})
def test_packed_verify_batch_perf(mesh_device, reset_seeds):
    import time

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    if _is_moe_model(model_path):
        pytest.skip(_MOE_UNSUPPORTED_REASON)

    if _is_pli_model(model_path):
        pytest.skip(_PLI_UNSUPPORTED_REASON)
    if mesh_device.get_num_devices() == 1:
        pytest.skip("single-device L1 limit: use a TP mesh (e.g. 1x4)")

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig

    # CI is a shared, time-boxed runner: perf numbers there are meaningless and
    # the full sweep (B up to 32, ctx=2048, 20 reps) on a 31B target blows the
    # job timeout (65k-token prefill + 4 B-values × 20 reps × 2 paths). Under
    # CI=true default to a light sweep so this still runs as a B>1 packed-verify
    # smoke/correctness check; local runs keep full fidelity. Any explicit
    # GEMMA4_BENCH_* override wins in both cases. NOTE: ctx must keep
    # max_seq_len >= the sliding window (1024) — the sliding-window decode slices
    # `window` keys from the cache, so a shorter cache overruns it. ctx=1024 gives
    # max_seq_len=1088 (>= 1024); do not lower it below the window.
    _ci = os.getenv("CI") == "true"
    Bs = [int(b) for b in os.environ.get("GEMMA4_BENCH_B", "1,8" if _ci else "1,8,16,32").split(",") if b.strip()]
    ctx = int(os.environ.get("GEMMA4_BENCH_CTX", 1024 if _ci else 2048))
    K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 3))
    P = K + 1
    reps = int(os.environ.get("GEMMA4_BENCH_ITERS", 3 if _ci else 20))
    # A packed-vs-alias argmax flip is a real bug only if the reference (alias)
    # preferred its token by more than this logit margin; smaller gaps are
    # near-ties that the ~1e-1 batched-SDPA noise is expected to flip.
    CONFIDENT_GAP = float(os.environ.get("GEMMA4_BENCH_CONFIDENT_GAP", 5.0))
    Bmax = max(Bs)

    block_size = 64
    # Key length S_k = max_seq_len: must cover positions 0..ctx+K and be a
    # multiple of the SDPA k_chunk (64). The additive mask zeroes out anything
    # past each row's causal bound, so over-allocating keys is harmless.
    max_seq_len = math.ceil((ctx + P) / block_size) * block_size
    blocks_per_user = max_seq_len // block_size
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=Bmax * blocks_per_user)

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=Bmax,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    page_table_torch = create_tt_page_table(Bmax, paged_attention_config)  # [Bmax, blocks_per_user], distinct per user

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=None,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table_torch,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=K,
    )
    mapper = target._replicate_to_mesh_mapper()
    tp = target.mesh_config.tp if target.mesh_config else 1
    H = target.layers[0].self_attn.config.num_attention_heads // tp
    window = target.hf_config.sliding_window
    vocab = target.vocab_size
    NEG = -1e9

    # Prefill Bmax users to length `ctx` (distinct KV blocks per user via the
    # page table). Content is irrelevant to the bench — packed and batch-alias
    # read the SAME prefilled cache — but must be deterministic: random prefill
    # produced runner-dependent near-tie argmax flips (BH 12B/31B CI: B=1 with
    # confident-flips=0 still failed a match-rate gate). Use a fixed in-vocab
    # arithmetic sequence, identical across users. warmup_prefill=False avoids
    # the TP>1 prefill-warmup sampling path (see the correctness test above).
    vocab_hi = min(2000, int(target.vocab_size) - 1)
    span = max(1, vocab_hi - 10)
    base = (torch.arange(ctx, dtype=torch.int32) % span) + 10
    in_pt = base.unsqueeze(0).expand(Bmax, ctx).contiguous()
    generator.prefill_forward_text(
        in_pt, page_table=page_table_torch, kv_cache=tt_kv_cache, prompt_lens=[ctx] * Bmax, warmup_prefill=False
    )

    c = ctx  # committed/anchor position; verify P fresh positions c..c+K
    S_k = max_seq_len
    # Deterministic verify candidates (also derived from the fixed prefill row).
    tokens_per_user = [int(in_pt[0, 0])] + [int(in_pt[0, min(1 + j, ctx - 1)]) for j in range(K)]

    def _from(t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(t, device=mesh_device, layout=layout, dtype=dtype, mesh_mapper=mapper)

    def _masks(B):
        # head-major rows h*P+p; per-user identical (same c) → tile across B.
        j = torch.arange(S_k)
        rf = torch.empty(P, S_k)
        rs = torch.empty(P, S_k)
        for p in range(P):
            upper = c + p
            rf[p] = torch.where(j <= upper, 0.0, NEG)
            rs[p] = torch.where((j <= upper) & (j > upper - window), 0.0, NEG)
        mf = rf.repeat(H, 1).reshape(1, 1, H * P, S_k).repeat(B, 1, 1, 1).to(torch.bfloat16)
        ms = rs.repeat(H, 1).reshape(1, 1, H * P, S_k).repeat(B, 1, 1, 1).to(torch.bfloat16)
        return _from(mf, ttnn.bfloat16, ttnn.TILE_LAYOUT), _from(ms, ttnn.bfloat16, ttnn.TILE_LAYOUT)

    def _run(call_fn, n_rows):
        """Compile (+read logits for correctness), capture, time `reps` replays."""
        logits, hidden = call_fn()
        ttnn.synchronize_device(mesh_device)
        lh = spec._logits_to_host(logits).reshape(n_rows, vocab).float().clone()
        logits.deallocate(True)
        hidden.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        logits, hidden = call_fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) / reps * 1e3
        ttnn.release_trace(mesh_device, tid)
        logits.deallocate(True)
        hidden.deallocate(True)
        return ms, lh

    logger.info(
        f"=== packed-verify batch bench | ctx={ctx} P={P} (K={K}) reps={reps} tp={tp} H_local={H} S_k={S_k} ==="
    )
    rows = []
    for B in Bs:
        pt_b = page_table_torch[:B].to(torch.int32)
        # ── packed: batch dim B, P folded into heads, fallback per-p KV write ──
        # rows user-major / position-minor: row u*P+p is user u's p-th candidate.
        x_p = _from(torch.tensor([tokens_per_user] * B, dtype=torch.int64).reshape(1, B * P), ttnn.uint32)
        pos_p = _from(torch.tensor([[c + p for u in range(B) for p in range(P)]], dtype=torch.int64), ttnn.uint32)
        mask_full, mask_slide = _masks(B)
        write_idxs = [_from(torch.full((B,), c + p, dtype=torch.int32), ttnn.int32) for p in range(P)]
        pt_packed = _from(pt_b, ttnn.int32)

        def _packed_call():
            return target.ttnn_packed_verify_forward(
                x=x_p,
                position_idx=pos_p,
                attn_mask_full=mask_full,
                attn_mask_sliding=mask_slide,
                packed_p=P,
                page_table=pt_packed,
                kv_cache=spec.tt_kv_cache,
                kv_write_idxs=write_idxs,
                embed_idx_full=None,
                embed_idx_sliding=None,
                hot_pt=None,
            )

        try:
            packed_ms, packed_lh = _run(_packed_call, B * P)  # first call compiles the packed SDPA
        except RuntimeError as e:
            if _is_l1_cb_overflow(e):
                pytest.skip(_L1_OVERFLOW_REASON)
            raise
        for t in (x_p, pos_p, mask_full, mask_slide, pt_packed, *write_idxs):
            t.deallocate(True)

        # ── batch-alias baseline ──────────────────────────────────────────────
        if B * P <= 32:
            # single-pass: B*P pseudo-users, each user's row replicated P times.
            x_a = spec._tokens_tensor([tokens_per_user[p] for u in range(B) for p in range(P)])
            pu_a, pi_a = spec._pos_tensors([c + p for u in range(B) for p in range(P)])
            pt_alias = _from(pt_b.repeat_interleave(P, dim=0), ttnn.int32)

            def _alias_call():
                return target.ttnn_verify_forward(
                    x=x_a, current_pos=pu_a, current_pos_cache=pi_a, page_table=pt_alias, kv_cache=spec.tt_kv_cache
                )

            alias_ms, alias_lh = _run(_alias_call, B * P)
            for t in (x_a, pu_a, pi_a, pt_alias):
                t.deallocate(True)
            # correctness: packed argmax vs batch-alias argmax per row. Both are
            # batched-SDPA paths whose per-user RoPE + cross-core reductions differ
            # by ~1e-1, so a divergence is only a bug at a CONFIDENT token — i.e.
            # when the reference (alias) logit gap between its argmax and packed's
            # pick is large. Near-ties (small gap) are expected to flip.
            pa = packed_lh.argmax(dim=-1)
            aa = alias_lh.argmax(dim=-1)
            n_match = int((pa == aa).sum())
            max_diff = float((packed_lh - alias_lh).abs().max())
            confident_flips = 0
            for r in range(B * P):
                if int(pa[r]) != int(aa[r]):
                    gap = float(alias_lh[r, int(aa[r])] - alias_lh[r, int(pa[r])])
                    if gap > CONFIDENT_GAP:
                        confident_flips += 1
            baseline_kind = f"alias(B*P={B*P})"
        else:
            # batch-alias can't fit B*(K+1)>32 users in one decode pass; the
            # no-packing cost is P sequential batch-B decodes.
            x_a = spec._tokens_tensor([tokens_per_user[0]] * B)
            pu_a, pi_a = spec._pos_tensors([c] * B)
            pt_alias = _from(pt_b, ttnn.int32)

            def _alias_call():
                return target.ttnn_verify_forward(
                    x=x_a, current_pos=pu_a, current_pos_cache=pi_a, page_table=pt_alias, kv_cache=spec.tt_kv_cache
                )

            one_ms, _ = _run(_alias_call, B)
            for t in (x_a, pu_a, pi_a, pt_alias):
                t.deallocate(True)
            alias_ms = one_ms * P
            n_match, max_diff, confident_flips = None, None, None
            baseline_kind = f"{P}x batch-{B} decode (B*(K+1)>32: no single-pass alias)"

        speedup = alias_ms / packed_ms if packed_ms > 0 else float("nan")
        rows.append((B, packed_ms, alias_ms, baseline_kind, speedup, n_match, max_diff, confident_flips))
        if n_match is not None:
            corr = f"corr {n_match}/{B*P} match, confident-flips={confident_flips} (max|Δlogit|={max_diff:.2f})"
        else:
            corr = "corr n/a (no single-pass alias reference at this B)"
        logger.info(
            f"[bench] B={B:>2}: packed={packed_ms:6.2f} ms | baseline({baseline_kind})={alias_ms:6.2f} ms | "
            f"speedup={speedup:4.2f}x | {corr}"
        )

    logger.info("=== summary (verify ms/iter; speedup = baseline/packed) ===")
    for B, p_ms, a_ms, kind, sp, nm, md, cf in rows:
        logger.info(f"  B={B:>2}  packed {p_ms:6.2f} ms  baseline {a_ms:6.2f} ms [{kind}]  →  {sp:4.2f}x")

    # Bug gate: only CONFIDENT-token flips are failures. Near-tie argmax
    # mismatches (reference gap ≤ CONFIDENT_GAP) are expected under bf16 /
    # batched-SDPA noise — BH CI showed B=1 at 2/4–3/4 match with
    # confident-flips=0 (max|Δlogit|≈3.7–4.6). Do not gate on raw match rate.
    #
    # B=1: zero confident flips (paired with matches_sequential gold).
    # B>1: allow ≤1 confident flip — WH@TP=8 has shown a single 1/32 confident
    # flip while BH@TP=4 was clean. Override: GEMMA4_BENCH_MAX_CONFIDENT_FLIPS.
    max_cf_b1 = 0
    max_cf_bgt1 = int(os.environ.get("GEMMA4_BENCH_MAX_CONFIDENT_FLIPS", "1"))
    for B, p_ms, a_ms, kind, sp, nm, md, cf in rows:
        if cf is None:
            continue
        max_cf = max_cf_b1 if B == 1 else max_cf_bgt1
        assert cf <= max_cf, (
            f"B={B}: {cf} confident-token flips packed vs batch-alias "
            f"(allowed≤{max_cf}; max|Δlogit|={md:.2f}; match {nm}/{B*P})"
        )


# ─────────────────── traced packed verify with per-layer inputs ───────────────
#
# The traced packed verify used to be unreachable for a per-layer-input target
# (E2B/E4B): PLI is computed on host from the candidate ids, and the verify
# forward built it with a fresh ``ttnn.from_torch(device=...)`` INSIDE the
# forward — a host write plus an allocation, both illegal during a trace capture.
#
# The fix is the same one the plain traced decode path has always used for
# ``pli_combined``: allocate the PLI buffer ONCE out-of-trace and refresh it per
# replay with ``copy_host_to_device_tensor``, so PLI enters the trace as data.
# See ``SpeculativeDecoder._pli_dev`` / ``_pli_refresh``.
#
# This test is the regression guard for that wiring. It asserts the traced chain
# reproduces the eager chain position-for-position over several iterations, which
# catches the three ways this breaks:
#   1. PLI silently dropped (layers run with pli_tt=None) — output diverges.
#   2. The trace bound to a buffer that is reallocated per call — replay N reads
#      stale/foreign PLI, so only iteration 1 matches.
#   3. Returning the freshly captured (never-executed) output buffer — iteration 1
#      is garbage while later ones are fine.
# It also runs enough iterations that later replays are *re*-replays, the case
# that surfaces trace-scratch corruption as a hang rather than a wrong value.


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 256_000_000})
def test_packed_verify_traced_pli_matches_eager(mesh_device, reset_seeds):
    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    if _is_moe_model(model_path):
        pytest.skip(_MOE_UNSUPPORTED_REASON)
    if not _is_pli_model(model_path):
        pytest.skip(
            "this test exists to cover the per-layer-input (PLI) trace path; "
            "set HF_MODEL to a PLI checkpoint (gemma-4-E2B/E4B) to run it"
        )

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    max_seq_len = 1024
    block_size = 64
    n_iters = 4  # >=3 so the last replays are re-replays of an already-replayed trace
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    assert target.hidden_size_per_layer_input, "expected a PLI target (config said so)"
    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = "The capital of France is"
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 24, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    # assistant_model unused — only the verify paths run.
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=None,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=3,
    )
    P = spec.draft_len + 1
    assert spec.target_needs_host_pli, "PLI target must report target_needs_host_pli"

    def _prefill():
        # warmup_prefill=False: see test_packed_verify_matches_sequential.
        generator.prefill_forward_text(
            in_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            warmup_prefill=False,
        )

    # Fixed candidate chain, reused verbatim by both runs so the two chains drive
    # identical tokens at identical positions (any divergence is the trace's).
    span = max(int(prefill_lens[0]) - 1, 1)
    chain = [anchor_token] + [int(in_pt[0, (j * 7 + 1) % span]) for j in range(n_iters * P)]

    def _run_chain(traced):
        _prefill()
        spec._pv_a_prev = -1
        spec._use_trace = traced
        argmaxes, hiddens = [], []
        pos = anchor_pos
        for it in range(n_iters):
            tokens = chain[it * P : it * P + P]
            positions = [pos + j for j in range(P)]
            try:
                lh, h = spec._verify(tokens, positions)
            except RuntimeError as e:
                if _is_l1_cb_overflow(e):
                    pytest.skip(_L1_OVERFLOW_REASON)
                raise
            argmaxes.append([int(torch.argmax(lh[j])) for j in range(P)])
            # The traced hidden is the PERSISTENT trace output — copy before the
            # next replay overwrites it, and never deallocate it.
            hiddens.append(spec._read_replica(h).float().reshape(P, -1).clone())
            if not traced:
                h.deallocate(True)
            pos += P
        return argmaxes, hiddens

    # GEMMA4_PV_SEED_FIRST=1 reproduces the demo's prologue: a batch=1 verify at the
    # anchor (spec.seed) BEFORE the packed chain. The demo forks 50/50 between two
    # trajectories with bit-identical seed+drafts, while this test — which goes
    # straight from prefill to packed verify — is byte-stable across processes. That
    # seed() call is the only identified structural difference, so this isolates it.
    if os.environ.get("GEMMA4_PV_SEED_FIRST") == "1":
        _prefill()
        spec._pv_a_prev = -1
        _h = spec.seed(anchor_token, anchor_pos)
        logger.info(f"[seed-first] seed_sum={spec._read_replica(_h).float().sum().item():.6f}")

    eager_ids, eager_h = _run_chain(traced=False)
    traced_ids, traced_h = _run_chain(traced=True)

    for it in range(n_iters):
        logger.info(f"iter {it}: eager={eager_ids[it]} traced={traced_ids[it]}")

    # Same program, same inputs, same weights ⇒ bit-identical argmax per row. A
    # per-iteration report matters: "only iteration 0 matches" means the trace is
    # bound to a reallocated buffer, "only iteration 0 differs" means the capture
    # branch returned its unexecuted output.
    bad = [it for it in range(n_iters) if eager_ids[it] != traced_ids[it]]
    assert not bad, (
        f"traced packed verify diverges from eager at iterations {bad}: " f"eager={eager_ids} traced={traced_ids}"
    )

    # Hidden states seed the drafter, so they must match too — a PLI mix-up can
    # leave argmax intact while corrupting the hidden that drives acceptance.
    for it in range(n_iters):
        md = (eager_h[it] - traced_h[it]).abs().max().item()
        logger.info(f"iter {it}: max|Δhidden| = {md:.3e}")
        assert md == 0.0, f"iter {it}: traced verify hidden differs from eager (max|Δ|={md:.3e})"


# ────────────────────────── determinism of greedy decode ──────────────────────
#
# Observed 2026-07-30: the spec-decode demo emits DIFFERENT text across identical
# greedy runs (3 runs at 300 tokens -> acceptance 0.76/0.42/0.42, two distinct
# output md5s), in both the traced and the eager loop.
#
# That should be impossible. Greedy speculative decoding commits exactly what plain
# greedy decode would emit: the TARGET's argmax picks every committed token, so what
# the drafter proposes can only change SPEED, never the text. Text changing means
# the target's own logits are not reproducible.
#
# Note what is already ruled out: test_packed_verify_traced_pli_matches_eager gets
# max|Δhidden| == 0.0 running the packed verify over a FIXED token sequence. So the
# verify is bit-reproducible for fixed inputs, and the nondeterminism must enter
# through the feedback loop (logits -> argmax -> next input), i.e. it takes a
# near-tie to become visible.
#
# This test separates the two candidate sources by running each chain TWICE from a
# fresh prefill in one process:
#   chain_b1     — batch=1 verify per step, the plain greedy decode path. Differs
#                  => the target itself is nondeterministic (CCL reduction order at
#                  tp=2 is the prime suspect) and every acceptance number recorded
#                  for this model is noise.
#   chain_packed — the P=K+1 packed verify driven by a FIXED token sequence, then
#                  re-driven by its own argmax output. Differs while chain_b1 is
#                  stable => the fault is spec-decode-specific: stale KV from
#                  rejected drafts (ttnn_verify_forward claims "rejected positions
#                  are simply overwritten on the next iteration").
# No drafter and no CME are involved in either chain.


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 256_000_000})
def test_target_greedy_determinism(mesh_device, reset_seeds):
    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    if _is_moe_model(model_path):
        pytest.skip(_MOE_UNSUPPORTED_REASON)

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    max_seq_len = 1024
    block_size = 64
    n_steps = int(os.environ.get("GEMMA4_DETERMINISM_STEPS", 48))
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    page_table = create_tt_page_table(1, paged_attention_config)
    # Same prompt the spec-decode demo uses, so this is comparable to those runs.
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Tell me about the history of computing in three sentences.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 32, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=None,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=3,
    )
    P = spec.draft_len + 1
    spec._use_trace = False  # eager: the demo showed the divergence in eager too

    def _prefill():
        generator.prefill_forward_text(
            in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos, warmup_prefill=False
        )

    def chain_b1():
        """Plain greedy decode: one batch=1 verify per token."""
        _prefill()
        spec._pv_a_prev = -1
        toks, tok, pos = [], anchor_token, anchor_pos
        for _ in range(n_steps):
            lh, h = spec._verify([tok], [pos])
            h.deallocate(True)
            tok = int(torch.argmax(lh[0]))
            toks.append(tok)
            pos += 1
        return toks

    def chain_packed(seed_chain):
        """Packed P-row verify, each iteration committing the target's own argmax.

        Driven by ``seed_chain`` so the inputs are identical across repeats; the
        OUTPUT argmaxes are what we compare.
        """
        _prefill()
        spec._pv_a_prev = -1
        out, pos, it = [], anchor_pos, 0
        full = [anchor_token] + list(seed_chain)
        while len(out) < n_steps:
            tokens = full[it * P : it * P + P]
            if len(tokens) < P:
                break
            try:
                lh, h = spec._verify(tokens, [pos + j for j in range(P)])
            except RuntimeError as e:
                if _is_l1_cb_overflow(e):
                    pytest.skip(_L1_OVERFLOW_REASON)
                raise
            h.deallocate(True)
            out.extend(int(torch.argmax(lh[j])) for j in range(P))
            pos += P
            it += 1
        return out[:n_steps]

    b1_a = chain_b1()
    b1_b = chain_b1()
    logger.info(f"chain_b1 run A: {b1_a}")
    logger.info(f"chain_b1 run B: {b1_b}")
    b1_div = next((i for i, (x, y) in enumerate(zip(b1_a, b1_b)) if x != y), None)

    pk_a = chain_packed(b1_a)
    pk_b = chain_packed(b1_a)
    logger.info(f"chain_packed run A: {pk_a}")
    logger.info(f"chain_packed run B: {pk_b}")
    pk_div = next((i for i, (x, y) in enumerate(zip(pk_a, pk_b)) if x != y), None)

    logger.info(
        "=== DETERMINISM VERDICT ===\n"
        f"  plain greedy (batch=1 verify): {'STABLE' if b1_div is None else f'DIVERGES at step {b1_div}'}\n"
        f"  packed verify (P={P}):         {'STABLE' if pk_div is None else f'DIVERGES at step {pk_div}'}\n"
        "  b1 diverges          -> target itself is nondeterministic (suspect CCL reduction order at tp=2)\n"
        "  only packed diverges -> spec-decode-specific: stale rejected-draft KV\n"
        "  both stable          -> nondeterminism needs the real drafter (CME bf16 topk ties) in the loop"
    )

    assert b1_div is None, f"plain greedy decode is NOT reproducible: diverges at step {b1_div}"
    assert pk_div is None, f"packed verify chain is NOT reproducible: diverges at step {pk_div}"


# ─────────────────── rejected-draft KV rollback (the real test) ────────────────
#
# test_target_greedy_determinism showed BOTH chains stable, in-process and across
# processes. But it drove the packed verify with the correct continuation, so every
# draft was ACCEPTED — it never produced a rejection, and therefore never exercised
# the rollback. That leaves the actual hypothesis untested.
#
# The claim under test is ttnn_verify_forward's: "Rejected positions are simply
# overwritten on the next iteration (KV rollback = position bookkeeping at
# batch=1)". A packed verify over [anchor, d1..dK] at p..p+K writes KV for ALL P
# rows before SDPA. When the drafts are wrong, positions p+1..p+K now hold KV for
# tokens that were never committed. Row 0 attends only up to p, so the committed
# token g[0] should be unaffected no matter how wrong the drafts are.
#
# If that holds, injecting deliberately WRONG drafts must reproduce the plain
# greedy chain exactly. If it does not, the target's output depends on rejected
# drafts — which is precisely the feedback path that would turn the drafter's
# documented CME bf16-topk tie-breaking noise into nondeterministic OUTPUT TEXT,
# explaining the 3-run demo divergence (acceptance 0.76/0.42/0.42, two md5s).
#
# No drafter here: the wrong drafts are fixed, so the test itself is deterministic.


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 256_000_000})
def test_rejected_draft_kv_does_not_corrupt_commit(mesh_device, reset_seeds):
    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    if _is_moe_model(model_path):
        pytest.skip(_MOE_UNSUPPORTED_REASON)

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    max_seq_len = 1024
    block_size = 64
    n_steps = int(os.environ.get("GEMMA4_DETERMINISM_STEPS", 40))
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Tell me about the history of computing in three sentences.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 32, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=None,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=3,
    )
    K = spec.draft_len
    spec._use_trace = False

    def _prefill():
        generator.prefill_forward_text(
            in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos, warmup_prefill=False
        )

    # Reference: plain greedy, one batch=1 verify per token.
    _prefill()
    spec._pv_a_prev = -1
    ref, tok, pos = [], anchor_token, anchor_pos
    for _ in range(n_steps):
        lh, h = spec._verify([tok], [pos])
        h.deallocate(True)
        tok = int(torch.argmax(lh[0]))
        ref.append(tok)
        pos += 1

    # Same chain, but every iteration proposes K deliberately WRONG drafts, so every
    # draft is rejected (m=0) and only the target's own argmax at row 0 is committed.
    # Advance by exactly one position per iteration, mirroring generate()'s m=0 case.
    BAD = [1, 2, 3][:K]  # fixed junk ids; never the real continuation
    _prefill()
    spec._pv_a_prev = -1
    got, tok, pos = [], anchor_token, anchor_pos
    n_rejected = 0
    for _ in range(n_steps):
        try:
            lh, h = spec._verify([tok] + BAD, [pos + j for j in range(K + 1)])
        except RuntimeError as e:
            if _is_l1_cb_overflow(e):
                pytest.skip(_L1_OVERFLOW_REASON)
            raise
        h.deallocate(True)
        g = [int(torch.argmax(lh[j])) for j in range(K + 1)]
        m, _committed = spec._accept_greedy(BAD, lh)
        n_rejected += K - m
        tok = g[0] if m == 0 else g[m]
        got.append(tok)
        pos += m + 1
        if m != 0:
            # A junk id happening to match the target's argmax would advance further
            # and break the 1:1 alignment with `ref`; bail loudly rather than compare
            # misaligned sequences.
            pytest.skip(f"junk draft was accepted at step {len(got)} (m={m}); pick different BAD ids")

    div = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), None)
    logger.info(f"plain greedy       : {ref}")
    logger.info(f"with junk drafts   : {got}")
    logger.info(f"rejected drafts    : {n_rejected} (expected {n_steps * K})")
    logger.info(
        "=== KV ROLLBACK VERDICT ===\n"
        f"  {'CLEAN — rejected-draft KV does not affect committed tokens' if div is None else f'CORRUPTION at step {div}'}\n"
        "  corruption => the target's output depends on REJECTED drafts, so drafter\n"
        "  noise (CME bf16 topk ties) propagates into output text — the demo divergence."
    )
    assert div is None, (
        f"rejected-draft KV corrupts the commit: chain diverges from plain greedy at step {div} "
        f"(ref={ref[max(0, div - 2):div + 3]} got={got[max(0, div - 2):div + 3]})"
    )


# ───────────────── partial acceptance (0 < m < K) — the untested case ──────────
#
# Coverage of the accept paths before this test:
#   m = 0   test_rejected_draft_kv_does_not_corrupt_commit — exact over 120 rejections
#   m = K   test_spec_decode_matches_greedy — passes, but only ever at 3.00/3
#   0<m<K   NOTHING
#
# Partial acceptance is where generate() does the interesting bookkeeping:
# ``committed = drafts[:m] + [g[m]]`` commits m+1 tokens at once and advances the
# anchor by m+1, and the packed verify's hot-block staging (``_pv_a_prev``, the
# rollover in ``_pv_host_inputs``) has to track a variable-length step. A defect
# there drops or duplicates committed tokens without ever raising.
#
# That matches what the demo actually emits: text with words missing mid-sentence
# ("Here isof computing", "of humanthat"), varying run to run with the drafter's
# acceptance pattern.
#
# This test forces a CHOSEN m every iteration with no drafter involved: propose the
# true continuation for the first m slots (guaranteed accepted, since they are the
# target's own argmax) and junk for the rest (guaranteed rejected). The committed
# chain must equal plain greedy for every m in 0..K.


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 256_000_000})
def test_partial_acceptance_matches_greedy(mesh_device, reset_seeds):
    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    if _is_moe_model(model_path):
        pytest.skip(_MOE_UNSUPPORTED_REASON)

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    max_seq_len = 1024
    block_size = 64
    n_steps = int(os.environ.get("GEMMA4_DETERMINISM_STEPS", 40))
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Tell me about the history of computing in three sentences.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 32, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=None,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=3,
    )
    K = spec.draft_len
    spec._use_trace = False

    def _prefill():
        generator.prefill_forward_text(
            in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos, warmup_prefill=False
        )

    # Reference: plain greedy, one batch=1 verify per token.
    _prefill()
    spec._pv_a_prev = -1
    ref, tok, pos = [], anchor_token, anchor_pos
    for _ in range(n_steps + K + 2):
        lh, h = spec._verify([tok], [pos])
        h.deallocate(True)
        tok = int(torch.argmax(lh[0]))
        ref.append(tok)
        pos += 1
    truth = [anchor_token] + ref  # truth[i] is the committed token at anchor_pos + i

    JUNK = [1, 2, 3]

    def chain_with_forced_m(want_m):
        """Chain where exactly ``want_m`` drafts are accepted each iteration."""
        _prefill()
        spec._pv_a_prev = -1
        out, pos_i = [], 0  # pos_i indexes `truth`
        while len(out) < n_steps:
            anchor = truth[pos_i]
            # First want_m drafts = the true continuation (accepted); rest = junk.
            drafts = [truth[pos_i + 1 + j] for j in range(want_m)] + JUNK[: K - want_m]
            try:
                lh, h = spec._verify([anchor] + drafts, [anchor_pos + pos_i + j for j in range(K + 1)])
            except RuntimeError as e:
                if _is_l1_cb_overflow(e):
                    pytest.skip(_L1_OVERFLOW_REASON)
                raise
            h.deallocate(True)
            m, committed = spec._accept_greedy(drafts, lh)
            if m != want_m:
                pytest.skip(f"forced m={want_m} but acceptance gave m={m} (junk id collided with argmax)")
            out.extend(committed)
            pos_i += m + 1
        return out[:n_steps]

    failures = []
    for want_m in range(0, K + 1):
        got = chain_with_forced_m(want_m)
        exp = ref[:n_steps]
        div = next((i for i, (x, y) in enumerate(zip(exp, got)) if x != y), None)
        status = "OK" if div is None else f"DIVERGES at {div}"
        logger.info(f"m={want_m}: {status}")
        if div is not None:
            logger.info(f"  expected {exp[max(0, div - 2):div + 4]}")
            logger.info(f"  got      {got[max(0, div - 2):div + 4]}")
            failures.append((want_m, div))

    logger.info(
        "=== PARTIAL ACCEPTANCE VERDICT ===\n"
        f"  {'all m in 0..K reproduce plain greedy' if not failures else f'BROKEN for m={[f[0] for f in failures]}'}\n"
        "  a failure only at 0<m<K localises the bug to the multi-token commit /\n"
        "  variable-length anchor advance, not to the verify itself."
    )
    assert not failures, f"spec-decode commit diverges from plain greedy at m={failures}"


# ─────────────── VARYING acceptance across iterations — the last gap ───────────
#
# test_partial_acceptance_matches_greedy holds m CONSTANT within a chain, so the
# anchor advances by a fixed m+1 every iteration and the packed verify's hot-block
# staging steps regularly. The real loop does not: m changes every iteration with
# whatever the drafter proposed, so the anchor advance is irregular and
# ``_pv_host_inputs``'s rollover test
#     roll = 1 if a == self._pv_a_prev + 1 else 0
# sees an irregular sequence of block indices (sometimes a == prev, sometimes
# prev+1), interleaved with ``_pv_seed_staging``'s committed-block reseed.
#
# That is the last untested combination, and it is exactly the one that would make
# the output depend on the ACCEPTANCE PATTERN — i.e. on drafter noise — which is
# the observed demo symptom (same config, acceptance 0.76/0.42/0.42, different
# text) that every other hypothesis has now failed to explain:
#   target determinism           STABLE at 48 and 300 steps, across processes
#   packed verify, fixed inputs  bit-exact
#   rejected-draft KV            exact over 120 rejections
#   drafter writes shared KV     refuted by inspection
#   constant partial acceptance  exact for m = 0..K
#
# The m sequence here is a fixed cycle, not random, so the test stays deterministic
# and reproducible while still being irregular.


def _decision_delta(a, b, k=32):
    """max|a - b| restricted to the DECISION-RELEVANT tokens (top-k union).

    max over the full 262144-wide row is the wrong statistic for "will the argmax
    agree": it is dominated by deep-negative junk logits where bf16 absolute error
    is large and utterly irrelevant to the decision. Measured on this model, the
    full-row max reads ~4-5 logits between two paths whose greedy tokens agree 86%
    of the time. Restricting to the union of each row's top-k puts the number in
    the region that actually competes for the argmax.
    """
    idx = torch.unique(torch.cat([torch.topk(a, k).indices, torch.topk(b, k).indices]))
    return float((a[idx] - b[idx]).abs().max())


def _summarise(label, vals, ulp=0.0625):
    """p50/p90/p99/max of a delta list, in absolute logits and in bf16 ULP.

    ULP is 0.0625 at the ~13-14 logit magnitudes these rows live at.
    """
    if not vals:
        logger.info(f"[pv-delta] {label}: no rows compared")
        return None
    d = sorted(vals)
    n = len(d)
    p50, p90, p99 = d[n // 2], d[int(n * 0.9)], d[min(n - 1, int(n * 0.99))]
    logger.info(
        f"[pv-delta] {label} over {n} rows: "
        f"p50={p50:.4f} ({p50 / ulp:.1f} ULP)  p90={p90:.4f} ({p90 / ulp:.1f} ULP)  "
        f"p99={p99:.4f} ({p99 / ulp:.1f} ULP)  max={d[-1]:.4f} ({d[-1] / ulp:.1f} ULP)"
    )
    return p50


def _report_deltas(row_deltas, pk_err=None, b1_err=None):
    """Report device-vs-device delta and, when available, device-vs-HF error.

    Only |device - HF| is authoritative. max|packed - batch1| measures DISAGREEMENT
    between two device paths that may share an error, so on its own it cannot say
    which one is wrong — the trap the first fp32 A/B fell into.
    """
    _summarise("max|packed - batch1|  (disagreement, NOT authoritative)", row_deltas)
    pk = _summarise("max|packed  - HF|     (packed accuracy)", pk_err or [])
    b1 = _summarise("max|batch=1 - HF|     (batch=1 accuracy)", b1_err or [])
    if pk is not None and b1 is not None:
        better = "packed" if pk < b1 else "batch=1" if b1 < pk else "tie"
        logger.info(
            f"[pv-delta] VERDICT: {better} is closer to HF at p50 "
            f"(packed {pk:.4f} vs batch=1 {b1:.4f}). A precision change should be judged "
            "on THIS, not on packed-vs-batch1 agreement."
        )


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 256_000_000})
def test_varying_acceptance_matches_greedy(mesh_device, reset_seeds):
    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    if _is_moe_model(model_path):
        pytest.skip(_MOE_UNSUPPORTED_REASON)

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    max_seq_len = 1024
    block_size = 64
    n_steps = int(os.environ.get("GEMMA4_DETERMINISM_STEPS", 80))
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Tell me about the history of computing in three sentences.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 32, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=None,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=3,
    )
    K = spec.draft_len
    spec._use_trace = False

    def _prefill():
        generator.prefill_forward_text(
            in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos, warmup_prefill=False
        )

    _prefill()
    spec._pv_a_prev = -1
    ref, tok, pos = [], anchor_token, anchor_pos
    # Keep the batch=1 row per step so a packed disagreement can be classified as a
    # near-tie (rounding) or a real error. ref_rows[k] is the verify AT position
    # anchor_pos+k, which emits the token living at anchor_pos+k+1.
    ref_rows = []
    for _ in range(n_steps + K + 2):
        lh, h = spec._verify([tok], [pos])
        h.deallocate(True)
        row = lh[0].float().clone()
        ref_rows.append(row)
        tok = int(torch.argmax(row))
        ref.append(tok)
        pos += 1
    truth = [anchor_token] + ref

    # ── optional CPU/fp32 ground truth ────────────────────────────────────────
    #
    # Comparing packed against batch=1 only shows they DIFFER, never which is
    # right. Both run on device; if the batch=1 decode SDPA also accumulates in
    # bf16 then the two can agree by sharing an error, and a change that makes one
    # MORE accurate will read as a larger delta against an inaccurate reference.
    # That is exactly how the first fp32 A/B misled: legacy tracked batch=1 for 104
    # rows while fp32 diverged at row 0, which says nothing about correctness.
    #
    # So score both device paths against HF on CPU in fp32, teacher-forced over the
    # same committed sequence. GEMMA4_PV_HF_REF=1 (needs ~20 GB RAM and a few
    # minutes to load); without it the test still runs and just reports the
    # device-vs-device delta, clearly labelled as non-authoritative.
    hf_rows = None
    if os.environ.get("GEMMA4_PV_HF_REF") == "1":
        import transformers
        from transformers import AutoConfig

        # Resolve the class from the checkpoint's declared architecture. Hardcoding
        # Gemma4UnifiedForConditionalGeneration is WRONG for E2B/E4B — those are the
        # plain Gemma4* architectures, and only 12B/31B are Unified (see
        # tests/hf_assistant_e2e_check.py::_hf_classes). Loading E2B into the
        # Unified class mismatches weights silently and yields garbage logits:
        # measured 1/85 greedy agreement with the device, which is what motivated
        # the gate below.
        _arch = (
            getattr(AutoConfig.from_pretrained(model_path, trust_remote_code=True), "architectures", None) or [None]
        )[0]
        _cls = getattr(transformers, _arch, None) if _arch else None
        assert _cls is not None, f"cannot resolve HF class for architecture {_arch!r}"

        prompt_ids = [int(t) for t in encoded[0][: prefill_lens[0]]]
        # Teacher-force prompt + the committed continuation. seq[i] is the token AT
        # position i, HF logits[i] predict position i+1, and the device verify at
        # anchor_pos+k (= L-1+k) therefore maps to hf_logits[base+k], base = L-1.
        seq = torch.tensor([prompt_ids + truth[1 : n_steps + K + 2]], dtype=torch.long)
        logger.info(f"[hf-ref] {_arch} on CPU (fp32), seq={tuple(seq.shape)} — slow")
        hf_full = _cls.from_pretrained(model_path, dtype=torch.float32).eval()
        with torch.no_grad():
            hf_logits = hf_full(input_ids=seq).logits[0].float()
        # Softcap: apply ONLY if the model did not already. Blindly applying tanh
        # would double-cap when HF does it internally; blindly skipping compares
        # capped device logits against raw HF ones (a ~40 logit offset). Detect it.
        _cap = getattr(target, "final_logit_softcapping", None)
        if _cap:
            _peak = float(hf_logits.abs().max())
            if _peak > _cap * 1.01:
                hf_logits = torch.tanh(hf_logits / _cap) * _cap
                logger.info(f"[hf-ref] applied softcapping={_cap} (HF peak was {_peak:.1f}, uncapped)")
            else:
                logger.info(f"[hf-ref] HF already softcapped (peak {_peak:.1f} <= {_cap}); not re-applying")
        base = len(prompt_ids) - 1
        hf_rows = [hf_logits[base + k] for k in range(min(len(ref_rows), hf_logits.shape[0] - base))]
        del hf_full, hf_logits

        # ── ACCEPTANCE GATE ───────────────────────────────────────────────────
        # The device batch=1 verify is a well-tested path that produces coherent
        # text and passes the repo's PCC suites. If it does NOT match this
        # reference, the reference is broken — not the hardware. Gate on that
        # before any accuracy number is emitted. This harness has already been
        # wrong twice (missing softcap, wrong model class), and each time it
        # printed a confident, plausible-looking verdict.
        # Drift check: an incremental bf16 decode chain compared against a single
        # fp32 teacher-forced pass accumulates KV-cache differences step by step.
        # If that is what we are seeing, agreement is tight EARLY and decays; if the
        # delta is flat, something is wrong at every position and drift is not the
        # explanation. Quartiles make the two cases trivially distinguishable.
        _q = max(1, len(hf_rows) // 4)
        for _qi in range(4):
            _lo, _hi = _qi * _q, min((_qi + 1) * _q, len(hf_rows))
            if _lo >= _hi:
                continue
            _qa = sum(1 for k in range(_lo, _hi) if int(hf_rows[k].argmax()) == ref[k])
            _qd = sorted(_decision_delta(ref_rows[k], hf_rows[k]) for k in range(_lo, _hi))[(_hi - _lo) // 2]
            logger.info(
                f"[hf-ref]   steps {_lo:3d}-{_hi - 1:3d}: greedy {_qa}/{_hi - _lo}, "
                f"median |batch=1 - HF| = {_qd:.4f} logits"
            )

        _agree = sum(1 for k in range(len(hf_rows)) if int(hf_rows[k].argmax()) == ref[k])
        _n = len(hf_rows)
        _d50 = sorted(_decision_delta(ref_rows[k], hf_rows[k]) for k in range(_n))[_n // 2] if _n else float("inf")
        _rate = _agree / _n if _n else 0.0
        # bf16 device vs fp32 CPU may legitimately flip a few genuine near-ties, so
        # allow 10%; but a median row delta above 1 logit is far beyond rounding.
        if _rate < 0.90 or _d50 > 1.0:
            logger.error(
                f"[hf-ref] GATE FAILED — reference REJECTED, accuracy numbers suppressed.\n"
                f"  batch=1 greedy agrees with HF on {_agree}/{_n} ({_rate:.1%}, need >=90%)\n"
                f"  median |batch=1 - HF| = {_d50:.4f} logits (need <=1.0)\n"
                f"  The device batch=1 path is well-tested, so this indicts the HARNESS: "
                f"check the resolved class ({_arch}), the softcap branch above, and the "
                f"base={base} alignment. Device-vs-device numbers below remain valid."
            )
            hf_rows = None
        else:
            logger.info(
                f"[hf-ref] GATE PASSED — batch=1 greedy agrees with HF on {_agree}/{_n} "
                f"({_rate:.1%}), median |batch=1 - HF| = {_d50:.4f} logits. "
                f"Accuracy numbers below are trustworthy."
            )

    JUNK = [1, 2, 3]
    # Irregular but fixed: block indices advance unevenly, and the m=K steps make
    # the anchor jump 4 while the m=0 steps creep by 1.
    M_CYCLE = [0, 3, 1, 0, 2, 3, 0, 1, 2, 0]

    _prefill()
    spec._pv_a_prev = -1
    got, pos_i, it, ms = [], 0, 0, []
    row_deltas = []
    pk_err, b1_err = [], []  # |device - HF| for the packed and batch=1 paths
    # try/finally: the in-loop assert below aborts on a divergence, and the delta
    # distribution is exactly what a precision A/B needs from a FAILING variant.
    # Reporting only on the success path would have made the comparison useless.
    try:
        while len(got) < n_steps:
            want_m = M_CYCLE[it % len(M_CYCLE)]
            anchor = truth[pos_i]
            drafts = [truth[pos_i + 1 + j] for j in range(want_m)] + JUNK[: K - want_m]
            try:
                lh, h = spec._verify([anchor] + drafts, [anchor_pos + pos_i + j for j in range(K + 1)])
            except RuntimeError as e:
                if _is_l1_cb_overflow(e):
                    pytest.skip(_L1_OVERFLOW_REASON)
                raise
            h.deallocate(True)
            # Compare ONLY rows whose context matches the reference chain.
            #
            # Row j attends to anchor + drafts[0..j-1]. For j <= want_m those drafts
            # are the true continuation, so the row is conditioned exactly as the
            # batch=1 chain was and the delta is genuine op-vs-op error. For
            # j > want_m the context contains the JUNK ids injected to force a
            # rejection, so a large delta there is CORRECT behaviour, not error.
            # Including those rows put ~40-logit junk-context deltas into the
            # distribution and made the p50/p90/max meaningless (they read ~700 ULP
            # while a valid-context position measured 1.125).
            for _j in range(want_m + 1):
                _k = pos_i + _j
                if _k < len(ref_rows):
                    row_deltas.append(_decision_delta(lh[_j].float(), ref_rows[_k]))
                    if hf_rows is not None and _k < len(hf_rows):
                        pk_err.append(_decision_delta(lh[_j].float(), hf_rows[_k]))
                        b1_err.append(_decision_delta(ref_rows[_k], hf_rows[_k]))

            m, committed = spec._accept_greedy(drafts, lh)
            # m > want_m is benign: a junk id happened to equal the target's argmax.
            # m < want_m is NOT — drafts[:want_m] ARE the target's own greedy
            # continuation, so rejecting one means the packed verify produced a
            # different argmax than the plain batch=1 chain at that position. Failing
            # loudly here is the point of the test; skipping would hide it.
            if m < want_m:
                # Classify: near-tie/rounding, or a real packed-SDPA error? Compare the
                # PACKED row against the BATCH=1 row for the same absolute position.
                # Small top-2 gap AND small packed-vs-batch1 delta => rounding flipped a
                # tie. Large gap or large delta => the packed verify is actually wrong,
                # which is a far more serious defect than a tie-break difference.
                k = pos_i + m  # ref_rows index of the verify at this row's position
                pk = lh[m].float()
                b1 = ref_rows[k]
                want_tok, got_tok = drafts[m], int(torch.argmax(pk))
                pv, pi = torch.topk(pk, 2)
                bv, bi = torch.topk(b1, 2)
                pk_gap = float(pv[0] - pv[1])
                b1_gap = float(bv[0] - bv[1])
                # How much the two ops disagree on the two contending tokens.
                delta_want = float(pk[want_tok] - b1[want_tok])
                delta_got = float(pk[got_tok] - b1[got_tok])
                max_abs = float((pk - b1).abs().max())
                # bf16 ULP at this magnitude, so the deltas can be read in units the
                # hardware actually resolves rather than as bare floats.
                import math as _math

                ulp = 2.0 ** (_math.floor(_math.log2(max(abs(float(bv[0])), 1e-9))) - 7)
                # The question is NOT "is the gap explainable by numerical difference"
                # (it almost always is) but "is the op's error SMALLER than the decision
                # margin". If the packed-vs-batch1 delta exceeds the top-2 gap, the
                # packed verify cannot reliably reproduce greedy at ANY position whose
                # margin is under that delta — an accuracy deficit, not a tie-break.
                if abs(delta_want) <= ulp and pk_gap <= 2 * ulp:
                    verdict = "TIE AT ULP — both within one ULP; unavoidable bf16 tie-break"
                elif max_abs > pk_gap:
                    verdict = (
                        f"ACCURACY DEFICIT — packed verify errs by up to {max_abs:.4f} "
                        f"({max_abs / ulp:.0f} ULP) while the decision margin is only "
                        f"{pk_gap:.4f} ({pk_gap / ulp:.0f} ULP). The error DOMINATES the "
                        "margin, so argmax will flip at any position with a gap below the "
                        "error. Not benign rounding — raise packed-SDPA precision."
                    )
                else:
                    verdict = "packed error is below the decision margin; disagreement is not explained by precision"
                logger.info(
                    "=== DISAGREEMENT DIAGNOSTIC ===\n"
                    f"  iter {it}, anchor offset {pos_i}, row {m}, abs position {anchor_pos + pos_i + m}\n"
                    f"  packed  top2: id={int(pi[0])} v={float(pv[0]):.6f} | id={int(pi[1])} v={float(pv[1]):.6f}"
                    f"  -> gap {pk_gap:.6f}\n"
                    f"  batch=1 top2: id={int(bi[0])} v={float(bv[0]):.6f} | id={int(bi[1])} v={float(bv[1]):.6f}"
                    f"  -> gap {b1_gap:.6f}\n"
                    f"  true token {want_tok}: packed {float(pk[want_tok]):.6f} vs batch1 "
                    f"{float(b1[want_tok]):.6f}  (delta {delta_want:+.6f})\n"
                    f"  packed picked {got_tok}: packed {float(pk[got_tok]):.6f} vs batch1 "
                    f"{float(b1[got_tok]):.6f}  (delta {delta_got:+.6f})\n"
                    f"  max|packed - batch1| over the whole 262144-wide row: {max_abs:.6f}\n"
                    f"  bf16 ULP at this magnitude: {ulp:.6f}  "
                    f"(top-2 gap = {pk_gap / ulp:.1f} ULP, op delta on the contended token = "
                    f"{abs(delta_want) / ulp:.1f} ULP, max row delta = {max_abs / ulp:.1f} ULP)\n"
                    f"  VERDICT: {verdict}"
                )
            assert m >= want_m, (
                f"iter {it} (anchor offset {pos_i}, forced m={want_m}): packed verify rejected a TRUE "
                f"continuation draft at row {m} — its argmax disagrees with plain greedy. "
                f"drafts={drafts} accepted={m}"
            )
            if m != want_m:
                pytest.skip(f"junk id collided with argmax at iter {it} (m={m} > want_m={want_m}); pick other JUNK ids")
            ms.append(m)
            got.extend(committed)
            pos_i += m + 1
            it += 1

    finally:
        _report_deltas(row_deltas, pk_err, b1_err)

    got = got[:n_steps]
    exp = ref[:n_steps]
    div = next((i for i, (x, y) in enumerate(zip(exp, got)) if x != y), None)
    logger.info(f"m sequence ({len(ms)} iters): {ms}")
    logger.info(
        "=== VARYING ACCEPTANCE VERDICT ===\n"
        f"  {'CLEAN — irregular acceptance still reproduces plain greedy' if div is None else f'DIVERGES at step {div}'}"
    )
    if div is not None:
        logger.info(f"  expected {exp[max(0, div - 3):div + 4]}")
        logger.info(f"  got      {got[max(0, div - 3):div + 4]}")
    assert div is None, f"varying acceptance diverges from plain greedy at step {div}"


# ──────────────── packed-verify cost vs context length (the sweep) ─────────────
#
# The packed verify measured 4.7x a plain decode at ~100-320 token contexts. Two
# structural reasons it could be context-dependent, pulling in OPPOSITE directions:
#
#   (a) S_k BUCKET PADDING makes it look bad at SHORT context.
#       `_pv_sk_bucket = 1024`, and the packed SDPA runs is_causal=False against an
#       explicit [1,1,H*P,S_k] mask, so it attends over the full padded S_k while a
#       plain decode stops at cur_pos. At context 128 that is 1024/128 = 8x wasted.
#       This shrinks to nothing as real context approaches the bucket.
#
#   (b) NO SLIDING-WINDOW SKIP makes it look bad at LONG context.
#       `_packed_verify_sdpa` passes sliding_window_size=None (decode.py:504,520),
#       where the ordinary decode path passes the real window (decode.py:310,329).
#       E2B is 28 sliding layers (window=512) + 7 full. So plain decode reads <=512
#       keys on 28/35 layers forever, while the packed path reads S_k on ALL of them.
#       This penalty GROWS without bound as context grows.
#
# Predicted key-positions read (35 layers, window 512, P=4):
#     ctx    plain            packed         ratio
#     128    35*128 =  4480   35*1024=35840   8.0x
#     512    28*512+7*512  =  17920  35840    2.0x
#    1024    28*512+7*1024 =  21504  35840    1.7x   <- predicted minimum
#    2048    28*512+7*2048 =  28672  71680    2.5x
#    4096    28*512+7*4096 =  43008 143360    3.3x
#
# So the prediction is a U-SHAPE with a minimum near the bucket size, NOT monotone
# improvement. If the measured ratio follows it, the fix is two-part: size S_k to the
# real context (kills a), and plumb sliding_window_size into the packed SDPA (kills
# b). If the ratio is flat instead, neither dominates and the cost is in the
# per-layer staging path, which is O(S2)=128 and context-independent.
#
# Times DEVICE work only — no host readback, no logits to host.


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 256_000_000})
def test_packed_verify_context_sweep(mesh_device, reset_seeds):
    import time

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    if _is_moe_model(model_path):
        pytest.skip(_MOE_UNSUPPORTED_REASON)

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig

    ctxs = [int(c) for c in os.environ.get("GEMMA4_SWEEP_CTX", "128,512,1024,2048,4096").split(",")]
    reps = int(os.environ.get("GEMMA4_SWEEP_REPS", 20))
    max_seq_len = max(ctxs) * 2
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    page_table = create_tt_page_table(1, paged_attention_config)
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=None,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=3,
    )
    P = spec.draft_len + 1
    window = target.hf_config.sliding_window
    n_layers = len(target.layers)
    n_slide = sum(1 for t in target.hf_config.layer_types if t == "sliding_attention")
    n_full = n_layers - n_slide
    logger.info(
        f"=== packed-verify context sweep | P={P} layers={n_layers} ({n_slide} sliding/{n_full} full) window={window} reps={reps} ==="
    )

    rows = []
    for ctx in ctxs:
        # Deterministic filler prefill of exactly `ctx` tokens.
        span = max(min(ctx, 64), 2)
        in_pt = ((torch.arange(ctx, dtype=torch.int32) % span) + 10).reshape(1, ctx)
        generator.prefill_forward_text(
            in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=[ctx], warmup_prefill=False
        )
        c = ctx - 1  # anchor position
        tok = int(in_pt[0, -1])
        tokens = [tok] * P

        def _time(fn):
            for _ in range(3):
                out = fn()
                for t in out if isinstance(out, tuple) else (out,):
                    t.deallocate(True)
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(reps):
                out = fn()
                for t in out if isinstance(out, tuple) else (out,):
                    t.deallocate(True)
            ttnn.synchronize_device(mesh_device)
            return (time.perf_counter() - t0) / reps

        try:
            # single-token verify — the plain-decode-shaped unit cost
            x1 = spec._tokens_tensor([tok])
            pu1, pi1 = spec._pos_tensors([c])
            pt1 = spec._page_table(1)
            pli1 = spec._pli_dev([tok])
            t1 = _time(
                lambda: target.ttnn_verify_forward(
                    x=x1,
                    current_pos=pu1,
                    current_pos_cache=pi1,
                    page_table=pt1,
                    kv_cache=spec.tt_kv_cache,
                    pli_stacked=pli1,
                )
            )
            for t in (x1, pu1, pi1, pt1):
                t.deallocate(True)
            if pli1 is not None:
                pli1.deallocate(True)

            # packed verify, P candidates folded into the query-head dim
            spec._pv_setup()
            spec._pv_a_prev = -1
            spec._pv_seed_staging(c)
            h = spec._pv_host_inputs(c, P)
            dev = spec._pv_device_inputs(tokens, h, with_pli=True)
            dev["pt"] = spec._page_table(1)
            tp = _time(lambda: spec._pv_call(dev, P, tokens))
            s_k = h["S_k"]
        except RuntimeError as e:
            if _is_l1_cb_overflow(e):
                logger.warning(f"ctx={ctx}: L1 overflow at S_k — skipping this point")
                continue
            raise

        # Predicted key-positions ratio from the two structural effects above.
        plain_keys = n_slide * min(ctx, window) + n_full * ctx
        packed_keys = n_layers * s_k
        rows.append((ctx, s_k, t1 * 1e3, tp * 1e3, tp / t1, packed_keys / plain_keys))
        logger.info(
            f"  ctx={ctx:5d}  S_k={s_k:5d}  T_single={t1*1e3:7.2f} ms  T_packed={tp*1e3:7.2f} ms  "
            f"ratio={tp/t1:5.2f}x  (predicted from key-reads: {packed_keys/plain_keys:5.2f}x)"
        )

    logger.info("=== SWEEP SUMMARY ===")
    logger.info(f"{'ctx':>6} {'S_k':>6} {'T_single':>9} {'T_packed':>9} {'measured':>9} {'predicted':>10}")
    for ctx, s_k, t1, tp, r, pr in rows:
        logger.info(f"{ctx:6d} {s_k:6d} {t1:8.2f}ms {tp:8.2f}ms {r:8.2f}x {pr:9.2f}x")
    if len(rows) >= 3:
        best = min(rows, key=lambda r: r[4])
        logger.info(
            f"minimum ratio {best[4]:.2f}x at ctx={best[0]}. "
            "U-shape (min near the S_k bucket) => fix BOTH: size S_k to real context, and pass "
            "sliding_window_size into _packed_verify_sdpa. Monotone decreasing => bucket padding only. "
            "Flat => cost is the per-layer staging path, not the SDPA."
        )


# ─────────────── is the DRAFTER bit-reproducible? (the last untested piece) ────
#
# Greedy spec decode diverges at the FIRST generated token across runs with an
# identical config and a fresh prefill (two distinct trajectories, sampled roughly
# at random). Everything else has been checked for determinism and is clean:
#   * single-token verify chain  — byte-identical at 48 and 300 steps, cross-process
#   * packed verify, fixed input — bit-exact traced vs eager (max|dhidden| = 0.0)
#   * rejected-draft KV          — commits exact over 120 forced rejections
#   * partial / varying acceptance — exact for every m
# The drafter is the one component never tested in isolation, and it is the only
# thing that can perturb the verify's INPUTS (its drafts become verify rows).
#
# This calls _draft twice with byte-identical arguments in ONE process and compares
# the proposed tokens. Any difference is drafter nondeterminism and explains the
# whole picture; identical output exonerates it and moves the search to the seed or
# the prefill.


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 256_000_000})
def test_drafter_is_deterministic(mesh_device, reset_seeds):
    model_path = os.getenv("HF_MODEL")
    assistant_path = os.getenv("GEMMA4_ASSISTANT_MODEL")
    if not model_path or not assistant_path:
        pytest.skip("set HF_MODEL and GEMMA4_ASSISTANT_MODEL to run")

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    max_seq_len, block_size = 1024, 64
    n_trials = int(os.environ.get("GEMMA4_DRAFTER_TRIALS", 8))
    pac = PagedAttentionConfig(block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size))
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=pac,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=assistant_path,
    )
    page_table = create_tt_page_table(1, pac)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Tell me about the history of computing in three sentences.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 32, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=3,
    )
    spec._use_trace = False
    generator.prefill_forward_text(
        in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos, warmup_prefill=False
    )
    # One seed hidden, reused verbatim for every trial, so the inputs are identical.
    anchor_hidden = spec.seed(anchor_token, anchor_pos)

    runs = []
    for _ in range(n_trials):
        drafts, _ = spec._draft(anchor_token, anchor_hidden, anchor_pos)
        runs.append(list(drafts))
        logger.info(f"  drafts = {drafts}")

    uniq = {tuple(r) for r in runs}
    logger.info(
        f"=== DRAFTER DETERMINISM: {len(uniq)} distinct result(s) over {n_trials} identical calls ===\n"
        f"  {'DETERMINISTIC — drafter exonerated; look at the seed or prefill' if len(uniq) == 1 else 'NONDETERMINISTIC — this is the root cause of the trajectory fork'}\n"
        f"  distinct: {sorted(uniq)}"
    )
    assert (
        len(uniq) == 1
    ), f"drafter is NONDETERMINISTIC: {len(uniq)} distinct draft sets from identical inputs: {sorted(uniq)}"


# ───────────── does the PACKED verify write the KV cache correctly? ────────────
#
# The drafter scores 0.98 when driven from a single-token-verify chain (export
# replay) but 0.02 in the live loop, at the SAME max_seq_len=1024, with the same
# weights, prompt and drafter. Seed source is not the cause (reseed, which is
# structurally identical to the export replay, gives 0.04) and neither is the read
# bound (KVBOUND=1 gives 0.03). What remains is the KV cache CONTENT.
#
# The two paths write KV differently:
#   single-token verify : paged_update_cache at one position
#   packed verify       : persistent staging + a loop-free embedding-gather merge,
#                         then paged_fill_cache of whole hot blocks
#
# A bug in the packed write path would be INVISIBLE to the target — it re-reads the
# hot block through staging — but fatal to the DRAFTER, which cross-attends to the
# committed cache. That matches every observation, including why
# test_rejected_draft_kv_does_not_corrupt_commit passes (it checks committed TOKENS,
# not cache bytes).
#
# This drives both paths over the SAME committed token sequence and diffs the cache.


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 256_000_000})
def test_packed_verify_writes_correct_kv(mesh_device, reset_seeds):
    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    if _is_moe_model(model_path):
        pytest.skip(_MOE_UNSUPPORTED_REASON)

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    from .test_spec_decode import _depage

    max_seq_len, block_size = 1024, 64
    n_steps = int(os.environ.get("GEMMA4_KVCHECK_STEPS", 24))
    pac = PagedAttentionConfig(block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size))
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=pac,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    page_table = create_tt_page_table(1, pac)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Tell me about the history of computing in three sentences.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 32, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=None,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=3,
    )
    K = spec.draft_len
    spec._use_trace = False

    def _prefill():
        generator.prefill_forward_text(
            in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos, warmup_prefill=False
        )

    def _snapshot(n_pos):
        """De-page the SHARED KV caches (what the drafter cross-attends to)."""
        out = {}
        for lt, idx in target.last_kv_layer_by_type.items():
            kc, vc = target.tt_kv_cache[idx]
            rep = lt == "full_attention"
            out[lt] = (
                _depage(kc, page_table, n_pos, block_size, mesh_device, rep),
                _depage(vc, page_table, n_pos, block_size, mesh_device, rep),
            )
        return out

    # ── reference: single-token verify per committed token ────────────────
    _prefill()
    spec._pv_a_prev = -1
    ref_tokens, tok, pos = [], anchor_token, anchor_pos
    for _ in range(n_steps):
        lh, h = spec._verify([tok], [pos])
        h.deallocate(True)
        tok = int(torch.argmax(lh[0]))
        ref_tokens.append(tok)
        pos += 1
    # Compare ONLY positions the reference chain actually wrote: it advances one
    # token per step from anchor_pos, so its last write is anchor_pos+n_steps-1.
    # Including one more position compares a packed junk-draft slot against a slot
    # the reference never touched — a guaranteed false positive (this is what made
    # an earlier version of this test report a nonexistent "3 positions differ").
    n_pos = anchor_pos + n_steps
    ref_kv = _snapshot(n_pos)

    # ── packed: same committed chain, advancing ONE token per iteration with
    # junk drafts (m=0) — exactly the live loop's dominant case at 0.02 accept.
    _prefill()
    spec._pv_a_prev = -1
    JUNK = [1, 2, 3][:K]
    tok, pos = anchor_token, anchor_pos
    for step in range(n_steps):
        try:
            lh, h = spec._verify([tok] + JUNK, [pos + j for j in range(K + 1)])
        except RuntimeError as e:
            if _is_l1_cb_overflow(e):
                pytest.skip(_L1_OVERFLOW_REASON)
            raise
        h.deallocate(True)
        tok = int(torch.argmax(lh[0]))  # row 0 = the committed token
        assert tok == ref_tokens[step], f"step {step}: packed committed {tok}, reference {ref_tokens[step]}"
        pos += 1
    packed_kv = _snapshot(n_pos)

    logger.info(f"=== KV cache: packed-verify writes vs single-token writes ({n_pos} positions) ===")
    worst = 0.0
    worst_pcc = 1.0
    for lt in ref_kv:
        for name, a, b in (("K", ref_kv[lt][0], packed_kv[lt][0]), ("V", ref_kv[lt][1], packed_kv[lt][1])):
            # PCC, not bit-equality. These are two DIFFERENT op sequences computing
            # the same K/V — packed multi-row RoPE + staging-gather +
            # paged_fill_cache, versus single-row RoPE + paged_update_cache — so a
            # few bf16 ULP of disagreement is the correct expectation. Earlier
            # revisions of this test asserted an absolute 1e-3 and reported a
            # nonexistent bug; chasing that with ever-looser absolute bounds is
            # threshold-tuning, so score it the way the rest of the repo does.
            d = (a - b).abs()
            per_pos = d.amax(dim=(0, 1, 3))
            fa, fb = a.flatten().double(), b.flatten().double()
            pcc = float(
                torch.corrcoef(torch.stack([fa, fb]))[0, 1] if fa.std() > 0 and fb.std() > 0 else torch.tensor(1.0)
            )
            worst = max(worst, float(d.max()))
            worst_pcc = min(worst_pcc, pcc)
            logger.info(f"  {lt:18s} {name}: PCC={pcc:.8f}  max|diff|={float(d.max()):.6f}")
            logger.info(
                "      tail |diff| by position  "
                + " ".join(f"{i}:{float(per_pos[i]):.3f}" for i in range(max(0, n_pos - 8), n_pos))
            )

    logger.info(
        "  VERDICT: "
        + (
            f"packed write path AGREES with single-token writes (min PCC={worst_pcc:.8f})"
            if worst_pcc >= 0.999
            else f"PACKED VERIFY WRITES A DIFFERENT CACHE (PCC={worst_pcc:.6f}, max|diff|={worst:.4f}) — "
            "invisible to the target (re-reads via staging) but fatal to the drafter"
        )
    )
    assert worst_pcc >= 0.999, f"packed verify's KV cache disagrees with single-token writes: min PCC={worst_pcc:.6f}"


def test_pli_device_matches_host(mesh_device, reset_seeds):
    """On-device PLI must match the host path that currently feeds ``pli_stacked``.

    This is the gate for the FUSED single-trace iteration (_fused_body_batched): that
    trace keeps the drafter's argmax/re-embed chain on device, so no draft token ever
    reaches the host and host PLI cannot be built for the verify. On-device PLI is the
    only thing blocking it -- and the fused trace is the only configuration that both
    sidesteps the two-traces-one-queue hang (there is nothing to alternate) and is
    projected to beat plain decode.

    The 4.38 GiB embed_tokens_per_layer table was long assumed to make this impractical;
    measured on a single p150a it uploads and gathers fine.
    """
    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    if _is_moe_model(model_path):
        pytest.skip(_MOE_UNSUPPORTED_REASON)
    if not _is_pli_model(model_path):
        pytest.skip("needs a PLI checkpoint (gemma-4-E2B/E4B)")

    import torch

    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    max_seq_len, block_size = 1024, 64
    generator, _, _ = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=PagedAttentionConfig(
            block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
        ),
        bounded_sliding_kv_cache=False,
    )
    model = generator.model[0]
    assert model.hidden_size_per_layer_input and model.per_layer_input_weights

    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
    rows = 4
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (1, rows), dtype=torch.int64)
    hidden = model.hf_config.hidden_size
    embeds = torch.randn(1, 1, rows, hidden, dtype=torch.float32) * 0.05

    # Host reference: the exact function that builds pli_stacked today. It returns a
    # list of [1, rows, pli_size], one per layer.
    host_list = model._compute_per_layer_inputs(ids, embeds.reshape(1, rows, hidden))
    host = torch.stack([h.reshape(1, rows, -1).float() for h in host_list], dim=0)

    ids_tt = ttnn.from_torch(
        ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper
    )
    emb_tt = ttnn.from_torch(
        embeds, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
    )
    dev = model.compute_pli_device(ids_tt, emb_tt)
    got = ttnn.to_torch(ttnn.get_device_tensors(dev)[0]).float()

    logger.info(f"  host={tuple(host.shape)}  device={tuple(got.shape)}")
    assert tuple(got.shape) == tuple(host.shape), f"shape mismatch: {tuple(got.shape)} vs {tuple(host.shape)}"
    pcc = torch.corrcoef(torch.stack([host.flatten(), got.flatten()]))[0, 1].item()
    logger.info(f"  on-device PLI vs host: PCC={pcc:.8f}  max|diff|={(host - got).abs().max().item():.6f}")
    assert pcc > 0.999, f"on-device PLI disagrees with host: PCC={pcc}"

# DFlash vs naive speculative decoding on tt-metal — feasibility, and synergy with L1 caching

Inputs: `dflash.pdf` (DFlash: Block Diffusion for Flash Speculative Decoding, Chen et al., ICML 2026),
`notes.md` (the spec-decode scoping notes), and this folder's measured compute-bound study
(`COMPUTE_BOUND_PROOF.md`). Platform context: Blackhole P150, Llama-3.1-8B, batch-1 decode.

## TL;DR
1. The metric that decides everything on tt-metal is **dispatches per accepted token**, because decode
   is dispatch-bound. Naive (autoregressive-draft) spec decode pays ~`(γ+1)` dispatches/cycle for
   `τ≈2-3` accepted tokens; DFlash pays ~`2` dispatches/cycle for `τ≈6.5`. DFlash wins on exactly the
   axis tt-metal is bottlenecked on.
2. The "1-row QKV projection" framing is the wrong reason naive spec decode is poor. Those idle matmul
   rows are ~free in wall-clock at batch-1. The real reason is **γ sequential dispatch-bound draft
   passes**, each paying tt-metal's large model-size-independent per-step overhead. DFlash collapses
   that to **one** parallel draft pass.
3. Feasibility: naive is *less code, low ceiling*; DFlash is *more code, high ceiling*, and its needed
   attention primitives already exist in tt-metal. DFlash's dominant gate is **obtaining a trained,
   target-aligned draft model** (port their HF checkpoint, do not train from scratch).
4. Synergy with **L1 KV cache: orthogonal, leaning mildly negative.** DFlash's verify reuses one KV
   read across `γ` query rows, which makes attention *more* compute-bound and hides the KV read *even
   more* — so L1-KV's read-latency edge is even more irrelevant, and DFlash adds rollback complexity to
   the L1 ring-write path.
5. The matched L1 lever for DFlash is **weights, not KV** (your intuition is correct): cache the small,
   frequently-run **draft model's weights** (+ shared embedding/LM-head + the persistent target-feature
   blob) in L1. Caveat: whether the draft fully fits L1 is a capacity question (a target-width 5-layer
   draft is ~GB-scale > aggregate L1; a narrower draft could fit). Note a weight-caching feature was
   built and **removed** on this branch (`8afb4bb`, `114e5a7`) — this would build on that.

---

## 1. The deciding metric: dispatches per accepted token
tt-metal batch-1 decode is dispatch-bound: per step a large fixed host cost (command-queue dispatch,
kernel launch, sampling) dominates, with sub-ms-to-~16ms device compute (`notes.md`; the ~31 ms fixed
per-step overhead isolated in `COMPUTE_BOUND_PROOF.md` §8c). On such a platform, latency is set by how
many **sequential device dispatches** you issue per output token, far more than by per-pass FLOPs or
bytes. Speculative decoding's whole value is reducing target steps per token; its whole risk on
tt-metal is **adding** dispatches via the draft.

## 2. Premise correction (mechanism, not conclusion)
Your conclusion — naive spec decode performs poorly — is right. The stated mechanism is not:
- "31 draft tokens via the inefficient 1-row QKV projection" implies the cost is wasted matmul rows.
  At batch-1 the linear ops are weight-streaming/dispatch-bound, so the 31/32 idle rows cost ~nothing
  in wall-clock (`notes.md` 28-30, `COMPUTE_BOUND_PROOF.md` §8c). Filling them is ~free, not the prize.
- The real penalty: a naive AR draft runs **γ sequential forward passes** (paper Eq. 2,
  `T_draft = γ·t_step`), each paying the per-step fixed dispatch overhead, which is essentially
  **model-size-independent**. A tiny draft model does not save the dominant term. That is what DFlash
  removes: all `γ` tokens in **one** forward pass (Eq. 3, `T_draft = t_parallel`, insensitive to `γ`).

## 3. Cost model on tt-metal (per spec cycle), from `L = (T_draft + T_verify)/τ`

| | naive AR-draft | DFlash (block diffusion) |
|---|---|---|
| draft | `γ` sequential dispatch-bound passes | **1** block pass, M=`γ` (fills the 32-row tile) |
| verify | 1 target pass, M=`γ` (fills the tile) | 1 target pass, M=`γ` |
| dispatches / cycle | ~`γ+1` | **~2** |
| acceptance `τ` | ~2-3 (shallow draft, saturates) | ~6.5 (deeper draft + target-feature conditioning) |
| **dispatches / accepted token** | `(γ+1)/τ` ≈ 5-9 | **`2/τ` ≈ 0.3** |
| measured speedup (GPU, paper) | EAGLE-3 ~2× | ~4.9× greedy / ~4.1× sampled |

`γ` = DFlash block size (16; 10 for LLaMA), comfortably under the 32-row tile. So on tt-metal **both
the draft and the verify are single tile-filling dispatches** — this is simultaneously the
`notes.md` "decode-matmul depad" win on both sides. Naive AR draft gets the depad on the verify side
only, while reintroducing `γ` sequential draft dispatches — the worst case for a dispatch-bound chip.

## 4. Feasibility — what each needs (tt-metal has no spec-decode path today)
Verified: `tt_transformers` has no speculative/EAGLE/Medusa/draft path; both start from scratch on the
verify + rollback plumbing.

**Shared (both):**
- **Verify** = M=`γ` target forward against the existing KV. Primitive exists:
  `chunked_scaled_dot_product_attention(chunk_start_idx=...)` (`sdpa.hpp:36`, used at
  `attention.py:1762`).
- **Accept/reject + rollback** of speculative KV writes. Hardest shared piece; it lands directly on
  the KV ring-write path and is *more* delicate with L1-KV (rejected tokens must not corrupt the ring;
  `notes.md` 39).

**Naive-only:**
- A small AR draft model (own weights + KV) or self-spec heads, reusing the existing decode path.
  Minimal new kernels — but only viable if the draft loop is **traced** (to strip dispatch), and even
  then `τ` is low and `γ` sequential-step latency drags it. *Low effort, low ceiling.*

**DFlash-only (higher effort, but where the payoff lives):**
1. **A trained, target-aligned block-diffusion draft model** — the dominant gate. Port their released
   HF checkpoints (Qwen3-4B/8B, LLaMA-3.1-8B) to tt-metal; do **not** train from scratch (~800K
   samples, H200-scale). The arch is a small 5-layer transformer (8 for Coder), sharing the target's
   embedding + LM head.
2. **Bidirectional (non-causal) block attention** over `[anchor + mask tokens]`. Primitive exists:
   `scaled_dot_product_attention(is_causal=False, attn_mask=...)` (`sdpa.hpp:20-21`). New work = the
   block mask + mask-token embeddings, not a new kernel.
3. **Target context-feature extraction + KV injection**: during the target prefill, grab hidden states
   from ~5 layers, fuse via a small projection, and inject into every draft layer's K/V cache
   (persistent, reused across the cycle). New plumbing in `model.py` (expose intermediate hidden
   states) + a projection matmul + a custom draft KV layout. Moderate.
4. **Block-diffusion sampling**: one forward pass denoises all masked positions (argmax over the
   block); straightforward once 1-3 exist.

## 5. Synergy with the L1 KV cache — orthogonal, leaning mildly negative
The L1 KV cache stores the **target** KV in L1, betting on a faster KV read. DFlash works against that
bet, not with it:
- **Verify deepens compute-boundedness.** Verifying `γ` query positions reuses **one** target KV read
  across `γ` query rows, raising attention arithmetic intensity (more compute per byte) and pushing
  the verify *further* into compute-bound (`notes.md` 33-37). The KV read was already 100% hidden
  behind compute at batch-1 (`COMPUTE_BOUND_PROOF.md` §5); under DFlash it is hidden with even more
  margin. So L1-KV's only physical edge (lower read latency) is *more* irrelevant under DFlash, not
  less. There is no read-bound regime here for faster L1 KV to win.
- **Rollback adds complexity to the L1 ring-write path.** Speculative KV writes that get rejected must
  be rolled back; the L1 ring/tier write path (`_write_adaptive_l1_tiers`, the on-device ring index)
  is exactly where that complexity concentrates. This is a net *cost* of combining them, not a benefit.
- **Capacity is unaffected either way.** DFlash does not change how much target KV must be resident; it
  changes how many query rows read it per cycle.

Conclusion: **DFlash and L1-KV are independent levers** (DFlash cuts dispatches/token; L1-KV is a KV
*capacity* play that does not affect latency). They compose without conflict at the capability level,
but DFlash makes the L1-KV *latency* thesis even more of a non-starter and adds rollback work. Treat
them as orthogonal; do not expect L1-KV to amplify DFlash.

## 6. The matched L1 lever for DFlash is WEIGHTS, not KV
Your instinct ("integrate with weight caching, not KV") is correct, with a precise rationale:
- **The dominant DRAM stream is weights, not KV** (`COMPUTE_BOUND_PROOF.md` §8c: 7.97 GB weights vs
  62 MB KV per step, **128:1**). Spec decode's main win is *amortizing the target weight stream over
  `τ` tokens* — the same effect as batching the verify. That win is algorithmic (fewer target passes),
  not something L1-KV touches.
- **The draft runs every cycle and is the per-cycle weight/dispatch-bound cost.** Where on-chip
  residency would actually help DFlash is keeping the **draft model's weights** (plus the shared
  embedding/LM-head it reuses, plus the small persistent target-feature blob it conditions on) in L1,
  so the frequently-run draft does not re-stream from DRAM each cycle. That is *weight* caching, the
  exact opposite stream from L1-KV.
- **Capacity caveat (honest).** A 5-layer draft at the target's hidden width is ~1.1 GB (bfp8) ≫
  aggregate L1 (~100 MB), so a full target-width draft does **not** fit L1 — weight-residency would
  need a *narrower* draft (spec-decode drafts usually are narrower than EAGLE/Medusa's single layer
  suggests is enough). Even unfitted, the draft's weight stream is ~7× smaller than a target pass, so
  it is the cheap part; L1-residency is a bonus that turns "cheap" into "free," contingent on a small
  enough draft. This is, once again, a **capacity** decision — the recurring theme of this folder.
- **Prior art on this branch:** a weight-caching/weight-sharding feature was implemented and then
  **removed** (`8afb4bb`, `114e5a7` "Remove weight caching feature"; `CODE_AUDIT.md` appendix). A
  DFlash-for-tt-metal effort that wants weight residency would resurrect/extend that path, scoped to
  the small draft model rather than the 8B target (where weights never fit L1).

So: **integrate DFlash with weight caching (draft-weight residency), not with the L1 KV cache.** The
L1-KV work and DFlash are best kept as separate, orthogonal levers.

## 7. Implementation task breakdown (DFlash path, recommended)
Ordered by dependency; each is a verifiable milestone.
1. **Spec-decode harness (shared):** target M=`γ` verify via `chunked_scaled_dot_product_attention`;
   greedy accept/reject; KV rollback for rejected tokens (start on the DRAM KV path, then port to the
   L1 ring). Validate correctness against plain decode (identical output, `τ` measured).
2. **Port a trained DFlash draft checkpoint** (LLaMA-3.1-8B block size 10, or Qwen3-8B block 16) into
   the `tt_transformers` model format; share target embedding + LM head.
3. **Draft forward:** non-causal block attention (`scaled_dot_product_attention(is_causal=False, mask)`)
   over `[anchor + (block-1) mask tokens]`; one pass → `γ` logits → argmax block.
4. **Target-feature extraction + KV injection:** expose ~5 target hidden layers from prefill; fuse via
   a projection; inject into draft layers' K/V. This is what lifts `τ` from ~2-3 to ~6.5.
5. **Trace both passes** (draft + verify) to remove dispatch; measure dispatches/accepted token and
   end-to-end tok/s vs the DRAM baseline.
6. **(Optional, separate project) draft-weight residency in L1** — only if a narrow-enough draft is
   used; reuse the removed weight-caching machinery scoped to the draft.

## 8. Confidence
- **Certain:** the deciding metric is dispatches/accepted token; DFlash collapses `γ` sequential draft
  passes to one; the non-causal and chunked-extend SDPA primitives already exist (cited).
- **Certain:** under DFlash the target KV read is hidden with even more margin, so L1-KV gives DFlash
  no latency benefit and adds rollback complexity (from the measured compute-bound result + `notes.md`).
- **Speculating:** the realized speedup on tt-metal (hinges on the draft-checkpoint port, `τ` transfer,
  and rollback cost), and whether a useful draft is narrow enough to be L1-weight-resident.

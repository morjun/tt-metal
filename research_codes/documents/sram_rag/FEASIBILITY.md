# Feasibility re-evaluation — In-SRAM RAG (revised proposal) vs DFlash

Re-evaluates the **revised** `proposal.md` (now with §2.1 distributed placement and §2.4 asynchronous
thresholding + micro-rollback) against this project's measured results
(`../l1_kv_cache_cache/COMPUTE_BOUND_PROOF.md`, `../l1_kv_cache_cache/notes.md`) and against the DFlash
path (`../dflash/DFLASH_FEASIBILITY.md`). Platform: Blackhole P150. Supersedes the prior review, which
judged the pre-revision proposal.

## Verdict (updated)
**The revision fixes the correctness-fatal flaw. In-SRAM RAG is now mechanistically sound and a
legitimate, novel research direction — gated on one empirical make-or-break question (the retrieval
quality of the in-core score).** It is no longer "not feasible." DFlash remains the stronger bet for
*raw decode speedup*, but the two now target **different axes**: DFlash makes decode 4-6× faster;
In-SRAM RAG adds a RAG capability at ~zero marginal decode latency (≈0 speedup by design). "Which is
more promising" therefore depends on the goal — see §5.

---

## 1. What the revision fixed (genuine credit)
The earlier version claimed the QKV projection *itself* computed the query-doc similarity (false:
matmul rows are independent). The revision replaces that with a coherent three-part mechanism:

1. **Free K_doc precompute in the projection's idle rows (§2.2).** `Y = X·W_qkv` with rows 1-9 = the
   8-9 resident doc vectors yields `doc_i·W_qkv`, i.e. each doc's projected **K** (and Q,V). This is
   genuinely free in wall-clock: the matmul processes the full 32-row tile and streams `W_qkv` once
   regardless of how many rows are populated (batch-1 linear ops are weight-streaming-bound,
   `COMPUTE_BOUND_PROOF.md` §8c). **This is the one correct use of "fill the idle rows"** — because the
   filled rows now produce a *useful byproduct* (K_doc) consumed downstream, not a discarded one.
2. **Explicit `Q_token · K_doc` dot (§2.4.1).** `score_i = (e_tok·W_q)·(e_doc_i·W_k)` is the actual
   similarity — the model's own attention logit between the current token and each doc-as-key. This is
   a real `[9×head_dim]·[head_dim]` tile MAC and replaces the false "projection = similarity" claim.
3. **Threshold hidden on the idle scalar core during the FPU-bound SDPA (§2.4.2-3).** The ~8900 ns
   SDPA is FPU-bound (measured: QK^T ≈ 77% of the envelope, FPU 68-84%, `COMPUTE_BOUND_PROOF.md` §4),
   so the scalar core (BRISC) is idle during it; a ~50 ns threshold check there is genuinely hidden.
   **This is the correct "zero-overhead" mechanism** — overlap with idle BRISC, not the (false) "free
   because matmul rows are idle." It is a correct application of this project's compute-bound result.

Also now defensible: **per-step zero DRAM bandwidth (§2.3).** The Top-1000 set is retrieved once per
query and resides in L1 across the whole generation (≈68 KB/core for 4096-dim vectors — trivial), so
*per decode step* the RAG adds no DRAM traffic. The one-time host→L1 load is amortized over all decode
steps (my earlier "reintroduces PCIe per step" objection was wrong; it is once per query).

## 2. The make-or-break risk: retrieval quality (the one thing to test first)
The score is the model's attention metric `(e_tok·W_q)·(e_doc·W_k)` applied to **document hidden-state
vectors**. Open questions, all unproven:
- **Untrained for retrieval.** `W_q, W_k` were trained for next-token attention, not document relevance.
  Using them as a retriever may or may not rank documents well.
- **Representation.** A document is many tokens; `e_doc` must be a single pooled hidden state (lossy),
  and the docs must live in the LLM's hidden space (not a separate encoder's space).
- **Per-head.** The score is per attention head (`n_q` of them) — needs pooling across heads.
- **Which layer.** The projection/SDPA happen at every layer; the proposal must fix which layer's
  `W_q/W_k` defines the score.

This is exactly what §3.2 (Recall@K vs Faiss/Milvus) measures, and it must be done **first, offline, in
pure Python — before any kernel work**. If the in-core metric does not retrieve competitively, no amount
of clever kernel hiding matters. High uncertainty; this is where the proposal lives or dies.

## 3. Remaining issues to fix in the writeup (not fatal, but needed)
1. **§2.2 wording.** State explicitly that the rows precompute **K_doc, consumed in §2.4.1**; delete any
   residual "projection into latent space = retrieval" implication.
2. **§2.4.4 micro-rollback semantics.** "Append the retrieved document's K,V" injects a single vector,
   but a real document is many tokens. Specify: either a 1-vector summary (weak context) or a real
   multi-token doc prefill (which is *not* "micro" — it is a chunked-prefill of the doc, the expensive
   path). And threshold injection **changes the output** — it is a deliberate behavior change, not
   "structural correctness." Frame it as a quality feature, not losslessness.
3. **Contribution 2 ("100% FPU utilization").** Reframe. At batch-1 a higher FPU-utilization counter
   does **not** imply speedup — the matmul is weight-streaming-bound, so filling rows changes the metric,
   not the wall-clock (`COMPUTE_BOUND_PROOF.md` §8c). The real contribution is "**free in-core K_doc
   precompute + score fully hidden behind SDPA**," not "100% utilization → faster."
4. **Hit-rate assumption.** The "1% hit" figure is unjustified (depends on threshold + corpus). If hits
   are frequent, each costs a recompute (and possibly a doc prefill), so the amortization claim needs a
   sensitivity analysis.

## 4. Feasibility on tt-metal (the revised mechanism)
| step | primitive / change | feasibility |
|---|---|---|
| dual-fetch docs into projection rows 1-9 | NCRISC reads L1-resident doc vectors into `X` | small kernel change ✓ |
| free K_doc via projection | reuse the existing QKV matmul (tile + weight stream unchanged) | correct, free ✓ |
| `Q_token · K_doc` score | a small tile MAC (`[9×dh]·[dh]`) | straightforward ✓ |
| threshold on BRISC during SDPA | scalar op on the idle 5th RISC | plausible ✓ (BRISC idle during FPU-bound SDPA) |
| micro-rollback / doc injection into KV | discard token + insert doc K,V + recompute | **hard**; same rollback complexity as spec-decode, lands on the KV ring-write path |

The mechanism is implementable; the hard engineering is the rollback/injection (shared with the
spec-decode path), and the hard *research* is §2 (quality).

## 5. Recalibrated comparison — different axes
| dimension | In-SRAM RAG (revised) | DFlash |
|---|---|---|
| goal | RAG re-ranking + dynamic context injection at ~0 marginal decode latency | make decode 4-6× faster |
| core mechanism sound now? | **yes** (free K_doc precompute + Q·K dot + BRISC-hidden threshold) | yes |
| end-to-end decode speedup | ~0 by design (free feature, not faster decode) | 4-6× (measured GPU) |
| novelty | **high** (first in-core fused RAG; zero-marginal-latency retrieval) | medium (port of a known method) |
| make-or-break risk | **retrieval quality** of the untrained in-core metric (testable offline) | porting a trained draft checkpoint |
| per-step DRAM cost | zero (docs resident in L1) | unchanged (weights still stream) |
| feasibility | feasible; hard parts = quality + rollback | feasible; hard part = checkpoint port |

The shared instinct ("use the idle batch-1 decode capacity") now lands correctly in **both**: DFlash
fills the matmul rows with *real tokens to verify* (cutting sequential dispatches → speed), and
In-SRAM RAG fills them with *docs to precompute K* and hides the score on the idle scalar core (→ free
capability). The earlier objection — "filling idle rows is zero-benefit" — applied only to the old
version, where the filled rows produced nothing useful; here they produce K_doc.

## 6. Recommendation
1. **Run the decisive cheap experiment first:** an offline Recall@K study of the in-core metric
   `(e_tok·W_q)·(e_doc·W_k)` (pure PyTorch, no tt-metal) vs Faiss on a standard RAG benchmark, with the
   pooling/layer choices fixed. This gates everything and needs no kernel work.
2. **If quality holds:** In-SRAM RAG is the more *novel/publishable* path — a genuinely new mechanism
   (zero-marginal-latency in-core retrieval + dynamic KV injection) well-matched to this hardware's
   FPU-bound SDPA window. Then build the kernel pipeline in §4 and the (shared) rollback path.
3. **If quality fails:** drop the retrieval-metric angle; DFlash is the safe, high-value path for raw
   decode speedup.
4. They are **not mutually exclusive** — different axes (speed vs capability) — and could even compose,
   though both add KV-rollback complexity, so sequence them rather than build at once.

Net: for an engineering speedup deliverable, **DFlash**. For a novel research contribution, the
**revised In-SRAM RAG** is now a real candidate, contingent on the offline quality study. The revision
moved it from "the math is wrong" to "the systems mechanism is right; does the metric retrieve?" — a
much better place to be.

## Confidence
- **Certain:** the revised mechanism is internally coherent; the free K_doc precompute and the
  BRISC-hidden threshold are correct uses of this project's measured compute-bound result; per-step DRAM
  is zero.
- **Certain:** DFlash is the stronger bet for raw decode speedup; the two optimize different axes.
- **Speculating (high uncertainty):** the retrieval quality of the untrained in-core metric — the
  make-or-break — which is untested and must be measured offline before committing kernel effort.

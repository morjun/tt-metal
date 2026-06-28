# Feasibility evaluation — In-SRAM RAG (revised proposal) vs DFlash

Evaluates the **revised** `proposal.md` (§2.1 distributed placement, §2.4 async thresholding +
micro-rollback) against this project's measured results (`../l1_kv_cache_cache/COMPUTE_BOUND_PROOF.md`,
`../l1_kv_cache_cache/notes.md`) and the DFlash path (`../dflash/DFLASH_FEASIBILITY.md`).
Platform: Blackhole P150.

> Verdict history (for honesty): v1 judged the original proposal "not feasible / not promising." v2
> over-corrected to "promising, different axis" by crediting the repaired mechanism but quietly
> retiring a still-valid objection (it optimizes a cheap, amortized sub-step of RAG). This version is
> the corrected, stable verdict: **the mechanism is sound and near-free, but it targets the wrong part
> of the RAG pipeline, so its expected impact is low.** Both the mechanism credit (v2) and the
> wrong-bottleneck critique (v1) are kept; the inflated "promising" framing is removed.

## Verdict
The revision fixes the correctness-fatal flaw, so In-SRAM RAG is now a **sound, near-zero-cost
mechanism**. But it is elegance applied to a step that was never expensive: it makes the in-core
**re-ranking of a host-prefiltered Top-1000** free, while the costly retrieval work (full-corpus ANN
search) stays on the host and the real end-to-end decode bottleneck (generation) is untouched. So for
**impact**, DFlash remains the stronger bet. In-SRAM RAG is worth pursuing only as (a) a niche
capability — per-token adaptive re-ranking/injection within the resident pool — and only if (b) its
in-core retrieval metric actually ranks documents well, which is unproven and must be measured first.

---

## 1. What the revision genuinely fixed (mechanism credit)
The original claimed the QKV projection *itself* yielded the similarity (false — matmul rows are
independent). The revision is a coherent, mostly-correct mechanism:
1. **Free K_doc precompute (§2.2).** `Y = X·W_qkv` with rows 1-9 = resident doc vectors yields each
   doc's projected **K** for free — the matmul processes the full 32-row tile and streams `W_qkv` once
   regardless of populated rows (batch-1 linear ops are weight-streaming-bound, `COMPUTE_BOUND_PROOF.md`
   §8c). Filling the idle rows is legitimate *here* because the result (K_doc) is consumed downstream.
2. **Explicit `Q_token · K_doc` dot (§2.4.1).** The real score, replacing the false "projection =
   similarity."
3. **Threshold hidden on the idle scalar core during the FPU-bound ~8900 ns SDPA (§2.4.2-3).** A
   correct use of the measured compute-bound result (BRISC is idle while the FPU runs SDPA).

Precision: the cost is **near**-zero, not zero — the `Q·K` dot is ~100 ns of FPU **serial** with the
SDPA on the same engine (~1% of the per-layer FPU window); only the ~50 ns BRISC threshold is truly
hidden. Per-step DRAM is genuinely zero (docs resident in L1, loaded once per query).

## 2. The decisive limitation — it optimizes a cheap, amortized sub-step (the real point)
This is why "near-free mechanism" does not translate to "promising system":
- **The expensive retrieval work stays on the host.** §2.1 step 1 keeps the full-corpus ANN search
  (Top-1000 out of millions/billions) on the host. That — plus the host→device transfer of candidates —
  is the hard, costly part of retrieval, and the proposal does not touch it.
- **What it makes free was already trivial.** Re-ranking 1000 candidates is ~1000 dot products —
  microseconds anywhere. Making that free saves a negligible absolute amount of time.
- **It is not even the end-to-end bottleneck.** In standard one-shot RAG, retrieval runs once per query
  and is amortized over the whole generation (hundreds of tokens), so generation dominates end-to-end —
  and generation is exactly what DFlash accelerates and what In-SRAM RAG leaves unchanged.

So the proposal inserts itself at the one point in the RAG pipeline that is both cheap and amortized. A
correct, near-zero-cost mechanism aimed there yields a near-zero-impact system. (This is the
"wrong-bottleneck" critique from v1, which the revision does not address and which v2 wrongly dropped.)

## 3. The make-or-break question — does the in-core metric actually retrieve? (unproven)
Stated honestly: the metric is **not proven inadequate, but it is unvalidated and plausibly weak.** The
score is the model's own attention geometry `(e·W_q)·(e·W_k)`.
- *Could work:* it is the model's native "what would I attend to" relevance, arguably better aligned
  with generation than an external cosine similarity.
- *May not:* `W_q,W_k` were trained for next-token attention, not document ranking; a document must be a
  single pooled hidden vector (lossy); the score is per-head (needs pooling); docs must live in the
  LLM's hidden space, not a sentence-encoder's.

This is testable offline (the §3.2 Recall@K plan) and **must be the first experiment** — pure PyTorch,
no kernels. If it fails, nothing else matters. High uncertainty.

## 4. The one real-but-narrow upside
To be fair to the proposal: per-token, generation-adaptive re-ranking/injection at ~zero marginal
latency is something one-shot host RAG cannot do cheaply (a host round trip per token would be
prohibitive). As the generation evolves, the model can pull in a different one of the resident
candidates. That is a genuine capability — but bounded: the candidate pool is a **one-shot host
retrieval**, so if the generation needs a document outside the original 1000, the mechanism cannot fetch
it (that would require per-token corpus search, which it does not do). So the upside is "adaptively
re-rank within a frozen pool," gated on §3.

## 5. Remaining writeup fixes (if pursued)
1. §2.2: say the rows precompute **K_doc consumed in §2.4.1**; drop any "projection = retrieval" wording.
2. §2.4.4 micro-rollback: a single K,V vector ≠ a multi-token document; specify whether injection is a
   1-vector summary (weak) or a real multi-token doc prefill (not "micro"). And it changes the output —
   frame as a behavior feature, not "structural correctness."
3. Contribution 2 ("100% FPU utilization"): reframe. Higher batch-1 FPU utilization does not imply
   speedup (the matmul is weight-streaming-bound). The real contribution is "free K_doc precompute +
   score hidden behind SDPA," not "100% utilization → faster."
4. "Zero overhead": say **near**-zero (~1% FPU for the dot; threshold hidden), not literally zero.

## 6. Feasibility on tt-metal (the mechanism is implementable)
| step | primitive / change | feasibility |
|---|---|---|
| dual-fetch docs into projection rows | NCRISC reads L1-resident doc vectors into `X` | small kernel change ✓ |
| free K_doc via projection | reuse the existing QKV matmul | correct, free ✓ |
| `Q·K_doc` score | small tile MAC (`[9×dh]·[dh]`) | ~100 ns FPU, serial ✓ |
| threshold on BRISC during SDPA | scalar op on the idle 5th RISC | plausible ✓ |
| micro-rollback / doc injection into KV | discard token + insert doc K,V + recompute | **hard**; same rollback complexity as spec-decode, on the KV ring-write path |

Feasible to build; the hard engineering is rollback/injection, and the hard *research* is §3.

## 7. vs DFlash — recalibrated
| dimension | In-SRAM RAG (revised) | DFlash |
|---|---|---|
| what it changes | makes in-core re-ranking of a host-prefiltered pool free; adds per-token adaptive injection | makes decode 4-6× faster |
| mechanism sound? | yes | yes |
| targets a real cost? | **no** — re-ranking is cheap & amortized; corpus ANN (expensive) stays on host; generation (the end-to-end bottleneck) untouched | **yes** — generation, the actual bottleneck |
| end-to-end impact | ~0 (free but marginal) | large (measured 4-6×) |
| novelty | high (in-core fused retrieval) but value-narrow | medium (port of a known method) |
| make-or-break | does the untrained in-core metric retrieve? (unproven) | porting a trained draft checkpoint |
| feasibility | feasible | feasible |

## 8. Recommendation
1. **For decode impact: DFlash.** It attacks the part that actually costs time (generation), with
   measured 4-6×.
2. **If the RAG direction is pursued:** run the **offline Recall@K study first** (pure PyTorch). If the
   in-core metric does not rank competitively, stop — no kernel work is justified.
3. **The genuinely high-impact RAG target is the one this proposal avoids:** move the coarse corpus ANN
   search itself **on-device**, so retrieval stops being a host round trip at all. That is much harder
   but attacks the part of RAG that is actually expensive. In-core re-ranking of a host-prefiltered pool
   is, by contrast, a free optimization of a cheap step.

## Confidence
- **Certain:** the wrong-bottleneck critique (re-ranking 1000 is trivial and amortized; the expensive
  corpus ANN stays on host; generation is the end-to-end bottleneck) — this is hardware-independent.
- **Certain:** the revised mechanism is internally sound and near-free; DFlash is the stronger
  impact bet.
- **Unknown (must measure):** whether the in-core attention metric retrieves well — the gate on any
  value at all.

# Feasibility review — "Zero-Overhead In-SRAM RAG via FPU Spatial Multiplexing" (vs DFlash)

Reviews `proposal.md` against this project's measured results
(`../l1_kv_cache_cache/COMPUTE_BOUND_PROOF.md`, `../l1_kv_cache_cache/notes.md`) and compares it to the
DFlash speculative-decode path (`../l1_kv_cache_cache/DFLASH_FEASIBILITY.md`). Platform: Blackhole P150.

## Verdict
**Not feasible as written, and not promising even if repaired. Pursue DFlash instead.** The proposal's
core mechanism (§2.2) rests on a linear-algebra error: the QKV projection cannot compute a
query-document similarity as a byproduct. Three further independent flaws each sink it on their own:
embedding-space mismatch, "zero-overhead = zero-benefit" (the idle FPU rows are free in wall-clock, so
filling them buys ~nothing), and optimizing a non-bottleneck while leaving the real RAG cost (index
search + host round trip) untouched. DFlash, by contrast, attacks the actual tt-metal bottleneck
(dispatches per accepted token), is algorithmically sound, and is measured at 4-6× on GPU.

---

## 1. The core mechanism does not compute what it claims (correctness-fatal)
§2.2 states that injecting 31 document vectors into the wasted rows of the QKV projection
`Y = X · W_qkv` makes "Row 0 yield the LLM's QKV vector, while Rows 1-31 inherently compute … an
in-core similarity dot-product for re-ranking."

A matmul's rows are independent. With `X ∈ R^{32×H}`, `W_qkv ∈ R^{H×3H'}`:
```
Y[i, :] = X[i, :] · W_qkv          (row i of the output depends only on row i of the input)
Y[0, :] = e_token · W_qkv          → the real QKV vector            ✓
Y[i, :] = d_i     · W_qkv          → document i projected by the WEIGHTS (a 3H'-vector)   ✗
```
- `Y[i,:]` is a **vector** (the doc projected through the model weights), not a **scalar** re-ranking
  score, and it **never involves the token** at all.
- A re-ranking score is `s_i = q · d_i` — a product whose two operands are the **query** and the
  **document**. In `X · W_qkv` the two operands are the **document** and the **weight matrix**. There is
  no query-document interaction anywhere in the tile.
- To actually produce the scores you need `s = D · q`, where `D ∈ R^{N×H}` stacks the candidate docs and
  `q ∈ R^H` is the query. That is a different matmul with **different operands** (q is the moving
  operand, D stationary). It **cannot fall out of `X · W_qkv`**, because `W_qkv` is fixed model weights,
  not the query.

So the "piggyback on QKV" idea computes 31 independent weight-projections of documents, not 31
similarities. The central claim is false, not merely inefficient.

## 2. Three more independent flaws
1. **Embedding-space mismatch (§2.2 "share identical dimensionality").** RAG document embeddings come
   from a separate encoder (typically 384/768/1024-dim, a different latent space), not the LLM's
   `Hidden_dim` (4096) residual stream. Even if shapes were forced to match, projecting a doc embedding
   through the LLM's `W_qkv` (trained for attention over KV positions) computes nothing meaningful for
   retrieval. "Seamlessly concatenated" does not hold.
2. **Zero-overhead = zero-benefit.** The proposal's premise (§1: "97% hardware idle … 31 of 32 rows
   wasted") is the exact misconception this project measured to be false in the dimension that matters:
   batch-1 linear ops are **weight-streaming / dispatch-bound**, so the idle rows cost ~nothing in
   wall-clock (`COMPUTE_BOUND_PROOF.md` §8c; `notes.md` 28-30). Filling them flips a utilization counter
   to 100% (Contribution 2, Evaluation 3.2) while changing end-to-end latency by ~0 — the matmul was
   never on the critical path, so reclaiming it yields no speedup. The "zero-overhead" is real precisely
   *because* it is also zero-benefit.
3. **Wrong bottleneck + leaky "zero DRAM" (§2.3, §3).** The retrieval compute itself (a `D·q` matvec
   over 1000 candidates ≈ a few M MACs) is negligible against an 8B-param forward. Real RAG latency is
   the ANN/index search over millions of documents and the host round trip — both of which the proposal
   keeps (host coarse-filtering to Top-1000) and does not touch. "Zero DRAM bandwidth" also ignores that
   getting candidates on-chip is a host→L1 transfer (the very PCIe cost it claims to remove) and that
   the vectors contend for the same scarce L1 capacity as the KV cache and circular buffers.

## 3. Steelman — the salvageable variant, and why it is still low-value
The defensible reinterpretation: run a **separate** retrieval matmul `s = D · q` with `D` resident in
L1, **co-scheduled into the FPU bubbles** that occur while weight tiles stream from DRAM (the FPU does
stall on the weight bus, so there is genuine idle compute to fill). This is sound in principle and is
the *correct* way to do in-SRAM retrieval (D stationary in L1, q the moving operand). But:
- It is a **different design** than the row-injection in §2.2 (which computes the wrong product).
- It still optimizes a **non-bottleneck** (retrieval compute is tiny; perfectly hiding it saves ~0).
- It does **not** address the real RAG cost (index search over the full corpus, host round trip), which
  remains on the host.
So even the strongest reading does not make it promising; it is a micro-optimization of a step that was
already cheap, dressed as a systems win.

## 4. vs DFlash
| dimension | In-SRAM RAG (proposal) | DFlash (spec decode) |
|---|---|---|
| core mechanism sound? | **No** — matmul rows are independent; QKV projection ≠ query·doc | Yes |
| attacks the real tt-metal bottleneck? | No — the matmul/retrieval compute is not the bottleneck | Yes — dispatches per accepted token |
| benefit if fully realized | ~0 end-to-end (free rows are free, not valuable) | 4-6× measured on GPU |
| addresses real RAG/serving cost | No (index search + host round trip untouched) | n/a (different problem) |
| needed primitives in tt-metal | new dual-fetch kernel; misframed | exist (non-causal block SDPA, chunked-extend verify) |
| main gate | the idea itself | port a trained draft checkpoint |
| feasibility | **not feasible as written** | feasible |

The shared instinct in both proposals — "fill the 31 wasted rows" — only pays off when the filled rows
do **more sequential work for you** (DFlash verify packs k real tokens, cutting sequential dispatches).
It does **not** pay off when the filled rows compute an **unrelated** quantity in cycles that were
already free (In-SRAM RAG). That distinction is the whole result of this project's compute-bound study.

## 5. Recommendation
- **Pick DFlash.** It is the one aligned with the measured reality of this hardware (dispatch-bound,
  weight-streaming-bound, compute-bound SDPA).
- If the RAG direction is still wanted, drop the "free piggyback on QKV" framing entirely and reframe as
  **(a)** a correct in-L1 `D·q` retrieval kernel hidden under weight-streaming bubbles (accepting it is a
  small, non-critical-path win), and **(b)** moving the *coarse* ANN search itself on-device (the actual
  latency source) — a much harder but genuinely impactful target. As written, neither contribution in §4
  of the proposal survives scrutiny.

## Confidence
- **Certain:** the §2.2 mechanism cannot compute query-doc similarity (direct linear algebra), and the
  idle-rows-are-free / wrong-bottleneck points follow from this project's measured results.
- **Certain:** DFlash is the sounder bet of the two.
- **Speculating:** only DFlash's realized tt-metal speedup (depends on the draft-checkpoint port and
  rollback cost), per `DFLASH_FEASIBILITY.md`.

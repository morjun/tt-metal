# FlashDecoding++Next vs current tt-metal — does tt-metal implement it, and how does the attention math differ?

Scope: compares the paper *FlashDecoding++Next: High Throughput LLM Inference With Latency and
Memory Optimization* (Dai et al., IEEE TC 2025) against the current tt-metal decode attention
(`scaled_dot_product_attention_decode`). Focus per the request: the **attention computation**.

## Bottom line
**No — tt-metal does not implement FlashDecoding++Next's attention contribution.** tt-metal's
flash-decode implements the *baseline* the paper improves on: **FlashDecoding-style synchronous
online (partial) softmax** with a running max and per-split rescale, plus a **cross-core
synchronous reduction**. The paper's central idea — **asynchronous softmax with a unified maximum**
that removes exactly that synchronization — is absent. The paper's other two contributions (flat
GEMM with double buffering, unified KV/activation memory) have only loose architectural analogs in
tt-metal via different mechanisms (fixed 32×32 tile matmul, circular-buffer double buffering, the
bank allocator), not the paper's specific techniques.

Important framing: the paper is GPU-specific (CUDA/Tensor Core, A100, shared memory, SM
scheduling). tt-metal is a different architecture (Tensix dataflow, NoC, SRAM-centric, no
Tensor/CUDA-core duality). So "does tt-metal implement it" means "does tt-metal use the same
*algorithmic* technique," not a line-for-line port.

---

## The paper's three contributions (summary)
1. **Asynchronous softmax with unified maximum (Section III) — the attention contribution.**
   Classic partial/online softmax (FlashAttention, FlashDecoding) splits the KV sequence, computes a
   partial softmax per split with that split's *own* max, then **synchronously updates** every prior
   partial result when a new, larger max appears (paper Eq. 2, Fig. 3b). The paper measures this
   synchronization at **~18.8% of attention compute** on an A100. Their fix: pick a single **unified
   scaling factor φ** (a fixed constant, not the running max) shared by all splits, so
   `softmax(x) = e^{x_i-φ} / Σ e^{x_i-φ}` (Eq. 3, valid for any φ). Each split then computes its
   numerator/denominator **independently and asynchronously**; the only cross-split step is a final
   plain sum (Eq. 4) — no max-based rescale. A **recomputation fallback** to the synchronous scheme
   handles the rare overflow when some `x_i - φ` leaves a safe range `[a,b]` (justified by the
   measured >99.99% of logits lying in a narrow band, Fig. 5).
2. **Flat GEMM with double buffering (Section IV).** Decode GEMMs are flat (M = batch ≪ 64). Libraries
   pad M to 64 → ~50% waste. The paper pads M to 8 (Tensor-Core minimum), uses small N tiles for
   parallelism, double-buffers K-tiles in shared memory to hide memory latency, and **selects CUDA
   Core vs Tensor Core** per shape (CUDA Core wins for very small M).
3. **Buffer reuse + unified memory (Section V).** Reuse 3 activation buffers per layer; unify the KV
   cache and activation pools (KV grows up in address, activations grow down) to remove ~22% peak
   memory.

---

## Attention computation — head-to-head

### Paper: asynchronous softmax, unified maximum
- Per-split contribution computed independently: `Σ e^{x_i-φ} v_i` and `Σ e^{x_i-φ}`, **no rescale
  against other splits**.
- Cross-split combine = a single addition of numerators and denominators, then one division (Eq. 4).
- Overflow handled by a fallback to the synchronous scheme only when needed.
- Net: removes the ~20% synchronous-update overhead; enables fine-grained pipelining.

### tt-metal: FlashDecoding synchronous online softmax (the baseline)
Verified in `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/`:

- **Running-max online softmax (intra-core, across K-chunks).** The compute kernel keeps a running
  max `cb_m_in` (`c_6`) and running sum `cb_l_in` (`c_7`) (`sdpa_flash_decode.cpp:91-92`). Each
  K-chunk: `reduce_c<...>(..., do_eltwise_max=true)` updates the running max
  (`sdpa_flash_decode.cpp:382`), `sub_exp_block_bcast_cols_inplace_reduce` computes
  `exp(QK - max)` and its row-sum (`:398`), and the running output accumulator is **rescaled by
  `exp(prev_max - cur_max)`** before adding the new chunk (the `SM_RESCALE` step). This is exactly
  the synchronous partial-softmax update of the paper's Fig. 3b, applied sequentially per chunk.
- **Cross-core synchronous reduction (inter-split).** When a head's KV context is split across
  `num_cores_per_head` cores, the **reducer** core (`do_reduce`) waits for `num_cores_per_head - 1`
  worker cores (`sdpa_flash_decode.cpp:183-184, 506-513`), then for each worker combines its partial
  `(m, l, O)` using a **max + `exp(Δmax)` rescale**:
  ```cpp
  // sdpa_flash_decode.cpp:520-527 (reducer combining a worker's partial)
  move_block<true>(cb_l_in, cb_prev_sum_2, Sq_chunk_t);
  max_block<vector_mode>(cb_m_in, cb_prev_max, cb_cur_max, Sq_chunk_t);   // unified? no — takes the max
  sub_exp_block<scale_fp32, vector_mode>(cb_m_in, cb_cur_max, cb_exp_max_diff_2, Sq_chunk_t);
  ```
  Helpers `max_block` (`compute_common.hpp:33`), `reduce_c` (`:64`), `sub_exp_block` (`:297`) are the
  online-softmax primitives. This is precisely the **synchronous cross-split update** the paper
  eliminates — tt-metal does it both across chunks (intra-core) and across cores (inter-split).
- **No unified-φ path.** There is no fixed-scale softmax option, no φ constant, no overflow
  recomputation fallback. The scale applied is the QK scale (`scale_fp32`), not a softmax-stabilizing
  unified max; stabilization is always via the running/elementwise max.

### Side-by-side
| aspect | FlashDecoding++Next (paper) | current tt-metal flash-decode |
|---|---|---|
| softmax stabilizer | **unified constant φ** (per Eq. 3) | **running / elementwise max** (`cb_m_in`) |
| partial-split combine | independent, then plain sum (async) | **max + `exp(Δmax)` rescale** (synchronous) |
| intra-core across chunks | no rescale (φ fixed) | per-chunk `SM_RESCALE` (`reduce_c` + `sub_exp`) |
| inter-core across splits | plain numerator/denominator add | reducer waits workers, max-rescale combine (`:506-549`) |
| overflow handling | recomputation fallback when `x-φ∉[a,b]` | n/a (max guarantees no overflow) |
| target overhead | ~18.8% softmax sync (A100) | carries that sync, intra- and inter-core |

---

## Would the paper's attention technique help tt-metal? (analysis)
Speculating, grounded in this repo's measured compute-bound study (`COMPUTE_BOUND_PROOF.md`):

1. **Decode is QK^T-matmul-bound on tt-metal, so softmax is the minority cost.** Measured FPU
   (QK_MM+PV_MM) is 68-84% of attention compute; SFPU softmax (SM_NORM+SM_RESCALE) is only 16-32%,
   and on the critical (longest) cores QK^T alone is ~77% of the envelope. The paper's async-softmax
   attacks the softmax-sync slice, which on tt-metal is already the smaller part and partly hidden.
   So the paper's headline ~1.14× decode speedup (measured on an A100 against FlashDecoding) would
   not transfer at face value — the bottleneck here is different.
2. **The cross-core synchronous reduction is the part most analogous to what the paper removes, and
   it only exists when `num_cores_per_head > 1`.** That is the batch-1 / long-context regime where a
   head's KV is split across cores (the reducer-waits-for-workers barrier, `:506-549`). At larger
   batch (`batch-32`), `num_cores_per_head → 1` (see `CODE_AUDIT.md` §3), so there is **no cross-core
   reduction to remove** — exactly the high-throughput regime the paper targets is where tt-metal's
   relevant synchronization disappears. Unified-φ would mainly help batch-1 long-context decode, by
   letting workers skip the max-rescale at combine and the reducer just sum partials.
3. **Feasibility caveats on tt-metal.** Unified-φ needs (a) a calibrated per-model/per-layer safe
   range for `x-φ` and (b) the recomputation fallback path; tt-metal compute kernels are statically
   compiled per program, so a data-dependent fallback adds a branch/recompile path and complicates
   the LLK pipeline. The win is bounded by point 1.

**Verdict:** adopting unified-max async softmax is a *valid* optimization tt-metal has not taken,
but on this hardware it targets a minority cost and its main lever (cross-core reduction) vanishes at
the batch sizes where throughput matters most. It does not change the L1-vs-DRAM latency verdict
(the KV read is hidden regardless of softmax scheme).

---

## The paper's other two contributions vs tt-metal (briefly)

### Flat GEMM with double buffering (Section IV)
- **Paper:** pad M to 8, CUDA-Core-vs-Tensor-Core selection, shared-memory double buffering.
- **tt-metal:** the Tensix matmul tile is **fixed 32×32**; M is padded to 32, not 8 (`attention.py:1296`
  pads batch to a 32-row tile). There is no CUDA/Tensor-core duality — one FPU matmul engine; the
  closest analog to the implementation-selection idea is the **HiFi2-vs-HiFi4 math-fidelity choice**
  tuned so the matmul keeps up with the DRAM weight stream (`attention.py:1246`). **Double buffering
  does exist**, but as circular-buffer prefetch: the reader (NCRISC) streams the next K/V chunk while
  the compute engine works the current one — the dataflow equivalent of the paper's shared-memory
  double buffer. The flat-M underutilization the paper targets is real on tt-metal too, but for SDPA
  the matmul rows are **heads** (which fill the tile), and for the linear ops M = batch (so batch-32
  fills it) — see `COMPUTE_BOUND_PROOF.md` §8c / `CODE_AUDIT.md` §3.1. Conclusion: **same problem
  acknowledged, different and not-equivalent mechanism; the paper's specific pad-to-8 + core-selection
  is not implemented (and not applicable).**

### Buffer reuse + unified memory (Section V)
- **Paper:** 3 reused activation buffers/layer; unified KV+activation pool with bidirectional address
  growth on GPU global memory.
- **tt-metal:** memory is managed by the **bank allocator** over distributed SRAM banks + DRAM, with
  programs declaring **circular buffers** at fixed addresses (compile-time placement) and the KV cache
  **pre-allocated** (persistent). Buffer reuse happens via CB lifecycle and allocator reuse, not the
  paper's explicit 3-buffer scheme; there is no unified KV/activation pool with bidirectional growth.
  The L1-KV work in this repo adds **per-core L1 KV tier allocation** (`CODE_AUDIT.md` §1), which is a
  *capacity* mechanism, unrelated to the paper's activation-redundancy goal. Conclusion: **different
  memory model; conceptual overlap (pre-allocation, reuse) but not the paper's technique.**

---

## Summary table
| paper contribution | implemented in tt-metal? | what tt-metal does instead |
|---|---|---|
| Async softmax, unified max | **No** | FlashDecoding synchronous online softmax (running max + per-chunk rescale + cross-core max-reduce) |
| Flat GEMM, pad-to-8 + double buffering + core selection | **No** (partial analog) | fixed 32×32 tile matmul (pad to 32); CB-prefetch double buffering; HiFi2/HiFi4 fidelity choice |
| Buffer reuse + unified KV/activation memory | **No** (partial analog) | bank allocator + compile-time CB placement + pre-allocated/persistent KV; L1-KV tiers add capacity |

## Confidence
- **Certain:** tt-metal uses running-max synchronous online softmax with a cross-core max-rescale
  reduction (read directly from the kernels, lines cited).
- **Certain:** tt-metal has no unified-φ / async-softmax / overflow-recomputation path.
- **Speculating:** the *benefit* of porting unified-max softmax to tt-metal is small and concentrated
  in batch-1 long-context (reasoned from the measured compute-bound split + the batch→core-allocation
  behavior, not benchmarked here).

## References (verified file:line)
- `ttnn/.../sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`: `cb_m_in`/`cb_l_in` `:91-92`;
  running-max `reduce_c` `:382`; `exp(QK-max)` `:398`; cross-core reduce loop `:506-549`
  (`max_block` `:523`, `sub_exp_block` `:527`).
- `ttnn/.../sdpa_decode/device/kernels/compute/compute_common.hpp`: `max_block` `:33`, `reduce_c`
  `:64`, `sub_exp_block_bcast_cols_inplace_reduce` `:123`, `sub_exp_block` `:297`.
- `models/tt_transformers/tt/attention.py`: batch→32-row M pad `:1296`; HiFi2 weight-matmul note `:1246`.
- Cross-refs: `COMPUTE_BOUND_PROOF.md` (compute-bound split, §8c weights), `CODE_AUDIT.md` §3
  (batch→core allocation).

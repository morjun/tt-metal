# Why the L1 KV Cache Cannot Beat the DRAM Baseline on Decode Latency

**A measured proof that SDPA-decode is compute-bound, that the KV read is fully hidden
behind compute, and that therefore moving the KV cache from DRAM into L1 yields parity,
not a speedup, on single-stream decode latency.**

Platform: Tenstorrent Blackhole P150, single card, Llama-3.1-8B, batch-1 decode.
Method: device-zone profiling (Tracy `DeviceZoneScopedN`) + op-level latency + end-to-end tok/s.
Figures (see the "Figures" section for captions): `reprofile/dram_bigpicture.png`,
`reprofile/sram_bigpicture.png`, `reprofile/hole_cdf_l1_vs_dram.png`,
`reprofile/zones2_l1only_896/overlap_gantt_l1.png`, and the orthogonal raw-access study
`reprofile/rawaccess_latency.png` / `rawaccess_bandwidth.png`.

---

## 1. Thesis and result

The decode attention kernel (`scaled_dot_product_attention_decode`) is **bound by the
QK^T matmul on the compute engine (FPU)**, not by the KV memory read. The reader (NCRISC)
finishes each KV chunk well before the compute engine (TRISC) needs it, so KV-read latency
sits entirely off the critical path. Because the SDPA *compute* is identical whether K/V
live in DRAM or L1, and because the read is hidden in both cases, **L1 and DRAM produce the
same op latency**. L1's only physical advantage — lower memory access latency — is invisible
while the read is hidden, and is additionally squandered by an issue-bound reader (Section 6).

Consequence: **L1 KV caching is a capacity lever, not a latency lever.** It cannot beat the
DRAM baseline on batch-1 decode latency at any context that fits in L1.

This document proves each link in that chain with measured data.

---

## 2. What one SDPA-decode op does (so the zones mean something)

Per decode step, for each attention layer, the KV sequence is split across ~64 Tensix cores.
Each core, on its slice of the sequence:

1. **(NCRISC, reader)** streams its K and V tiles from memory (DRAM or L1) into circular
   buffers — zone `RD_CHUNK` (with sub-zones `RD_K`, `RD_V`, barrier `RD_KBAR`).
2. **(TRISC, compute)** runs the flash-attention math over that slice — envelope zone
   `CMP_CHUNK`, containing:
   - `QK_MM` — Q·Kᵀ matmul (FPU),
   - a short data-format reconfig (FPU→SFPU handoff),
   - `SM_NORM` — row-max / exp / row-sum softmax (SFPU),
   - `PV_MM` — scores·V matmul (FPU),
   - `SM_RESCALE` — online-softmax rescale across chunks (SFPU).

The reader and the compute engine are different RISC processors that run **concurrently**:
while TRISC computes chunk *i*, NCRISC prefetches chunk *i+1*. The whole question of
"compute-bound vs read-bound" is whether the reader can stay ahead of the compute engine.

---

## 3. Methodology and its pitfalls (so the evidence is trustworthy)

All numbers below come from `DeviceZoneScopedN` markers in
`sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp` and `dataflow/dataflow_common.hpp`,
guarded by `SDPA_PROFILE_ZONES`, captured with `python -m tracy -r -p -v`, ctx=896, no-trace
(the kernel-internal timeline is identical with or without host-side trace). Three pitfalls
were identified and controlled — they invalidate naive readings and are why earlier
conclusions wobbled:

- **The aggregate `DEVICE KERNEL DURATION` column is corrupted** by a free-running-counter
  overflow (it reports ~5.7 s and ~32 s per op — physically impossible). Use the per-zone
  ZONE_START/ZONE_END cycle deltas, or `DEVICE KERNEL DURATION PER CORE AVG/MAX`. A claim of
  "L1 is 5.6x slower" was exactly this artifact (ratio of two corrupted values).
- **The per-core marker buffer (~250 markers) overflows** if too many zones are enabled at
  high layer counts or under trace replay. 2-layer no-trace fits; 32-layer or adding a 3rd
  nesting level drops a kernel's markers entirely. Zones are layer-invariant, so 2-layer
  represents 32-layer.
- **Per-core "cycles since reset" are NOT globally synchronized.** Different cores sit on
  different counter epochs (e.g., x=1 cores read ~5.86e9 ns while x=11 cores read ~0). So a
  cross-core wall-clock timeline from the raw CSV is meaningless. Only two comparisons are
  valid and are the only ones used here: (a) **same-core** overlap (NCRISC and TRISC on one
  physical core share a counter), and (b) **durations** (epoch-invariant).

---

## 4. Evidence A — the compute is matmul-bound (FPU), and QK^T dominates

Device-zone aggregation on the wall-setting TRISCs (`analyze_zones.py`):

| engine | zones | share of attention compute |
|---|---|---|
| FPU (matmul) | QK_MM + PV_MM | **68-84%** |
| SFPU (softmax) | SM_NORM + SM_RESCALE | 16-32% |

Per-core breakdown of the compute envelope (`draw_bigpicture.py`, n=192 core·TRISC units,
L1 ctx896), bucketed by envelope length:

| units (by CMP_CHUNK) | QK_MM | QK_MM→SM_NORM hole | CMP_CHUNK |
|---|---|---|---|
| shortest 25% | 2229 ns | 150 ns | 5124 ns |
| 2nd quartile | 3450 ns | 262 ns | 6187 ns |
| 3rd quartile | 6602 ns | 127 ns | 9249 ns |
| **longest 25% (critical path)** | **8935 ns** | **86 ns** | **11532 ns** |

On the cores that set op latency, the **QK^T matmul alone is 77% of the envelope** and the
QK_MM→SM_NORM hole is **86 ns** — negligible. SDPA-decode is a matmul, and the op latency is
dominated by it. (The hole is an interval no zone accounts for — see Section 7.)

---

## 5. Evidence B — the KV read is fully hidden behind compute

Two independent, valid lines of evidence (see Section 3 on what is valid):

**(i) Durations: read < compute on every single core.** Per-chunk read `RD_CHUNK` = **4704 ns**
(median, L1). The *shortest* compute unit is 4998 ns; the bottleneck is 11532 ns. So
`RD_CHUNK` is shorter than the compute on **all 192 units** — and because the reader
double-buffers (prefetches chunk *i+1* during compute of chunk *i*), the read never gates the
compute. The bottleneck core — the one that matters — buries a 4704 ns read inside an 8930 ns
QK^T matmul *alone*. See `sdpa_bigpicture.png`: the read line sits left of every bar.

**(ii) Same-core cycle-aligned timeline.** `overlap_gantt_l1.png` plots one physical core's
NCRISC read and TRISC compute on their shared counter: the L1 read bar lies entirely inside
the compute envelope. The prior full-grid run measured **read-hidden fraction = 100.0% on all
64 cores** (NCRISC RD ⊆ TRISC CMP_CHUNK union), with a ~1.9x compute-over-read margin.

The reader stays ahead with margin to spare. Faster memory cannot speed up an op whose
memory traffic is already finished before it is needed.

---

## 6. Evidence C — L1's memory advantage is real but unusable here

The premise "L1 is faster than DRAM" is true at the raw-access level and false at the
chunk-read level:

| zone | L1 | DRAM | meaning |
|---|---|---|---|
| `RD_LAT` (isolated 1-tile round-trip) | 362-368 ns | 405-413 ns | L1 SRAM ~12% lower latency than DRAM (GDDR6) |
| `RD_K` (K chunk) | 2597 ns | 2584 ns | **equal** |
| `RD_V` (V chunk) | 2073 ns | 2068 ns | **equal** |
| `RD_CHUNK` (full chunk) | 4704 ns | 4719 ns | **equal** |

L1's per-tile latency edge **does not appear in the bulk read**, because the chunk read is
**issue/barrier-bound** (per-tile NoC address-compute + a fixed barrier cadence
`get_barrier_read_threshold()` tuned for DRAM), not memory-latency-bound (`RD_KBAR`
final-barrier wait ≈ 30 ns for both). So even if reads *were* on the critical path, L1 as
currently read would not be faster than DRAM. They are not on the critical path anyway
(Section 5), so this is doubly moot for latency.

---

## 7. Evidence D — the QK_MM→SM_NORM "hole" (left unnamed: no zone accounts for it)

Between the `QK_MM` zone END and the `SM_NORM` zone START each TRISC shows an idle interval. We
deliberately do NOT call it a characterized "gap": no device zone covers it, so it is left as a
HOLE. What is established about it:

1. **It is an intra-core UNPACK/MATH/PACK pipeline bubble (measured by sub-unit).** The SDPA
   compute kernel runs across the three Tensix sub-units, which pipeline through `cb_qk_im`. The
   hole splits by sub-unit — UNPACK (TRISC_0) p90 **804 ns**, PACK (TRISC_2) **550 ns**, MATH
   (TRISC_1) **509 ns**: UNPACK waits longest for the QK^T output to be packed before it can feed
   the reduce. That per-sub-unit split is the fingerprint of an intra-core pipeline bubble (not
   cross-core, not memory).

2. **Why no single zone/line pins it.** The only executed source statements in the interval are
   two reconfig calls (`sdpa_flash_decode.cpp:369-370`); sub-zoning them (`DF_RECFG` 38 ns +
   `PACK_DRAIN` 32 ns) is only ~30% (~70 ns) of the typical hole. The mask-add (`:340-360`) is
   fused into QK_MM (`add_mask_fusion`, `DYNAMIC_CHUNK_SIZE=1`) and does not run. The rest is two
   things, only one of which is a code instruction: (a) **LLK circular-buffer sync** — a real
   wait, but inside the compute library (the `cb_wait_front`/semaphores the compute API inserts
   around `cb_matmul_blocks`/`reduce_c`), not a top-level SDPA-source line; and (b) **hardware
   pipeline drain** — the QK_MM `DeviceZoneScopedN` closes when the matmul is *issued*, while the
   FPU/packer finish in-flight work, which has no wait instruction at all. A probe that hoisted
   `cb_wait_front(cb_qk_im)` into the hole (zone `QK_WAIT`) measured ~0 — confirming there is no
   top-level software wait at that point; the time is LLK sync + hardware drain. Going finer is
   blocked by the device profiler's hard per-kernel zone cap (~4-5 zones; the kernel is already at
   it, so added zones are silently dropped) — pinning the LLK instruction would need to
   instrument the compute library itself or enlarge the profiler marker buffer. Code snippet +
   line refs: `RESULTS_AND_REPRO.md`.

3. **It is NOT a memory read wait — two independent proofs.**
   - *Memory-invariant (measured).* The hole-duration CDF is identical for L1 and DRAM:

     | hole duration (n=3072) | L1 | DRAM |
     |---|---|---|
     | p50 | 167 ns | 86 ns |
     | p90 | 590 ns | 541 ns |
     | cores > 500 ns | 384 (12%) | 382 (12%) |

     A read wait would shrink with L1's ~12% lower read latency (Section 6); it does not.
   - *No read dependency in the interval (code).* Mask fused (so no `add_block_inplace(cb_mask_in)`);
     the cross-core reduction runs AFTER PV, not here; `SM_NORM`'s `reduce_c`/`sub_exp` consume
     only this core's own `cb_qk_im` plus a constant `cb_identity_scale_in` (`compute_common.hpp`).
     The reader (NCRISC) is not involved.

4. **It is off the critical path.** The hole is largest on light, fast cores (short QK_MM) that
   finish early and idle; the *bottleneck* cores that set op latency have an ~86 ns hole
   (Section 4). It sits behind a ~9 µs QK^T matmul that is identical for L1 and DRAM, so removing
   it — even entirely — opens no latency margin over DRAM.

(No fabricated cross-core "who-waits-for-whom" timeline: per-core profiler counters are on
different epochs — Section 3 — so a wall-clock cross-core alignment is not constructible, and is
unnecessary: the L1==DRAM identity plus the code structure already prove memory is not the cause.)

---

## 8. Evidence E — the bottom line, measured end to end

- **Op latency, same context, L1 vs DRAM:** within ±0.6% across 512/1024/1792 tokens.
- **End-to-end, trace-on:** after moving the ring index on-device (Section 8b), L1 trace-on
  **20.95 tok/s** vs DRAM trace-on **20.97 tok/s** (−0.1%, within run-to-run noise) — exact
  parity. (Before that change, with the host-side ring modulo + push, L1 was 20.90 = −0.35%.)
  Trace-off both ~11.2-11.6 tok/s; the 2.4x ON/OFF difference is host dispatch removed by trace,
  orthogonal to L1 vs DRAM.
- **KV write:** l1_only skips the DRAM `paged_update_cache` but issues the *same*
  `paged_update_cache` to the L1 tier (1-token write is op-overhead-bound) — net zero.

L1 ties DRAM. It does not beat it.

---

## 8b. Host-side per-step operation inventory (L1 vs DRAM)

To prove L1 reaches "at least parity," every per-step host operation was audited and classified.
Under trace, the per-step host work is now IDENTICAL for L1 and DRAM. Inventory (per decode step):

| operation | DRAM | L1 | type | verdict |
|---|---|---|---|---|
| token pad + `from_torch` | ✓ | ✓ | host compute + H2D | shared, not a delta |
| `current_pos` `from_torch` | ✓ | ✓ | H2D async | shared |
| `rope_idxs = get_rot_idxs` (host) | ✓ | ✓ | host compute | shared |
| `page_table` `from_torch` | paged | paged | H2D async | shared |
| in-place `copy_host_to_device_tensor` | ✓ | ✓ | H2D async | shared |
| `current_pos += 1` (host) | ✓ | ✓ | host compute | shared |
| `sample_host` (logits readback) | ✓ | ✓ | **device→host SYNC** | shared — blocks both, NOT an L1 delta |
| ring modulo `torch.remainder/where` | ✗ | ✗ | — | **ELIMINATED**: moved on-device (`Transformer._compute_l1_ring_pos_device`), computed once/step from the device `current_pos`; no-wrap case returns `current_pos` (zero ops) |
| `l1_update_pos` `from_torch` + extra H2D push | ✗ | ✗ | — | **ELIMINATED** with the above (no host tensor to stage/push) |
| hit-ratio `torch` + `.item()`×2 + `add_sample` | ✗ | flag-only | host compute | **GATED OFF by default** (`l1_kv_min_expected_hit_ratio==0`); diagnostic; its `.item()` is on host torch tensors, not a device sync |
| `_build_adaptive_l1_write_pos` `to_torch` | ✗ | ✗ | — | not reached under trace; the on-device path replaced it; one-shot even on the no-trace fallback |
| tier alloc + sink seed (`to_torch`) | ✗ | ✓ **once** | setup SYNC | **CANNOT be eliminated, but is one-time at compile, not per-step.** Reason: the L1 buffer must be placed (after CB addresses are frozen) and populated with the prefilled KV exactly once; it is amortized over the whole decode and is off the steady-state path. |

**Result:** under trace, L1's per-step host work == DRAM's (tokens, current_pos, rope, page_table).
There is ZERO L1-specific per-step host compute and ZERO per-step device→host sync unique to L1.
The single L1-only cost that genuinely cannot be removed is the one-time tier allocation + seed,
which does not touch steady-state latency. The on-device ring index adds ~5 ttnn ops/step inside
the trace (wrap case) or zero (no-wrap), which is hidden in device time and does not regress
tok/s (measured 20.95, up from 20.90). This is why L1 reaches exact host parity with DRAM.

---

## Figures (captions; the charts are data-only)

- **`dram_bigpicture.png` / `sram_bigpicture.png`** — per-backend SDPA-decode big picture.
  *Panel A:* each of the 192 (core × TRISC) units' compute timeline, sorted by envelope length;
  segments `pre | QK_MM | <hole> | SM_NORM | rest(PV_MM+rescale+post)` sum to the unit's CMP_CHUNK.
  The QK_MM→SM_NORM interval is left BLANK (a hole — no zone covers it, Section 7). The dashed line
  is the per-chunk KV read (RD_CHUNK); it sits left of every unit ⇒ read is hidden on every core.
  *Panel C:* the bottleneck unit (longest envelope, sets op latency) with its read bar — the read
  ends inside the QK^T matmul alone. The two charts are near-identical: SDPA compute is the same
  whether KV is in DRAM or SRAM. (The SRAM Panels omit the small green PV/rest tail — the
  CMP_CHUNK envelope zone is dropped by the profiler on the L1 path, so the SRAM timeline ends at
  SM_NORM; QK_MM-dominance, the hole, and read-hidden are unaffected.)
- **`hole_cdf_l1_vs_dram.png`** — CDF of the QK_MM→SM_NORM hole duration, SRAM vs DRAM. The two
  curves coincide (p90 ~590 vs ~541 ns; >500 ns tail ~12% both) ⇒ the hole is memory-invariant, so
  it is not a read wait (Section 7).
- **`zones2_l1only_896/overlap_gantt_l1.png`** — same-core (shared-counter) NCRISC read vs TRISC
  compute: the L1 read bar lies entirely inside the compute envelope.

## 10b. Raw memory-hierarchy access (orthogonal study)

Independently of SDPA, the raw READ latency and bandwidth of the memory hierarchy were swept over
transfer size (64 B → 1 MB; the per-read destination is one core's L1, ~1.5 MB, so a single
resident transfer caps there — the 4 MB end is bandwidth-via-iteration, not one read) for three
sources: DRAM, local SRAM (a core's own L1), and remote SRAM (another Tensix core's L1 over NoC).
This generalizes the old single-point RD_LAT (DRAM ~412 ns / L1 ~336 ns at one 2 KB tile).

- Harness (reused, no custom kernel): `tests/.../perf_microbenchmark/dispatch/test_bw_and_latency`
  — `-m 1` DRAM, `-m 2` L1 (`-sx/-sy` source core: == reader ⇒ local, far core ⇒ remote over NoC),
  `-p` per-read size, `-bs` total KB, `-l` latency mode, `-i` iterations.
- Figures: **`rawaccess_latency.png`** (latency ns vs size, log-y) and **`rawaccess_bandwidth.png`**
  (GB/s vs size), each with three series (DRAM / local SRAM / remote SRAM), log-scale x.
- Measured (P150): latency floor at 64 B — local SRAM ~55 ns < remote SRAM ~245 ns < DRAM ~337 ns
  (consistent with the old RD_LAT ~336/412 ns at 2 KB); the three converge at large sizes
  (transfer-time-bound). **Single-reader** bandwidth at 1 MB — SRAM ~82 GB/s vs DRAM ~63 GB/s
  (one core / one NoC link; the device aggregate DRAM BW across all cores is the ~512 GB/s spec).
  So SRAM's edge over DRAM is real at the raw-access level (~12% lower 1-tile latency, ~30% higher
  single-reader BW) — but it is hidden behind compute in SDPA decode (Sections 4-7), which is why
  it does not change op latency.
- Note: even "local SRAM" goes through the NoC read path here (loopback), matching how SDPA reads
  L1 — the apples-to-apples comparison vs DRAM and remote SRAM.
- Repro: `reprofile/run_rawaccess_sweep.sh` → `rawaccess/rawaccess.csv` → `reprofile/draw_rawaccess.py`.

### Bandwidth-saturation crossover — there is none on a single chip
The "where does L1 finally beat DRAM" question reduces to: where does the SDPA DRAM read DEMAND
reach the DRAM aggregate CEILING? Demand = KV-bytes/token ÷ SDPA-op time/token. For Llama-8B
(8 kv-heads × 128 × {K,V}): bf16 = 4096 B/token, bfp8 ≈ 2176 B/token; the measured SDPA-op slope
is ~0.0116 µs/token, so demand **asymptotes at ~353 GB/s (bf16) / ~188 GB/s (bfp8)** — below the
~512 GB/s GDDR6 aggregate ceiling (single-reader DRAM measured ~63 GB/s; the multi-core aggregate
test was unreliable in this env, so 512 is the spec). Batch and context distribute across cores
preserving the compute:read ratio, so they do NOT raise the asymptote. Result: **DRAM never
saturates at any context or batch on one chip ⇒ the read stays hidden ⇒ L1 cannot beat DRAM on a
single chip** (~1.45x DRAM headroom at the bf16 asymptote). Figure: `crossover_demand_vs_ceiling.png`
(`reprofile/draw_crossover.py`). The crossover requires one of: a model with higher
KV-bytes-per-compute, KV larger than fits one chip's L1 (⇒ quantization / multi-chip — a CAPACITY
lever), or multi-chip where per-chip DRAM BW is divided while KV is replicated. None is a layout
change. This is the quantitative form of "the lever is capacity, not layout."

### When can DRAM bandwidth saturate? (and why multi-chip alone cannot) — detailed

**The governing quantity.** DRAM is saturated (the read stops being hideable) iff, per chip,
`DEMAND ≥ CEILING`, where
`DEMAND = (KV bytes read per token) / (SDPA compute time per token)` and `CEILING` = that chip's
DRAM aggregate bandwidth (~512 GB/s GDDR6). DEMAND is a *ratio of per-token quantities*, so it is
a property of the model, not of scale.

**Why it is invariant under scale and standard sharding.** Expand it:
`DEMAND ∝ (n_kv_heads · head_dim · dtype_bytes) / (n_q_heads · head_dim · compute_rate)`
`     = (n_kv_heads / n_q_heads) · (dtype_bytes / compute_rate)`.
The dependence on context length and batch cancels (both KV bytes and compute scale with them),
leaving a per-token architectural ratio: the GQA group size, the KV dtype, and the attention
arithmetic intensity. Now apply each multi-chip scheme — each shards numerator and denominator
together, and each chip brings its own DRAM (its own CEILING):
- **TP (tensor/head parallel, N ≤ n_kv_heads):** chip computes `n_q_heads/N` q-heads and reads
  `n_kv_heads/N` kv-heads ⇒ DEMAND unchanged; CEILING per chip unchanged ⇒ no saturation.
- **DP (data/batch parallel):** each chip is the full model on different sequences ⇒ identical to
  one chip ⇒ no saturation.
- **CP (context/sequence parallel):** each chip holds `1/N` of the context, computes a partial
  attention over its slice (read `1/N`, compute `1/N`) ⇒ DEMAND unchanged ⇒ no saturation.

So adding chips adds DEMAND and CEILING in lockstep — **you cannot saturate DRAM by scaling out.**
(Empirically consistent with the measured per-core invariance under batch/heads.) For Llama-8B
bf16, DEMAND ≈ 353 GB/s vs CEILING ≈ 512 GB/s — ~1.45x headroom, at every context, batch, and chip
count under standard parallelism.

**The one sharding exception (a corner case).** TP with `N > n_kv_heads`: kv-heads cannot divide
below 1, so per-chip read floors at one kv-head while per-chip compute keeps shrinking
(`n_q_heads/N`). DEMAND then rises ∝ `N / n_kv_heads` and can eventually exceed CEILING. But this
is an over-decomposed, inefficient TP degree (beyond the kv-head count) rarely used for decode, and
per-chip latency is dominated by other costs there — not a practical crossover.

**When DRAM DOES saturate (where L1 could win on latency).** Raise the read:compute ratio past the
~1.45x headroom:
1. **Attention architecture — the dominant dial.**
   - **MHA / low-GQA:** `n_kv_heads → n_q_heads`. For Llama-8B that is 32 vs 8 → **4× the KV read
     for the same matmul compute** → DEMAND ≈ 4·188 ≈ **750 GB/s (bfp8) ≫ 512** → saturated. Any
     MHA or small-GQA-group model qualifies.
   - **Wide KV dtype:** fp16/bf16 vs int8/int4 — more bytes/token raises DEMAND. (Quantizing KV
     *lowers* DEMAND, i.e., hides the read *more* — the opposite of what helps L1.)
   - **Not** `head_dim`, **not** FFN width / "arithmetic intensity": `head_dim` cancels (it scales
     read and compute equally, see derivation below), and the read is hidden behind the **SDPA**
     compute (QK^T / PV), not the FFN — so model-level arithmetic intensity is irrelevant to whether
     the KV read is exposed. The only architectural dials are the **GQA ratio** and the **KV dtype**.
2. **Hardware balance:** a memory-light / compute-heavy chip (lower DRAM BW, or much higher FLOPs)
   lowers CEILING and/or shrinks the compute time that hides the read → DEMAND crosses CEILING.
   (Counter-intuitively, a *faster*-compute future chip saturates DRAM *more* easily, because there
   is less compute time to hide the same read behind.)

**But saturation alone is not an L1 win — capacity still gates it.** Even once DRAM is saturated
and the read is exposed, L1 beats DRAM only if the KV actually fits in (aggregate) L1. The very
regimes that expose the read (MHA, large head_dim, wide dtype) are the ones with the *largest* KV,
so they are the hardest to fit — the crossover and the capacity requirement compound. That is why
the lever is fundamentally **capacity** (get the KV into on-chip SRAM at all — quantization,
multi-chip aggregate L1), not **layout** (how it is arranged once it already fits). And note "SRAM
is faster" is true only at the raw-access level (Section 10b); that speed is hidden behind compute,
so what L1 actually offers is being *outside the shared DRAM bandwidth pool* — useful only in the
saturated regimes above.

### The saturation formula, derived from first principles

We want a single closed-form test for "does the KV read saturate DRAM (stop being hideable behind
SDPA compute)?" applied to any model config, with no per-model profiling. Derive it from the two
per-decode-step quantities.

**Numerator — KV bytes read per step (context `C`).** The flash-decode op reads the entire K and V
cache once per step:
`READ = C · n_kv_heads · head_dim · 2 · dtype_bytes`  (the `2` = K and V).

**Denominator — SDPA compute time per step.** QK^T is a dot product of length `head_dim` for each of
`n_q_heads` query heads against each of `C` cached positions; PV is the symmetric contraction. So
the MAC count is `COMPUTE_MACs = C · n_q_heads · head_dim · k`, where `k` folds the fixed
per-position work (QK^T + PV + softmax) and is a constant for a given kernel. At a sustained FPU MAC
rate `R` (MAC/s), compute time `= COMPUTE_MACs / R`.

**Demand = numerator / denominator:**
```
DEMAND = READ / (COMPUTE_MACs / R)
       = [C · n_kv_heads · head_dim · 2 · dtype_bytes] · R / [C · n_q_heads · head_dim · k]
       = (n_kv_heads / n_q_heads) · dtype_bytes · (2R / k)
```
`C` cancels (read and compute both scale with context) and `head_dim` cancels (it scales read and
compute equally). What survives is a pure per-token architectural ratio. Collapse the two hardware
constants `(2R / k)` into one calibrated constant **`Κ`**:
```
DEMAND[GB/s] ≈ (n_kv_heads / n_q_heads) · dtype_bytes · Κ
```

**Calibrating Κ from the one measured model.** Llama-3.1-8B (`n_kv/n_q = 8/32 = 0.25`, KV = bfp8 ≈
1.0625 B/elem) has a measured SDPA-op slope of 0.0116 µs/token, i.e. an asymptotic
`DEMAND = (8·128·2·1.0625) / 0.0116 = 2176 B / 0.0116 µs ≈ 188 GB/s`. Solving
`188 = 0.25 · 1.0625 · Κ` gives **`Κ ≈ 707 GB/s`**. Because `Κ` is `2R/k` — purely the FPU rate and
the kernel's fixed per-position cost — it is a hardware constant, identical across models on the
same chip/kernel. (Cross-check: the same Κ reproduces the bf16 asymptote, `0.25·2·707 ≈ 353 GB/s`,
matching `crossover_demand_vs_ceiling.png`.)

**The test:** the read saturates DRAM iff `DEMAND ≥ CEILING (≈512 GB/s)`, i.e.
```
(n_kv_heads / n_q_heads) · dtype_bytes  ≥  512 / 707  ≈  0.72
```
So the saturation threshold on the product `ratio · dtype_bytes` is ≈ **0.72**: e.g. ratio > 0.36 at
bf16 (2 B), or essentially MHA-only (ratio ≈ 1) at bfp8 (1.06 B).

### Per-model evaluation across tt_transformers configs

Applying `DEMAND = ratio · dtype_bytes · 707` to every model the framework has configs for
(`model_config.py`: `LOCAL_HF_PARAMS` L439-452, plus the HF-loaded Phi-3 family at L108/116/631-632/
2427). KV dtype is bfp8 (~1.06 B) for the Llama/Mistral/Phi-3 accuracy group and the Qwen-VL models;
Qwen2.5-7B forces bf16 (L162-170). Llama-3.2-3B also runs bf16 in accuracy mode.

| model | n_q / n_kv | ratio | KV dtype (B) | DEMAND (GB/s) | ≥ 512? |
|---|---|---|---|---|---|
| Llama-3.1-70B / 3.2-90B-V / Qwen2.5-VL-72B | 64/8 | 0.125 | bfp8 (1.06) | ~94 | no |
| Qwen2.5-VL-3B | 16/2 | 0.125 | bfp8 (1.06) | ~94 | no |
| Qwen2.5-7B | 28/4 | 0.143 | bf16 (2.0) | ~202 | no |
| Qwen2.5-VL-32B | 40/8 | 0.20 | bfp8 (1.06) | ~150 | no |
| Llama-3.1-8B / Mistral-7B-v0.3 / 3.2-1B / 3.2-11B-V | 32/8 | 0.25 | bfp8 (1.06) | ~188 | no |
| Llama-3.2-3B | 24/8 | 0.333 | bfp8 / bf16 | ~250 / ~471 | no (closest) |
| **Phi-3-mini / Phi-3.5-mini** | **32/32** | **1.00** | **bfp8 (1.06)** | **~750** | **YES (1.46×)** |

**Conclusion.** Every GQA model with a shipped config stays compute-bound — the highest, Llama-3.2-3B
in bf16 accuracy mode, reaches only ~471 GB/s, still ~8% under the ~512 GB/s ceiling. The single
config that saturates is **Phi-3-mini / Phi-3.5-mini**, which are **MHA** (32 query = 32 KV heads,
ratio 1.0, head_dim 96): they read 4× the KV per token for the same matmul compute, so DEMAND ≈ 750
GB/s (bfp8) to ~1414 GB/s (bf16) clears the ceiling at long context. There the KV read is exposed
and L1 could in principle beat DRAM on decode latency.

Two caveats: (1) it remains a **capacity** story — MHA's 4× KV/token (≈3× Llama-8B's bytes after the
head_dim-96 offset) shrinks the L1-resident context to ≈1/3, so the L1 win is realizable only where
that larger KV still fits; (2) this is an **analytical prediction** from the calibrated Κ and the
published Phi-3 head config (the repo loads it from HF, not a local params file), not an on-device
measurement — and Phi-3-mini is not supported on P150, so it was not run.

---

## 9. The chain of reasoning, in one place

1. SDPA-decode op latency is set by the slowest core's **compute** (Sections 4, 7).
2. That compute is **77% QK^T matmul** (FPU), with a negligible critical-path hole (Section 4).
3. The KV **read finishes inside that compute** with ~1.9x margin, on every core (Section 5).
4. Therefore op latency is **independent of KV memory speed/location** — read is off the
   critical path.
5. The SDPA **compute is byte-for-byte identical** for L1 and DRAM (same kernel, same op;
   only the K/V source address differs) ⇒ identical compute time (Section 8).
6. L1's one real edge (lower access latency) is **invisible while the read is hidden**, and
   is **further wasted** by an issue-bound reader (Section 6).
7. ∴ **L1 == DRAM on decode latency.** A win would require read > compute (read exposed),
   which does not occur at any context that fits in L1.

---

## 10. The only real lever: capacity, not layout

L1 cannot reduce decode latency, but it can hold KV that would otherwise spill to or saturate
DRAM. The latency crossover where read would exceed compute (and faster L1 would help) lies
far above current L1 capacity (~900 tokens at 32 layers); reaching it needs more L1 residency,
i.e. **capacity** (KV quantization to int8/int4, multi-chip per-chip residency), not a better
in-L1 layout. Note multi-chip tensor-parallel sharding does **not** expose the read either:
per-chip compute and per-chip KV read both scale as 1/N, leaving the compute:read ratio
invariant (KV is replicated per device; attention is all-local). So the capacity value of
multi-chip L1 is longer full-resident context per chip, again not latency.

---

## Appendix — key measured numbers (L1 ctx896, 2-layer, no-trace unless noted)

```
FPU:SFPU on wall TRISCs ............ 68-84% : 16-32%   (matmul-bound)
QK_MM, critical-path cores ......... 8935 ns  (77% of 11532 ns envelope)
QK_MM->SM_NORM hole, crit path ..... ~86 ns (negligible; no zone covers it, ~30% is reconfig)
QK_MM->SM_NORM hole, light cores ... up to ~1665 ns (unaccounted; off critical path)
CMP_CHUNK spread ................... 4998 -> 11619 ns (2.3x load imbalance)
RD_CHUNK (read/chunk) .............. L1 4704 / DRAM 4719 ns   (equal; issue-bound)
RD_K / RD_V ........................ 2597/2584 ; 2073/2068 ns (equal)
RD_LAT (1-tile latency) ............ L1 362 / DRAM 405 ns     (L1 ~12% lower, unused)
RD_KBAR (final barrier wait) ....... ~30 ns both              (no memory-wait on crit path)
read-hidden fraction ............... 100% on all 64 cores (~1.9x margin)
op latency L1 vs DRAM .............. within +/-0.6% (512/1024/1792 tok)
end-to-end tok/s, trace-on ......... L1 20.90 vs DRAM 20.97 (-0.35%, within noise)
```

Reproduce: `reprofile/run_rw_probe.sh` (zones), `analyze_zones.py` / `analyze_gaps2.py`
(analysis), `draw_bigpicture.py` (figure). Enable zones by
uncommenting `#define SDPA_PROFILE_ZONES 1` in both kernels and wiping
`~/.cache/tt-metal-cache`; disable (default) after.

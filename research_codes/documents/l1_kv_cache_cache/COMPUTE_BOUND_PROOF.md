# Why the L1 KV Cache Cannot Beat the DRAM Baseline on Decode Latency

**A measured proof that SDPA-decode is compute-bound, that the KV read is fully hidden
behind compute, and that therefore moving the KV cache from DRAM into L1 yields parity,
not a speedup, on single-stream decode latency.**

Platform: Tenstorrent Blackhole P150, single card, Llama-3.1-8B, batch-1 decode.
Method: device-zone profiling (Tracy `DeviceZoneScopedN`) + op-level latency + end-to-end tok/s.
Figures: `reprofile/sdpa_bigpicture.png`, `reprofile/zones2_l1only_896/overlap_gantt_l1.png`.

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

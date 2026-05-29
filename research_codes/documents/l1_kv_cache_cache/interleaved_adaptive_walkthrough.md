# Interleaved-Adaptive L1 KV Cache: Walkthrough and Findings

> Adds an interleaved-layout option to the adaptive L1 KV cache (`--l1_kv_interleaved_adaptive`),
> as a higher-capacity, higher-throughput alternative to the HEIGHT_SHARDED tiers.
> Pure-Python change (5 files, +52 lines); no kernel/op/build changes.
> Llama-8B, Blackhole P150, batch-1 (`simple_text_demo.py`).
> Last updated: 2026-05-30.

This doc answers four questions raised in review:
1. Does it work in **non-l1-only** mode?
2. Is the **output correct** (not gibberish)?
3. If it doesn't read the headroom map, in what sense is it **"adaptive"**?
4. If `paged_update_cache` already writes interleaved, **how is it used in the sharded config**?

---

## 1. What the change is

`attention.py::_build_adaptive_l1_memcfg_tiers` gained an early branch: when
`l1_kv_interleaved_adaptive` is set, it returns **one** `L1_MEMORY_CONFIG` (interleaved,
all-banks, `allocator_id=0`) tier of `l1_kv_total_size` tokens, instead of N
HEIGHT_SHARDED tiers on disjoint low-CB cores. Everything downstream is unchanged: the
ring-write path, the attention-sink seeding, `l1_only_mode`, and the n-tier SDPA reader
all operate on the tier list regardless of layout.

The sharded path is untouched (the branch early-returns only when the flag is set), so
**both layouts coexist and are flag-selected** for A/B comparison.

### Why it needed zero kernel/op/build changes (Q4 mechanism)

Both the write and read sides of the SDPA decode op are **layout-agnostic**:

- **Write — `paged_update_cache`.** It writes into whatever L1 tensor it is handed and
  derives the bank/shard addressing from that tensor's own buffer. In the **sharded**
  config it writes into the sharded tier tensors (`attention.py:1192/1225`); in the
  **interleaved** config it writes into the interleaved tier; in the legacy non-adaptive
  path it writes into the interleaved `l1_kv_cache`. Same op, three layouts — it does not
  special-case layout. (This is the direct answer to Q4: the sharded config uses the
  *same* `paged_update_cache`, just pointed at sharded tensors.)
- **Read — n-tier reader.** `sdpa_decode_program_factory.cpp:876` builds each tier's
  reader via `TensorAccessorArgs(*l1_k_tiers[ti]->buffer())` — i.e. **from the tensor's
  actual buffer layout**. The kernel reads through `TensorAccessor`
  (`reader_decode_all.cpp`), which transparently handles interleaved or sharded. The
  tile-id formula (`head_base * tier_size_tiles * DHt + local_row * DHt + col`) is the
  logical flat tile index either way; the accessor maps it to banks. The non-adaptive
  interleaved path already feeds an interleaved tensor as "tier 0" through this exact op,
  which is the proof it works.

So swapping the tier tensor's `MemoryConfig` from sharded to interleaved is sufficient;
the op, program factory, and kernels adapt automatically.

---

## 2. The "adaptive" naming is currently a misnomer (Q3 — corrected)

**It does not read the headroom map.** The branch sizes the cache to
`l1_kv_total_size` (= `l1_kv_window_size + l1_kv_sink_size`, user-set), and returns
before any per-core gate, cumulative-depth gate, or headroom lookup runs.

What it *does* reuse from the adaptive code path is the **machinery**, not the sizing:
the ring buffer, attention-sink seeding, `l1_only_mode`, the per-step write-position
hoist, and the n-tier SDPA reader. The genuinely "adaptive" part — headroom-driven
per-core auto-sizing — exists *because* sharded tiers must each fit their cores' L1 gap.
A single uniform interleaved buffer has no per-core tiers to fit, so that sizing logic
doesn't directly apply.

Consequences / honesty:
- The flag name `l1_kv_interleaved_adaptive` overclaims. It is really **"interleaved L1
  KV using the streaming (sink+ring / l1_only) machinery, fixed user-set size."**
  Recommend renaming to `l1_kv_interleaved` (deferred — touches the 5 plumbed files).
- Auto-sizing *could* be added (compute the max interleaved tokens that fit the tightest
  per-core gap). **But the vanilla headroom map under-predicts the l1_only capacity**: it
  predicts ~400 tokens, yet l1_only allocates 896+ clash-free, because `l1_only_mode`
  shrinks the SDPA CB region (no DRAM-read CBs) and frees L1 the vanilla map never saw.
  So genuine auto-sizing needs an l1_only-specific headroom basis — left as future work.

---

## 3. Capacity and throughput (measured)

`simple_text_demo.py -k "performance and batch-1"`, Llama-8B BH P150:

| profile | L1 KV capacity (clash-free) | steady tok/s | output |
|---|---:|---:|---|
| DRAM baseline | — | 11.7 | coherent |
| sharded adaptive (l1_only) | 256 (max ~320) | 7.1 | degenerate → coherent after §7 ceil fix |
| interleaved non-adaptive | 400 (432 clashes) | ~11 | coherent |
| **interleaved-adaptive, l1_only (after §7 fixes)** | **416 → 896+** | 11.6 → 10.9 | **byte-matches DRAM baseline** |
| **interleaved-adaptive, non-l1-only** | **416** (tested) | **10.5** | **coherent** |

(The l1_only "degenerate" results below were the state BEFORE the §7 root-cause fixes;
after them, l1_only output is byte-identical to the DRAM baseline.)

Why interleaved beats sharded on **both** axes:
- **Capacity**: an interleaved buffer is uniform across banks, so it shares the thin top
  band with the model's own interleaved buffers (the same reason non-adaptive interleaved
  reaches 400). Sharded tiers can't use the 64 dense cores at all (one sharded tile-row =
  all 32 layers = 278 KB/core > their ~280 KB gap).
- **Throughput**: interleaved L1 reads stripe across many banks (DRAM-like parallelism);
  sharded reads serialize at the single source core's NoC port (perf-doc "Problem 2").

Caveat on the large numbers: at this benchmark's ~130-token decode, `l1_only` clamps
`cur_pos`, so windows beyond ~150 are never *exercised* — 768/896 are **allocation
ceilings**, not throughput-at-depth. Long-decode behavior is unmeasured.

---

## 4. non-l1-only works; l1_only degrades quality for BOTH layouts (Q1 + Q2)

Generated text, same harness:

- **DRAM baseline**: "…Sriracha is a spicy condiment made from chili peppers, vinegar,
  and garlic." — coherent.
- **interleaved-adaptive, non-l1-only (hybrid L1+DRAM)**: "…provide information within a
  specific topic area of law, but I don't have personal opinions." — **coherent**, stops
  at EOS naturally (iter 67), no clash, ~10.5 tok/s.
- **interleaved-adaptive, l1_only**: "As a computer program is not a computer program is
  not a computer program is not…" — **degenerate repetition** (not gibberish; grammatical
  but looping).
- **sharded adaptive, l1_only**: "…science any science task computer that requires science
  a lot of computer science strength." — **also degenerate**.

Conclusion:
- **Q1 (non-l1-only):** Yes — it runs clash-free, coherent, ~10.5 tok/s (vs 11.7
  baseline; slightly slower because hybrid still does DRAM reads). This is the
  **recommended mode** for the interleaved-adaptive cache today.
- **Q2 (correctness):** non-l1-only is coherent. **l1_only degenerates for *both* the
  sharded and interleaved layouts** — so the degeneration is an `l1_only_mode` problem
  (context dropping / attention-sink not anchoring as intended for this prompt+window),
  **not** introduced by the interleaved change. It is layout-independent and pre-existing.
  Needs separate investigation (suspect sink seeding or the `cur_pos` clamp dropping the
  prompt); do not trust l1_only output quality until fixed.

---

## 5. How to run

```
# Recommended: coherent, high-capacity, ~10.5 tok/s
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" \
  --use_adaptive_l1_kv_cache --l1_kv_interleaved_adaptive --l1_kv_window_size 480

# l1_only variant: higher allocation ceiling + faster, but output currently degenerate
#   (add --l1_kv_only_mode) — quality blocked on the l1_only investigation above.
```
`--l1_kv_headroom_json` is accepted but **ignored** by this path (see §2). Capacity =
`--l1_kv_window_size + --l1_kv_sink_size` (sink defaults to 32 under
`--use_adaptive_l1_kv_cache`).

---

## 7. Root-cause + fix for the l1_only degeneration (RESOLVED)

The l1_only degeneration ("…is not a computer program is not…", "…taste or taste…") was
**two bugs**, both now fixed; l1_only output is byte-identical to the DRAM baseline.

**Bug A — boundary tile read stale DRAM (the dominant one).**
`dataflow_common.hpp::read_kv_mask_chunks_n_tier` computed the L1 fresh-window upper bound
with **floor**: `fresh_hi_tile = (cur_pos+1)/32`. That excludes the partial tile that
*contains* `cur_pos`, so that boundary tile fell through to the DRAM fallback
(`k_reader`/`v_reader`). In l1_only the decode path skips DRAM writes, so that DRAM is
**stale** → the model's most-recent ~32 tokens read garbage → it can't track its own
generation → repetition loop. Fix: use **ceil** for `fresh_hi_tile` (clamped to
`total_l1_tiles`). The boundary tile's positions ≤ cur_pos were decode-written to the ring;
positions > cur_pos are causally masked by SDPA, so reading the whole tile from L1 is
correct. This alone makes both sharded and interleaved l1_only *coherent*.

**Bug B — prefill body not L1-resident (faithfulness / true DRAM-free).**
`seed_adaptive_l1_sinks` seeded only the 32-token attention sink. With `decode_start_pos=0`
forced in l1_only, the kernel treats all of `[0, cur_pos]` as L1-resident, but prefill
`[sink, prompt_len)` was never written to L1 → read as zeros. With Bug A fixed the model
limps along on sink + recent tokens (coherent but **lossy** StreamingLLM — output diverges
from baseline, e.g. "the city of Paris" instead of the true continuation). Fix: when the
cache is a single tier sized ≥ the sequence (the interleaved-adaptive case), seed the
**full prefill** `[0, capacity)` from DRAM instead of just the sink. Then `[0, prompt_len)`
is L1-resident, decode extends it linearly (no wrap), and l1_only attends to the full
context — output **byte-matches the DRAM baseline**, with **zero DRAM reads in decode** (so
the DRAM writes are correctly skipped — the "skip DRAM entirely" goal).

Both fixes are layout-independent in the kernel; Bug B's seed currently covers the
single-tier (interleaved) case. Multi-tier sharded still gets the sink-only seed, so
sharded l1_only is coherent-but-lossy (Bug A fixed, Bug B not) until a cross-tier
full-prefill seed is added.

Files: `dataflow_common.hpp` (ceil), `attention.py::seed_adaptive_l1_sinks` (full seed).

## 8. Open items

1. ~~l1_only quality~~ — **FIXED** (§7); l1_only interleaved-adaptive == DRAM baseline.
2. **Naming**: rename `l1_kv_interleaved_adaptive` → `l1_kv_interleaved` (not headroom-adaptive).
3. **Genuine auto-sizing**: needs an l1_only-aware headroom basis (vanilla map under-predicts ~2x).
4. **Long-decode validation**: exercise windows past the ring-wrap point (capacity < seq),
   where StreamingLLM loss is expected and faithfulness will degrade by design.
5. **Sharded l1_only faithfulness**: add cross-tier full-prefill seed (Bug B for multi-tier).
6. **Hybrid (option 1)** still available if sharded locality is ever wanted on top of the band.

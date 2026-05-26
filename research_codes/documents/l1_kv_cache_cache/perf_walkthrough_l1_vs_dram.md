# Adaptive L1 KV Cache: Performance Walkthrough and Why L1-Only Is Still Slower Than DRAM

> Captures the optimization journey from a 5.75 tok/s correctness baseline to a 11.5 tok/s
> steady-state L1-only mode, the diagnostic data that exposed the real bottleneck, and the
> architectural reasons we cannot freely shrink `runtime_pad_bytes` to use all of L1.
> Llama-8B, Blackhole P150, batch=1 latency test (`simple_text_demo.py`).
>
> Last updated: 2026-05-25.

---

## 1. Scoreboard

DRAM-only baseline = pure `paged_update_cache` to interleaved DRAM + DRAM reads in SDPA decode.
Adaptive L1 KV cache = `paged_update_cache` to HEIGHT_SHARDED L1 tiers + L1 reads in SDPA decode
(with optional DRAM fallback for tiles outside the L1 ring window).

| Configuration | Mode | Cache | 128th tok | Avg | tok/s |
|---|---|---|---:|---:|---:|
| **DRAM baseline** | — | none | **84.2 ms** | 85.2 ms | **11.73** |
| + Phase 5 (disjoint-core filter) | hybrid | T=1 sink-only | 91.8 ms | 106 ms | 9.43 |
| Phase 4b initial (l1_only_mode, naive cur_pos plumb) | l1_only | T=2 | 107 ms | 302 ms | 3.31 |
| Phase 4b + kernel-side cur_pos clamping | l1_only | T=2 | 90.2 ms | 156 ms | 6.40 |
| + Phase 6 (hoist `flat_pos`, alias `k_heads`) | l1_only | T=2 | **87.5 ms** | 104 ms | 9.59 |
| + trace mode | l1_only | T=2 disjoint | **86.9 ms** | 102 ms | 9.81 |
| Trivially-fast (sink-only, no ring writes) | l1_only | T=1 sink only | **84.8 ms** | 102 ms | 11.79 |

Two important nuances:

1. The "trivially-fast" config is not an honest baseline win — `sink_size == total_capacity ⇒ ring_cap = 0`,
   `_build_adaptive_l1_write_pos` returns `None`, and the model does **zero** decode-time L1 writes.
   The kernel iterates only the 32 sink tiles. It's doing less work than baseline, not faster work.

2. With real ring writes (T=2, sink=32, ring=32), the best we measured is **86.9 ms steady state in trace mode**,
   versus DRAM baseline **84.2 ms steady state in trace mode** — a **2.7 ms gap that the optimizer can't close
   from the Python side**. The detailed breakdown in §4 explains why.

---

## 2. Optimization journey (chronological)

This is a compressed log of what we tried, in order. Each phase has its own working doc in
`research_codes/documents/l1_kv_cache_cache/`; this file is the index + analysis.

### Phase 0 — Correctness restoration

The original adaptive scheme had two pre-existing bugs that produced incorrect (sometimes degenerate)
output once decoding crossed the ring's wrap point:

1. **Ring-write / direct-read mismatch in the SDPA decode reader.** The Python write path stored token K/V
   at `L1[cur_pos % T_total]` (ring), but the SDPA decode reader naively returned `L1[gst]` for every
   sequence tile `gst`. After `cur_pos ≥ T_total`, the slot was overwritten but the reader still treated it
   as the old token. Fix: ring-aware fresh-window math in `dataflow_common.hpp::read_kv_mask_chunks_n_tier`
   (compute `fresh_lo_tile`, `fresh_hi_tile`, map `gst → flat_tile = gst % total_l1_tiles`).

2. **Prefill positions not seeded into L1.** The L1 adaptive tier tensors were zero-initialised, and prefill
   K/V never touched them. Decode-time reads for prefill positions (sink range) returned zeros, breaking
   attention. Fix: pass `decode_start_pos` (cur_pos at start of decode) as a runtime arg so the kernel knows
   to only treat post-prefill tiles as "fresh in L1".

After this phase, output matched DRAM baseline byte-for-byte but throughput was 5.75 tok/s (worst of run).

### Phase 1 — Eliminate per-step host syncs in the L1 write path

`_write_adaptive_l1_tiers` had `int(ttnn.to_torch(flat_l1_pos_tensor)…)` — a host sync per attention layer
per decode step (32 syncs/token at ~2-3 ms each). Replaced with on-device per-tier offset compute + the
`UINT32_MAX` skip-sentinel that `paged_update_cache` already honors (writer/reader kernels at
`reader_update_cache_interleaved_start_id.cpp:72-74`, `writer_…:60-62`).

Result: 138 → 123 ms per token. The host syncs were the dominant cost.

### Phase 2 — CB-clash bisect

Lowered `runtime_pad_bytes` from 1024 KiB toward 256 KiB to fit more KV. Below ~580 KiB:

```
Statically allocated circular buffers in program 49 clash with L1 buffers on core range
[(x=0,y=0) - (x=7,y=8)]. L1 buffer allocated at 691456 and static circular buffer region ends at 722944
```

Program 49 is the SDPA decode kernel; its CB region top is at byte 722944 on cores y=0..7. The "L1 buffer at
691456" turned out to be the 63-core L-shape model intermediate (QKV/embedding path) — see §3 for why this
sets the floor on `runtime_pad_bytes`.

### Phase 3 — Shrink-all cumcap policy

Refactored `_build_adaptive_l1_memcfg_tiers` to seed every viable tier at 1 tile-row and grow by
token-efficiency, instead of greedily dropping low-efficiency tiers. Mainly a structural fix; the
clash threshold from Phase 2 still gates total capacity.

### Phase 4a — Attention sinks for the adaptive path

`_prefill_write_l1_cache` only ever seeded the fixed-window cache. The adaptive tier tensors stayed zero.
Built `seed_adaptive_l1_sinks` (Python helper) that, post-`_post_compile_allocate_l1_kv`:
1. Slices DRAM K/V[0:sink_size) → `ttnn.to_torch` to host (one-time at warmup).
2. Re-creates Tier 0's K and V tensors with the sink data baked in.

Kernel now reads sink tiles unconditionally and ring tiles only within the freshness window. Output
quality improved (model recovers from limited context using sink anchoring per StreamingLLM).

### Phase 4b — L1-only inference mode

Added `l1_only_mode` config flag. When enabled, the kernel internally clamps `cur_pos` to `total_l1_tokens - 1`
so the chunk-iteration loop walks only L1-resident positions; DRAM K/V reads disappear from SDPA decode.
Plumbed through reader, writer, compute kernels (all three must clamp identically or chunk count diverges
and CB push/pop drifts).

Initial naive plumbing (cur_pos_ids list path) caused massive recompile and bad performance (3.31 tok/s).
Kernel-side clamping (preserving the cur_pos_tensor path = same program cache as hybrid) brought it back
to 6.40 tok/s; later optimizations pushed it to 9.81 tok/s in trace mode.

### Phase 5 — Disjoint-core filter

Originally Tier 2 had 27 cores including (8-10, 0-4) which overlap with the 63-core L-shape buffer.
Filtered out the L-shape footprint via `l1_kv_avoid_cores`, leaving disjoint cores at y=9 (11 cores)
and (8-10, 5-8) (12 cores). Smaller per-tier core count = fewer paged_update_cache fanout.

Coincidentally exposed Phase 4a's key win (sink-only T=1 = 9.43 tok/s) because the 32-token disjoint
config had `sink == total ⇒ ring_cap = 0 ⇒ no decode-time writes`. We initially mistook this for "L1
caching beats DRAM". §6 explains why this comparison was misleading.

### Phase 6 — Cheaper per-step write path

Three layered optimizations:
1. **Skip DRAM K/V write in `l1_only_mode`** (Python). Saves 2 `paged_update_cache` per layer × 32 = 64 dispatches per token.
2. **Alias `k_heads_l1 = k_heads_1BKD`** (skip the `ttnn.mul(_, 1.0)` clone) when DRAM write is skipped.
3. **Hoist `_build_adaptive_l1_write_pos` out of attention layer code** into `model.prepare_inputs_decode`.
   Adaptive total capacity is propagated back to `ModelArgs` after `_post_compile_allocate_l1_kv`. The
   `l1_update_pos` tensor is computed ONCE per decode step (not 32×). **This single change cut steady-state
   token cost from 103 ms to 88 ms — 15 ms saved/token.**

### Trace mode test

Re-ran with `enable_trace=True` for the batch-1 perf test. Our adaptive path is now trace-compatible
(all host syncs eliminated except one-time decode-start-pos capture). Trace amortized another ~3 ms/token
of Python dispatch overhead, landing l1_only_mode at **86.9 ms steady state**.

But baseline DRAM in trace mode is **84.2 ms steady state**. The remaining 2.7 ms is **device-side cost
the Python optimizer cannot touch**.

---

## 3. Why `runtime_pad_bytes` cannot be small — the L1 layout invariant

### 3.1 The shared-bank algo-space counter

tt-metal's `BankManager` tracks a single shared address space per `AllocatorID`. Every sharded buffer —
regardless of which cores it physically lives on — consumes `size_per_bank` from that one counter. Even when
two sharded buffers occupy disjoint core sets, they still advance the same counter.

Implication: cumulative L1 KV allocation across all tiers cannot exceed
`bank_allocatable_bytes - runtime_pad_bytes - safety_margin_bytes`.

### 3.2 The 63-core L-shape buffer

Llama-8B's QKV/embedding path allocates a sharded intermediate buffer on a 63-core L-shape (cores
`(0..10, 0..4) ∪ (0..7, 5)`). Its bottom address (`bottom_so_far`) on its cores is determined by what's
already allocated before it runs. The L-shape is allocated **during warmup**, *before*
`_post_compile_allocate_l1_kv` runs.

So our KV-tier allocations happen *after* the L-shape, layered on top in the allocator's bottom-up flow.
But the shared counter affects *future* L-shape allocations: as the counter grows, later sharded
allocations on the same cores land at higher addresses.

### 3.3 Program 49's CB region top is fixed at byte 722944

Program 49 is the SDPA decode kernel. Its statically-allocated circular buffers occupy
`[722944, L1_top]` on cores `(0..7, 0..8)` — a fixed window at the top of L1. This is determined by
the SDPA program's CB sizing (Q tile-size buffers, attention sink, mask, intermediate accumulators…),
which is itself determined by the model and is not adjustable from our adaptive-cache path.

### 3.4 The collision

When `runtime_pad_bytes` is small, the cumulative KV cache uses more algo-space. The shared counter
advances higher. **Subsequent** sharded allocations (e.g., re-allocations of the L-shape across
decode steps, or other sharded intermediates that get freed/re-allocated) end up at addresses
above 722944 on the L-shape's L1 cores → overlap with program 49's CBs → fatal assert at program
launch:

```
Statically allocated circular buffers in program 49 clash with L1 buffers on core range
[(x=0,y=0) - (x=7,y=8)]. L1 buffer allocated at 691456 and static circular buffer region ends at 722944
```

Empirically, with the current model + sink_size=32 + adaptive path, we found:
- `runtime_pad_bytes ≥ ~580 KiB`: safe
- `runtime_pad_bytes ≤ ~520 KiB`: program 49 CB clash (L-shape pushed too high)
- At even smaller pads: a different clash on program 67 / core (0,0) for a different intermediate

### 3.5 The headroom JSON IS conservative per-core. So why isn't that enough?

The JSON's `gap_bytes_free_headroom` per core is the WORST CASE — `L1_top - max(cb_region_end across all
programs measured on that core)`. It IS conservative for what it represents: the highest address that's
guaranteed not to overlap any program's CBs on that core.

The reason we can't simply set `runtime_pad_bytes = 0` and pack KV up to that per-core max is more
subtle: **the tt-metal `BankManager` advances a *shared* counter across ALL sharded buffer allocations**,
regardless of which cores each buffer occupies. Every sharded buffer's start address is essentially
`current_bank_counter`. The counter grows monotonically with every allocation.

A sharded buffer's per-core footprint is `start_addr + size_per_bank` on every core it occupies.
The constraint that must hold is

    start_addr + size_per_bank  ≤  cb_region_end(core)  for every core in the buffer's grid.

The JSON gives us `cb_region_end(core)`. What it does NOT tell us is the SEQUENCE of allocations the
model will perform at runtime — specifically, which intermediates get allocated AFTER our adaptive KV
tiers, on WHICH cores, and HOW BIG they are. We control our own KV tier sizes, but the model's own
intermediate buffers (QKV outputs, attention residuals, etc.) are out of our control.

#### The actual failure mode: shared-counter inflation, not per-core overflow

Here's how a tight `runtime_pad_bytes` causes clashes even when KV tiers are on disjoint cores:

1. Pre-decode state. Allocator counter is at some baseline B₀ (model weights, persistent caches).
2. `_post_compile_allocate_l1_kv` allocates our KV tiers. Each tier advances the counter by its size.
   Cumulative: B₀ + KV_total.
3. Decode forward begins. The model's QKV-projection produces a sharded intermediate ("L-shape", 63
   cores spanning `(0..10, 0..4) ∪ (0..7, 5)`). The allocator places it at `start = B₀ + KV_total`.
4. The L-shape's per-core address must satisfy `start + L_shape_size_per_bank ≤ cb_region_end(c)` for
   every L-shape core c. The TIGHTEST core's cb_region_end sets the ceiling.
5. The JSON tells us cb_region_end(0,0) = 722944 (Program 49's CB region floor). If
   `B₀ + KV_total + L_shape_size > 722944`, we clash.

Our KV tiers are on disjoint cores (y=9 and (8..10, 5..8)). They never PHYSICALLY occupy (0,0). But
they consume from the SHARED counter, which raises the L-shape's start address. Eventually
KV_total grows enough that L-shape's address exceeds (0,0)'s headroom.

#### So why "can't we just trust JSON 100%"?

We DO trust the JSON's per-core cb_region_end. The reason we can't pack to that ceiling is:

- The JSON tells us per-core constraints.
- The allocator forces all sharded buffers to share a single counter (so all start addresses are aligned to that counter).
- We can predict our own KV tier sizes but not the model's intermediates (sizes, cores, sequence of allocation).
- `runtime_pad_bytes` is the empirical slack in the shared counter that leaves room for those intermediates to land below their cores' cb_region_end.

In other words: the JSON is the per-core ceiling; `runtime_pad_bytes` is the per-step slack that
ensures whatever the model allocates after our KV tiers still fits under that ceiling.

### 3.6 Effective KV capacity given this constraint

- bank_allocatable = 1_470_080 B
- safety_margin = 64 KiB = 65_536 B
- runtime_pad = 580 KiB = 593_920 B  (empirically the floor before clashes appear)
- algo_budget = 810_624 B ≈ 2.9 tile-rows of K+V across 32 layers
- → max 2 tile-rows fit (third is dropped or shrunk)

For our 27-core overlap tier: 2 tile-rows × 27 cores × tile_height = 192 tokens.
For our 11-core disjoint tier: 2 tile-rows × 11 cores × tile_height / GQA-step = 64 tokens.

**That's our ceiling on adaptive L1 KV cache size given the current model + allocator constraints.**

### 3.7 What it would take to break the constraint

1. **Per-bank-isolated allocations**. The allocator would need to track per-bank counters
   *independently* (no shared bank counter), so allocations on disjoint cores don't push each other's
   start addresses up. Requires a tt-metal allocator rewrite or a separate sub-device for KV tiers.
2. **Predict every model intermediate's allocation**. Augment the JSON with per-intermediate
   information (size, cores, allocation order). Then size `runtime_pad_bytes` exactly. Possible but
   model-specific and brittle to model changes.
3. **Move the L-shape (and any other tight-headroom-core intermediates) to a different core grid**
   that doesn't overlap any SDPA-decode core. Model-side change to the QKV/embedding path.
4. **Shrink program 49's CB region** so cb_region_end is higher (i.e., more headroom on all SDPA
   decode cores). Model-side change to SDPA decode kernel internals. Subtle and risky.

None of these is a quick win. The current `runtime_pad_bytes ≈ 580 KiB` is the practical floor without
substantial model or allocator surgery.

---

## 4. Why L1-only is still slower than DRAM baseline — the actual bottleneck

After all the host-sync removals, trace mode, disjoint cores, and cur_pos clamping:
- L1-only T=2 disjoint + trace: **86.9 ms steady state**
- DRAM baseline + trace: **84.2 ms steady state**
- Gap: **2.7 ms/token**

The intuition "L1 SRAM is faster than DRAM" would predict the opposite. Why doesn't it apply?

### 4.1 Side-by-side device-side timings (trace mode, steady state)

From `TT_L1_KV_PERF=1` instrumentation:

| Op | DRAM baseline | L1-only T=2 | Delta |
|---|---:|---:|---:|
| `paged_update_cache` write block (per layer, both K+V) | 0.208 ms | 0.203 ms | **≈ identical** |
| `sdpa_call` (per layer) | **0.404 ms** | **0.598 ms** | **+0.194 ms** |
| `model_forward` (per token) | 149.9 ms | 159.4 ms | +9.5 ms |

**The L1 writes are NOT slower** — they are essentially identical to DRAM writes (and even
marginally faster on average, by 5 µs).

**The L1 reads inside SDPA decode are 48% slower than DRAM reads.** Across 32 layers,
that's 0.194 × 32 = **+6.2 ms per token**, which fully accounts for the observed gap.

### 4.2 Why are L1 reads slower than DRAM reads here?

Naive intuition: L1 SRAM has lower latency than DRAM. *Locally accessed* L1 (same-core reads) is indeed ~10× faster. But our L1 KV cache is **HEIGHT_SHARDED across 11 source cores**, and our SDPA decode runs on **64 worker cores**. The access pattern is:

```
SDPA decode cores (64)
        │
        │ NoC reads of K/V tile T
        ▼
   ┌────────────┐
   │  Source    │  ◄── tile T lives on exactly one core
   │  L1 core   │  ◄── this core's NoC port serializes 64 concurrent reads
   └────────────┘
```

vs. DRAM:

```
SDPA decode cores (64)
        │
        │ NoC reads of K/V tile T
        ▼
   ┌────────────┐
   │ DRAM bank  │  ◄── tile T's bank
   │ controller │  ◄── multi-port memory controller; parallel reads
   └────────────┘   ◄── interleaved bank striping (consecutive tiles → diff banks)
```

DRAM channels are **multi-port** and the cache is **interleaved across 12 banks**. A broadcast-read of
the same tile gets handled in parallel hardware. A sharded-L1 source core's NoC port handles requests
serially.

The user's intuition is correct on **bandwidth** (NoC link aggregate > DRAM channel aggregate), but
this access pattern is **broadcast** (1 source, many readers), where DRAM's parallel-read infrastructure
wins.

### 4.3 What is *not* the bottleneck (rule-out evidence)

- **Python dispatch overhead** — Eliminated by trace mode. Trace amortized ~3 ms/token of Python cost,
  but the L1-only vs DRAM gap remained at 2.7 ms. The bottleneck is below the Python layer.
- **K+V write fusion** — The `paged_update_cache` timing data shows L1 writes are NOT slower than DRAM
  writes. Fusing K+V into one op would save Python overhead (already amortized by trace) and a bit of
  device-side scheduling, but not the +6.2 ms in SDPA decode reads.
- **Disjoint vs overlap cores** — Disjoint (11 cores) vs overlap (27 cores) tested at the same
  cache size: ~89 vs ~90 ms steady state. Tier core count is a minor factor; the source-core NoC
  port contention dominates.
- **Cache size** — Tested T=1 vs T=2 disjoint with sink_size=0 (ring active in both): 102 ms vs 105 ms.
  Bigger cache adds a small amount of work but doesn't dramatically slow things down. Hit-ratio gains
  from a bigger cache barely register because L1-read-per-tile is intrinsically slower than DRAM-read-per-tile.

### 4.4 What would eliminate the gap

1. **Head-sharded L1 cache.** Shard K/V by KV-head across the cores that already host `k_heads_1BKD`.
   Each SDPA core processing head H reads K/V[H] from the core holding H. Reduces concurrent readers
   per source from 64 to ~8. Expected: 1-3 ms/token saved, ~70% confidence to beat baseline by a tight
   margin. Implementation: 1-3 days, requires changes to allocator, paged_update_cache call path, and
   SDPA decode read formula.

2. **Replicated L1 cache.** Each SDPA core has its own copy of K/V. Reads become truly local SRAM
   (sub-ns latency). Memory cost: 8× the cache size, but with our small caches this still fits.
   Expected: 5+ ms/token saved, ~85% confidence to beat baseline meaningfully. Implementation: similar
   effort to head-sharded.

3. **Use L1 cache only as overflow for very long sequences.** For batch=1 with seq ≤ 1024, DRAM is
   already efficient; L1 helps mainly when DRAM bandwidth becomes saturated (large batch, long seq).
   For this latency benchmark, DRAM may simply be the right choice.

---

## 5. Current best configuration

```
runtime_pad_bytes = 768 KiB    # tested floor for disjoint-cores adaptive
l1_kv_avoid_cores = L-shape footprint  # (0..10, 0..4) ∪ (0..7, 5)
l1_kv_sink_size = 32 tokens     # 1 tile, StreamingLLM default
l1_kv_only_mode = True          # skip DRAM reads + DRAM writes
enable_trace = True             # batch-1 perf test default is False — we manually enabled

# Result: T=2 disjoint, 11 cores × 2 tile-rows = 64 tokens (32 sink + 32 ring)
# Steady state: 86.9 ms / 128th token, ~11.5 tok/s
# Avg: 102 ms (dragged by ~3 s compile of unique programs on 1st decode step)
# DRAM baseline (trace, no adaptive): 84.2 ms / 128th token, 11.73 tok/s
```

---

## 6. The misleading "9.43 tok/s baseline"

For most of this session we worked against an apparent "best adaptive" of 9.43 tok/s
(`runtime_pad=1024 KiB`, T=1 sink-only). It LOOKED like L1 caching working. It was an illusion.

**`T=1 sink-only` config has `ring_cap = total_capacity - sink_size = 0`. The Python `_build_adaptive_l1_write_pos`
returns None, the model skips all decode-time L1 writes, and the kernel reads only 32 sink positions.**
That config is effectively *DRAM-only* (no ring caching) with a tiny sink-read overhead. It doesn't
exercise the adaptive scheme at all.

The honest comparison requires `ring_cap > 0`. Once we have real ring writes (T=2 sink+ring, T=3
sink+bigger ring, etc.), per-token cost stabilizes around 86-90 ms — bounded by the L1 sharded read
latency, regardless of cache size.

---

## 7. Pointer index

| Topic | Reference |
|---|---|
| L1 layout / sharded vs interleaved | `l1_layout_restriction.md` |
| Per-core validate framework | `per_core_validate_walkthrough.md`, `per_core_validate_patch_design.md` |
| Adaptive tier architecture | `l1_kv_cache_architecture.md` |
| CB usage analysis | `sdpa_cb_sram_utilization_analysis.md`, `l1_cb_map_analysis.md` |
| SDPA decode core grid | `blackhole_p150_sdpa_decode_cores.md`, `program_grid_assignment_analysis.md` |
| Decode SRAM tradeoff | `l1_kv_decode_sram_tradeoff.md` |
| Historical session notes | `adaptive_l1_kv_headroom_session_summary.md`, `challenges.md` |

Code touched in this session:
- `models/tt_transformers/tt/attention.py` — adaptive write path, sink seed, disjoint-core filter,
  L1-only flag plumbing, `_build_adaptive_l1_memcfg_tiers` shrink-all rewrite.
- `models/tt_transformers/tt/model.py::prepare_inputs_decode` — adaptive flat_pos hoist.
- `models/tt_transformers/tt/generator.py` — `_post_compile_allocate_l1_kv` propagates
  `l1_kv_adaptive_total_capacity` to ModelArgs.
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/`:
  - `device/kernels/dataflow/dataflow_common.hpp` — ring-aware n-tier read function,
    sink-tile handling, decode_start_pos clamp.
  - `device/kernels/dataflow/reader_decode_all.cpp`, `writer_decode_all.cpp` — l1_only_mode
    runtime arg + clamp.
  - `device/kernels/compute/sdpa_flash_decode.cpp` — l1_only_mode clamp (mirror of reader).
  - `device/sdpa_decode_program_factory.{cpp,hpp}` — runtime arg plumbing.
  - `device/sdpa_decode_op.{cpp,hpp}`, `sdpa_decode.{cpp,hpp}`, `sdpa_decode_pybind.cpp` — public API.
- `tt_metal/impl/allocator/bank_manager.cpp` — per-core query diagnostic (gated by `TT_METAL_LOG_L1_KV_DIAG`).

---

## 8. Summary

The adaptive L1 KV cache scheme works **correctly** and is **competitive** with DRAM, but at this
hardware configuration it does not beat the DRAM baseline. The fundamental obstacles are:

1. **L1 SRAM is fast locally but our adaptive cache is sharded across remote cores**. Cross-NoC
   reads from a single source core serialize at the source's NoC port, while DRAM channels handle
   broadcast reads in parallel.

2. **`runtime_pad_bytes` cannot be small** without colliding with model-intermediate buffers and
   SDPA decode's CB region. The L1 KV cache is capped at roughly 2 tile-rows of cumulative
   algo-space (~544 KiB shared bank counter), which translates to 64-192 tokens depending on core
   layout — far below the ~5 MiB of L1 we might naïvely think is "available".

3. **L1 writes are competitive** with DRAM writes (per-call cost is nearly identical), so write
   optimization is not the lever to pull. The lever is **reducing concurrent readers per source
   core**, which requires either head-sharded L1, replicated L1, or accepting the architecture's
   intended workload pattern (DRAM for broadcast reads, L1 for partitioned reads).

The remaining ~2.7 ms gap to baseline could potentially be closed by **head-sharded** or **replicated**
L1 cache layouts (§4.4), but those are 1-3 day projects and carry implementation risk.

# L1 KV Cache Modes — Comprehensive Reference

> The single source of truth for the L1 KV cache feature in `tt_transformers` after the
> mode cleanup. Replaces the legacy fixed-window / `use_adaptive_l1_kv_cache` /
> `l1_kv_interleaved_adaptive` flag sprawl with one `l1_kv_mode` selector.
> Llama-8B, Blackhole P150, batch-1 (`simple_text_demo.py`). Last updated: 2026-05-30.

---

## 1. The four modes (quick reference)

`--l1_kv_mode` selects the KV cache layout; `--l1_kv_only_mode` is an orthogonal add-on.

| mode | KV storage | sizing | DRAM in decode | use case |
|---|---|---|---|---|
| `dram` (default) | DRAM only | — | read + write | baseline |
| `interleaved` | 1 interleaved L1 buffer (all banks) | `--l1_kv_window_size` (+ sink) | hybrid: read DRAM unless `l1_only` | highest capacity, fastest L1 reads |
| `sharded` | N HEIGHT_SHARDED tiers (low-CB cores) | auto from `--l1_kv_headroom_json` | hybrid unless `l1_only` | per-core locality experiments |
| `hybrid` | sharded tiers + interleaved tier | window + headroom | — | **not yet implemented** |

`--l1_kv_only_mode` (StreamingLLM): SDPA decode attends only to L1 (clamped `cur_pos`),
and the decode-time DRAM writes are skipped. Applies to interleaved/sharded/hybrid.

### Measured (Llama-8B BH P150, batch-1, l1_only)

| mode | L1 KV capacity (clash-free) | steady tok/s | output vs DRAM baseline |
|---|---:|---:|---|
| dram | — | 11.7 | (reference) |
| interleaved | 416 → 896+ | 11.6 → 10.9 | **byte-identical** |
| sharded | 256 (≈320 max) | 7.1 | coherent; lossy past wrap (cap < seq) |

Interleaved wins on both capacity and throughput: it is uniform across banks (coexists
with the model's own interleaved buffers in the thin top band), and its reads stripe
across all banks (no sharded source-core serialization). `l1_only` additionally shrinks
the SDPA CB footprint (no DRAM-read CBs), freeing L1 for more KV.

---

## 2. Flags

| flag | applies to | meaning |
|---|---|---|
| `--l1_kv_mode {dram,interleaved,sharded,hybrid}` | all | layout selector (default `dram`) |
| `--l1_kv_only_mode` | non-dram | StreamingLLM L1-only decode (skip DRAM reads/writes) |
| `--l1_kv_window_size N` | interleaved, hybrid | cache capacity in tokens (capacity = N + sink) |
| `--l1_kv_sink_size N` | non-dram | pinned attention-sink tokens (default 32 under non-dram) |
| `--l1_kv_headroom_json PATH` | sharded, hybrid | offline per-core headroom map for sharded sizing |
| `--l1_kv_safety_margin`, `--l1_kv_min_viable_tokens`, `--l1_kv_min_expected_hit_ratio` | sharded | sharded sizing knobs |

Removed in the cleanup: `--use_adaptive_l1_kv_cache`, `--l1_kv_interleaved_adaptive`,
`--l1_kv_use_sharded` (all subsumed by `--l1_kv_mode`).

Internally, `ModelArgs.use_adaptive_l1_kv_cache` is now a derived flag
(`= l1_kv_mode in {interleaved,sharded,hybrid}`) that gates the post-compile allocation
hook and the forward-path L1 routing — it is not user-facing.

---

## 3. Shared architecture (all non-dram modes)

All non-dram modes funnel through one code path; only the tier *layout* differs.

### 3.1 Tier model
`attention.l1_kv_tiers` is a list of `(k_tensor, v_tensor, token_start, tok_count)`.
- `interleaved`: exactly one tier, `L1_MEMORY_CONFIG` (interleaved across all banks),
  covering `[0, capacity)`.
- `sharded`: N tiers, each `HEIGHT_SHARDED` on a distinct core class, covering
  contiguous token ranges `[token_start_i, token_start_i + tok_count_i)`.

### 3.2 Post-compile allocation (timing)
The L1 KV cache is allocated **after** the decode programs compile (CB regions frozen),
not at `__init__`. The generator (`generator.py::_post_compile_allocate_l1_kv`) runs a
warmup decode step, then calls `attention.allocate_l1_kv_cache(headroom_map)` →
`_build_adaptive_l1_memcfg_tiers` (mode branch) → allocates tier tensors. This guarantees
the cache lands above every program's CB top, avoiding the CB-clash that init-time
allocation could trigger. There is no init-time L1 KV allocation anymore.

### 3.3 Write path — `paged_update_cache` (layout-agnostic)
`ttnn.experimental.paged_update_cache` writes into whatever L1 tensor it is handed and
derives bank/shard addressing from that tensor's own buffer. The same op writes the
interleaved tier, the sharded tiers, and the DRAM cache — it does not special-case layout.
That is why swapping a tier's `MemoryConfig` between sharded and interleaved is free.

### 3.4 Read path — n-tier SDPA reader + `TensorAccessor`
The SDPA decode op takes `l1_k_tensors`/`l1_v_tensors` + `l1_tier_token_starts`/`_counts`.
The program factory builds each tier's reader via
`TensorAccessorArgs(*l1_k_tiers[i]->buffer())` — **from the tensor's actual layout** — so
`reader_decode_all.cpp` reads interleaved or sharded tiers transparently. The tile-id
formula (`head_base * tier_size_tiles * DHt + local_row * DHt + col`) is the logical flat
tile index either way; the accessor maps it to banks.

### 3.5 Ring + attention sink
Decode writes at flat slot `sink + (pos - sink) % ring_cap` (`ring_cap = capacity - sink`).
With `capacity >= sequence` there is no wrap and the slot equals `pos` (linear cache).
The first `sink` tokens are pinned (the ring write skips them) so the model keeps its
StreamingLLM anchor tokens even after wrap.

### 3.6 l1_only mechanics (and the correctness fixes)
With `--l1_kv_only_mode`, SDPA attends only to L1: the kernel clamps `cur_pos` to
`capacity-1`, `decode_start_pos` is forced to 0, and decode-time DRAM writes are skipped.
Two fixes make this byte-faithful (commits d0d6445, 28e02bc):
1. **Boundary tile** (`dataflow_common.hpp`): the fresh-window upper bound uses **ceil**,
   so the partial tile containing `cur_pos` reads from L1 (its ≤cur_pos positions are
   ring-written; >cur_pos are causally masked) instead of the stale DRAM fallback.
2. **Full-prefill seed** (`seed_adaptive_l1_sinks`): seeds the entire prefill `[0, prompt)`
   from DRAM into L1 (per-tier: each tier from `DRAM[token_start_i : +tok_count_i]`), so
   `decode_start_pos=0` is valid — the whole prompt is L1-resident.
Result (capacity ≥ sequence): l1_only output is byte-identical to the DRAM baseline with
zero DRAM reads/writes in decode.

---

## 4. Mode deep-dives

### 4.1 `dram` (baseline)
No L1 KV cache. KV lives in DRAM (`layer_past`), written by `paged_update_cache` every
decode step and read by SDPA decode. `l1_kv_tiers` is empty; the forward L1 branches are
inert. This is the correctness/throughput reference (~11.7 tok/s).

### 4.2 `interleaved`
**Allocation:** one `L1_MEMORY_CONFIG` tier of `l1_kv_total_size` (`window + sink`) tokens,
rounded up to a tile, `allocator_id=0`. Spans all 110 banks; per-bank footprint is
`total / 110` — thin and uniform.

**Why it reaches high capacity:** uniform per-bank occupancy coexists with the model's own
interleaved L1 buffers in the same thin band above the CB region. There are no per-core
sharded tiles to fit, so the dense (tall-CB) cores are used at the same thin rate as the
low-CB cores. Capacity is bounded by the tightest core's gap; with `l1_only` the SDPA CB
region shrinks, raising the ceiling further (clash-free past 896 tokens here).

**Throughput:** interleaved L1 reads stripe across banks (DRAM-like parallelism), so it
matches/beats the DRAM baseline — no sharded source-core NoC serialization.

**Sizing:** manual via `--l1_kv_window_size` (it does **not** read the headroom map). This
is intentional: a uniform buffer has no per-core tiles to auto-fit, and the vanilla
headroom map under-predicts the l1_only capacity ~2×.

**When to use:** the recommended high-capacity / high-throughput L1 KV mode.

### 4.3 `sharded`
**Allocation:** N `HEIGHT_SHARDED` tiers, one per "headroom class" of cores, auto-sized by
`_build_adaptive_l1_memcfg_tiers` from `--l1_kv_headroom_json`:
- *Per-core gate*: each core's tile-rows = `floor((gap - safety) / bytes_per_tile_row)`,
  where `bytes_per_tile_row = per_tile_bytes · DHt · num_layers · 2`. Dense (tall-CB) cores
  with a small gap get 0 rows and are excluded.
- *Per-tier allocators*: each tier gets its own L1 allocator id (`1..K`, provisioned in
  `l1_banking_allocator.cpp`), so tiers on disjoint cores don't push each other below their
  CB floor. Core-aware dependency subtraction in `bank_manager.cpp` keeps cross-tier and
  tier-vs-model placement correct.

**Capacity:** ~256 (default) up to ~320 clash-free. Lower than interleaved because the
coarse per-core granularity (one tile-row carries all 32 layers ≈ 278 KB/core) can't use
the 64 dense cores at all, and the model's interleaved buffers cap how much sharded KV fits.

**Throughput:** lower (~7 tok/s at 256) — wide tiers add `paged_update_cache` fanout and
each KV tile reads from a single source core whose NoC port serializes the 64 SDPA readers
(perf-doc "Problem 2").

**l1_only faithfulness:** the cross-tier full-prefill seed makes it faithful when
capacity ≥ sequence; below that the ring wraps and drops earliest context (StreamingLLM).

**When to use:** per-core locality / Problem-2 experiments; not the throughput or capacity
winner.

### 4.4 `hybrid` (planned, not implemented)
Sharded tiers on the low-CB cores **plus** an interleaved tier filling the dense cores'
thin band — to stack capacity beyond interleaved-alone. `_build_adaptive_l1_memcfg_tiers`
raises `NotImplementedError` for now. The allocation feasibility was validated separately
(interleaved 320 + sharded 256 = 576 tokens clash-free); the remaining work is the
read-path token-range partitioning across the mixed tiers. See
`interleaved_adaptive_walkthrough.md` §1 and the "hybrid" option in
`perf_walkthrough_l1_vs_dram.md`.

---

## 5. How to run each mode

```bash
source python_env/bin/activate ; export TT_METAL_HOME=$PWD
HM=research_codes/documents/l1_kv_cache_cache/comparison/headroom_map_global_minimum.json
B='pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"'

# 1. DRAM baseline
TT_LOGGER_LEVEL=Info $B

# 2. Interleaved (+ l1_only). Capacity = window + sink. Headroom JSON not needed.
TT_LOGGER_LEVEL=Info $B --l1_kv_mode interleaved --l1_kv_window_size 768 --l1_kv_only_mode

# 4. Sharded (+ l1_only). Auto-sized from headroom JSON; no window needed.
TT_LOGGER_LEVEL=Info $B --l1_kv_mode sharded --l1_kv_headroom_json $HM --l1_kv_only_mode

# 3. Hybrid — raises NotImplementedError (placeholder).
```

Drop `--l1_kv_only_mode` for the hybrid-DRAM variant (keeps DRAM reads; prefill served
from DRAM, decode ring from L1). Inspect a run: `grep "\[User 0\]" <log> | tail -1`
(generated text), `grep -oE "Iteration 199:.*tok/s" <log> | tail -1`, `grep -c clash <log>`.

Standalone correctness probe (interleaved tier vs DRAM, decode range):
`HF_MODEL=... python tests/test_l1_kv_cache_model_path.py` (PCC should be 1.0).

---

## 6. Code map

| concern | location |
|---|---|
| mode selector / derived flag | `model_config.py::ModelArgs`, `attention.py::__init__` |
| tier build (mode branch) | `attention.py::_build_adaptive_l1_memcfg_tiers` |
| tier allocation + seed | `attention.py::allocate_l1_kv_cache`, `_allocate_adaptive_l1_kv_tiers`, `seed_adaptive_l1_sinks` |
| post-compile hook | `generator.py::_post_compile_allocate_l1_kv` |
| ring write position | `attention.py::_build_adaptive_l1_write_pos`, `_write_adaptive_l1_tiers` |
| forward write/read | `attention.py::forward_decode` (l1_kv_tiers branches) |
| SDPA n-tier reader | `sdpa_decode/device/kernels/dataflow/dataflow_common.hpp::read_kv_mask_chunks_n_tier`, `reader_decode_all.cpp` |
| per-tier L1 allocators | `tt_metal/impl/allocator/l1_banking_allocator.cpp`, `bank_manager.cpp` (core-aware subtraction) |

---

## 7. Related docs
- `interleaved_adaptive_walkthrough.md` — interleaved implementation + the l1_only fixes (§7).
- `perf_walkthrough_l1_vs_dram.md` — original sharded analysis, Problem 1 / Problem 2.
- `comparison/headroom_map_global_minimum.json` — offline headroom map for sharded sizing.

# Code Audit — L1 KV Cache + Raw Memory Access Bandwidth Profiling

Purpose: a manual-audit map of every code change this work introduced, with file, region,
role, and the focus point to verify. Two sections: (1) L1 KV cache (the feature), (2) raw
memory access bandwidth profiling (the measurement tooling).

## How to read / verify this doc
- The `l1-kv-cache` branch sits ~1868 commits past its merge-base with `main`, so a raw
  `git diff main...HEAD` is meaningless (it includes thousands of unrelated upstream commits).
  This audit instead maps the feature by its **code markers** in the current working tree.
- Authoritative change set = the 104 commits by author `morjun`
  (`git log --author=morjun --oneline`). The feature also once contained a **weight-sharding**
  path that was **removed** (commit `c62f6fb` "Remove weight caching feature") — not audited here.
- Line numbers are current as of this audit; if they drift, grep the **marker string** given
  with each entry (line numbers move, markers do not).
- **Not part of this feature (do not audit as ours):** `ttnn/.../transformer/sdpa_windowed/*`
  is an upstream Qwen2.5-VL op (authored by TT, used in `models/demos/qwen25_vl/`), unrelated
  to L1 KV. It only appears in the branch diff because it landed upstream after the old merge-base.

## Verify-by-grep cheat sheet
```bash
# Python feature surface
grep -rn "l1_kv\|l1_only\|_compute_l1_ring_pos_device\|_write_adaptive_l1_tiers" models/tt_transformers/tt/
# Kernel/op feature surface
grep -rn "l1_only_mode\|num_l1_tiers\|ring_wrapped\|find_tier\|l1_decode_start_pos" \
  ttnn/cpp/ttnn/operations/transformer/sdpa_decode/
# Profiling scaffolding state (should be OFF = commented)
grep -rn "SDPA_PROFILE_ZONES" ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/
```

---

# Section 1 — L1 KV Cache

## 1.0 What the feature does (data flow, for orientation)
A decode-time path that keeps the K/V cache resident in on-chip L1 instead of DRAM, as an
adaptive **N-tier** (≤5) ring buffer with an optional pinned **attention sink** and a
StreamingLLM-style **l1_only** mode. Lifecycle per run:
1. **Config** (`model_config.py`, `conftest.py`, demo): `--l1_kv_mode` etc. select the layout.
2. **Post-compile allocation** (`generator.py` → `attention.py`): after the decode graph
   compiles, probe live per-core L1 headroom, allocate per-core KV tiers, and **seed** the
   sink/prefill from DRAM.
3. **Per-step ring write** (`model.py` + `attention.py`): compute the flat L1 ring write index
   (on-device, trace-safe) and write K/V into the tiers; skip the DRAM write in l1_only mode.
4. **SDPA read** (`sdpa_decode` op + reader kernel): the reader reads K/V from the L1 tiers
   (ring-mapped) instead of DRAM, clamping `cur_pos` in l1_only mode.

## 1.1 Python orchestration layer

### `models/tt_transformers/tt/model.py` (class Transformer)
| Region | Marker / function | Role | Audit focus |
|---|---|---|---|
| L266 | `prepare_decode_inputs_host()` | Host input prep; now does **no** ring modulo (delegated on-device) | Confirm no host `torch.remainder`/`from_torch`/H2D for ring pos; hit-ratio block opt-in only |
| L458 | `_compute_l1_ring_pos_device()` | Trace-safe on-device ring write index | No-wrap fast path + ring formula (below) |
| L510 | `ttnn_decode_forward()` | Calls ring-pos compute when host left it `None`; tracks `built_ring_pos` for dealloc | Tensor lifecycle: only dealloc a **newly built** tensor, never `current_pos` |

Focus snippets (verified):
```python
# model.py:479-480  — no-wrap fast path: whole context fits, ring never wraps -> reuse current_pos (0 ops)
if T >= self.args.max_seq_len:
    return current_pos
# model.py:488-496  — sink + ring index:  wrapped = (pos<sink) ? pos : sink + ((pos-sink) % ring)
ring = T - sink
...
ringed = ttnn.remainder(shifted, float(ring))   # L490
```
Audit point: the device formula here **must match** the kernel's `find_tier` ring map
(§1.2, `dataflow_common.hpp`) and the writer's per-tier offset math (`_write_adaptive_l1_tiers`).

### `models/tt_transformers/tt/attention.py` (class Attention) — the core of the feature
| Region | Marker / function | Role | Audit focus |
|---|---|---|---|
| ~L92-172 | `__init__` L1 block | Reads config, sets `use_adaptive_l1_kv_cache`, default sink=32 for adaptive, `l1_kv_avoid_cores` | Default logic; sink+window invariant assert |
| L495 | `_allocate_adaptive_l1_kv_tiers()` | Allocate per-core HEIGHT_SHARDED K/V tiers; set `l1_kv_adaptive_total_capacity` | OOM skip-tier handling; idempotent guard (`if self.l1_kv_tiers: return`) |
| L586 | `seed_adaptive_l1_sinks()` | Seed sink (hybrid) or full prefill (l1_only) into tiers from DRAM | **Host sync** via `ttnn.to_torch()` (one-time, acceptable); full-vs-sink branch on `l1_only_mode` |
| L709 | `_build_adaptive_l1_memcfg_tiers()` | Per-core mem-config; two budget gates (per-core headroom + cumulative allocator space) | Budget formula; interleaved = single tier; greedy packing order |
| L1146 | `_write_adaptive_l1_tiers()` | Ring writes to N tiers, no host sync; `-1` sentinel for out-of-range tiles | Single-tier fast path (`token_start==0`); sentinel = `0xFFFFFFFF` in kernel |
| L1236 | `forward_decode()` L1 block | Clone/alias K/V for L1 write, gate DRAM write, build SDPA L1 kwargs | Aliasing vs double-free; `skip_dram_kv_write` gate; l1_only SDPA flags |

Focus snippets (verified):
```python
# attention.py:1365-1366  — l1_only skips the DRAM KV write entirely
skip_dram_kv_write = self.l1_kv_only_mode and self.l1_kv_tiers and not page_table and l1_write_enabled
if not skip_dram_kv_write:
    ttnn.experimental.paged_update_cache(keys, k_heads_1BKD, ...)   # DRAM write (baseline path)
```
Audit point: in l1_only mode the same `paged_update_cache` is still issued to the **L1 tier**
(`_write_adaptive_l1_tiers`), so "skip DRAM write" is not a free win — verify the L1 write cost.

### `models/tt_transformers/tt/generator.py` (class Generator)
| Region | Marker / function | Role | Audit focus |
|---|---|---|---|
| ~L93-106 | `__init__` L1 fields | `l1_kv_needs_alloc`, `_l1_kv_warmup_done`, safety margin, min-viable-tokens | Activation detection covers both legacy window and adaptive |
| ~L504-575 | warmup/alloc integration in decode | Run **one untimed warmup decode** to compile L1 path before timed loop / trace capture | Warmup-vs-JSON branch; this is the fix that made trace capture work |
| L636 | `_post_compile_allocate_l1_kv()` | Orchestrates: probe headroom → per-layer alloc → seed → propagate capacity | Headroom source (JSON vs live); capacity propagated to `ModelArgs` for host pre-compute |

Audit point: trace-safety hinges on the warmup forward happening **after** tier alloc/seed and
**before** `begin_trace_capture`; without it the L1 ops are uncompiled at capture and it crashes.

### Config + entry points
| File | Region | Role | Audit focus |
|---|---|---|---|
| `model_config.py` | `ModelArgs.__init__` ~L466-486 | 8 L1 flags + `use_adaptive_l1_kv_cache = l1_kv_mode in (interleaved, sharded, hybrid)` | Defaults keep `dram` baseline unchanged |
| `common.py` | model-creation wrapper ~L685-711 | Thread 8 flags into `ModelArgs` | Pass-through completeness |
| `demo/conftest.py` | CLI args ~L80-157 | `--l1_kv_*` pytest options | Names/defaults match `model_config.py` |
| `demo/simple_text_demo.py` | create wrapper ~L339-383; warmup ~L1189-1214 | Pass flags; `DECODE_WARMUP_ITERS` env warmup | `enable_trace` restored to True; warmup gated on env |

Config flags (defaults): `l1_kv_mode=dram`, `l1_kv_window_size=0`, `l1_kv_sink_size=0`,
`l1_kv_min_expected_hit_ratio=0.0`, `l1_kv_safety_margin=65536`, `l1_kv_min_viable_tokens=64`,
`l1_kv_only_mode=False`, `l1_kv_headroom_json=None`.

## 1.2 Device op + kernels (`ttnn/.../transformer/sdpa_decode/`)

### Op API surface (new params, default to the DRAM baseline)
| File | Region | New params | Audit focus |
|---|---|---|---|
| `sdpa_decode.hpp` | struct, ~L29-40 | `l1_sink_size`, `l1_min_expected_hit_ratio`, `l1_k_tensor/v_tensor` (legacy tier-0), `l1_k_tensors/v_tensors` (N-tier), `l1_tier_token_starts/counts`, `l1_decode_start_pos`, `l1_only_mode` | All default to empty/0/false → stock call unchanged |
| `sdpa_decode.cpp` | invoke, ~L83-112 | Packs tier-0 K/V at optional-input [4,5], extra tiers at [6,7],[8,9]… | Index chaining correctness |
| `sdpa_decode_pybind.cpp` | ~L71-80 | Exposes all params to Python | Backward-compatible defaults |
| `sdpa_decode_op.cpp` | `create_program` ~L345-399; `compute_program_hash` ~L407-426 | Extract tiers from optional inputs; **hash includes L1 params** | Cache invalidation when tier count / sink / ratio change |
| `sdpa_decode_program_factory.cpp` | compile args ~L869-882; runtime args ~L1034-1155; override ~L1185-1369 | `MAX_L1_TIERS=5`; per-tier `(k_addr,v_addr,start_tile,size_tiles)`; trailing `(decode_start_pos, sink_tile_count, l1_only_mode)` | Arg ordering must match the reader's parse order |

### Reader kernel — `kernels/dataflow/reader_decode_all.cpp`
| Region | Marker | Role | Audit focus |
|---|---|---|---|
| L61-62 | `num_l1_tiers`, `use_l1_kv_cache` | Compile-time tier count + accessor chaining | Offset chaining threads 5 tier-pairs even when unused |
| L90-130 | tier runtime-arg parse | Reads per-tier addr/geometry + `l1_decode_start_pos_arg` (L127), `l1_sink_tile_count_arg` (L130) | Parse order must mirror program factory |
| ~L170-185 | l1_only `cur_pos` clamp | `cur_pos = min(cur_pos, total_l1_tokens-1)` | Must match writer + compute clamps exactly |
| ~L447-709 | `num_l1_tiers==N` dispatch | 5 compile-time branches calling `read_kv_mask_chunks_n_tier<…,N>` | Each branch passes correct readers; placeholders for unused |

### Reader core logic — `kernels/dataflow/dataflow_common.hpp`
This is the **correctness heart** of the L1 read path. `read_kv_mask_chunks_n_tier()` (~L669).
| Region | Marker | Role | Audit focus |
|---|---|---|---|
| ~L725-770 | ring window calc | Compute `fresh_lo/hi_tile`, `ring_wrapped` | Sink/ring partition; ceil on boundary tile; clamp to `decode_start_pos_tile` |
| L744 | `ring_wrapped` | `!(cur_pos+1 <= sink_tokens + ring_tokens)` | The no-wrap fast-path predicate (mirrors model.py:479) |
| ~L771-779 | hit-ratio gate | `enable_l1_reads` if expected hit ratio ≥ threshold | Gating arithmetic |
| ~L808-856 | `find_tier` lambda | Map global seq tile → (tier, local tile), with ring modulo | **Ring map** (below); tier range checks |
| ~L858-980 | K/V read loops | Per-tile tier dispatch; DRAM fallback on `0xFFFFFFFF` | L1 tile-id math `head_base*size*DHt + local_row*DHt + col` |

Focus snippet (verified — the ring map and its no-wrap fast path):
```cpp
// dataflow_common.hpp (find_tier): sink is direct-mapped; ring wraps within the ring region only
if (gst < sink_tile_count) {
    flat_tile = gst;                                   // sink: identity
} else if (gst >= fresh_lo_tile && gst < fresh_hi_tile && ring_tile_count > 0) {
    if (!ring_wrapped) {
        flat_tile = gst;                               // no-wrap fast path: skip the SW-divide modulo
    } else {
        flat_tile = sink_tile_count + ((gst - sink_tile_count) % ring_tile_count);  // ring wrap
    }
} else {
    return 0xFFFFFFFFu;                                // not in L1 -> DRAM fallback
}
```

### Writer + compute clamps (must be identical to reader)
| File | Region | Role |
|---|---|---|
| `kernels/dataflow/writer_decode_all.cpp` | ~L84-89 | l1_only `cur_pos` clamp (mirror) |
| `kernels/compute/sdpa_flash_decode.cpp` | ~L161-164 | l1_only `cur_pos` clamp (mirror) |

Audit point: all three kernels (reader/writer/compute) clamp `cur_pos` to
`total_l1_tokens-1` independently; if they disagree, chunk iteration / CB push-pop desync.

## 1.3 Correctness-critical checkpoints (the short list to verify)
1. **Ring formula agreement** across 3 sites: `model.py:488-496` (device index), writer offset
   math in `_write_adaptive_l1_tiers` (attention.py), and `find_tier` in `dataflow_common.hpp`.
2. **No-wrap fast path** agreement: `model.py:480` (`T >= max_seq_len`) and
   `dataflow_common.hpp:744` (`ring_wrapped`) must classify the same runs as no-wrap.
3. **cur_pos clamp** identical in reader/writer/compute (l1_only mode).
4. **Sentinel** `-1` (float) → `int32` → `0xFFFFFFFF` round-trips for "skip this tier write".
5. **Tensor lifecycle** in `ttnn_decode_forward`: dealloc only the **built** ring tensor.
6. **Trace-safety**: warmup forward before `begin_trace_capture`; host path creates no
   config-varying tensors per step.
7. **Program-cache hash** includes L1 params (`sdpa_decode_op.cpp`) so tier changes recompile.

---

# Section 2 — Raw Memory Access Bandwidth Profiling

This section is measurement tooling, not a model code change. Three pieces: a reused
microbenchmark, an in-situ probe in the SDPA reader (gated, off by default), and the
sweep/plot scripts.

## 2.1 Reused microbenchmark — `test_bw_and_latency`
- Path: `tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency.cpp`
  (+ kernel `kernels/bw_and_latency.cpp`).
- **Modified on this branch only for the MeshDevice / distributed host API** (call-site
  modernization); the core measurement loop is upstream/unchanged. Built with
  `./build_metal.sh --build-tests` (the prebuilt binary was ABI-stale and had to be rebuilt).
- Relevant flags used by the sweep:

| Flag | Meaning | Use in sweep |
|---|---|---|
| `-m` | source: `1`=DRAM, `2`=L1, `3`=all-DRAM banks | DRAM vs SRAM select |
| `-sx -sy` | L1 source core (X,Y) | local SRAM = reader core; remote SRAM = far core |
| `-rx -ry` | reader/worker core (X,Y) | fixed reader at (1,1) |
| `-p` | page size = per-read transfer bytes | swept 64B…1MB |
| `-bs` | total transfer KB (`page_count = bs*1024/p`) | sets total moved |
| `-l` | latency mode (single outstanding read + barrier) | latency series |
| `-i` | iterations | amortize (2000) |

Single-reader caveat documented in the doc: this measures one reader's BW (~63 GB/s DRAM,
~82 GB/s SRAM at 1MB), not the ~512 GB/s aggregate; the multi-core aggregate test
(`8_dram_adjacent_core_read`) was broken in this env, so the aggregate ceiling uses the GDDR6 spec.

## 2.2 In-situ RD_LAT probe in the SDPA reader (gated, OFF)
Separate from the microbench: an isolated single-tile read-latency probe compiled into the
actual SDPA reader, used for the per-access L1-vs-DRAM latency numbers (DRAM ~412 / L1 ~336 ns).
- `dataflow_common.hpp:560-566` — DRAM single-tile `RD_LAT` probe.
- `dataflow_common.hpp:789` — L1 single-tile `RD_LAT` probe.
- Both guarded by `#if defined(SDPA_PROFILE_ZONES)`; the macro is **commented out**
  (`dataflow_common.hpp:12`) → probe is **OFF** by default, zero cost. Toggling requires
  `rm -rf ~/.cache/tt-metal-cache` (build-key hash excludes kernel source).

## 2.3 SDPA device-zone instrumentation (adjacent; OFF)
Used for the compute-bound proof (not "raw access" per se, but the same scaffolding):
- `kernels/compute/sdpa_flash_decode.cpp:29` — `SDPA_PROFILE_ZONES` macro (commented/OFF);
  zones `CMP_CHUNK`, `QK_MM`, `SM_NORM`, `PV_MM`, `SM_RESCALE`.
- `dataflow_common.hpp` — reader zones `RD_CHUNK`, `RD_K`, `RD_V`, `RD_KBAR`, `RD_LAT`.
- Profiler caps ~4-5 distinct zones per kernel; adding more silently drops the later ones.
- **State to verify after any profiling session:** both `SDPA_PROFILE_ZONES` defines commented
  out, cache wiped. (Grep cheat sheet above.)

## 2.4 Sweep + plot scripts (untracked, under `reprofile/`)
| File | Role | Output |
|---|---|---|
| `reprofile/run_rawaccess_sweep.sh` | Sweep `-p` 64B…1MB for DRAM / local-SRAM (src=reader (1,1)) / remote-SRAM (src far (7,6)); latency (`-l`) and bandwidth runs | `rawaccess/rawaccess.csv` (cols: source, size_bytes, latency_ns, bw_gbs) |
| `reprofile/draw_rawaccess.py` | Plot the CSV | `rawaccess_latency.png` (log-y), `rawaccess_bandwidth.png` |

Measured shape (P150, single card): latency@64B local-SRAM 55 ns < remote-SRAM 245 ns <
DRAM 337 ns (converge at large size = transfer-bound); single-reader BW@1MB SRAM ~82 vs DRAM
~63 GB/s. Note "local SRAM" still traverses the NoC (loopback), matching how SDPA reads L1.

---

# Section 3 — How a single Blackhole chip processes batch (SDPA core allocation, e.g. `batch-32`)

This section is **upstream mechanism**, not a morjun change — but it is documented here because
(a) the audit question asks how the chip maps batch to cores, and (b) the L1-KV reader's per-core
tier addressing (`l1_kv_head_base` in `dataflow_common.hpp`, §1.2) is indexed by exactly the
`(cur_batch, cur_head)` this allocator computes. All line refs are verified in
`ttnn/.../sdpa_decode/device/sdpa_decode_program_factory.cpp`.

Batch enters the decode step in **two unrelated places**; do not conflate them:

## 3.1 Linear ops (QKV / O / MLP / LM-head): batch = matmul rows (M)
`forward_decode` activation is `x: (seq_len=1, 1, batch, dim)` (`attention.py:1240`), and it is
reshaped so batch is the **M (row) dimension** of the projection matmul, padded to one 32-row tile:
```python
# attention.py:1295-1296
xqkv_fused = ttnn.reshape(
    xqkv_fused, (1, 1, self.batch_size_per_device_group, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3]))
```
So `batch-1` fills 1 of 32 tile rows (31 idle); **`batch-32` fills the tile exactly** (M=32), against
the *same* shared weight tensor. This is the throughput lever for the weight-bound step
(see `COMPUTE_BOUND_PROOF.md` §8c). It is a per-matmul shape, not a core-assignment.

## 3.2 SDPA decode: batch is sharded across cores (the part the question asks about)
The decode SDPA op processes **all B users in one invocation** and partitions the Tensix grid into
per-batch, per-head groups. The allocation math (verified):

```cpp
num_cores_available      = grid_size.x * grid_size.y;                                   // :184  (pool; grid from SDPA_DECODE_PROGCFG)
TT_FATAL(num_cores_available >= B, ...);                                                // :206  (need ≥1 core per user)
max_num_cores_for_compute = program_config->max_cores_per_head_batch * B * num_kv_heads;// :212  (default max_cores_per_head_batch=16, sdpa_config.hpp:18)
num_cores_per_batch  = min(num_cores_available, max_num_cores_for_compute) / B;         // :213
num_cores_per_head   = max(1, num_cores_per_batch / num_kv_heads);                      // :215  (context-split cores per kv-head)
num_heads_per_core   = max(1, ceil(num_kv_heads / num_cores_per_batch));                // :216  (kv-heads each core covers)
num_output_cores     = B;                                                               // :218  (one output/reducer per user)
num_active_cores     = num_cores_per_head * num_kv_heads * B / num_heads_per_core;       // :219
```

**The hierarchy:** the grid is divided into `B` contiguous **per-batch blocks** of
`num_cores_per_batch` cores each; within a block the cores split into `num_kv_heads` head groups of
`num_cores_per_head` cores; and each head group flash-decodes one kv-head by **splitting the KV
context across its cores** and reducing partial softmax results into one reducer core. Per-core
identity at runtime (verified `:1055-1058`, 1D core index `i`):
```cpp
cur_batch          = i / num_cores_per_batch;                 // which user this core serves
cur_head           = (i % num_cores_per_batch) / num_cores_per_head;  // which kv-head
core_num_in_reduce = i % num_cores_per_head;                  // position within a head group
do_reduce          = (i % num_cores_per_head) == 0;           // head-group reducer (:770-771)
do_output          = (i % num_cores_per_batch) == 0;          // per-batch output core (:796-797)
```
Core layout is a 1D list `[batch0: output, workers…][batch1: output, workers…]…` (`:228-285`); cores
beyond `num_active_cores` go idle (`core_group_idle`).

**How it shifts from `batch-1` to `batch-32`** (num_kv_heads = 8 for Llama-8B): the core pool is
fixed, so growing `B` *divides* it — `num_cores_per_batch` shrinks, the per-head context split
(`num_cores_per_head`) collapses toward 1, and `num_heads_per_core` rises above 1 (one core then
serves several kv-heads). Worked example for a pool of `P` active cores:

| config | num_cores_per_batch | num_cores_per_head | num_heads_per_core | shape of the work |
|---|---|---|---|---|
| `batch-1`  | `P` (≈64 observed) | `P/8` (≈8) | 1 | 8 heads × ~8 cores/head, heavy **context-split** per head |
| `batch-32` | `P/32` | `max(1, P/256)` → **1** | `ceil(8/(P/32))` (≥1) | ~1 core per (user, head-bundle); **no context-split**, each core does a whole head |

So the same silicon that, at `batch-1`, throws ~8 cores at each head to slice the KV context, at
`batch-32` instead gives each user a thin slice of the grid and has each core compute one (or a few)
full heads end-to-end, with one output core per user (`num_output_cores = B = 32`). The total active
cores grow with B until the pool is exhausted (capped at `num_cores_available`; once
`num_cores_per_batch` hits 1, extra users would need `num_cores_available ≥ B`, asserted at `:206`).

**Read the real numbers per run:** the factory logs the resolved scheme — grep the run for
`Parallelization scheme` (`log_debug` at `:287-296`: `num_cores_per_batch`, `num_cores_per_head`,
`num_heads_per_core`, `num_active_cores`, `core_group`). The exact pool depends on the model's
`SDPA_DECODE_PROGCFG` (`compute_with_storage_grid_size`, `max_cores_per_head_batch`) and on
`q_heads_parallel_factor` (`:121`, which can parallelize GQA query-heads like extra batch).

**Why this matters for the rest of the docs:** because batch shards across cores (read and compute
per core both fixed by heads-per-core, independent of B), the per-core compute:read ratio is
batch-invariant — the mechanism behind the strict "batch cannot saturate DRAM" proof in
`COMPUTE_BOUND_PROOF.md`. And the L1-KV reader keys its tier offset off `cur_batch`/`cur_head`
(`l1_kv_head_base`, §1.2), so per-batch KV correctness rides on this same assignment.

---

## Appendix — exclusions and removed code (so the audit is complete)
- **`sdpa_windowed`** (`ttnn/.../transformer/sdpa_windowed/*`): upstream Qwen2.5-VL op, not this
  feature. Verified author is TT-upstream, absent from morjun's commits. Do not audit as ours.
- **Weight-sharding path**: implemented then removed (`c62f6fb`). Residual touches in matmul
  kernels (`bmm_large_block_*`, `reader_bmm_*`) from that effort were reverted; if any remain in
  the working tree they are not part of the L1 KV feature.
- **Diagnostic/host-timer plumbing** (`ttnn/cpp/ttnn/util/timer.hpp`, `ttnn/ttnn/timer.py`,
  `l1_kv_perf.timed(...)` call sites, `visualize_timers.py`): profiling aids; opt-in, off the
  decode critical path.

# Profile SDPA-decode internals: FPU vs SFPU split + read/compute overlap timeline

## Context

We established (validated, see `performance-analysis.md` §2) that SDPA-decode is
**compute-bound**: L1 == DRAM op latency within ±2%, KV read fully hidden behind compute.
Two questions remain open and are the actionable next step:

- **(A) FPU vs SFPU:** within the attention compute, is softmax (SFPU: exp / reduce /
  recip) the bottleneck, or are the two matmuls (FPU: QK^T, PV)? This decides whether a
  softmax lever (`EXP_APPROX_MODE`, math approx, fp32 acc) can actually reduce the
  compute that dictates decode latency.
- **(B) Read/compute overlap timeline:** measure the exact intervals where NoC KV read
  (NCRISC) and compute (TRISC) run, on a shared timebase, to confirm read ⊆ compute and
  quantify the hiding margin.

No hardware FPU/SFPU utilization counter exists on this build (confirmed). The only path
is manual `DeviceZoneScopedN` zones inside the kernels. The NoC event profiler crashes on
this Blackhole build (`kernel.cpp:293` binary-cache mismatch when `PROFILE_NOC_EVENTS`
flips), so goal B is done instead by overlaying NCRISC reader zones and TRISC compute
zones — all RISCs on a core share one cycle counter, so they overlay directly. Both goals
are served by **one** kernel-instrumentation pass and **one** profiled run; the resulting
`profile_log_device.csv` contains every zone.

Intended outcome: a per-zone breakdown (FPU-total vs SFPU-total, ns and ratio) and a
read-vs-compute overlap fraction + Gantt, written into `performance-analysis.md`, plus a
clear yes/no on whether softmax tuning is worth pursuing.

## Key facts (from exploration, file:line)

- Compute kernel `ttnn/cpp/.../sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`,
  k-chunk loop **284-467**. FPU: QK^T `cb_matmul_blocks` **307-323** (+optional mask add
  326-345); PV `cb_matmul_blocks` **392-411**. SFPU: `reduce_c<MAX>` **365**, `sub_exp`
  **376-381**, `reduce_c<SUM>` **389-390**, online rescale **418-448**, `recip` ~**549**
  (post-loop, one-shot). The kernel does NOT currently include the profiler header.
- Reader `ttnn/cpp/.../sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp`: under
  `--l1_kv_mode dram` the **plain non-paged branch (438+)** runs (cb_reserve → noc_async_read
  loop → `noc_async_read_barrier()` → cb_push_back for K then V), NOT the n-tier helper.
  Instrument both this branch and `dataflow_common.hpp::read_kv_mask_chunks_n_tier`
  (K 747-854, V 862-924) so either selected path is covered.
- `DeviceZoneScopedN(name)` from `tools/profiler/kernel_profiler.hpp`, RAII; use the plain
  variant on TRISC (the Main/guaranteed variant is unavailable on TRISC). Budget **250
  optional markers / core / RISC / invocation**; a zone = 2 markers; overflow sets
  `DROPPED_ZONES` and silently drops. Markers reset each op invocation (each decode step).
- Chunk count = `seq_len / k_chunk_size`, `k_chunk_size = min(512, largest_pow2_divisor(seq_len))`
  (`sdpa_decode.cpp:15-28`), split across cores. **seq_len 1792 = 2^8·7 → k_chunk_size 256
  → 7 chunks total**, a handful per core. 5 zones/chunk × 2 = 10 markers; ≤7 chunks/core →
  ≤70 markers ≪ 250. No gating needed. (`prompt_ctx1792.json` already exists.)
- Output `generated/profiler/reports/<ts>/profile_log_device.csv`: cols core_x, core_y,
  RISC, timer_id, time[cycles], run_host_id, zone name, type(ZONE_START/END), src line/file.
  ns = cycles*1000/1350. Reuse `research_codes/tracy/analyze_tracy_profile.py::parse_device_profile()`
  (→ DeviceZone{core,risc,start_cycle,end_cycle,duration_cycles,zone_name,run_host_id}) and
  `research_codes/visualize_riscv_timeline.py` (Gantt).
- These kernels JIT-recompile on edit (no `build_metal.sh`), but the persistent on-disk
  kernel cache can serve a stale binary; force fresh compile (below).

## Approach

All zone code is guarded behind a compile define `SDPA_PROFILE_ZONES`, injected only when
env `SDPA_PROFILE_ZONES=1` is set. This keeps the production build byte-identical AND
changes the kernel hash so the JIT recompiles fresh (sidesteps the stale-binary cache).

### 1. Goal A zones — `sdpa_flash_decode.cpp` (inside loop 284-467)
Add `#include "tools/profiler/kernel_profiler.hpp"` (guarded). Wrap each contiguous
compute block in a nested RAII scope, keeping `reconfig_data_format`/`pack_reconfig`
setup calls OUTSIDE the zones:

| Zone | Wraps | lines | engine |
|---|---|---|---|
| `QK_MM` | QK^T matmul (+mask add) | 307-345 | FPU |
| `SM_MAXEXP` | reduce_max + sub_exp | 354-382 | SFPU |
| `SM_SUM` | reduce_sum | 384-390 | SFPU |
| `PV_MM` | PV matmul | 392-411 | FPU |
| `SM_RESCALE` | online-softmax rescale | 418-448 | SFPU |

(Optional one-shot `SM_RECIP` around `recip` ~549, outside the per-chunk comparison.)

### 2. Goal B zones — reader + a compute envelope
- Reader (both `reader_decode_all.cpp:438+` plain branch AND `dataflow_common.hpp`
  747-924): per chunk, `RD_K` around the K reserve→read→`noc_async_read_barrier()`→push,
  `RD_V` around the V block (2 zones/chunk, NCRISC).
- Compute: add `CMP_CHUNK` wrapping the whole loop-body interior (284-467) as the
  per-chunk compute envelope on TRISC. Total TRISC = 6 zones/chunk = 12 markers; ≤84 at 7
  chunks — safe.

Overlay: per core, NCRISC `RD_K/RD_V` intervals vs the union of TRISC `CMP_CHUNK` (and the
QK/PV/SM zones) intervals on the same cycle base. Because the reader runs one chunk ahead
(double-buffered via cb_reserve/push), expect `RD[i+1]` to overlap `CMP_CHUNK[i]`.

### 3. Inject define + force fresh compile — `sdpa_decode_program_factory.cpp`
When env `SDPA_PROFILE_ZONES=1`: add `"SDPA_PROFILE_ZONES"` to the compute `defines` map
(~979) and to the reader `CreateKernel` defines (after ~1011). For the run, also set
`TT_METAL_KERNEL_CACHE_DISABLE=1` (or `rm -rf $TT_METAL_HOME/built/`) to guarantee no stale
binary.

### 4. Run + parse — extend `reprofile/run_one.sh` (or add `run_zones.sh`)
Same tracy invocation, fixed to DRAM / ctx1792 / 2 layers, with the two env vars:
```
TT_METAL_KERNEL_CACHE_DISABLE=1 SDPA_PROFILE_ZONES=1 \
PATH=$PWD/python_env/bin:$PATH TT_METAL_HOME=$PWD PYTHONPATH=$PWD/tools \
python -m tracy -r -p -v -m pytest \
  "models/tt_transformers/demo/simple_text_demo.py::test_demo_text[blackhole-mesh_device0-device_params0-performance-batch-1]" \
  --input_prompts research_codes/documents/l1_kv_cache_cache/reprofile/prompt_ctx1792.json \
  --max_seq_len 2048 --max_generated_tokens 8 --num_layers 2 \
  --instruct 0 --paged_attention 0 --l1_kv_mode dram
```
Copy `profile_log_device.csv` (NOT ops_perf). New small analysis script
(`reprofile/analyze_zones.py`) using `parse_device_profile()`:
- **Goal A:** group `duration_cycles` by zone_name across all cores/invocations;
  `FPU_total = QK_MM + PV_MM`, `SFPU_total = SM_MAXEXP + SM_SUM + SM_RESCALE`; report ns,
  per-chunk avg, ratio. Larger one answers A.
- **Goal B:** per core, overlap_fraction = covered(RD ∩ union(CMP)) / total(RD); report
  compute−read margin (ns). Render Gantt via `visualize_riscv_timeline.py`.

### 5. Follow-on lever (separate experiment, NOT this measurement)
If SFPU dominates: flip `EXP_APPROX_MODE` (~979) / `math_approx_mode` / `fp32_dest_acc_en`
(ComputeConfig ~1005-1006) in the program factory, re-run the same harness, compare
`SFPU_total`. Keep out of the measurement pass.

## Files to modify
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp` — Goal A zones (guarded).
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp` — Goal B reader zones, plain branch (guarded).
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` — Goal B reader zones in `read_kv_mask_chunks_n_tier` (guarded).
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp` — inject `SDPA_PROFILE_ZONES` define under env; (later) EXP_APPROX lever.
- `research_codes/documents/l1_kv_cache_cache/reprofile/run_one.sh` (+ new `analyze_zones.py`) — run + parse.
- Reuse: `research_codes/tracy/analyze_tracy_profile.py`, `research_codes/visualize_riscv_timeline.py`.

## Verification
1. **Fresh compile:** run log shows `sdpa_flash_decode.cpp` compiling; zone names appear in CSV.
2. **No drops:** every expected zone name present with count = chunks/core (marker math ≤84 ≪ 250); no `DROPPED_ZONES`.
3. **Zone sum vs op:** per-core Σ`CMP_CHUNK` ≈ SDPA TRISC device-kernel-duration (cross-check vs `extract.py` ops_perf SDPA number) within a few %.
4. **Ordering:** per chunk on timeline = QK_MM → SM_MAXEXP → SM_SUM → PV_MM → SM_RESCALE.
5. **Overlap:** overlap_fraction ≈ 1.0 and read_total ≤ compute_total (consistent with L1==DRAM). A leak past compute would contradict the prior finding → investigate (likely a barrier outside the zone).
6. **Determinism:** run twice; per-chunk averages stable within a few %.
7. Write results table + Gantt reference into `performance-analysis.md`; update memory `l1-kv-sdpa-compute-bound`.

## Risks / hygiene
- Core production kernels: ALL zone code guarded by `#ifdef SDPA_PROFILE_ZONES`; default build byte-identical. Remove scaffolding or leave permanently behind the off-by-default define after the investigation; never leave bare zones in the hot loop.
- Few chunks/core (1-7) → rely on aggregation across 64 attention cores × 8 steps × 2 layers for sample count, not per-core.
- If a longer context is ever profiled (>~25 chunks/core), gate zones to the first N chunks to stay under 250 markers.

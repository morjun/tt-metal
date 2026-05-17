# Adaptive L1 KV Cache — Headroom Estimation Session Summary

> Blackhole (P150) · Llama 3.1 8B · `l1-kv-cache` branch
> Last updated: 2026-05-17

---

## 1. Background & Motivation

The Adaptive N-tier L1 KV Cache stores a portion of the KV attention history in each
Tensix core's L1 SRAM instead of DRAM, reducing NoC & DRAM traffic for SDPA decode. The amount
of KV tokens that can safely fit in L1 depends on how much L1 is *free* after the model's
static Circular Buffers (CBs) and transient runtime allocations have claimed their regions.

**The core problem that triggered this work:**

```
RuntimeError: Statically allocated circular buffers in program N clash with L1 buffers
on core range [...]. L1 buffer allocated at 106240 and static CB region ends at 110976.
```

The L1 KV tensor was being placed by the bottom-up allocator at an address *inside* the
static CB region of a later program (e.g., the RoPE embedding program), causing a hard
crash. The root cause was **inaccurate headroom estimation**.

---

## 2. Key Discoveries from Profiling Analysis

### 2.1 Discrepancy Between 1-layer and 32-layer CB Footprints

Offline profiling (via `TT_METAL_LOG_L1_CB_MAP=1`) showed that `--num_layers 32` has a
significantly larger CB footprint than `--num_layers 1` on the same hardware:

| Core group | 32-layer CB end (worst-case pid) | 1-layer CB end |
|---|---|---|
| SDPA cores (x=0–7, y=0–7) | ~1,249,664 B (pid 5, SDPA) | ~1,180,160 B |
| Gather cores (y=8) | larger | smaller |

**Why?** The `max_seq_len` parameter — and specifically the derived `chunked_prefill_max_seq_len` —
scales with model depth (num_layers). A deeper model allocates larger scratch CBs for
the SDPA chunked prefill path (program pid 41), which only appears in 32-layer runs. (Not sure)

### 2.2 What the Live Scan Misses (Top-Down Allocations)

The runtime API `get_l1_headroom_per_core()` measures the gap between the CB region top
and the current **top-down** (temporary output buffer) watermark. However, between decode
steps the temporary buffers are freed, so the between-step scan under-reports true peak usage:

| Core group | Transient top-down missing | Source |
|---|---|---|
| SDPA active (y=0–7) | ~4,352 B | attention output intermediates |
| Gather/concat (y=8) | ~99,136 B | nlp_concat_heads output |
| Non-SDPA (x=8–10, col) | ~65,792 B | MLP activation outputs |
| Idle row (y=9) | ~142,144 B | pipeline flush buffers |

### 2.3 The CB Clash Root Cause (Suspected)

The HEIGHT_SHARDED L1 KV tensor is placed by the **bottom-up allocator** starting at the
first free address above existing allocations. Before any program has run (pre-compile),
that address can be very low (e.g., `106,240`). When a program later tries to register
its static CBs (which reach up to e.g. `110,976`), the runtime detects the conflict.

**Critical constraint**: the KV tensor must be allocated **after** the compile decode step
has run and raised the bottom-up watermark above `max(cb_region_end)` across all programs.

---

## 3. Dual-Path Headroom System (Implemented)

### Architecture Overview

```
--l1_kv_headroom_json provided?
       │
       ├─ YES → Path A (JSON): load gap_bytes_free_headroom per core from offline profile
       │         Most accurate: captures both CB static footprint and peak transient allocs
       │
       └─ NO  → Path B (Live Scan): improved mid-step measurement
                 Scan AFTER warmup decode step, while output tensors are still alive
                 Captures more top-down buffers than a between-step scan
```

### Execution Timeline (Both Paths)

| Decode call | JSON path | Live-scan path |
|---|---|---|
| Step 1 (compile) | Normal decode → raises L1 watermark past all CB tops | Same |
| Step 2 | **Allocate from JSON** → decode normally | **Decode first** → scan live → allocate |
| Step 3+ | Decode with L1 KV tiers active | Same |

> **Why step 1 must run first:** Before step 1, the L1 bottom-up watermark is low.
> The KV tensor would land inside a later program's CB region. Step 1 dispatches every
> program (embedding, attention, MLP, …), permanently raising the watermark above
> `max(cb_region_end)` across all programs (~1.25 MiB on SDPA cores).

---

## 4. Files Changed

### CLI & Config Chain

| File | Change |
|---|---|
| `models/tt_transformers/demo/conftest.py` | Added `--l1_kv_headroom_json` CLI option |
| `models/tt_transformers/demo/simple_text_demo.py` | Wired `l1_kv_headroom_json` in both `prepare_generator_args` signature and the **inner** `create_tt_model()` call (Bug 1 fix) |
| `models/tt_transformers/tt/common.py` | Added `l1_kv_headroom_json=None` param, forwarded to `ModelArgs` |
| `models/tt_transformers/tt/model_config.py` | Added `self.l1_kv_headroom_json` field to `ModelArgs` |

### Generator Logic

| File | Change |
|---|---|
| `models/tt_transformers/tt/generator.py` | Added `_l1_kv_warmup_done` state; restructured `_decode_forward_no_trace_text` dispatch for both paths; added `_post_compile_allocate_l1_kv()` (headroom source selection); added `_load_headroom_json()` static helper |

### Bug Fixes Applied in This Session

| Bug | Symptom | Fix |
|---|---|---|
| **`AttributeError: _run_decode_forward_text`** | Crashed on step 2 | Replaced phantom helper call with inlined warmup decode body |
| **JSON path not selected** | Log showed "live scan" even with `--l1_kv_headroom_json` | `l1_kv_headroom_json` was not passed to inner `create_tt_model()` call inside the submesh loop in `simple_text_demo.py:383` |
| **CB clash at 106,240 < 110,976** | `RuntimeError: circular buffers clash` | JSON path was allocating KV **before** step 1 ran (watermark still low). Fixed by moving allocation to after `_decode_compile_done=True`, i.e., on step 2+ |

---

## 5. Key Implementation Details

### `_decode_forward_no_trace_text` Dispatch Logic

```python
_has_json = bool(getattr(self.model_args[0], "l1_kv_headroom_json", None))

if self.l1_kv_needs_alloc and self._decode_compile_done:
    if _has_json:
        # JSON path: step 1 already ran, watermark raised, allocate from JSON
        self._post_compile_allocate_l1_kv()
    elif self._l1_kv_warmup_done:
        # Safety net (live-scan path, already done)
        self._post_compile_allocate_l1_kv()
    else:
        # Live-scan path: run decode body first, scan at end, allocate
        self._l1_kv_warmup_done = True
        self._decode_compile_done = True
        # ... inlined decode body (prepare_inputs + ttnn_decode_forward) ...
        self._post_compile_allocate_l1_kv()  # scan while output tensors alive
        return tt_logits_warmup

self._decode_compile_done = True
# [normal decode body falls through here]
```

### `_load_headroom_json` — JSON Format

```python
# Input JSON (produced by TT_METAL_LOG_L1_CB_MAP profiling):
# {
#   "(0,0)": {
#     "cb_region_end_bytes": 1249664,
#     "lowest_top_down_addr": 1568512,
#     "top_down_size_bytes": 4352,
#     "gap_bytes_free_headroom": 318848,   ← this field is consumed
#     "max_l1_size": 1572864
#   }, ...
# }
#
# gap_bytes_free_headroom = lowest_top_down_addr - cb_region_end_bytes
# This is the "safe zone" between the highest static CB and the peak transient buffer.
```

### Headroom → Tier Sizing (in `attention.py`)

```python
# Cost formula per core (K+V, all 32 layers):
bytes_per_tile_row = tile_size * head_dim * elem_bytes * num_layers * 2

# Available tile-rows per core:
net = headroom_bytes - safety_margin_bytes   # safety_margin default: 64 KiB
tile_rows = net // bytes_per_tile_row
```

Cores are bucketed by `tile_rows` into tiers (up to 5 tiers). Each tier becomes one
HEIGHT_SHARDED L1 tensor covering that set of cores.

---

## 6. Verification Commands

```bash
# Path A: offline JSON (most accurate, zero warmup overhead)
TT_LOGGER_LEVEL=Info \
pytest models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" \
  --use_adaptive_l1_kv_cache \
  --l1_kv_headroom_json research_codes/documents/l1_kv_cache_cache/comparison/headroom_map_global_minimum.json \
  2>&1 | tee /tmp/adaptive_json.log

# Path B: improved live scan
TT_LOGGER_LEVEL=Info \
pytest models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" \
  --use_adaptive_l1_kv_cache \
  2>&1 | tee /tmp/adaptive_live.log
```

### Log Signatures to Confirm Correct Path

| Log line | Meaning |
|---|---|
| `[L1 KV] Using offline headroom JSON: <path> (110 cores)` | Path A active ✓ |
| `[L1 KV] Using live headroom scan (improved: mid-step measurement)` | Path B active |
| `[L1 KV adaptive] Tier: X cores × Y tile-rows = Z tokens` | Tier allocated |
| `[L1 KV] Adaptive L1 KV cache allocation complete.` | All tiers done |
| *(no CB clash error)* | Allocation within safe bounds ✓ |

---

## 7. Remaining Tasks

### Immediate (Next Run)

- [ ] **Run Path A end-to-end** with the fixed `simple_text_demo.py` and confirm the log
  shows `"Using offline headroom JSON"` (not "live scan") and no CB clash error.
- [ ] **Run Path B end-to-end** and confirm the warmup decode completes before allocation.
- [ ] **Check tier allocation counts**: the log should show 3 tiers (56 cores × 1 tile-row,
  8 cores × 2 tile-rows, 36 cores × 4 tile-rows) matching previous working runs.

### Medium Term

- [ ] **OOM handling audit**: `attention.py` catches `RuntimeError` on tier allocation and
  skips tiers — verify the fallback path logs a warning and continues rather than silently
  losing tokens.
- [ ] **Refresh the JSON profile**: the current `headroom_map_global_minimum.json` was
  captured with a specific `max_seq_len` and `num_layers=32`. Re-capture if those change.
- [ ] **Profile Path B accuracy vs Path A**: compare headroom values printed in the log
  for each core group; confirm Path B is within ~4 KiB of Path A on SDPA cores, and
  within ~99 KiB on gather cores.

### Long Term

- [ ] **Build verification**: confirm `cmake --build build_Release -t sdpa_decode` still
  passes (C++ tier-dispatch kernel changes from earlier in the project).
- [ ] **Performance benchmarking**: measure tokens/s improvement from L1 KV hits vs
  baseline DRAM-only (the primary motivation for this entire feature).
- [ ] **Trace-mode support**: the dual-path dispatch is only in `_decode_forward_no_trace_text`.
  The `enable_trace=True` path in `decode_forward_text` has a different compile→capture→replay
  flow; L1 KV allocation timing there needs separate analysis.

---

## 8. Headroom JSON Reference Data

From `comparison/headroom_map_global_minimum.json` (32-layer, worst-case per program):

| Core (representative) | `cb_region_end_bytes` | `gap_bytes_free_headroom` | `worst_case_pid` |
|---|---|---|---|
| (0,0) — SDPA | 1,249,664 | 318,848 | 5 (SDPA) |
| (0,1)–(7,7) — SDPA | 1,225,088 | 343,424 | 41 (chunked prefill) |
| (0,8)–(7,8) — gather | ~723,000 | ~849,920 | varies |
| (8,0)–(10,8) — column | ~360,832 | ~1,212,032 | varies |
| (0,9)–(8,9) — idle row | ~218,816 | ~1,354,048 | varies |

> `gap_bytes_free_headroom` = `lowest_top_down_addr − cb_region_end_bytes`
> This is the value passed to the tier allocator as available bytes per core.

---

## 9. Related Documents in This Directory

| File | Content |
|---|---|
| `workflow_overview.md` | End-to-end code flow for all 3 KV workflows (DRAM, fixed-window, adaptive) |
| `l1_kv_cache_architecture.md` | Architecture deep-dive: shard layouts, tier math, kernel interface |
| `l1_cb_map_analysis.md` | Profiling methodology and CB map parse results |
| `blackhole_p150_sdpa_decode_cores.md` | Core grid assignment for SDPA decode on Blackhole P150 |
| `comparison/headroom_map_global_minimum.json` | Offline headroom profile (32-layer, global minimum per core) |
| `comparison/headroom_adaptive_scan.txt` | Live-scan baseline for comparison |
| `impl/` | C++ kernel source snapshots |

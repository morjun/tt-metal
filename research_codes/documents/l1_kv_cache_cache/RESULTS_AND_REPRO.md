# L1 KV cache — results and reproduction commands

Companion to `COMPUTE_BOUND_PROOF.md`. Records the measured results for the two demands and the
exact commands/scripts that produced them. Platform: Blackhole P150, single card, Llama-3.1-8B,
batch-1 decode, ctx896.

---

## Results summary

### Demand 1 — host parity (ring modulo moved off the host)
- The L1 ring write index is now computed **on-device** once per step
  (`Transformer._compute_l1_ring_pos_device`, `model.py`), with a no-wrap fast path that returns
  `current_pos` unchanged. The host `torch.remainder`/`where`, the `from_torch`, and the extra
  H2D push are removed (commit `3da90257006`).
- **Measured (trace on, l1_only window 960):** trace captures cleanly with the modulo inside the
  trace; steady **20.95 tok/s vs DRAM 20.97** (was 20.90 with the host modulo). Output coherent
  (correctness preserved).
- Per-step host work is now identical to DRAM; no L1-specific per-step host compute and no
  per-step device→host sync. Only un-removable L1 cost = one-time tier alloc + seed (compile-time,
  not per-step). Full inventory: `COMPUTE_BOUND_PROOF.md` §8b.

### Demand 2 — the QK_MM→SM_NORM hole is NOT a memory read wait
The interval between the `QK_MM` zone end and the `SM_NORM` zone start is left as a HOLE — no zone
covers it, so it is not labelled a characterized "gap." It is small on most cores (p50 ~85 ns)
with a tail (~12% of cores >500 ns, up to ~1665 ns), and it is NOT a memory read wait:
- **Memory-invariant (measured).** The hole-duration CDF is the same for L1 and DRAM —
  p90 590 ns (L1) vs 541 ns (DRAM); >500 ns tail 384 cores (12%) L1 vs 382 (12%) DRAM. A read
  wait would shrink with L1's ~12% lower per-tile latency (RD_LAT 362 vs 405 ns); it does not.
- **No read dependency in the interval (code).** Mask fused via `DYNAMIC_CHUNK_SIZE=1`; the
  cross-core reduction is post-PV; SM_NORM reads only this core's own `cb_qk_im` + a constant.
- A wall-clock cross-core aligned timeline is **infeasible** (per-core profiler counters on
  different epochs ~1e12 cycles apart; Tracy applies no skew correction) — and not needed: the
  L1==DRAM identity + code structure settle the read question.
- Chart: `reprofile/sdpa_bigpicture.png` (Panel B = the L1-vs-DRAM hole-duration CDF; the hole is
  drawn as blank white space in Panels A/C). Also `reprofile/zones2_l1only_896/overlap_gantt_l1.png`.

#### What code is (and is NOT) in the hole
`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`,
the interval between the QK_MM zone close (`:338`) and the SM_NORM zone open (`:380`):

```cpp
{
    SDPA_ZONE("QK_MM");                                  // :320  FPU QK^T matmul
    cb_matmul_blocks(cb_q_in, cb_k_in, cb_qk_im, ...,    // :321  ISSUES the matmul
                     add_mask_fusion, mask_cb_to_use, cb_zero_in);
}                                                        // :338  <-- QK_MM zone END (FPU still packing cb_qk_im)
/* QK += MASK */
if (!add_mask_fusion) {                                  // :341  add_mask_fusion=TRUE (causal last chunk,
    ... add_block_inplace(cb_qk_im, cb_mask_in, ...);    //        DYNAMIC_CHUNK_SIZE=1) -> this block is SKIPPED
}
reconfig_data_format(cb_qk_im, cb_identity_scale_in);    // :369  the ONLY executed code in the hole...
pack_reconfig_data_format(cb_cur_max);                   // :370  ...= DF_RECFG ~38ns + PACK_DRAIN ~32ns (~70ns, ~30%)
{
    SDPA_ZONE("SM_NORM");                                // :381  SFPU softmax
    reduce_c<PoolType::MAX, ...>(cb_cur_max, ...);       // :382  reads cb_qk_im (own) + const -> no reader/cross-core dep
```

- The ONLY executed statements in the hole are the two reconfig calls at `:369-370`. Measured as
  the `DF_RECFG`+`PACK_DRAIN` sub-zones, they are ~70 ns ≈ **30%** of the typical hole.
- The mask-add (`:340-360`) does NOT run (fused into QK_MM at `:335`), so there is no
  `cb_mask_in` read in the interval.
- The remaining **~70% (and the >1000 ns tail) corresponds to NO source line.** It is the implicit
  FPU matmul pipeline drain: the `QK_MM` `DeviceZoneScopedN` closes at `:338` when the matmul is
  *issued*, but the FPU keeps computing/packing `cb_qk_im`; `SM_NORM`'s `reduce_c` (`:382`) cannot
  read `cb_qk_im` until that finishes. There is no "wait for drain" instruction — it is hardware
  scheduling. That absence of a covering zone/statement is exactly why it is left as a hole.

### Standing conclusion
SDPA-decode is compute-bound (QK^T matmul ~66-77% of the envelope), the KV read is fully hidden
(RD_CHUNK ~4.7 µs < every core's compute; 100% read-hidden), and L1 == DRAM compute. So L1 ties
DRAM on decode latency; the lever is capacity, not layout. See `COMPUTE_BOUND_PROOF.md`.

---

## Reproduction commands

All from repo root `/home/masterjunmo/codes/tt-metal`. Single card:
`TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150`.

### Parity / functional (trace on, no zones)
```bash
./python_env/bin/tt-smi -r            # reset device between runs
TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 TT_LOGGER_LEVEL=Info \
  ./python_env/bin/pytest -q models/tt_transformers/demo/simple_text_demo.py \
    -k "performance and batch-1" \
    --l1_kv_mode interleaved --l1_kv_window_size 960 --l1_kv_only_mode
# DRAM baseline: drop the three --l1_kv_* flags.
# Read steady-state: average "Iteration N: Xms" for N>=5 -> tok/s = 1000/mean_ms.
```

### Device-zone profiling (hole / read-hidden)
Zones are guarded by `SDPA_PROFILE_ZONES` (compile-time, comment OUT by default) in BOTH
`sdpa_flash_decode.cpp` and `dataflow_common.hpp`. The build-key hash excludes kernel source, so
after toggling a macro you MUST wipe the kernel cache. Profiler caps distinct zones per kernel
(~5); adding more silently drops the later-declared ones (swap out PV_MM/SM_RESCALE to make room).
```bash
# 1) uncomment '#define SDPA_PROFILE_ZONES 1' in the compute kernel (and dataflow for reader zones)
# 2) set enable_trace = False at simple_text_demo.py:931 (zone runs are no-trace; trace overflows
#    the per-core marker buffer)
# 3) capture (run_zones32.sh wraps tracy + copies CSVs; 'wipe' recompiles kernels with zones):
bash research_codes/documents/l1_kv_cache_cache/reprofile/run_zones32.sh l1only wipe 2
bash research_codes/documents/l1_kv_cache_cache/reprofile/run_zones32.sh dram   ""   2
#    -> writes reprofile/zones2_{l1only,dram}_896/profile_log_device.csv
# 4) restore: re-comment both macros, set enable_trace=True, wipe cache.
```

### Analysis + charts (read-only, no device)
```bash
P=research_codes/documents/l1_kv_cache_cache/reprofile
./python_env/bin/python $P/analyze_gaps2.py $P/zones2_l1only_896/profile_log_device.csv L1-only
./python_env/bin/python $P/analyze_gaps2.py $P/zones2_dram_896/profile_log_device.csv DRAM
./python_env/bin/python $P/draw_bigpicture.py     # -> sdpa_bigpicture.png (Demand 2 big picture)
```

### Read-wait probe (RD_LAT / RD_CHUNK, both compute + reader zones)
```bash
# uncomment SDPA_PROFILE_ZONES in BOTH kernels, enable_trace=False, then:
bash research_codes/documents/l1_kv_cache_cache/reprofile/run_rw_probe.sh
#    -> reprofile/zones_rw_{l1only,dram}/profile_log_device.csv (RD_LAT, RD_K, RD_V, RD_CHUNK)
```

### Notes / gotchas
- `tt-smi -r` between runs; killing a run wedges the device (next run hangs at prefill).
- 2-layer represents 32-layer (zones are layer-invariant; 32 layers overflow the marker buffer).
- The aggregate `DEVICE KERNEL DURATION` column is corrupted (timestamp overflow) — use per-zone
  ZONE_START/END deltas or `DEVICE KERNEL DURATION PER CORE AVG/MAX`.
- Build (only if a non-kernel rebuild is needed): `./build_metal.sh` (never `cmake --build`).

### Commits (branch l1-kv-cache)
- `322b99515bb` Enable ttnn trace for L1 KV cache decode (warm up L1 ops before begin_trace_capture)
- `a68085aa844` Gate per-step L1 hit-ratio computation behind opt-in
- `3da90257006` Compute L1 ring write position on-device (eliminate host modulo + push)

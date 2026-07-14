# L1 KV Cache — Grand Master Plan

Date: 2026-06-12. Branch: `l1-kv-cache`. Target HW: P150 (Blackhole), single card, 32-layer, batch-1 decode.

## STATUS (2026-06-12)

Task 3 DONE and verified on P150: L1-only decode now runs under trace at ~48 ms/step / ~20.9 tok/s
(parity with DRAM trace-on ~45 ms / 22 tok/s; ~2.3x over L1 trace-off ~109 ms / ~9 tok/s). Fix was a
warm-up ordering bug, not a structural incompatibility — see `l1_kv_trace_support` memory and the
generator.py/model.py diff. Verified with `--l1_kv_mode interleaved --l1_kv_only_mode` (single 32-token
tier). Recommended follow-up: re-confirm with the 992-token window used in the original profiling.
Tasks 1 and 2 remain as below.

## Bottom line

The three tasks are not independent. Tasks 2 and 3 are the **same fix**: making the L1 KV decode
step trace-capturable. Trace is the dominant performance lever (2.4x), not a marginal one. Task 1
as literally stated **cannot be measured from the current profiling data** — the SM_NORM/QK_MM zones
were not captured. Priority order: **Task 3 (trace) → re-measure → Task 1 (zones) only if a gap
remains → Task 2 micro-opts (mostly absorbed by trace).**

---

## Context and corrections to the initial exploration

Three claims from the first-pass exploration are wrong and were verified against source/data:

1. **"Measurable SM_NORM→QK_MM gap in zones2_*_896 data."** False. `SDPA_PROFILE_ZONES` is
   commented out at `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp:29`.
   The CSVs contain only reader zones (`RD_K`, `RD_V`, `RD_KBAR`, `RD_CHUNK`, `RD_LAT`) and generic
   per-RISC envelopes (`TRISC-KERNEL`, `NCRISC-KERNEL`, ...). `SM_NORM`/`QK_MM`/`PV_MM`/`SM_RESCALE`
   are absent. The gap must be re-profiled with the macro enabled.

2. **"L1 is 5.6x slower than DRAM on SDPA."** False. That ratio (32.1e9 / 5.72e9) divides two
   corrupted values. The `DEVICE KERNEL DURATION [ns]` column reads ~5.7s and ~32s per op — physically
   impossible for one decode op — and only n=16 decode rows exist for an 896-token run. The column is
   an absolute timestamp leak / aggregation artifact, not a duration. This contradicts the established,
   directly-measured result that L1 == DRAM on SDPA latency (SDPA is compute-bound; KV read is hidden
   behind compute). Trust the prior measurement, not these CSVs.

3. **"`.item()` calls at model.py:349,352 are device→host syncs."** False. They operate on
   `current_pos`, which is already a host torch tensor in `prepare_decode_inputs_host`. They are cheap
   host-CPU ops, not device stalls. The line 349 `.mean().item()` feeds only a logging sample and can
   be dropped outright.

The one decisive, verified fact (`TRACE_HOWTO.md`, this dir): trace ON ~45 ms/step (~22 tok/s) vs
trace OFF ~109 ms/step (~9 tok/s). The L1-kv branch hardcodes trace OFF at
`models/tt_transformers/demo/simple_text_demo.py:931`. The DRAM baseline path supports trace. So any
A/B that pits trace-capable DRAM against trace-disabled L1 attributes a 2.4x dispatch gap to "L1 being
slower." Closing this is the whole game.

---

## Task 3 (do first) — Trace support for the L1 KV path

Goal: capture and replay the full L1 decode step so per-step host dispatch is eliminated, matching the
DRAM path. Template to mirror: the existing DRAM trace flow in
`models/tt_transformers/tt/generator.py` (`_capture_decode_trace_text` ~856-911,
`_decode_forward_trace_text` ~913-955) and the in-place tensor update via
`copy_host_to_device_tensor` (`models/tt_transformers/tt/common.py:446-472`).

Blockers and fixes:

1. **Per-step `ttnn.from_torch(l1_update_pos, ...)`** — `models/tt_transformers/tt/model.py:329-338`.
   Creates a fresh device tensor every step (allocation + H2D transfer); a new buffer address breaks
   replay. Fix: pre-allocate `l1_update_pos_tt` once (alongside `tokens`/`current_pos`/`rot_mat_idxs`
   in the trace input set) and update in place with `ttnn.copy_host_to_device_tensor`. This is the
   same mechanism the other persistent inputs already use. (This also resolves Task 2.)

2. **`l1_write_enabled` is a Python bool driving control flow** — set at `model.py:352-354`, branched
   on in `models/tt_transformers/tt/attention.py` (~1341, 1365, 1375). Trace captures exactly one
   branch; if the value flips between steps the replay runs the wrong path. Fix (choose one):
   - Simplest/recommended for the perf benchmark: make the write policy **static** — always write while
     within capacity (set `l1_kv_min_expected_hit_ratio = 0.0`, the default already yields
     `l1_write_enabled = True`). The adaptive gating is an optimization that is incompatible with a
     single captured trace; keep it only in the no-trace debug path.
   - If adaptive gating must survive trace: promote it to a 1-element device tensor and replace the
     Python `if` with a device-side `ttnn.where`/sentinel so the kernel no-ops when disabled (the tier
     code at `attention.py:1146-1226` already uses a `-1` skip sentinel — extend that pattern).

3. **`_build_adaptive_l1_write_pos` host sync** — `attention.py:1097-1144`, does
   `ttnn.to_torch(current_pos)` mid-forward (line ~1123); forbidden inside trace. It already returns
   `None`/is guarded inside trace. Fix: make pre-computed `l1_update_pos` **mandatory** on the trace
   path (assert non-None when `l1_kv_tiers` exist) so this function is never entered during capture or
   replay.

4. **Tier structure immutability** — `l1_kv_tiers` (token_start, tok_count) is allocated once after
   compile (`generator.py` `_post_compile_allocate_l1_kv`, ~636-709, sets `l1_kv_needs_alloc=False`).
   Already trace-safe; no change. Verify no tier reallocation can fire after capture.

5. **Re-enable the control** — revert the debug override at `simple_text_demo.py:931` and make the
   CLI flag honored (or just default batch-1 to trace ON as upstream does, parametrize at line 441).

Acceptance: L1 trace ON runs to completion, output tokens match the no-trace L1 run bit-for-bit (same
greedy/seeded sample), steady-state ms/step drops from ~100+ to ~45-50.

---

## Task 2 — Eliminate extra host-device round trips

Largely **subsumed by Task 3 fix #1** (in-place `l1_update_pos` update removes the per-step
allocation + H2D round trip). Remaining cleanups, in no-trace and trace paths alike:

1. Drop the logging-only `expected_hit_ratio.mean().item()` (`model.py:349`) under the perf benchmark,
   or gate it behind a debug flag — it adds host work with no functional role.
2. The ring modulo (`torch.remainder`/`torch.where`, `model.py:319-327`) is cheap host CPU on a
   host tensor; leave it on host (computing it on-device would add device ops and a dependency that
   complicates trace). Keep ring math host-side, only the **result tensor** goes device-side in place.
3. Confirm `current_pos` increment stays on host (`simple_text_demo.py:1276`) for the host-sampled
   path; no device round trip needed there.

Acceptance: with trace OFF, L1 ms/step within noise of the DRAM no-trace baseline (ties the
"capacity, not latency" expectation from prior measurement).

---

## Task 1 — Quantify the TRISC SM_NORM→QK_MM gap (do only if a gap remains post-trace)

This is a measurement task that currently has no data. Steps:

1. Enable zones: uncomment `#define SDPA_PROFILE_ZONES 1` at
   `sdpa_flash_decode.cpp:29`, `rm -rf ~/.cache/tt-metal-cache`, rebuild with `./build_metal.sh`.
2. Re-profile both DRAM and L1 (same trace setting on both — fairness) with the device profiler,
   regenerate the gantt.
3. Measure per-iteration cycles from `SM_NORM` end to next-chunk `QK_MM` start on TRISC_0/1/2, and
   overlay reader zones (`RD_K` start/`RD_KBAR` end) to test the read-wait hypothesis: if the gap is
   covered by an outstanding `RD_K`/`RD_KBAR`, it is reader latency for K chunk i+1; if the reader is
   idle during the gap, it is a compute-side pipeline bubble (CB sync / data-format reconfig at
   `sdpa_flash_decode.cpp:298-320`), not a read wait.

Prior expectation (from established measurement): SDPA is compute-bound and KV read is hidden behind
compute, so the gap is more likely an inter-chunk pipeline bubble than a read stall. Confirm or refute
with the enabled zones rather than the corrupted op-duration CSVs.

Note: the L1 CSV already shows `RD_K`/`RD_V` = 512 vs DRAM 2048 (4x fewer DRAM reads from L1 cache
hits) while `RD_CHUNK` = 2048 in both. That is the expected L1 hit signature and confirms the cache
works; it does not by itself change SDPA latency, consistent with the compute-bound finding.

---

## Verification (end-to-end)

- `TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 TT_LOGGER_LEVEL=Info ./python_env/bin/pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"`
- Compare steady-state (skip first ~5 warmup) ms/step and tok/s across four cells: {DRAM, L1} x {trace
  ON, trace OFF}. Success = L1 trace-ON ties or beats DRAM trace-ON; L1 trace-ON ~2x faster than L1
  trace-OFF.
- Correctness: token-accuracy or exact-match check that L1 trace output == L1 no-trace output.
- Build with `./build_metal.sh` (never `cmake --build`).

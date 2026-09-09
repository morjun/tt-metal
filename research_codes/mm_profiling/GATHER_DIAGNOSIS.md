# Matched gather diagnosis

Status: investigation in progress; **no causal verdict and no Item 19 implementation**.
Last updated 2026-09-09: 400-replay K3/4/5 and warm-up qualification complete; actual-capture graph inspection added.

## Stage status against the plan

| Plan stage | State | Where |
|---|---|---|
| §2 matched workload | **done** — both harnesses run `_make_fused_k_body`; AST identity and bit-identical K=1/K=3 outputs checked before timing | this file, "Validation" |
| §3A K sweep | **done** for K = 1-8 at 20 replays, six requested counterbalanced rounds; K=1 has five complete pairs | `MEASUREMENT_RECORD.md` §6P.23 |
| §3A same-arm control | **done** at K = 1, 3, six rounds | §6P.23 |
| §3A replay-count control | **done** at K = 1,2,3,8 x 20/100/400 replays | §6P.25 |
| §3A warm-up control | **done** at K3, 400 replays, warm-ups 3/20; no resolved differential change | 2026-09-09 campaign below |
| §3B captured-program inspection | **partial** — graph inventories at K=3,4,5 and capture manifests at all K; recorded op arguments and input addresses compared in actual-capture graphs; compiled configs, complete lifetimes and actual payload bytes still pending | §6P.24 |
| §3C qualified traced instrumentation | **not run** — needs the profiler-capacity work below | — |
| §3D targeted intervention and reversal | **not started** | — |

## Measured results, current estimators

Unprofiled, matched body, `down_proj` pinned in both arms, ring minus multicast:

| K | 20 replays | 400 replays |
|---:|---:|---:|
| 1 | +6.00 | **+3.17 [2.98, 3.44]** |
| 2 | +22.23 | **+20.13 [19.69, 20.91]** |
| 3 | +30.21 | **+29.84 [29.44, 30.14]** |
| 4 | +18.21 | **+18.08 [17.93, 18.22]** |
| 5 | +34.78 | **+33.80 [33.49, 34.07]** |
| 6 | +40.19 | not measured |
| 7 | +38.74 | not measured |
| 8 | +37.35 | **+37.90 [37.61, 38.57]** |

The 400-replay estimates are more precise. A systematic upward bias at 20
replays is not established: round-paired 20-minus-400 delta intervals include
zero at every tested K. Prefer 400 replays for precision; K3/4/5 have now been repeated there (2026-09-09), confirming the K4 dip.
The K1/2/8 entries are from 2026-09-08. K6/7 remain unmeasured at 400 replays.

The replay-count control rejects a purely batch-charged explanation of the
measured delta over K = 1,2,3,8: the per-trace delta is approximately invariant
from 20 to 400 replays (ratios 0.52, 0.97, 0.90, 0.97 against 0.05 predicted
for a batch-charged cost). A small batch-charged residual remains plausible, but its nonzero magnitude
and responsibility for low-K changes have not been established.

## Device and instrument constraints found in the campaign

1. **AICLK is not pinnable.** `tt-smi` exposes no clock control; there is no
   power-state API in the tree; the driver's `power_policy` parameter is
   read-only at runtime and governs the idle floor, not the loaded droop. A
   1350 -> 1343 MHz droop was observed once, within a cell, and stopped the
   same-arm stage. Recovery is measurement discipline, not configuration.
2. **Clock handling now has two distinct gates.** `valid_clock` covers drift
   within a cell. Pair acceptance additionally requires both arms to report the
   same clock, because a mismatched pair injects the full frequency difference
   into the delta while a matched one only rescales it. `--on-clock-change
   retry` re-runs a drifted cell, preserves every rejected capture as
   `<cell>.rejected<N>.jsonl`, and excludes rather than halts if it never
   qualifies. Default remains `abort`.
3. **The board wedged once**, mid-campaign, at device open: both boards visible
   on the PCI bus with healthy links, char devices present, but every ioctl
   returning `ENODEV`. `tt-smi -r` could not open the board to reset it;
   recovery required privileged intervention. No cell data was affected — the
   run halted on its first cell. Cause not established. The campaign opens and
   closes one process per cell, which is worth watching if it recurs.

## Reproduction

Use the specdec virtual environment, while importing this worktree:

```sh
source ~/codes/tt-metal-gemma4-specdec/python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD:$PWD/ttnn:$PWD/tools
python research_codes/mm_profiling/run_gather_campaign.py --stage validate
python research_codes/mm_profiling/run_gather_campaign.py --stage baseline
python research_codes/mm_profiling/run_gather_campaign.py --stage replays --on-clock-change retry
python research_codes/mm_profiling/run_gather_campaign.py --stage warmup
python research_codes/mm_profiling/run_gather_campaign.py --stage same --on-clock-change retry
python research_codes/mm_profiling/run_gather_campaign.py --stage graph --ks 3 4 5
python research_codes/mm_profiling/analyze_gather_campaign.py generated/gather_diagnosis --stage baseline
python research_codes/mm_profiling/analyze_gather_campaign.py generated/gather_diagnosis --stage same
python research_codes/mm_profiling/analyze_gather_campaign.py generated/gather_diagnosis --stage replays
```

The commands above reproduce the original stage choices. New campaigns must
use a new `--output` directory; the updated launcher refuses incompatible
settings or legacy directories lacking a settings manifest. Current qualification
commands appear below.

`--stage` selects which cells the analyzer reads; it defaults to `baseline`.
`--ks` overrides the stage's K list. `--on-clock-change retry` is described
under "Device and instrument constraints".

Run campaigns sequentially with exclusive device-0 access. A process-level lock
prevents these launchers overlapping; it does not exclude unrelated applications.
One live trace is used at a time. A completed cell has a `.done.json` sidecar;
failed/incomplete cells stop the campaign and require inspection before retry.
Never silently overwrite them. Record any recovery or exclusion.

The launcher removes inherited GEMMA4 experiment settings and explicitly sets
tp=1, batch=1, BF16, tuner=1, activation chaining=0. The fixture uses context 512,
maximum sequence length 1024, token 12345, and seeded hidden/KV inputs. Both arms
relocate exactly four down_proj weights after the ledger's initial DRAM capture.
The matched body includes device argmax and hidden/token recurrence. The legacy
`test_trace_command_stream_size` is a different, fixed-input workload.

Every timed batch follows at least ten seconds of idle and three stable telemetry
samples two seconds apart. Qualification requires one reported clock and a
temperature range at most 1 degree C; it times out after 120 seconds. Pre/post
telemetry is preserved. These samples do not rule out transient clock changes.
Raw THROTTLER/FAULTS fields are retained in snapshots; missing telemetry is not
evidence of no throttling.

Results are microseconds per complete trace. Bootstrap sampling uses independent
process pairs, not individual trace replays. K fits are descriptive and their
coefficients are resampled in complete round blocks. Never use the fitted shape
alone as a causal explanation. Pilot data is kept separately from the campaign.

## Validation completed before the timing campaign

- The extracted body AST is identical to the original ledger body's AST.
- Device checks passed for both arms at K=1 and K=3: eager and three successive
  traced replays produced bit-identical token and hidden outputs.
- CPU tests cover thermal qualification, recurrence/ownership, long internal
  stalls, invalid replay ranges, and independent-sample bootstrap behavior.

## Observations requiring correction in earlier records

1. The legacy command-size probe repeats fixed inputs without the external
   argmax. Its sizes/counts cannot establish those of the full recurrent ledger.
2. The local burst analyzer used the retired gap-segmentation algorithm. It now
   refuses measurements without explicit, evidenced replay ranges, and refuses
   named attribution without verified launch identities. A telescoping sum does
   not validate a boundary. Existing metadata with only `ops` is insufficient.
3. RoPE's inspected program factory chooses its core set from input/output shard
   specifications or shape-based work splitting. It does not directly query the
   L1 watermark. The earlier discrete-cost observation does not establish a
   watermark-induced grid change; indirect input/layout changes remain to test.
4. TRACE allocated bytes are distinct from actual command payload bytes. Equal
   allocation cannot exclude different commands, consumption timing, or state.
5. A ~23 us plateau is not yet established outside the original K range and is
   not a mechanism. Per-K differences of separate captures are not timestamps of
   individual steps. The profiled +16.8 us reshard number is not an unprofiled
   causal decomposition.
6. Batch-length variation with K was asserted as a thermal/DVFS confound. That
   is withdrawn: no clock variation was observed in accepted pairs, and a clock
   state common to both arms rescales a paired delta rather than adding to it
   (about 0.16 us on a 30 us delta for a 0.5% droop, against the 12 us it was
   invoked to explain). §6P.25 addressed the batch-length question directly and
   by the correct route.
7. Graph counts, capture manifests and peak-L1 metrics being equal up to scale
   at K=3,4,5 does **not** establish identical captured programs. It excludes
   count- and inventory-level differences only.
8. The legacy per-class table's "resolved" test compares a delta against
   replay-to-replay spread within an arm. It does not test the differential
   profiler perturbation later measured at about 6.5 us/step between arms at
   the same K. Against that systematic the matmul row (-26.3) has roughly 4x
   margin, the summed reshard rows (+16.8) about 2.6x, and the RoPE row
   (+4.66) **none** — a profiler effect of the measured size could produce it.
   RoPE has never been measured on the matched body: the matched campaign
   times whole traces and the graph stage records inventories only. Do not
   quote +4.66 as evidence that the ring changes untouched operations.

## Pending gates

- K3/4/5 qualification at 400 replays: **complete**, with the K4 dip confirmed.
  K6/7 at 400 replays are optional if needed for a specific causal prediction.
- Warm-up control: **complete**; no resolved change from 3 to 20 warm-ups.
- Complete §3B: effective program configurations, runtime arguments, buffer
  addresses and lifetimes, binary residency, synchronization commands, and
  **actual** trace payload bytes rather than allocated bytes.
- Qualify complete traced diagnostic captures and their arm-specific
  perturbation. §6P.21 measured that perturbation as **differential between the
  arms** (about 6.5 us/step at K=1), so profiled captures are usable for
  within-arm structure and not for arm comparison until re-qualified.
- Run a targeted intervention and reversal supporting a causal mechanism.

No mid-run profiler drain is permitted. Source inspection found
`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`, feeding both profiler allocation and
`PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC` JIT definitions; availability in the
loaded build must be checked before use. This is not the nonexistent
`PROFILER_DRAM_BUFFER_SIZE` macro. Current Python and build artifacts are symlinked
to the specdec checkout: never rebuild through those links for an isolated probe.

The investigation is complete only after a causal intervention succeeds and its
reversal restores the effect. Instrument limitations are blockers, not a reason
to implement Item 19 or declare an inferred cause.


## 2026-09-09 qualification campaign — complete

New artifacts: `generated/gather_qualification_20260909`. Existing captures are
preserved. `--replays` now controls timed counts for every timing stage; a
settings manifest rejects incompatible resumption. Warm-up order reverses by
round, alongside existing arm counterbalancing.

```sh
python research_codes/mm_profiling/run_gather_campaign.py --stage baseline --ks 3 4 5 --replays 400 --rounds 6 --output generated/gather_qualification_20260909 --on-clock-change retry
python research_codes/mm_profiling/run_gather_campaign.py --stage warmup --ks 3 --replays 400 --rounds 6 --output generated/gather_qualification_20260909 --on-clock-change retry
```

Run sequentially. Primary predeclared contrasts: within-round delta(K4)-delta(K3)
and delta(K4)-delta(K5), with process-round bootstrap intervals. Warm-up contrast:
delta(warmup20)-delta(warmup3) at K3, also paired by round. These are differences
of paired deltas, not differences of marginal medians. Six process rounds remain
a small sample; intervals do not bound systematic arm-specific effects.

Warm-ups precede stabilization idle in this harness. This control tests execution
history retained across that idle; it does not establish a thermally warm start.
A dip persisting at 400 replays directs program inspection to K3/4/5; otherwise
retire the dip as a diagnostic clue and focus on the robust K3 regression.
No Item 19 implementation or attribution is authorized by a curve shape alone.


### Qualified K3/4/5 timing result

Six accepted process pairs per K; 400 replays; all reported clocks 1350 MHz;
no clock retries or pair exclusions. Full output: `baseline_analysis.json` in
the qualification directory.

| K | Ring-minus-mcast, us/trace | 95% bootstrap CI |
|---:|---:|---|
| 3 | +29.836 | [29.444, 30.139] |
| 4 | +18.083 | [17.932, 18.216] |
| 5 | +33.796 | [33.493, 34.070] |

Round-paired contrasts: delta(K4)-delta(K3) = **−11.891** [−12.003, −11.293]
us; delta(K4)-delta(K5) = **−15.693** [−15.972, −15.463] us. The K4 dip survives
400-replay qualification. These intervals describe independent-round variation,
not systematic instrument uncertainty, and do not identify a causal mechanism.

Read-only graph/source inspection and the next actual-capture metadata fields:
[`GATHER_CAPTURE_INSPECTION.md`](GATHER_CAPTURE_INSPECTION.md). The old eager graph
shows a ring-K4 address difference but cannot establish the timed trace's layout.


### Warm-up and capture inspection results

At K3 and 400 replays, six paired rounds give +29.864 [29.367, 30.198] us
with three warm-ups and +29.578 [29.337, 29.789] with twenty. The paired
change is **−0.307 [−0.810, +0.392] us/trace**: no resolved differential effect.
All 24 warm-up cells completed, with no clock retries or pair exclusions.

Two untimed actual-capture graph rounds (12 cells) followed. Recorded operation
arguments match across K3/4/5 within each arm. Ring K4's input-address difference
appears in round 0 but not round 1; therefore the old address pattern is not a
repeatable K4-specific invariant. Details, raw paths and the preserved failed
initial instrumentation attempt are in `GATHER_CAPTURE_INSPECTION.md`.

Next: collect ordered dispatch-node metadata and actual command bytes without
reading device tensors during trace capture, then localize a candidate and
perform a controlled intervention/reversal. The source's unordered RoPE-cache
construction is a possible address-layout confound; record/control that order.
No cause has been established and Item 19 remains deferred.

## Standard wall-clock reporting — 2026-09-09

`analyze_gather_campaign.py` now always includes absolute multicast/gather
summaries in JSON for A/B timing stages, alongside the original paired deltas.
Use `--markdown REPORT.md` for a readable table and within-arm K comparisons.
Same-arm controls are not relabeled as multicast/gather: their physical arm
alternates by round, so that stage retains its slot-paired analysis.

```sh
python research_codes/mm_profiling/analyze_gather_campaign.py generated/gather_qualification_20260909 --stage baseline --markdown generated/gather_qualification_20260909/baseline_wallclock.md
```

The qualification directory also contains `warmup_wallclock.md`,
`legacy_baseline_wallclock.md` (20-replay K1–8), and
`replay_control_wallclock.md` (20/100/400 controls). Each includes source-group
settings; campaigns are not pooled. Absolute values are host replay-loop wall
time plus final synchronization, divided by replay count; setup is excluded.
Arm medians use the same accepted pairs as the paired-delta estimator.

Adjacent-K increments and centered midpoint residuals are computed within common
process rounds at identical replay/warm-up settings and equal reported clocks
across K. Cross-K clock exclusions are reported. Missing K values are not bridged
as one-step increments. Separate-capture differences are not individual-step
timestamps, and bootstrap intervals do not bound systematic measurement error.

| K | Multicast us/trace | Gather us/trace | Paired delta us/trace |
|---:|---:|---:|---:|
| 3 | 3164.042 | 3193.876 | +29.836 |
| 4 | 4224.550 | 4242.645 | +18.083 |
| 5 | 5270.317 | 5304.051 | +33.796 |

At 400 replays, multicast increments K3→4 and K4→5 are 1060.510 and
1045.736 us; gather increments are 1048.667 and 1061.360 us. Relative to
`T4 − (T3+T5)/2`, multicast is **+7.427 [7.199, 7.492] us** and gather is
**−6.346 [−6.552, −6.142] us**, each over six matched rounds with no cross-K
clock exclusions. Both arms contribute to the differential dip relative to this
local linear reference. This is not a causal decomposition or evidence that an
individual fourth step has either cost. Exact cause remains open.

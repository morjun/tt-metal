# Pre-registration: final-use placement intervention (gate 2)

Written **after** gate 1's host-side capture and **before** any timing cell of the
intervention has been run or looked at. Timestamp at the end.

## What the intervention is

`GEMMA4_DISABLE_TERMINAL_TOPDOWN=1` with `GEMMA4_TERMINAL_TOPDOWN_TRACE=1`.
The allocator's terminal-use exception

    bool binary_top_down = !extra_data_[i].next_use_idx[ExtraData::kBinary].has_value();

is disabled for the measured pinned trace only: binaries with no later use are
allocated bottom-up through the existing allocator. The L1 region size, the
eviction rule, the reset rule and every synchronization decision are untouched.

## What gate 1 established (host-side, no timing)

- The harness's initial DRAM trace is **byte-identical** in all 12 (K, arm) pairs.
- Both arms produce correct output under base and under the intervention at every K.
- **Containment is exact**: interior decision differences are 0, and the recorded
  command stream first diverges at precisely the byte where the final step begins.
- The override fires on 45-48 binaries per cell, all inside the final step.
- The targeted quantity moves. Final-step memory-reuse-bound ("tight") wait count,
  ring minus multicast:

| K | 3 | 4 | 5 | 6 | 7 | 8 |
|---|--:|--:|--:|--:|--:|--:|
| base | +4 | **-1** | +1 | +2 | +1 | **-1** |
| intervention | +3 | **+2** | +2 | +3 | +3 | **+1** |

  **The intervention removes the negatives at K=4 and K=8**, which are exactly the
  two K at which §6P.36 found the ring arm asymmetrically favoured. Under the
  intervention the differential is positive at every K in 3..8.
- The multicast final-step allocator reset is removed at K=3, 4, 6 and 8, and is
  unchanged at K=5 and 7. The ring arm's resets are all interior and are never
  changed by a final-step-only override, as expected.

## The predictions, stated before the timings are seen

Measured baseline penalty (ring - multicast, us/trace), §6P.23/§6P.26:
K=3 +29.88, K=4 +18.21, K=5 +34.78 (K=4/5 at 400 replays per §6P.26), with the
**dip depth** defined as in §6P.35: `mean(K3, K5) - K4`.

**H1 -- the mechanism explains the dip.** If the final-step reset/wait asymmetry
causes the dip, then removing that asymmetry should remove or reduce the dip:

  - the K=4 penalty rises toward its K=3 and K=5 neighbours;
  - the **paired dip depth decreases** by an amount comparable to its baseline
    value (baseline 13.88 in §6P.35's units);
  - the sign of the change in dip depth is **negative** in a majority of the six
    counterbalanced rounds.

**H0 -- the mechanism does not explain the dip.** The paired dip depth is
unchanged within its round-to-round range, or increases, as it did under §6P.35's
forced-send intervention (+3.27, all six rounds positive).

**What would NOT establish H1.** A generic speedup or slowdown in both arms at all
K. The intervention adds and removes commands, so an absolute per-arm response is
expected and carries no information about the mechanism. Only the **paired K=4 dip
change** discriminates.

**Directional detail.** Gate 1 predicts the differential becomes positive at K=4,
i.e. the ring arm should lose its K=4 relative advantage. So under H1 the K=4
penalty should **increase**, not decrease. A K=4 penalty that decreases under the
intervention is evidence against H1 even if the dip changes.

## Protocol, fixed in advance

- Unprofiled, `GEMMA4_DIAG_MODE=timing`, warmup 3, **400 replays**, no trace dump
  and no allocmap active.
- K in {3, 4, 5}, both arms, both conditions = 12 cells per round.
- **Six counterbalanced rounds**; within a round the condition order alternates so
  base and intervention are not systematically first.
- Occupancy sampled before, during and after every cell. Contaminated cells are
  flagged, never silently dropped, and another user's processes are never signalled.
- Reported quantities, all three: **absolute per-arm response**
  (intervention - base, per arm per K), **paired differential response**
  (change in ring - multicast, within round), and the **paired K=4 dip change**
  with its observed min-max range across the six rounds, labelled as a range and
  not a confidence interval.

## Withdrawal condition

If the timing cells show any round in which base and intervention differ in output
correctness, or if the occupancy log shows a foreign device holder during any cell,
that round is reported as contaminated and the campaign is repeated rather than
reinterpreted.
2026-09-10T19:54:17+09:00

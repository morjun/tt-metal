# Pre-declared K=10 prediction, written before the K=10 capture completed

Timestamp below. The K=3..9 captures fix two periodic schedules:

- multicast arm: 3 allocator resets at every EVEN interior step >= 2, plus 1 in the final step
- ring arm: 1 allocator reset at each interior step congruent to 1 mod 3 (steps 1, 4, 7, ...)
- final-step memory-reuse-bound wait count: multicast 4 (K odd) / 6 (K even); ring 8 (K = 0 mod 3) / 5 otherwise

Predictions for K=10, made before the data exists:

| quantity | multicast | ring | ring - multicast |
|---|--:|--:|--:|
| allocator resets in the trace | 13 | 3 | -10 |
| final-step memory-reuse-bound waits | 6 | 5 | -1 |

If the observed values differ, the periodic description is wrong and every
K-extrapolation built on it in this section must be withdrawn.
2026-09-10T19:09:26+09:00

## Correction to the pre-registration claim

At the moment of writing, `cwmap/k10_a0/alloc.jsonl` ALREADY EXISTED on disk (the
multicast cell had finished). It had not been read or analysed, but "written before
the data existed" is false for the multicast column. Only the **ring** column was
genuinely pre-registered: `k10_a1` was still running.

Treat the multicast K=10 numbers as an unexamined-data prediction, which is weaker,
and the ring K=10 numbers as pre-registered.

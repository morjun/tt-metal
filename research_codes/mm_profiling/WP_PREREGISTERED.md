# Pre-registration: wait-placement experiment (6P.39)

Written after the audit gate passed and before any timing cell was run. K=3 ring only.

## Conditions (all four timed, four fresh processes per round)

| condition | allocator | added waits |
|---|---|---|
| `ringbase` | original (terminal-use top-down) | none |
| `bottomup` | `GEMMA4_DISABLE_TERMINAL_TOPDOWN` | none |
| `early` | bottom-up | nodes **413, 449**, targets 44990 / 48950 |
| `late` (control) | bottom-up | nodes **445, 481**, same targets |

`early` restores the two waits at the positions they occupy under the original
allocator. `late` adds the same two commands where their targets are already
satisfied, so it carries the added-command cost with no added ordering constraint.

**Control deviation, recorded.** The audit asked for the control "immediately
after the existing equivalent waits", i.e. nodes 416 and 452. That is not
implementable: both already carry their own waits, and a node holds exactly one
stall command sequence with one patched target, so injecting there would REPLACE
a required wait. 445 and 481 are the nearest LATER wait-free nodes. Verified
equal cost: both conditions add exactly 128 bytes (2 x 64 B; the stall sequence
size is program-independent).

## Audit gate, passed before timing

DRAM trace byte-identical in all four; only the intended nodes change and only by
gaining a wait; no other node's wait changes; command-byte delta exactly +128 for
`early` and `late` and 0 for `bottomup`; all four produce correct output.

## Predeclared prediction

If delaying these two waits contributes to the K=3 ring speedup, **`early` is
slower than `late`**. The paired `early - late` response estimates sensitivity to
these two wait positions under bottom-up allocation. It is **not** automatically
an additive component of the ring/multicast gap.

**No response rejects this particular candidate, not all synchronization
effects.** A response that is present but far smaller than 7.59 us leaves most of
the bottom-up ring speedup unexplained, and that must be said rather than
absorbed.

## Anchors that must reproduce

`bottomup - ringbase` must reproduce §6P.38's ring absolute response of
**-7.59 us** [-8.47, -7.18]. If it does not, the campaign is void.

## Protocol

Six paired rounds, condition order reversed on even rounds, 400 replays,
unprofiled, fresh process per cell, explicit environment cleanup before each,
endpoint clock recorded, per-cell occupancy sampled. All cells preserved and
flagged; none dropped. Primary outcome is absolute K=3 ring trace time. No dip
criterion, no K sweep. No automatic Item 19 transition follows any result.
2026-09-10T22:29:25+09:00

# Pre-registration — drafter L1 `down_proj` pinning + `gather_in0` on Gemma-4 **12B**

Written **before** the first cell, 2026-09-17, at `ff2d19494a7`. The point of writing it first is
that the predicted outcome here is a **negative** one, and a negative result is only citable if the
prediction preceded the measurement.

## Question

Does the E2B drafter-scope intervention — `down_proj` pinned L1 WIDTH_SHARDED, with and without
the `gather_in0` ring — reproduce on the **12B** drafter at 1x1, bf16, K=3?

## The arithmetic, committed in advance

Per-bank pin cost is `K x 32 x per_core_N x elem`, `per_core_N = Nt / cores`. Cost falls as the
grid grows but bottoms out when N is exhausted.

12B `down_proj`: K=8192, N=1024 -> Kt=256, Nt=32. `_pick_grid(32)` = 8x4 = **32 cores** (64 does
not divide 32), so `per_core_N = 1` and the charge is at its floor:

> **8192 x 32 x 2 = 524,288 B = 512.0 KiB / bank / layer**

E2B for contrast: K=2048 -> 128.0 KiB/bank/layer, 8 cores.

Both L1 arms share this floor: the ring's weight spec is `[k, n//cores]` (`matmul_tuning.py:224`)
and `shard_l1_width`'s is `ShardSpec(grid, [k, shard_w])` (`weight_placement.py:372`) — **full K on
every core** in both. Lowering it needs split-K with a cross-core reduction, which contradicts the
ring's design and is not implemented anywhere in this codebase.

## Predictions (falsifiable, stated before running)

| # | prediction | how it is refuted |
|--:|---|---|
| P1 | **n=4 and n=3 are impossible**: 2048 and 1536 KiB/bank exceed the ~1496 KiB top of allocatable L1 | any FIT at n>=3 |
| P2 | **n\* = 0 at the default SDPA cap** (512 KiB vs a 320 KiB allowance) | a FIT at n=1, unset `GEMMA4_SDPA_MAX_CORES` |
| P3 | **n\*′ = 1 at `GEMMA4_SDPA_MAX_CORES=1`** (~590 KiB allowance) | no FIT at n=1 even with the knob |
| P4 | failures at n>=3 surface as an **allocator OOM**; at n=1,2 as a **CB clash** | any other exception class |
| P5 | the ring is **grid-eligible** for this shape (`_pick_grid == _pick_grid_2 == 32`) | a `[gather] SKIP` for 8192x1024 |
| P6 | the 12B op table has **no `TopK`** (dense head, `use_ordered_embeddings: false`) | `TopK` present in the per-op CSV |

**If P2 and P3 both hold, the headline deliverable is the capacity wall itself**, not a timing
number, and the campaign stops after the gate unless HYBRID is pursued.

## Cells

**G0** — `test_profile_eager_step`, `GEMMA4_L1_ARM=dram`. Confirms the harness runs a **non-CME**
assistant at all (never previously exercised) and prints the tuner's grid for K=8192,N=1024.

**G1** — the ladder: `GEMMA4_L1_ARM=l1_sharded GEMMA4_L1_ONLY=down_proj
GEMMA4_L1_WEIGHT_BUDGET_MB=128`, `GEMMA4_L1_LAYERS` in {`0`, `0-1`, `0-2`, `0-3`} x
`GEMMA4_SDPA_MAX_CORES` in {unset, 1}. Fresh process per cell.

**G2** — ring engagement at the best n, `GEMMA4_GATHER_IN0=1`.

Classification is fixed in advance: **FIT** / allocator-OOM / CB-clash / config-bug / hard-crash.

## Fixed environment

```
MESH_DEVICE=P150  TT_VISIBLE_DEVICES=0  ARCH_NAME=blackhole  PYTHONHASHSEED=0
GEMMA4_ASSISTANT_MODEL=google/gemma-4-12B-it-assistant
GEMMA4_TUNE_MATMULS=1  GEMMA4_SHARD_ACTIVATIONS=0
unset: GEMMA4_PRECISION GEMMA4_MM_BLOCK_CAP GEMMA4_WEIGHTS_IN_L1 GEMMA4_KEEP_GATHER_IN0
```

bf16 throughout — the no-precision-change rule is in force, so **bfp8 is excluded** even though it
would fit (272 KiB/bank would clear the default budget).

## Two traps that would silently corrupt an arm

1. **Budget truncation.** Each `down_proj` is exactly 16.00 MiB and the default budget is 32 MB
   with a `<=` test, so `l1_sharded` would pin **exactly 2 of 4** while still reporting
   "L1 pinned". Hence `GEMMA4_L1_WEIGHT_BUDGET_MB=128` in every gate cell, and a `memory_config()`
   readback rather than trusting the placement log.
2. **The interleaved arm holds more.** `l1` charges `16 MiB / 110 banks` = 152 KiB/bank/layer and
   can hold 4 where sharded holds 0. Both L1 arms must be pinned to the same `GEMMA4_L1_LAYERS`.

## Recording

`gemma4-12b/MEASUREMENT_RECORD.md` §5 (capacity), §6 (wall clock), §7 (per-op) — **not** the E2B
parent. Tags `CFG-DRAFT-12B-tp1` and, if the SDPA knob is used, `CFG-DRAFT-12B-tp1-sdpa1`.
All cells preserved and flagged; none dropped.

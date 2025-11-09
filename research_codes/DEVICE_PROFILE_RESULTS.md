# Device Profile Analysis Results

## Overview
This document contains the **correct** device-level profiling analysis for weight loading scenarios using Tenstorrent's device profiler, with proper parallel work calculation.

## Hardware Configuration
- **Device**: Tenstorrent Blackhole P150A
- **Clock Frequency**: 1.35 GHz
- **Tensix Cores**: 130 (13×10 grid)
- **RISC Processors per Core**: 5 (1 BRISC + 1 NCRISC + 3 TRISC)
- **Maximum Theoretical Parallelism**: 130 cores × 5 RISCs = **650x**

## Profiling Method
- **Tool**: `TT_METAL_DEVICE_PROFILER=1` environment variable
- **Output**: `generated/profiler/.logs/profile_log_device.csv`
- **Measurement**: Cycle-accurate timestamps at 1.35 GHz
- **Correct Calculation**: Parallel work = timeline span × num_cores (NOT sum of zone durations!)

## Test Configuration
- **Model**: Single linear layer (4096 → 4096)
- **Weight**: 64 MB (4096 × 4096 × FP32)
- **Large batch**: B=256 (1 forward pass)
- **Mini-batch**: b=32 (8 forward passes)

---

## Large Batch Scenario (B=256, 1 forward pass)

### Python-level Timing
```
Forward (ms): 0.239
```

### Device Profile Results

**Operation Analysis**:
- Total zones: 1,280
- Wall clock time: **0.167 ms**
- Cores used: **128 cores** (98.5% of 130 available)

### Parallel Work Breakdown (Correct Calculation)

**Formula**: Work = timeline_span × num_cores

| Component | Timeline Span | Cores | Parallel Work | % of Total |
|-----------|---------------|-------|---------------|------------|
| **Weight Streaming (BRISC)** | 0.167 ms | 128 | 21.38 ms | 35.2% |
| **NoC Communication (NCRISC)** | 0.153 ms | 128 | 19.62 ms | 32.3% |
| **Computation (TRISC)** | 0.154 ms | 128 | 19.74 ms | 32.5% |
| **TOTAL** | - | - | **60.74 ms** | 100.0% |

**Effective Parallelism**: 363.7x (60.74 ms / 0.167 ms)
- Theoretical maximum: 650x
- **Efficiency: 56.0%**

**Interpretation**:
- All components run in **parallel** with high overlap
- Wall clock (0.167 ms) < Python time (0.239 ms) ✓
- Sync overhead: 0.072 ms (30.1% of Python time)
- Work distribution is balanced: ~33% each component

---

## Mini-batch Scenario (b=32, 8 forward passes)

### Python-level Timing
```
Forward sum (ms): 1.440
Per-forward: 0.180 ms
```

### Device Profile Results

**Operation Analysis**:
- Total zones: 14,400 (8 forwards)
- Total wall clock time: **1.486 ms** (for all 8 forwards)
- Per-forward average: **0.186 ms**
- Cores used: **130 cores** (100% utilization!)

### Parallel Work Breakdown (8 forwards total)

| Component | Timeline Span | Cores | Parallel Work | % of Total |
|-----------|---------------|-------|---------------|------------|
| **Weight Streaming (BRISC)** | 1.868 ms | 130 | 242.84 ms | 34.0% |
| **NoC Communication (NCRISC)** | 1.866 ms | 130 | 242.56 ms | 33.9% |
| **Computation (TRISC)** | 1.794 ms | 128 | 229.69 ms | 32.1% |
| **TOTAL** | - | - | **715.09 ms** | 100.0% |

**Effective Parallelism**: 382.8x (715.09 ms / 1.868 ms)
- Theoretical maximum: 650x
- **Efficiency: 58.9%**

### Per-forward Average

| Component | Parallel Work (per forward) |
|-----------|----------------------------|
| **Weight Streaming** | 30.36 ms |
| **NoC Communication** | 30.32 ms |
| **Computation** | 28.71 ms |
| **TOTAL** | **89.39 ms** |

**Per-forward Parallelism**: 481.3x (efficiency: 74.0%)

**Interpretation**:
- Mini-batch uses **MORE cores** than large batch (130 vs 128)
- Higher parallelism efficiency per forward (74.0% vs 56.0%)
- But requires 8 forwards to process same amount of data

---

## Comparison: Large Batch vs Mini-batch

### Per-forward Work (Fair Comparison)

| Component | Large Batch | Mini-batch (per fwd) | Ratio |
|-----------|-------------|---------------------|-------|
| **Weight Streaming** | 21.38 ms | 30.36 ms | 1.42x |
| **NoC Communication** | 19.62 ms | 30.32 ms | 1.55x |
| **Computation** | 19.74 ms | 28.71 ms | 1.45x |
| **TOTAL WORK** | 60.74 ms | 89.39 ms | **1.47x** |
| **Wall Clock** | 0.167 ms | 0.186 ms | 1.11x |
| **Parallelism** | 363.7x | 481.3x | 1.32x |

**Key Finding**: Mini-batch performs **47% more work per forward** despite having:
- Batch size 1/8 of large batch (32 vs 256)
- Same or more cores used (130 vs 128)

### Why Mini-batch Uses More Work?

**Root Cause**: **Mini-batch uses the same number of cores (128-130) as large batch, despite processing 1/8 the data!**

Expected behavior:
- Large batch (B=256): 128 cores → 2 samples per core
- Mini-batch (b=32): Should use 16 cores → 2 samples per core

Actual behavior:
- Mini-batch (b=32): Uses **130 cores** → 0.25 samples per core!

**This is 8x over-parallelization**, causing:
1. More weight streaming work (each core loads weights)
2. More NoC communication (inter-core coordination)
3. Less efficient per-core utilization

### Total Time Comparison (256 elements)

| Metric | Large Batch | Mini-batch | Ratio |
|--------|-------------|------------|-------|
| **Device time** | 0.167 ms | 1.486 ms | **8.9x slower** |
| **Python time** | 0.239 ms | 1.440 ms | **6.0x slower** |
| **Total Work** | 60.74 ms | 715.09 ms | **11.8x more work** |

---

## Key Findings

### 1. Incorrect Parallelization Strategy

**Mini-batch uses 8x more cores than needed:**
- Expected: 32 samples → 16 cores (with 2 samples/core)
- Actual: 32 samples → 130 cores (with 0.25 samples/core!)

This causes:
- 47% more work per forward
- 8.9x slower total time for same amount of data

### 2. Work Distribution is Consistent

Both scenarios show similar distribution:
- Weight streaming: ~34%
- NoC communication: ~33%
- Computation: ~33%

This indicates **balanced parallel execution**, not a weight streaming bottleneck.

### 3. Parallelism Efficiency

- Large batch: 56.0% efficiency (363.7x / 650x)
- Mini-batch per-forward: 74.0% efficiency (481.3x / 650x)

Mini-batch is actually **more efficient per-forward** when normalized by work, but performs unnecessary work due to over-parallelization.

### 4. Weight Streaming is NOT the Bottleneck

Weight streaming occupies:
- Large batch: 35.2% of total work
- Mini-batch: 34.0% of total work

**Computation (32%) is similar** to weight streaming, indicating balanced workload.

The real bottleneck is the **inefficient resource allocation strategy** that uses 8x more cores than necessary for mini-batch.

### 5. User's Hypothesis Validation

**Hypothesis**: "If weight streaming is not a bottleneck, mini-batch and large batch should have similar weight streaming times"

**Result**:
- Mini-batch weight streaming per forward: 30.36 ms
- Large batch weight streaming: 21.38 ms
- Difference: **42% more work** in mini-batch

**Conclusion**: Weight streaming time is NOT similar because mini-batch uses **8x more cores than needed**, causing each core to load weights even though it processes very little data (0.25 samples/core).

This confirms that:
1. Weight streaming itself is NOT inefficient (balanced at 34% of work)
2. The problem is **over-parallelization** - using 130 cores for 32 samples
3. If mini-batch used only 16 cores, weight streaming time would be ~2.4 ms (8x less), which is LESS than large batch (21.38 ms)!

---

## Conclusion

### The Real Problem: Over-Parallelization

**Mini-batch uses 8x more cores than necessary:**
- 130 cores for 32 samples = 0.25 samples/core
- Should use 16 cores for 32 samples = 2 samples/core (same as large batch)

This causes:
- **47% more work per forward** (89.39 ms vs 60.74 ms)
- **8.9x slower total time** (1.486 ms vs 0.167 ms)
- **11.8x more total work** (715 ms vs 61 ms)

### Weight Streaming is Well-Optimized

Weight streaming occupies **34% of total work**, balanced with:
- NoC communication: 33%
- Computation: 33%

This indicates **efficient parallel execution**, not a weight streaming bottleneck.

### The Fix

To make mini-batch efficient, ttnn should:
1. **Scale cores with batch size**: 32 samples → 16 cores (not 130)
2. **Maintain samples-per-core ratio**: Keep 2 samples/core like large batch
3. **Result**: Mini-batch would be ~8x faster with ~8x less work

### Bottom Line

**Weight streaming is NOT the bottleneck.** The issue is ttnn's sharding strategy that fails to reduce core usage proportionally with batch size, causing massive over-parallelization and wasted work.

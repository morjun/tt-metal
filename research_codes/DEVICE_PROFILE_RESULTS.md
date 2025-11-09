# Device Profile Analysis Results

## Overview
This document contains device-level profiling analysis for weight loading scenarios using Tenstorrent's device profiler, with proper parallel work calculation.

**Generated automatically by `run_full_device_profile.py`**

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
- Wall clock time: **0.167 ms**
- Cores used: **128 cores**

### Parallel Work Breakdown (Correct Calculation)

**Formula**: Work = timeline_span × num_cores

| Component | Parallel Work | % of Total |
|-----------|---------------|------------|
| **Weight Streaming (BRISC)** | 21.38 ms | 35.2% |
| **NoC Communication (NCRISC)** | 19.64 ms | 32.3% |
| **Computation (TRISC)** | 19.76 ms | 32.5% |
| **TOTAL** | **60.78 ms** | 100.0% |

**Effective Parallelism**: 363.9x (60.78 ms / 0.167 ms)
- Theoretical maximum: 650x
- **Efficiency: 56.0%**

**Interpretation**:
- All components run in **parallel** with high overlap
- Wall clock (0.167 ms) vs Python time (0.239 ms)
- Work distribution is balanced: ~33% each component

---

## Mini-batch Scenario (b=32, 8 forward passes)

### Python-level Timing (8 forwards total)
```
Total: 1.439 ms
Per-forward: 0.180 ms
```

### Device Profile Results

**Operation Analysis**:
- Total wall clock time: **1.862 ms** (timeline span, includes overlaps)
- Cores used: **130 cores** (100% utilization!)

### Parallel Work Breakdown (8 forwards total)

| Component | Parallel Work (total) | Parallel Work (per forward) | % of Total |
|-----------|----------------------|---------------------------|------------|
| **Weight Streaming (BRISC)** | 242.12 ms | 30.27 ms | 271.6% |
| **NoC Communication (NCRISC)** | 241.85 ms | 30.23 ms | 271.3% |
| **Computation (TRISC)** | 229.10 ms | 28.64 ms | 257.0% |
| **TOTAL** | **89.13 ms** | **89.13 ms** | 100.0% |

**Effective Parallelism**: 382.9x (total), 480.5x (per forward)
- Theoretical maximum: 650x
- **Efficiency: 58.9% (total), 73.9% (per forward)**

**Interpretation**:
- Mini-batch uses **MORE cores** than large batch (130 vs 128)
- Higher parallelism efficiency per forward (73.9% vs 56.0%)
- But requires 8 forwards to process same amount of data

---

## Comparison: Large Batch vs Mini-batch

### Per-forward Work (Fair Comparison)

| Component | Large Batch | Mini-batch (per fwd) | Ratio |
|-----------|-------------|---------------------|-------|
| **Weight Streaming** | 21.38 ms | 30.27 ms | 1.42x |
| **NoC Communication** | 19.64 ms | 30.23 ms | 1.54x |
| **Computation** | 19.76 ms | 28.64 ms | 1.45x |
| **TOTAL WORK** | 60.78 ms | 89.13 ms | **1.47x** |
| **Wall Clock** | 0.167 ms | 0.185 ms | 1.11x |
| **Parallelism** | 363.9x | 480.5x | 1.32x |

**Key Finding**: Mini-batch performs **46.6% more work per forward** despite having:
- Batch size 1/8 of large batch (32 vs 256)
- Same or more cores used (130 vs 128)

### Why Mini-batch Uses More Work?

**Root Cause**: Mini-batch uses the same number of cores (130) as large batch (128), despite processing 1/8 the data!

Expected behavior:
- Large batch (B=256): 128 cores → 2.0 samples per core
- Mini-batch (b=32): Should use ~16 cores → 2 samples per core

Actual behavior:
- Mini-batch (b=32): Uses **130 cores** → 0.25 samples per core!

**This is ~8x over-parallelization**, causing:
1. More weight streaming work (each core loads weights)
2. More NoC communication (inter-core coordination)
3. Less efficient per-core utilization

### Total Time Comparison (256 elements)

| Metric | Large Batch | Mini-batch (×8) | Ratio |
|--------|-------------|-----------------|-------|
| **Device time** | 0.167 ms | 1.484 ms | **8.9x slower** |
| **Python time** | 0.239 ms | 1.439 ms | **6.0x slower** |
| **Total Work** | 60.78 ms | 713.04 ms | **11.7x more work** |

---

## Key Findings

### 1. Incorrect Parallelization Strategy

**Mini-batch uses ~8x more cores than needed:**
- Expected: 32 samples → ~16 cores (with 2 samples/core)
- Actual: 32 samples → 130 cores (with 0.25 samples/core!)

This causes:
- 46.6% more work per forward
- 8.9x slower total time for same amount of data

### 2. Work Distribution is Consistent

Both scenarios show similar distribution:
- Weight streaming: ~35% (large), ~34% (mini)
- NoC communication: ~32% (large), ~34% (mini)
- Computation: ~33% (large), ~32% (mini)

This indicates **balanced parallel execution**, not a weight streaming bottleneck.

### 3. Parallelism Efficiency

- Large batch: 56.0% efficiency (363.9x / 650x)
- Mini-batch per-forward: 73.9% efficiency (480.5x / 650x)

Mini-batch is actually **more efficient per-forward** when normalized by work, but performs unnecessary work due to over-parallelization.

### 4. Weight Streaming is NOT the Bottleneck

Weight streaming occupies:
- Large batch: 35.2% of total work
- Mini-batch: 34.0% of total work

**Computation is similar** to weight streaming, indicating balanced workload.

The real bottleneck is the **inefficient resource allocation strategy** that uses ~8x more cores than necessary for mini-batch.

---

## Conclusion

### The Real Problem: Over-Parallelization

**Mini-batch uses ~8x more cores than necessary:**
- 130 cores for 32 samples = 0.25 samples/core
- Should use ~16 cores for 32 samples = 2 samples/core (same as large batch)

This causes:
- **46.6% more work per forward** (89.13 ms vs 60.78 ms)
- **8.9x slower total time** (1.484 ms vs 0.167 ms)
- **11.7x more total work** (713.0 ms vs 60.8 ms)

### Weight Streaming is Well-Optimized

Weight streaming occupies **~35% of total work**, balanced with:
- NoC communication: ~32%
- Computation: ~33%

This indicates **efficient parallel execution**, not a weight streaming bottleneck.

### The Fix

To make mini-batch efficient, ttnn should:
1. **Scale cores with batch size**: 32 samples → ~16 cores (not 130)
2. **Maintain samples-per-core ratio**: Keep ~2 samples/core like large batch
3. **Result**: Mini-batch would be ~8x faster with ~8x less work

### Bottom Line

**Weight streaming is NOT the bottleneck.** The issue is ttnn's sharding strategy that fails to reduce core usage proportionally with batch size, causing massive over-parallelization and wasted work.

---

## Raw Data

### Large Batch
```json
{
  "wall_clock_ms": 0.167012,
  "python_ms": 0.239,
  "weight_streaming_ms": 21.38,
  "noc_communication_ms": 19.64,
  "computation_ms": 19.76,
  "total_work_ms": 60.78,
  "parallelism": 363.9,
  "efficiency_pct": 56.0
}
```

### Mini-batch
```json
{
  "wall_clock_ms": 1.862499,
  "weight_streaming_ms": 242.12,
  "noc_communication_ms": 241.85,
  "computation_ms": 229.1,
  "total_work_ms": 89.13,
  "parallelism": 382.9,
  "efficiency_pct": 58.9,
  "per_fwd_wall_ms": 0.185497,
  "per_fwd_python_ms": 0.179875,
  "per_fwd_work_ms": 89.13,
  "per_fwd_parallelism": 480.5,
  "per_fwd_efficiency_pct": 73.9,
  "python_ms": 0.179875
}
```

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
| **Weight Streaming (BRISC)** | 21.33 ms | 35.2% |
| **NoC Communication (NCRISC)** | 19.61 ms | 32.3% |
| **Computation (TRISC)** | 19.74 ms | 32.5% |
| **TOTAL** | **60.68 ms** | 100.0% |

**Effective Parallelism**: 364.2x (60.68 ms / 0.167 ms)
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
- Total wall clock time: **1.874 ms** (timeline span, includes overlaps)
- Cores used: **130 cores** (100% utilization!)

### Parallel Work Breakdown (8 forwards total)

| Component | Parallel Work (total) | Parallel Work (per forward) | % of Total |
|-----------|----------------------|---------------------------|------------|
| **Weight Streaming (BRISC)** | 243.66 ms | 30.46 ms | 271.7% |
| **NoC Communication (NCRISC)** | 243.39 ms | 30.42 ms | 271.4% |
| **Computation (TRISC)** | 230.37 ms | 28.80 ms | 256.9% |
| **TOTAL** | **89.68 ms** | **89.68 ms** | 100.0% |

**Effective Parallelism**: 382.8x (total), 479.8x (per forward)
- Theoretical maximum: 650x
- **Efficiency: 58.9% (total), 73.8% (per forward)**

**Interpretation**:
- Mini-batch uses **MORE cores** than large batch (130 vs 128)
- Higher parallelism efficiency per forward (73.8% vs 56.0%)
- But requires 8 forwards to process same amount of data

---

## Comparison: Large Batch vs Mini-batch

### Per-forward Work (Fair Comparison)

| Component | Large Batch | Mini-batch (per fwd) | Ratio |
|-----------|-------------|---------------------|-------|
| **Weight Streaming** | 21.33 ms | 30.46 ms | 1.43x |
| **NoC Communication** | 19.61 ms | 30.42 ms | 1.55x |
| **Computation** | 19.74 ms | 28.80 ms | 1.46x |
| **TOTAL WORK** | 60.68 ms | 89.68 ms | **1.48x** |
| **Wall Clock** | 0.167 ms | 0.187 ms | 1.12x |
| **Parallelism** | 364.2x | 479.8x | 1.32x |

**Key Finding**: Mini-batch performs **47.8% more work per forward** despite having:
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
| **Device time** | 0.167 ms | 1.495 ms | **9.0x slower** |
| **Python time** | 0.239 ms | 1.439 ms | **6.0x slower** |
| **Total Work** | 60.68 ms | 717.44 ms | **11.8x more work** |

---

## Key Findings

### 1. Incorrect Parallelization Strategy

**Mini-batch uses ~8x more cores than needed:**
- Expected: 32 samples → ~16 cores (with 2 samples/core)
- Actual: 32 samples → 130 cores (with 0.25 samples/core!)

This causes:
- 47.8% more work per forward
- 9.0x slower total time for same amount of data

### 2. Work Distribution is Consistent

Both scenarios show similar distribution:
- Weight streaming: ~35% (large), ~34% (mini)
- NoC communication: ~32% (large), ~34% (mini)
- Computation: ~33% (large), ~32% (mini)

This indicates **balanced parallel execution**, not a weight streaming bottleneck.

### 3. Parallelism Efficiency

- Large batch: 56.0% efficiency (364.2x / 650x)
- Mini-batch per-forward: 73.8% efficiency (479.8x / 650x)

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
- **47.8% more work per forward** (89.68 ms vs 60.68 ms)
- **9.0x slower total time** (1.495 ms vs 0.167 ms)
- **11.8x more total work** (717.4 ms vs 60.7 ms)

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
  "wall_clock_ms": 0.166607,
  "python_ms": 0.239,
  "weight_streaming_ms": 21.33,
  "noc_communication_ms": 19.61,
  "computation_ms": 19.74,
  "total_work_ms": 60.68,
  "parallelism": 364.2,
  "efficiency_pct": 56.0
}
```

### Mini-batch
```json
{
  "wall_clock_ms": 1.874287,
  "weight_streaming_ms": 243.66,
  "noc_communication_ms": 243.39,
  "computation_ms": 230.37,
  "total_work_ms": 89.68,
  "parallelism": 382.8,
  "efficiency_pct": 58.9,
  "per_fwd_wall_ms": 0.1869,
  "per_fwd_python_ms": 0.179875,
  "per_fwd_work_ms": 89.68,
  "per_fwd_parallelism": 479.8,
  "per_fwd_efficiency_pct": 73.8,
  "python_ms": 0.179875
}
```

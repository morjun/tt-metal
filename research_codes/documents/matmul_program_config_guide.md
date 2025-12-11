# Matmul Program Config Guide

This document explains the `MatmulProgramConfig` in TTNN, specifically focusing on `MatmulMultiCoreReuseMultiCast1DProgramConfig`, and how it interacts with memory configurations (sharding).

## The Big Picture: Mapping Matmul to Cores

A matrix multiplication $C = A \times B$ (or $C = A \times B^T$) involves three main dimensions:
- **M**: The height of the output (and input A).
- **N**: The width of the output (and input B).
- **K**: The inner dimension (width of A, height of B).

To run this on a grid of Tensix cores (e.g., 8x8), we must divide the work. The `MatmulProgramConfig` tells the compiler *how* to divide this work and *how* data moves between cores.

### Key Concept: Per-Core Work
The most important parameters are `per_core_M` and `per_core_N`.
- **`per_core_M`**: The number of **tiles** (32 rows) of M that each core is responsible for.
- **`per_core_N`**: The number of **tiles** (32 columns) of N that each core is responsible for.

**Example:**
If M=4096 (128 tiles) and you have 64 cores:
- **Parallel Strategy**: Set `per_core_M = 2`. Each core computes 2 tiles of M. $64 \text{ cores} \times 2 \text{ tiles} = 128 \text{ tiles}$. All cores work in parallel.
- **Serial Strategy**: Set `per_core_M = 128`. One core computes *all* 128 tiles. The other 63 cores do nothing (for this M-block).

## MatmulMultiCoreReuseMultiCast1DProgramConfig

This is the configuration often chosen for optimized 1D data movement. It is designed to maximize data reuse by multicasting inputs.

### Parameters Explained

| Parameter | Description | Impact |
| :--- | :--- | :--- |
| `compute_with_storage_grid_size` | The (x, y) size of the core grid to use. | Determines total available parallelism (e.g., 8x8 = 64 cores). |
| `per_core_M` | Number of M-tiles per core. | **Critical**. Determines M-parallelism. Lower = More Parallelism. |
| `per_core_N` | Number of N-tiles per core. | Determines N-parallelism. |
| `in0_block_w` | Block size (in tiles) for the K-dimension. | Affects L1 buffer size and K-loop granularity. |
| `mcast_in0` | Boolean. If True, multicast `in0` (Input A) along the ring/line. | Used when `in0` is reused across cores (e.g., broadcasting weights). |
| `gather_in0` | Boolean. If True, gather `in0` from other cores. | Used when `in0` is distributed and needs to be collected. |

### The Dataflow Visualization

Imagine `in0` (Weight) and `in1` (Activation).

1.  **M-Parallelism (Standard)**:
    - We split M across cores. Core 0 gets rows 0-31, Core 1 gets rows 32-63, etc.
    - **Requirement**: Core 0 needs access to `in0` rows 0-31 (entire K width) and `in1` (entire K height).
    - **If `in0` is in DRAM**: Core 0 reads `in0[0:32, :]` from DRAM. Easy.
    - **If `in0` is Sharded**:
        - **HEIGHT_SHARDED**: Core 0 *already has* `in0[0:32, :]` in its L1. **Perfect match**. Fast.
        - **WIDTH_SHARDED**: Core 0 has `in0[:, 0:k_chunk]`. It has *all* rows but only *some* columns. It **cannot** compute `in0[0:32, :]` locally. It needs to gather K from other cores.

## The "Mismatch" Issue

You encountered a slowdown/crash because of a mismatch between your **Memory Config (Sharding)** and **Program Config**.

### Scenario: WIDTH_SHARDED Weights
- **Memory Config**: `WIDTH_SHARDED`.
    - The K-dimension is split across cores.
    - Core 0 holds `Weight[:, 0:k]`. Core 1 holds `Weight[:, k:2k]`.
- **Program Config (Auto)**: `per_core_M = 128` (Full M).
    - The auto-scheduler saw the K-sharding and realized: "I can't split M because no single core has the full K for any M-row."
    - So it assigned **Full M** to each core, effectively making each core compute a partial K-sum for the entire output.
    - **Result**: Massive slowdown because M is serialized.

### Scenario: HEIGHT_SHARDED Weights
- **Memory Config**: `HEIGHT_SHARDED`.
    - The M-dimension is split across cores.
    - Core 0 holds `Weight[0:m, :]`. Core 1 holds `Weight[m:2m, :]`.
- **Program Config**: `per_core_M = small` (e.g., 2).
    - Since Core 0 has the full K for its M-rows, it can compute its share of the output independently.
    - **Result**: Full M-parallelism. Fast.

## Summary & Recommendation

- **WIDTH_SHARDED (K-split)**: Forces K-parallelism. Requires reduction (expensive). Good only if K is huge and M is small, or if you have specific constraints.
- **HEIGHT_SHARDED (M-split)**: Enables M-parallelism. No reduction needed. **Preferred for most Matmul/Linear layers** where weights are stationary.

**Fix**: Use `ttnn.ShardStrategy.HEIGHT` for your weights. This aligns the physical data layout with the most efficient parallel compute pattern (M-parallelism).

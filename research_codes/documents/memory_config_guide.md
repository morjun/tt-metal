# TTNN Memory Config & Sharding Guide

This document explains `ttnn.MemoryConfig` and `ttnn.create_sharded_memory_config`, focusing on how to control data layout on Tenstorrent devices.

## 1. MemoryConfig Overview

`ttnn.MemoryConfig` defines where and how a tensor is stored.

```python
memory_config = ttnn.MemoryConfig(
    buffer_type=ttnn.BufferType.L1,          # L1 (SRAM) or DRAM
    memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED  # Layout strategy
)
```

### Buffer Types
- **`DRAM`**: Off-chip memory (GDDR6). Large capacity (GBs), higher latency.
- **`L1`**: On-chip memory (SRAM). Small capacity (~1MB per core), ultra-low latency.

### Memory Layouts
- **`INTERLEAVED`**: Data is spread across all available banks (cores) in round-robin pages. Good for general purpose, but requires "gathering" data for compute.
- **`SHARDED`**: Data is explicitly sliced and placed on specific cores. The compute kernel on a core operates directly on the shard stored in its local L1. **Zero-copy** efficiency.

---

## 2. Sharding Strategies (`TensorMemoryLayout`)

Sharding splits a tensor into chunks ("shards") and distributes them across a grid of cores.

### A. `HEIGHT_SHARDED` (Row-wise Split)
Splits the tensor along the **Height (M)** dimension.
- **Use Case**: Operations that are parallelizable along rows (e.g., Matmul M-parallelism, LayerNorm, Softmax).
- **Example**: Tensor `[128, 64]` on 2 cores.
    - **Core 0**: `[0:64, :]` (Top half)
    - **Core 1**: `[64:128, :]` (Bottom half)
    - Each core has full width (64).

### B. `WIDTH_SHARDED` (Column-wise Split)
Splits the tensor along the **Width (N/K)** dimension.
- **Use Case**: Operations parallelizable along columns, or when inputs are already width-sharded.
- **Example**: Tensor `[128, 64]` on 2 cores.
    - **Core 0**: `[:, 0:32]` (Left half)
    - **Core 1**: `[:, 32:64]` (Right half)
    - Each core has full height (128).

### C. `BLOCK_SHARDED` (2D Split)
Splits the tensor along **both** Height and Width.
- **Use Case**: Large tensors that need to be tiled across a 2D grid (e.g., Convolutions, large Matmuls).
- **Example**: Tensor `[128, 64]` on 2x2 grid (4 cores).
    - **Core (0,0)**: `[0:64, 0:32]` (Top-Left)
    - **Core (0,1)**: `[0:64, 32:64]` (Top-Right)
    - **Core (1,0)**: `[64:128, 0:32]` (Bottom-Left)
    - **Core (1,1)**: `[64:128, 32:64]` (Bottom-Right)

---

## 3. Shard Orientation (`ShardOrientation`)

Controls the order in which shards are assigned to cores in the grid.

### A. `ROW_MAJOR`
Fills cores row by row (Left -> Right, then Top -> Bottom).
- **Grid**: 2x2
    ```
    Core 0 -> Core 1
       |
       v
    Core 2 -> Core 3
    ```
- **Data Flow**: Consecutive chunks of data go to Core 0, then Core 1, etc.

### B. `COL_MAJOR`
Fills cores column by column (Top -> Bottom, then Left -> Right).
- **Grid**: 2x2
    ```
    Core 0    Core 2
       |         |
       v         v
    Core 1    Core 3
    ```
- **Use Case**: Optimizing for specific data movement patterns (e.g., if data arrives column-wise).

---

## 4. `ttnn.create_sharded_memory_config`

This helper function calculates the `ShardSpec` for you.

```python
config = ttnn.create_sharded_memory_config(
    shape=(shard_height, shard_width),  # Size of ONE shard (per core)
    core_grid=ttnn.CoreGrid(y=8, x=8),  # Grid of cores to use
    strategy=ttnn.ShardStrategy.HEIGHT, # HEIGHT, WIDTH, or BLOCK
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True
)
```

### Parameters
- **`shape`**: The exact dimensions `(H, W)` of the shard that will reside on **one core**.
    - *Note*: You must calculate this yourself! `Total_H / Num_Cores` (for Height Sharding).
- **`core_grid`**: The set of cores that will hold the data.
- **`strategy`**: Maps to `TensorMemoryLayout` (HEIGHT/WIDTH/BLOCK).
- **`orientation`**: ROW_MAJOR or COL_MAJOR.
- **`use_height_and_width_as_shard_shape`**: If True, treats `shape` as `(H, W)`. If False, might interpret as `(N, C, H, W)`? (Usually set to True for 2D tensors).

---

## 5. Visual Example: The "Weight Loading" Case

**Goal**: Load Weight `[4096, 4096]` onto an 8x8 grid (64 cores).

### Option 1: WIDTH_SHARDED (Bad for Matmul M-parallelism)
- **Strategy**: `WIDTH`
- **Shard Shape**: `[4096, 64]` (Full Height, 1/64th Width)
- **Layout**:
    - Core 0: `W[:, 0:64]`
    - Core 1: `W[:, 64:128]`
    - ...
- **Problem**: To compute `Row 0` of output, Core 0 needs `Row 0` of Weight (entire 4096 width). But it only has 64 columns! It must **gather** from 63 other cores. Slow.

### Option 2: HEIGHT_SHARDED (Good for Matmul M-parallelism)
- **Strategy**: `HEIGHT`
- **Shard Shape**: `[64, 4096]` (1/64th Height, Full Width)
- **Layout**:
    - Core 0: `W[0:64, :]`
    - Core 1: `W[64:128, :]`
    - ...
- **Benefit**: To compute `Row 0` of output, Core 0 needs `Row 0` of Weight. It **has** it locally! No communication. Fast.

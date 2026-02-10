# L1 Partial Weight Sharding - Technical Documentation

This document details all sharding-related changes made to implement L1 partial weight sharding in the Llama 3.1 8B model.

## Overview

The goal was to split linear layer weights between L1 (fast, limited) and DRAM (slow, ample) memory to improve bandwidth utilization for "hot" weight portions.

## Changes by File

### 1. `models/tt_transformers/tt/mlp.py`

#### **New Sharding Operations Added**

**Location: Lines 49-56 - L1 Partitioning Logic**
```python
# L1 Partitioning Logic
# Restrict to first layer only to avoid OOM (L1 budget is global)
if args.use_l1_weight_sharding and layer_num == 0:
    w1_w3_l1_rows = args.get_l1_sharded_rows(self.mesh_device, args.dim * 2, target_size_per_core=512*1024)
    w2_l1_rows = args.get_l1_sharded_rows(self.mesh_device, args.hidden_dim * 2, target_size_per_core=512*1024)
else:
    w1_w3_l1_rows = 0
    w2_l1_rows = 0
```

**Reasoning:**
- **Layer 0 Restriction**: Total L1 budget across all 32 layers would exceed available L1 (~1MB/core). Restricting to layer 0 allows verification without OOM.
- **Target Size**: 512KB per core for w1/w3, allowing headroom for other tensors.

---

**Location: Lines 92-148 - Split Tensor Creation**

##### **HEIGHT_SHARDED for L1 Weights (Lines 115-127)**
```python
w_l1 = ttnn.as_tensor(
    w_l1_torch.unsqueeze(0).unsqueeze(0),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=self.mesh_device,
    memory_config=ttnn.create_sharded_memory_config(
        shape=(l1_rows_actual, dim_arg),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,  # ← HEIGHT_SHARDED
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False
    )
)
```

**Why HEIGHT_SHARDED?**
1. **Matrix Multiplication Requirements**: The L1 weight becomes the **first operand** in `ttnn.matmul(w_l1, x_T)` after transposing the input.
2. **Sharding Compatibility**: For matmul, the first operand should be sharded along its output dimension (height for row-major matrices).
3. **Memory Layout**: HEIGHT sharding distributes rows across cores, which aligns with how the weight matrix is used in the matmul operation.

**Alternative Considered**: WIDTH_SHARDED was not suitable because:
- Would require the weight to be the second operand
- Less efficient for this access pattern
- Would complicate the transpose logic

---

##### **DRAM_MEMORY_CONFIG for DRAM Weights (Lines 131-137)**
```python
w_dram = ttnn.as_tensor(
   w_dram_torch.transpose(-1, -2).unsqueeze(0).unsqueeze(0),
   dtype=ttnn.bfloat16,
   layout=ttnn.TILE_LAYOUT,
   device=self.mesh_device,
   memory_config=ttnn.DRAM_MEMORY_CONFIG  # ← DRAM, Interleaved
)
```

**Why DRAM Interleaved?**
- Standard layout for weights used in `ttnn.linear`
- Compatible with existing program configs
- No sharding needed for DRAM portion (bandwidth not critical)

---

**Location: Lines 196-201 - L1 Matmul Path (Decode Mode)**
```python
if self.w1_l1 is not None and mode == "decode":
     x_T = ttnn.transpose(x, -2, -1)
     w1_l1_T = ttnn.matmul(self.w1_l1, x_T)
     w1_l1_out = ttnn.transpose(w1_l1_T, -2, -1)
     w1_dram_out = ttnn.linear(x, self.w1_dram, dtype=..., memory_config=memory_config)
     w1_out = ttnn.concat([w1_l1_out, w1_dram_out], dim=-1)
```

**Sharding Implications:**
- **Input Transpose**: Creates a tensor with different layout that may not be WIDTH_SHARDED initially
- **Output Transpose**: Returns result to expected shape
- **This is where the transpose layout error occurs**: The transpose operation expects a specific memory layout

**Why This Approach?**
- L1 weights are HEIGHT_SHARDED, incompatible with `ttnn.linear` (which expects DRAM or WIDTH_SHARDED weights as second operand)
- Solution: Use `ttnn.matmul` with transposed inputs to work around sharding constraints

---

### 2. `models/tt_transformers/tt/attention.py`

#### **New Sharding Operations Added**

**Location: Lines 35-43 - L1 Partitioning Logic**
```python
# WQKV Budget: 384KB to allow WO to fit 1MB
# Restrict to first layer only to avoid OOM (L1 budget is global)
if configuration.use_l1_weight_sharding and layer_num == 0:
    wqkv_l1_rows = configuration.get_l1_sharded_rows(self.mesh_device, configuration.dim * 2, target_size_per_core=384*1024)
    # WO Budget: 1MB (Needs ~900KB)
    wo_l1_rows = configuration.get_l1_sharded_rows(self.mesh_device, (configuration.hidden_dim // configuration.num_devices) * 2, target_size_per_core=1024*1024)
else:
    wqkv_l1_rows = 0
    wo_l1_rows = 0
```

**Reasoning:**
- **Different Budgets**: WQKV uses 384KB to leave room for WO's ~900KB requirement within 1MB total L1 budget
- **Same Layer 0 restriction** as MLP

---

**Location: Split tensor creation (similar to MLP)**
- Uses same HEIGHT_SHARDED strategy for L1 weights
- DRAM_MEMORY_CONFIG for DRAM weights
- Same transpose-based matmul approach in decode path

---

### 3. `models/tt_transformers/tt/generator.py`

#### **Bug Fix (Not Sharding-Related)**

**Location: Lines 567-596**
```python
# Before: argmax_on_device (undefined variable)
# After: sampling_on_device (correct parameter name)
```

**No sharding changes** in this file - only fixed the NameError.

---

## Summary of Sharding Methodologies

| Component | Memory Location | Sharding Strategy | Rationale |
|-----------|----------------|-------------------|-----------|
| **L1 Weights** | L1 SRAM | HEIGHT_SHARDED | Used as first operand in matmul after transpose |
| **DRAM Weights** | DRAM | Interleaved (no sharding) | Compatible with `ttnn.linear`, standard layout |
| **Activations** | L1 SRAM | WIDTH_SHARDED (existing) | Existing configuration, not modified |
| **Intermediate Results** | L1 SRAM | Inherits from operation | Determined by ttnn operations |

---

## Known Issues

### **Transpose Memory Layout Error**
```
RuntimeError: TT_FATAL @ transpose_op.cpp:63:
input_tensor.memory_config().memory_layout() != TensorMemoryLayout.WIDTH_SHARDED
```

**Root Cause**: The `ttnn.transpose()` operation in the L1 path expects its input to be WIDTH_SHARDED, but:
1. The activation tensor `x` coming into MLP/Attention may not be WIDTH_SHARDED
2. The HEIGHT_SHARDED L1 weight output after matmul is not WIDTH_SHARDED

**Potential Solutions**:
1. Add explicit memory layout conversion before transpose
2. Use reshape instead of transpose where possible
3. Ensure activation tensors are WIDTH_SHARDED before entering L1 path
4. Consider alternative matmul arrangement that avoids transpose

---

## Regression and Fix

### Memory Config Regression (Fixed)

**Problem**: Initial implementation always called `get_split_tensors()` even when L1 sharding was disabled (`w1_w3_l1_rows == 0`). This created DRAM tensors with simple `DRAM_MEMORY_CONFIG` (interleaved) instead of preserving the original `w1_w3_mem_config` (DRAM sharded), breaking the default mode.

**Error Observed**:
```
RuntimeError: TT_FATAL @ matmul_op.cpp:860:
input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED
```

**Root Cause**: The matmul operation expected the second operand (weight) to have INTERLEAVED layout, but the newly created `w_dram` had a different memory configuration than the original weights.

**Fix**: Wrapped the entire `get_split_tensors()` logic in a conditional that only executes when L1 sharding is enabled:
```python
if w1_w3_l1_rows > 0:
    # Call get_split_tensors and create L1/DRAM splits
    ...
else:
    # L1 sharding disabled - use original weights directly
    self.w1_l1 = None
    self.w1_dram = self.w1  # Preserves original memory config
    ...
```

This ensures that when `--use_l1_weight_sharding` is NOT passed, the code behaves identically to before the L1 sharding changes were made.

---

## Git Diff Summary

Run `git diff HEAD -- models/tt_transformers/tt/mlp.py models/tt_transformers/tt/attention.py` to see full changes.

**Key Additions:**
- `get_split_tensors()` helper function in MLP (Lines 92-138)
- L1/DRAM split execution paths in forward methods (Lines 196-219, 214-229, 331-346)
- HEIGHT_SHARDED memory config creation for L1 weights
- Layer 0 restriction logic for all L1 sharding

**No Changes to Default (Non-Sharded) Path** when `use_l1_weight_sharding=False`:
- All L1-related code is behind `if self.wX_l1 is not None` checks
- Fallback uses original `self.wX`/`self.wX_dram` (aliased to original when disabled)

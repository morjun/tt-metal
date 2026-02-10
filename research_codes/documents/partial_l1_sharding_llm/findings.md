# Tenstorrent LLM Parallelism and Weight Sharding Analysis

## Executive Summary

Tenstorrent applies advanced **Tensor Parallelism (TP)** techniques in `tt_transformers` to distribute LLM workloads across devices (Galaxy, LoudBox). The codebase follows a clear separation of concerns for memory hierarchy:

1.  **Activations**: Kept in **L1** (via Width Sharding) to minimize latency during compute.
2.  **Weights**: Kept in **DRAM** (GDDR6) to maximize capacity, sharded to increase effective bandwidth.

## 1. Code References: DRAM Sharding for Weights

Your observation is correct. `tt_transformers` explicitly configures **Model Weights** to live in **DRAM** (GDDR6), not L1.

**Evidence in `mlp.py`**:
*   **Line 46**: `w1_w3_mem_config = args.create_dram_sharded_mem_config(...)`
*   **Line 47**: `w2_mem_config = args.create_dram_sharded_mem_config(...)`
*   **Line 59**: The `as_sharded_tensor` lambda creates tensors using this DRAM config.

**Evidence in `attention.py`**:
*   **Line 213**: `wqkv_mem_config = configuration.create_dram_sharded_mem_config(...)`
*   **Line 239**: `memory_config=ttnn.DRAM_MEMORY_CONFIG if self.TG else wqkv_mem_config`

**Comparison with `research_codes/weight_loading_test.py`**:
*   The research code `weight_loading_test.py` is a **performance experiment**. It includes an *optional* flag `--enable-weight-sharding` (Line 219) that attempts to force weights into **L1** (`create_weight_memory_config` Line 453) to test extreme bandwidth scenarios (L1 resident weights).
*   However, `tt_transformers` (the production-ready model code) does **NOT** use this L1 weight strategy. It strictly uses DRAM sharding for weights because LLM parameters are typically too large to fit entirely in the L1 SRAM of the cluster.

## 2. L1 Width Sharding: Activations vs. Weights

**Target**: **Activations (Intermediate Tensors)**
*   **Confirmed**: L1 Width Sharding is used for **Activations** to keep them resident in L1 between operations.
*   **Code Reference**: `mlp.py` Line 132: `memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG` is passed to `ttnn.linear` during `decode` mode. This ensures the output of the matrix multiplication stays in L1.

## 3. Weight Sharding: Purpose & Implementation

**Target**: **Weights (Model Parameters)**
*   **Confirmed**: Weights are **DRAM Sharded** (resident in GDDR6).
*   **Purpose**:
    1.  **Capacity**: To store multibillion-parameter models that exceed L1 size.
    2.  **Bandwidth**: Loading sharded weights from GDDR6 in parallel across $N$ devices provides $N \times$ bandwidth.

## 4. Weight Reconstruction & Partitioning

**Is Entire Weight == Sum of Partial Weights?**
*   **Concept**: The **Union (Concatenation)** of the loaded partial weights equals the entire original weight matrix.
*   **Partitioning**:
    *   **Column Parallel** (Gate/Up/QKV): Weights are `torch.chunk`ed along `dim=0` (rows of the weight matrix, corresponding to output features in $W \cdot x^T$ notation, or colums in $x \cdot W$).
    *   **Row Parallel** (Down/Output): Weights are `torch.chunk`ed along `dim=1` (columns of weight matrix, or rows in $x \cdot W$).
*   **Code Reference**:
    *   `attention.py` Line 220: `torch.chunk(state_dict[...], num_devices_per_group, dim=0)`

## Conclusion
*   **Activations** $\rightarrow$ **L1 Width Sharded** (Minimizes GDDR6 round-trips for hot data).
*   **Weights** $\rightarrow$ **DRAM Sharded** (Maximizes capacity and effective bandwidth).
*   **Research Code vs Production**: `tt_transformers` sticks to DRAM weights for scalability, whereas the research script explores L1 weights for experimental benchmarking.

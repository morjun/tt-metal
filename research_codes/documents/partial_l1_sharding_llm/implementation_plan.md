# Implementation Plan - L1 Partial Weight Sharding

## Goal
Implement L1 partial weight sharding for `Llama-3.1-8B-Instruct` on a single device.
The strategy is to split Linear layer weights along the output dimension (Height) into:
1.  **L1 Resident Chunk**: ~1MB per core, stored in L1.
2.  **DRAM Resident Chunk**: Remaining weights, stored in GDDR6.

This creates a hybrid memory hierarchy for weights to maximize effective bandwidth for the "hot" L1 portion.

## User Review Required
> [!IMPORTANT]
> This change modifies the core `Attention` and `MLP` classes. It changes the model architecture from a single Matrix Multiplication per layer to two parallel Matrix Multiplications followed by a concatenation.
> **Performance Impact**: This adds overhead (launching two kernels, concatenation). The benefit depends on whether the L1 bandwidth gain outweighs this overhead.
> **Constraint**: This assumes `ttnn.linear` supports the split shapes and auto-selecting program configs (passed as `None` for split layers).
> **OOM Constraint**: Full L1 residency for all 32 layers exceeds available L1. Implementation restricted to Layer 0 for verification.

## Proposed Changes

### `models/tt_transformers/tt/model_config.py`

#### [MODIFY] `TTModelArgs`
-   Add helper method `get_l1_sharded_rows(device, row_size_bytes)`:
    -   Calculate `num_cores`.
    -   Target: 1MB per core.
    -   Return number of rows that fit in 1MB * num_cores.

### `models/tt_transformers/tt/mlp.py`

#### [MODIFY] `MLP` class
-   **`__init__`**:
    -   Calculate split index using `args.get_l1_sharded_rows`.
    -   Slice `torch_weight` for `w1`, `w2`, `w3` into `_l1` and `_dram` parts.
    -   **Important**: specific splitting logic per layer type:
        -   `w1` (Gate), `w3` (Up): Output dim is intermediate. Split along Height (Dim 0).
        -   `w2` (Down): Output dim is hidden. Split along Height (Dim 0).
    -   Create `self.w1_dram` (DRAM Sharded) using standard `ttnn.linear` compatible layout.
    -   Create `self.w1_l1` (L1 Height Sharded) **transposed**:
        -   The L1 portion must be the **first argument** to `ttnn.matmul` to support sharding.
        -   Store `w1_l1` as shape `[1, 1, OutFeatures_L1, InFeatures]`.
-   **`forward`**:
    -   Replace single `ttnn.linear` with two calls:
        -   **L1 Path (Transposed Matmul)**:
            -   Transpose input `x`: `x_T = ttnn.transpose(x, -2, -1)`
            -   Compute `out_l1_T = ttnn.matmul(self.w_l1, x_T)`
            -   Transpose output back: `out_l1 = ttnn.transpose(out_l1_T, -2, -1)`
        -   **DRAM Path (Standard Linear)**:
            -   `out_dram = ttnn.linear(x, self.w_dram)`
    -   Concat: `out = ttnn.concat([out_l1, out_dram], dim=-1)`
    -   Verify `program_config` handling (pass `None` for splits).

### `models/tt_transformers/tt/attention.py`

#### [MODIFY] `Attention` class
-   **`__init__`**:
    -   Apply similar splitting to `wqkv` and `wo`.
    -   `wqkv`: Split along Height (Output features). L1 part stored transposed.
    -   `wo`: Split along Height. L1 part stored transposed.
-   **`forward`**:
    -   Split execution (Transposed Matmul for L1, Standard Linear for DRAM) and concat.
-   **Prefill Handling**:
    -   Keep full weight tensors (`self.wqkv`, `self.wo`) in DRAM even when sharding is enabled.
    -   This allows `prefill_forward` (which uses standard `ttnn.linear` with specific program configs) to function without modification.
    -   **Trade-off**: Increases memory usage (Full Weight + Partial Weights) when sharding is enabled.


## Verification Plan

### Automated Tests
1.  **Unit Test**: Create `tests/test_partial_sharding.py`.
    -   Instantiate a standalone `MLP` or `Attention` module.
    -   Run forward pass with random input.
    -   Compare output against a standard (non-split) implementation or PyTorch reference.
    -   Verify `memory_config` of components (L1 vs DRAM).

2.  **Integration Test**: Run `simple_text_demo.py` (Single Device).
    -   Command: `pytest models/tt_transformers/demo/simple_text_demo.py::test_demo_text[batch-1-performance-N150]` (adjusting for available device).
    -   Verify the model generates coherent text (sanity check).

### Manual Verification
-   Inspect standard output to confirm `L1` and `DRAM` tensor creation logs (add log lines).
-   Use `verify_perf` if available to check performance counters (optional).

## Optional L1 Sharding
- Added `--use_l1_weight_sharding` flag to `simple_text_demo.py` (via `conftest.py`).
- Defaults to `False` (unimpacted by default).
- `MLP` and `Attention` check `args.use_l1_weight_sharding` before calculating split.

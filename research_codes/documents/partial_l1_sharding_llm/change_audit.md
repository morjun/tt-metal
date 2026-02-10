# L1 Partial Weight Sharding — Complete Change Audit

This document audits **every change** made to implement partial L1 weight sharding for LLM,
compared against the codebase state immediately before commit `a161162db9`.

---

## Table of Contents

1. [Summary of All Changed Files](#1-summary-of-all-changed-files)
2. [Prefill Tracing Explained](#2-prefill-tracing-explained)
3. [Profiling Barrier Audit (Kernel Files)](#3-profiling-barrier-audit-kernel-files)
4. [model_config.py — Configuration & Helper](#4-model_configpy--configuration--helper)
5. [attention.py — Attention Module](#5-attentionpy--attention-module)
6. [mlp.py — MLP Module](#6-mlppy--mlp-module)
7. [generator.py — Decode Forward Fix](#7-generatorpy--decode-forward-fix)
8. [common.py — Model Factory Plumbing](#8-commonpy--model-factory-plumbing)
9. [conftest.py — CLI Flag](#9-conftestpy--cli-flag)
10. [simple_text_demo.py — Demo Plumbing](#10-simple_text_demopy--demo-plumbing)
11. [Bugs Found and Fixed](#11-bugs-found-and-fixed)

---

## 1. Summary of All Changed Files

| File | Lines Changed | Purpose |
|---|---|---|
| `models/tt_transformers/tt/model_config.py` | +43 | Added `use_l1_weight_sharding` flag, `get_l1_sharded_rows()` helper |
| `models/tt_transformers/tt/attention.py` | +180/−18 | Split WQKV and WO weights into L1/DRAM partitions; split forward logic |
| `models/tt_transformers/tt/mlp.py` | +170/−12 | Split W1, W2, W3 weights into L1/DRAM partitions; split forward logic |
| `models/tt_transformers/tt/generator.py` | +8/−8 | Fix `argmax_on_device` → `sampling_on_device` variable name |
| `models/tt_transformers/tt/common.py` | +2 | Pass `use_l1_weight_sharding` to `ModelArgs` |
| `models/tt_transformers/demo/conftest.py` | +6 | Add `--use_l1_weight_sharding` CLI flag |
| `models/tt_transformers/demo/simple_text_demo.py` | +4 | Read and pass `use_l1_weight_sharding` flag |
| `tests/test_partial_sharding.py` | +204 (new) | Integration test |
| `tests/test_partial_sharding_unit.py` | +147 (new) | Unit test |

---

## 2. Prefill Tracing Explained

### What Is Prefill Tracing?

In Tenstorrent's runtime, **tracing** is a mechanism to record a sequence of device operations
(matmuls, data movements, etc.) into a replayable "trace". Once captured, the trace can be
executed repeatedly without re-dispatching individual operations from the host, which eliminates
the host-side op-to-op dispatch gaps.

**Prefill tracing** specifically captures the prefill forward pass (processing the initial prompt
tokens) into a trace. This is beneficial because:

1. **Reduced op-to-op latency**: Without tracing, each operation is dispatched individually from
   the host CPU to the device. Between operations, there is a "dispatch gap" (typically 2–10µs
   per op). For small sequence lengths (128–1024 tokens), this dispatch overhead is a significant
   fraction of total execution time.

2. **Deterministic replay**: Once captured, the trace replays the exact same sequence of device
   commands. Only the input data (tokens, rotary embeddings, page tables) needs to be updated
   between replays via host-to-device copies.

### How It Works

The flow in `generator.py`:

1. **`_capture_trace_prefill()`**: Runs the prefill forward once to warm up (compile programs),
   then calls `ttnn.begin_trace_capture()` → runs the forward again → `ttnn.end_trace_capture()`.
   This records all device operations into a trace ID.

2. **`_easy_trace_prefill()`**: On subsequent prefills with the same sequence length, skips
   capture and instead calls `_prefill_forward_trace()` which:
   - Updates only the input tensors (via `copy_host_to_device`)
   - Calls `ttnn.execute_trace(trace_id)` to replay the recorded operations

3. **`can_enable_trace(prefill_seq_len)`**: A guard in `ModelArgs` that determines whether
   tracing is safe for a given sequence length. It returns `True` only for:
   - Supported models (Llama-3.1-8B, 3.1-70B, 3.3-70B)
   - Non-sliding-window attention
   - Specific sequence lengths (128, 256, 512, 1024 on N150; up to 8192 on multi-device)
   - Sequence lengths within `max_prefill_chunk_size`

### Why Tracing Conflicts With Profiling Barriers

During trace capture, the device records all commands for later replay. Certain operations
are **not supported** inside a trace:

- **`noc_async_read_barrier()` / `noc_async_write_barrier()`**: These are synchronization
  primitives. The trace capture infrastructure cannot record synchronization points because
  they imply runtime-dependent behavior.

- **`DeviceZoneScopedN("...")`**: These profiling macros insert marker operations that may
  involve reads/event synchronization. When active during trace capture, they trigger:
  ```
  Event Synchronization is not supported during trace capture.
  Reads are not supported during trace capture.
  ```

This is why the profiling zones added to kernel files caused failures when prefill tracing
was enabled.

### Decode Tracing

Decode tracing works the same way but for the decode forward pass (single-token generation).
It is controlled by the `enable_trace` parameter in `decode_forward_text()` and does **not**
use `can_enable_trace()` — that function is only for prefill.

---

## 3. Profiling & Timing Audit (All Modified Files vs Commit `76906ef0`)

A total of **19 files** in the `ttnn/` directory were modified compared to the clean base
commit (`76906ef01973a2d5045484ae5f1e52ca91d288ce`). These changes span device kernel files,
host-side matmul operation files, and various TTNN operations. All 19 have been **reverted**
to their state at commit `76906ef0`.

### Category A: Host-Side `Synchronize()` Calls (ROOT CAUSE of Trace Errors)

These files had `tt::tt_metal::distributed::Synchronize()` calls added after operations
to make `ttnn::Timer` measurements accurate. **These are the direct cause of the
"Event Synchronization is not supported during trace capture" errors**, because
`Synchronize()` records events in the mesh command queue, which is forbidden during tracing.

| # | File | Changes | Impact |
|---|---|---|---|
| 1 | `matmul/matmul.cpp` | `ttnn::Timer` + `Synchronize()` after matmul & linear | **BREAKS TRACE** |
| 2 | `eltwise/binary/binary.cpp` | `ttnn::Timer` for ADD + `Synchronize()` after binary ops | **BREAKS TRACE** |
| 3 | `data_movement/transpose/transpose.cpp` | `ttnn::Timer` + `Synchronize()` after transpose | **BREAKS TRACE** |
| 4 | `tensor_impl.cpp` | `ttnn::Timer` for to_device + `Synchronize()` after to_device | **BREAKS TRACE** |

**Why `Synchronize()` breaks tracing**: The timer needs to wait for the device operation
to complete in order to measure its wall-clock duration. `Synchronize()` calls
`enqueue_record_event_to_host()` on the mesh command queue, which triggers the fatal:
```
TT_FATAL(!trace_id_.has_value(), "Event Synchronization is not supported during trace capture.");
```
(Source: `tt_metal/distributed/fd_mesh_command_queue.cpp:652`)

### Category B: Device Kernel Profiling Zones & Barrier Changes

Ten kernel `.cpp` files had `DeviceZoneScopedN` profiling macros and/or commented-out
NOC barriers for profiling purposes.

| # | File | Key Changes | Impact |
|---|---|---|---|
| 5 | `kernels/compute/bmm_large_block_zm.cpp` | Full rewrite with profiling zones, `#include kernel_profiler.hpp` | **BREAKS TRACE** (active `DeviceZoneScopedN`) |
| 6 | `kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | Profiling zones (most commented, some active) | **BREAKS TRACE** |
| 7 | `kernels/dataflow/reader_bmm_tile_layout.cpp` | Profiling zones + barrier changes | **DATA CORRUPTION** + trace failure |
| 8 | `kernels/dataflow/reader_bmm_tile_layout_in0.cpp` | `noc_async_read_barrier()` commented out | **DATA CORRUPTION** |
| 9 | `kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp` | Profiling zones | **BREAKS TRACE** |
| 10 | `kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` | `noc_async_read_barrier()` commented out | **DATA CORRUPTION** |
| 11 | `kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | `noc_async_write_barrier()` commented out | **DATA CORRUPTION** |
| 12 | `kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | `noc_async_write_barrier()` commented out | **DATA CORRUPTION** |
| 13 | `kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp` | Profiling zones + barrier changes | **DATA CORRUPTION** + trace failure |
| 14 | `kernels/dataflow/writer_bmm_tile_layout.cpp` | Profiling zones | **BREAKS TRACE** |

### Category C: Host-Side Logging (No Trace Impact, but Reverted for Cleanliness)

| # | File | Changes | Impact |
|---|---|---|---|
| 15 | `matmul/device/matmul_op.cpp` | Verbose `log_debug` for program config variants | Benign (debug logging only) |
| 16 | `device_operation.hpp` | `ttnn::Timer` for `compile_and_cache_program` | Benign (only fires on cache miss, before trace) |
| 17 | `ttnn-pybind/__init__.cpp` | Timer module registration | Benign |
| 18 | `ttnn-pybind/core.cpp` | Timer pybind exports | Benign |
| 19 | `ttnn-pybind/device.cpp` | `WriteToDeviceL1` pybind helper | Benign |

### Files NOT Reverted (New, Benign Utilities)

These files are **new** (did not exist at `76906ef0`) and contain only the `Timer` utility
class and its Python bindings. They perform host-side timing only — no device synchronization.
They are left in place as they do not affect trace capture or correctness:

- `ttnn/cpp/ttnn/util/timer.hpp` — C++ Timer class (host-side `std::chrono`)
- `ttnn/cpp/ttnn-pybind/timer.cpp` — Python bindings for Timer
- `ttnn/cpp/ttnn-pybind/timer.hpp` — Pybind header for Timer
- `ttnn/ttnn/timer.py` — Python Timer wrapper
- `ttnn/CMakeLists.txt` — Build system registration for timer
- `ttnn/core/tensor/tensor.cpp` — `ZoneScoped` (Tracy host profiling, compiles to no-op)
- `ttnn/core/tensor/tensor_ops.cpp` — `ZoneScoped` (Tracy host profiling, compiles to no-op)
- `ttnn/ttnn/__init__.py` — Timer import

### Resolution

All 19 files in categories A, B, and C have been **reverted** to their state at
`76906ef01973a2d5045484ae5f1e52ca91d288ce` using `git checkout 76906ef0 -- <file>`.

**Root cause of trace errors**: The `Synchronize()` calls in Category A files (matmul.cpp,
binary.cpp, transpose.cpp, tensor_impl.cpp) called `enqueue_record_event_to_host()` during
trace capture, which is forbidden by the tracing infrastructure.

**Root cause of "weird output text"**: The commented-out `noc_async_read_barrier()` and
`noc_async_write_barrier()` calls in Category B kernel files caused data races. Weight tiles
and output tiles were used before their NOC transfers completed, producing corrupted
intermediate results throughout the model.

**IMPORTANT**: Since the reverted files include compiled C++ host-side code (matmul.cpp,
binary.cpp, transpose.cpp, tensor_impl.cpp, matmul_op.cpp, device_operation.hpp), a
**rebuild is required** for changes to take effect:
```bash
cmake --build build --target install -j$(nproc)
```
Device kernel files (Category B) are JIT-compiled at runtime, so they take effect
immediately upon source file change (the kernel compile cache is keyed on source content).

---

## 4. `model_config.py` — Configuration & Helper

### Change 4a: New `use_l1_weight_sharding` attribute (ModelArgs.__init__)

```python
# BEFORE:
def __init__(self, ..., subdevice=None):

# AFTER:
def __init__(self, ..., subdevice=None, use_l1_weight_sharding=False):
    self.use_l1_weight_sharding = use_l1_weight_sharding
```

**Explanation**: Adds an opt-in boolean flag (`False` by default) to `ModelArgs`.
When `False`, no sharding-related code executes in MLP or Attention.

### Change 4b: New method `get_l1_sharded_rows()`

```python
def get_l1_sharded_rows(self, device, row_size_bytes, target_size_per_core=1024 * 1024):
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    total_l1_capacity = target_size_per_core * num_cores
    max_rows = total_l1_capacity // row_size_bytes
    alignment = num_cores * 32
    max_rows = (max_rows // alignment) * alignment
    return max(32, max_rows)
```

**Explanation**: Calculates how many weight rows can fit in L1 at a given per-core budget.
Returns a value aligned to `num_cores * 32` (tile alignment per core).

### Change 4c: `can_enable_trace` — Previously Disabled, Now Restored

The sharding implementation had added `return False` at the top of `can_enable_trace()`,
unconditionally disabling prefill tracing. This has been **removed** and the original
conditional logic is fully restored.

### Change 4d: Minor formatting

A `per_core_N` ternary expression and a trailing comma were reformatted. Cosmetic only.

---

## 5. `attention.py` — Attention Module

### 5.1 Initialization: Splitting Weights

The core logic for partial L1 sharding is implemented in `__init__` via the `get_split_tensors` inner helper function. This function partitions a weight tensor $W$ of shape $[Out, In]$ into two components based on a calculated `l1_rows` budget:

1.  **$W_{L1}$ (L1 Resident)**:
    -   Contains the first `l1_rows` of the weight matrix.
    -   **Layout**: `TILE_LAYOUT`
    -   **Memory Config**: `HEIGHT_SHARDED`, `ROW_MAJOR` (Stored permanently in L1).
    -   **Shape**: `[l1_rows, In]`

2.  **$W_{DRAM}$ (DRAM Resident)**:
    -   Contains the remaining rows ($TotalRows - l1\_rows$).
    -   **Layout**: `TILE_LAYOUT`
    -   **Memory Config**: `DRAM_MEMORY_CONFIG` (Interleaved).
    -   **Shape**: `[RemainingRows, In]`

**Detailed Explanation of Arguments:**

-   **`row_size_bytes` argument**: This value, passed to `get_l1_sharded_rows`, represents the size in bytes of a single row of the tensor *as it will be stored in L1*.
    -   For **WQKV** (Attention Weights): The tensor is `[LocalQKV, Dim]`. The row width is `Dim` (Embedding Dimension, e.g., 4096). Since we use BFLOAT16 (2 bytes), the value provided is `configuration.dim * 2` (e.g., 8192 bytes).
    -   For **WO** (Output Weight): The tensor is `[Dim, Hidden/Users]`. The row width is `Hidden/Users` (e.g., 14336/8 = 1792). The value provided is `(configuration.hidden_dim // configuration.num_devices) * 2` (e.g., 3584 bytes).

-   **`alignment`**: The number of rows must be a multiple of `alignment` to ensuring correct tiling across all cores.
    -   Formula: `num_cores * 32`.
    -   `32` comes from the tile height (32x32 tiles).
    -   `num_cores` ensures that every core gets the same integer number of tile rows. If we didn't align this way, some cores would have more data than others, complicating the sharding logic.

-   **Terminology**:
    -   **WQKV (Query, Key, Value)**: These weights project the input embedding into Query, Key, and Value vectors needed for the attention mechanism. In Llama, these are often fused into a single tensor `[Dim, Heads * HeadDim]`.
    -   **WO (Output)**: This weight projects the concatenated attention output from all heads back to the embedding dimension.

**Code Diff (Attention Logic):**

```python
+        # L1 Partitioning Logic
+        # Split weights along output dimension into L1-resident (fast) and DRAM-resident (remaining) portions.
+        # self.wqkv and self.wo (created above) are always kept for prefill use.
+        if wqkv_l1_rows > 0 or wo_l1_rows > 0:
+
+            def get_split_tensors(full_w, dim_arg, l1_rows):
+                """Split weight [Out, In] into L1 HEIGHT-sharded and DRAM INTERLEAVED portions."""
+                total_rows = full_w.shape[0]
+                grid = self.mesh_device.compute_with_storage_grid_size()
+                num_cores = grid.x * grid.y
+                alignment = num_cores * 32
+
+                l1_rows_actual = min(total_rows, l1_rows)
+                l1_rows_actual = (l1_rows_actual // alignment) * alignment
+
+                w_l1_torch = full_w[:l1_rows_actual, :]
+                w_dram_torch = full_w[l1_rows_actual:, :]
...
+            # Split WQKV: qkv_cat is [1, 1, Dim, LocalQKV], transpose to [LocalQKV, Dim] for height splitting
+            qkv_cat_local = qkv_cat.squeeze(0).squeeze(0).transpose(-1, -2)
+            self.wqkv_l1, self.wqkv_dram = get_split_tensors(qkv_cat_local, configuration.dim, wqkv_l1_rows)
```

### 5.2 Forward Pass: Hybrid Execution

In `forward_decode`, the matrix multiplication is split into two parallel paths that are later concatenated.

**Path A: L1-Resident Calculation (Transposed Matmul)**
Because $W_{L1}$ is stored as a sharded tensor of shape `[Rows, Cols]`, we perform a "transposed matmul" sequence to align dimensions and keep data in L1:
1.  **Transpose Input $X$**: `[Batch, 1, In]` $\to$ `[Batch, 1, In, 1]` $\to$ `[Batch, 1, 1, In]` (via `transpose(-2, -1)`).
2.  **Matmul**: $W_{L1} \times X^T$ produces partial output $Y_{L1}^T$.
3.  **Transpose Output**: $Y_{L1}^T \to Y_{L1}$ via `transpose(-2, -1)` to return to original shape.

**Path B: DRAM-Resident Calculation (Standard Linear)**
Standard linear operation using the remainder of the weights ($W_{DRAM}$):
1.  Linear: $X \times W_{DRAM} \to Y_{DRAM}$ (using default generic kernel).
2.  **Note**: This path does *not* use explicit transposes.

**Combination**:
The results are concatenated along the last dimension (width/features):
$$ Y_{Total} = \text{Concat}(Y_{L1}, Y_{DRAM}, \text{dim}=-1) $$

---

## 6. `mlp.py` — MLP Module

### 6.1 Initialization

The MLP module applies the same `get_split_tensors` logic to three weight matrices:

-   **W1 (Gate Projection)**: Projects the input embedding up to the hidden dimension. It acts as the "Gate" in the SwiGLU activation function.
-   **W3 (Up Projection)**: Also projects input up to the hidden dimension (often called "Up"). It is multiplied element-wise with the swish-activated Gate output.
-   **W2 (Down Projection)**: Projects the intermediate hidden state back down to the embedding dimension.

**Target Budgets & Space Specification**:
The code explicitly specifies the target L1 space for these shards in `mlp.py`.
-   **W1 & W3**: `target_size_per_core=128 * 1024` (128KB). They share the same `l1_rows` calculation input (`args.dim * 2` row size).
-   **W2**: `target_size_per_core=64 * 1024` (64KB). Uses `(args.hidden_dim // args.num_devices) * 2` row size.

**Code Diff (MLP Logic):**

```python
+        if args.use_l1_weight_sharding and layer_num == 0:
+            w1_w3_l1_rows = args.get_l1_sharded_rows(self.mesh_device, args.dim * 2, target_size_per_core=128 * 1024)
+            w2_l1_rows = args.get_l1_sharded_rows(
+                self.mesh_device, (args.hidden_dim // args.num_devices) * 2, target_size_per_core=64 * 1024
+            )
+
+            self.w1_l1, self.w1_dram = get_split_tensors("w1", args.dim, w1_w3_l1_rows)
+            self.w2_l1, self.w2_dram = get_split_tensors("w2", args.hidden_dim // args.num_devices, w2_l1_rows)
+            self.w3_l1, self.w3_dram = get_split_tensors("w3", args.dim, w1_w3_l1_rows)
```

### 6.2 Forward Pass

For each linear layer (W1, W3, W2), the forward logic checks if L1 sharding is enabled (`self.wX_l1 is not None` and `mode == "decode"`):

1.  **L1 Path (Transposed Matmul)**:
    -   Transpose `X` to `X_T`.
    -   Atomic Matmul: `Y_L1_T = ttnn.matmul(W_L1, X_T)`.
    -   Transpose Output `Y_L1 = ttnn.transpose(Y_L1_T, -2, -1)`.
2.  **DRAM Path (Standard Linear)**: `Linear(X, W_dram) -> Y_dram` (No transposes).
3.  **Merge**: `Concat([Y_l1, Y_dram], dim=-1)`

This hybrid approach allows the most frequently accessed top rows of the weights to stay in the fast L1 cache, while the bulk of the parameters reside in DRAM, optimizing for the limited L1 capacity of the Tenstorrent architecture.

---

## 7. L1 SRAM Layout Visualization

To visualize the impact of Partial L1 Sharding, consider the layout of a single Tensix Core's L1 Memory (1MB).

### 7.1 Without Sharding (Base Case)
Weights are streamed from DRAM through circular buffers (CBs). They do not persist in L1.

**Note on Circular Buffers**: In this scenario, we often allocate **larger** circular buffers (e.g., double or triple buffered) because the entire 1MB (minus reserve) is available. This maximizes bandwidth by hiding DRAM latency.

```text
+------------------------------------------------------+ 0KB
|             L1 Memory (1MB Total)                    |
|                                                      |
|  Reserved (System / Mailbox / FW)                    |
+------------------------------------------------------+ ~100KB
|                                                      |
|  [CB: Inputs]  (Activations from neighbors/DRAM)     |
|                                                      |
|  [CB: Weights] (LARGE Double-buffer, e.g. 100s KB)   |  <-- Can be large
|                                                      |
|  [CB: Outputs] (Results to neighbors/DRAM)           |
|                                                      |
|  [CB: Intermediates] (Math intermediate tiles)       |
|                                                      |
+------------------------------------------------------+ ~800KB
|                                                      |
|               FREE SPACE / UNUSED                    |
|                                                      |
+------------------------------------------------------+ 1MB
```

### 7.2 With Partial L1 Weight Sharding
Top rows of WQKV, WO, W1, W2, W3 are statically allocated in L1. This drastically reduces the available space.

**Impact on Circular Buffers**:
1.  **Static Shards**: Consume a significant fixed portion (e.g., ~400KB total).
2.  **No Weight CB for L1 Part**: The L1-resident calculation (`Path A`) does **not** need a Weight CB; it uses the static shard directly.
3.  **Smaller DRAM Weight CB**: For the `Path B` (DRAM) calculation, we still need a Weight CB, but we must make it **smaller** (e.g., single buffered or smaller blocks) to fit in the remaining space.
4.  **Reduced Free Space**: The margin for error is very small. Over-allocating static shards (like the 1MB WO target we fixed) causes the "Memory Clash" because the remaining space isn't enough even for the minimal CBs needed for `Path B` and Inputs/Outputs.

```text
+------------------------------------------------------+ 0KB
|             L1 Memory (1MB Total)                    |
|  Reserved (System / Mailbox / FW)                    |
+------------------------------------------------------+ ~100KB
|  [Static] WQKV L1 Shard                              |
+------------------------------------------------------+
|  [Static] WO L1 Shard                                |
+------------------------------------------------------+
|  [Static] MLP W1/W3 L1 Shards                        |
+------------------------------------------------------+
|  [Static] MLP W2 L1 Shard                            | <--- (Missed in previous version)
+------------------------------------------------------+ ~560KB (e.g.)
|                                                      |
|  [CB: Inputs]                                        |
|  [CB: DRAM Weights] (MUST be smaller now)            | <-- Smaller CB
|  [CB: Outputs]                                       |
|  [CB: Intermediates]                                 |
|                                                      |
+------------------------------------------------------+ ~980KB
|         Useable Free Space (Example: ~20KB)          | <-- Very tight!
+------------------------------------------------------+ 1MB
```

**Key Constraint**: The sum of `Reserved + Static Shards + Circular Buffers` must be $< 1MB$. The "Memory Clash" bugs occur when this limit is exceeded.

---

### 7.3 Code Audit: Hybrid Strategy & CB Sizing

The "Sharded Case" uses a hybrid strategy that treats L1 and DRAM resident weights differently to fit within memory constraints.

**1. L1 Resident Part (The "Transposed" Path)**
-   **Weight Shape**: `[Out_Rows, In_Cols]`.
-   **Operation**: `Matmul(W_L1, X_T)`.
-   **Memory**: Statically allocated in L1. **No Weight Circular Buffer** is required because the weight is already resident.
-   **Efficiency**: High math utilization, zero DRAM bandwidth for weights.

**2. DRAM Resident Part (The "Standard" Path)**
-   **Weight Shape**: `[In_Cols, Out_Rows]` (Transposed relative to L1 part).
-   **Operation**: `ttnn.linear(X, W_DRAM)`.
-   **Config**: Uses **default** configuration (Generic Kernel), unlike the optimized `ATTN_OUTPUT_PROGCFG` used in the non-sharded case.
-   **CB Implication**:
    -   *Non-Sharded (Optimized)*: `in0_block_w=4` $\to$ **~512KB CB**. (Would crash if used with static shards).
    -   *Sharded DRAM (Generic)*: Default `in0_block_w` (usually 1 or 2) $\to$ **~128KB CB**.
-   **Memory**: The smaller default CB allows this path to run alongside the large static L1 shards without causing an OOM.

**Conclusion**: The user correctly recalled a "special one with transposes" — this is the **L1 Path**. The **DRAM Path** remains a standard linear op, but crucially relies on arguably less-optimized (smaller) buffer sizes to ensure memory safety.

---

## 8. `generator.py` — Decode Forward Fix

### Change 7a: `argmax_on_device` → `sampling_on_device`

Six references to `argmax_on_device` in `_decode_forward_trace_text()` were changed to
`sampling_on_device`. This fixes a `NameError` where the variable had been renamed upstream.
This is **not related to L1 sharding** but was bundled in the same commit.

---

## 8. `common.py` — Model Factory Plumbing

Pass `use_l1_weight_sharding` from the demo/test down to `ModelArgs`.

---

## 9. `conftest.py` — CLI Flag

Adds `--use_l1_weight_sharding` (default `False`) to pytest options.

---

## 10. `simple_text_demo.py` — Demo Plumbing

Reads the CLI flag and passes it to `prepare_generator_args()`.

---

## 11. Bugs Found and Fixed

### Bug 1 (Critical): WQKV `ttnn.linear` in disabled path — attention.py

**Symptom**: `RuntimeError: Input tensor B must have INTERLEAVED memory layout, got: WIDTH_SHARDED`

**Cause**: The `else` branch (sharding disabled) was missing `program_config` and
`compute_kernel_config`, and used `DRAM_MEMORY_CONFIG` instead of `L1_WIDTH_SHARDED_MEMORY_CONFIG`.
Without `program_config`, the matmul kernel couldn't handle the DRAM-sharded weight tensor.

**Fix**: Restored all three original parameters.

### Bug 2 (Critical): WO matmul in disabled path — attention.py

**Symptom**: Would have caused a similar crash at the WO matmul step.

**Cause**: Changed from `ttnn.matmul` to `ttnn.linear` with `core_grid` and `program_config`
commented out.

**Fix**: Restored to `ttnn.matmul` with original `core_grid` and `program_config`.

### Bug 3 (Regression): KV cache guard removed — attention.py

**Symptom**: Double-initialization of KV cache when used with vLLM.

**Cause**: `if not use_paged_kv_cache:` guard was removed.

**Fix**: Guard restored.

### Bug 4 (Regression): `can_enable_trace` disabled — model_config.py

**Symptom**: Prefill tracing never activates, reducing prefill performance.

**Cause**: `return False` was added at top of `can_enable_trace()`.

**Fix**: `return False` removed; original conditional logic restored.

### Bug 5 (Committed + Working tree): Profiling changes across 19 files

**Symptom**: (a) "Event Synchronization is not supported during trace capture" — breaks
prefill tracing. (b) "Weird output text" — silent data corruption.

**Cause**: Two independent issues across 19 files modified since commit `76906ef0`:
- **Trace errors**: `tt::tt_metal::distributed::Synchronize()` calls added to `matmul.cpp`,
  `binary.cpp`, `transpose.cpp`, and `tensor_impl.cpp` to make `ttnn::Timer` measurements
  accurate. These call `enqueue_record_event_to_host()` which is forbidden during trace capture.
- **Data corruption**: `noc_async_read_barrier()` and `noc_async_write_barrier()` calls
  commented out in multiple kernel dataflow files for profiling.
- **Additional trace failures**: `DeviceZoneScopedN` profiling zones and
  `#include "tools/profiler/kernel_profiler.hpp"` in kernel compute/dataflow files.

**Fix**: All 19 files reverted to their state at commit `76906ef0`. See Section 3 for
the full file list. **Rebuild required** for host-side C++ changes to take effect.

### Bug 6 (Working tree): `can_enable_trace(1)` in decode path — generator.py

**Symptom**: Decode tracing always disabled.

**Cause**: A new block in `decode_forward_text()` called `can_enable_trace(1)`, which
always returns `False` (seqlen=1 is not in allowed list), defeating decode tracing.

**Fix**: Block removed.

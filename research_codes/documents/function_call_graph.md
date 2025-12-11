# Function Call Graph for `weight_loading_test.py`

This document traces the execution flow of key `ttnn` APIs used in `weight_loading_test.py`, from the Python interface down to the low-level `tt-metal` implementation.

## 1. Device Management

### `ttnn.open_device`
*   **Python**: `ttnn.open_device` (alias for `ttnn._ttnn.device.open_device`) in `ttnn/ttnn/device.py`
*   **C++ Binding**: `ttnn::open_mesh_device` in `ttnn/cpp/ttnn-pybind/device.cpp`
*   **C++ Implementation**: `ttnn::open_mesh_device` in `ttnn/core/device.cpp`
*   **tt-metal**: `MeshDevice::create_unit_mesh` in `tt_metal/distributed/mesh_device.cpp`
    *   Calls `CreateDevices` in `tt_metal/tt_metal.cpp` to initialize `Device` instances.
    *   Initializes `DevicePool`.

### `ttnn.close_device`
*   **Python**: `ttnn.close_device` (alias for `ttnn._ttnn.device.close_device`) in `ttnn/ttnn/device.py`
*   **C++ Binding**: `ttnn::close_device` in `ttnn/cpp/ttnn-pybind/device.cpp`
*   **C++ Implementation**: `ttnn::close_device` in `ttnn/core/device.cpp`
*   **tt-metal**: `MeshDevice::close` in `tt_metal/distributed/mesh_device.cpp`
    *   Calls `CloseDevices` in `tt_metal/tt_metal.cpp`.

### `ttnn.synchronize_device`
*   **Python**: `ttnn.synchronize_device` (alias for `ttnn._ttnn.device.synchronize_device`)
*   **C++ Binding**: `ttnn::device::synchronize_device` in `ttnn/cpp/ttnn-pybind/device.cpp`
*   **tt-metal**: `tt::tt_metal::distributed::Synchronize` in `tt_metal/distributed/mesh_device.cpp` (implied)

## 2. Tensor Creation & Data Transfer

### `ttnn.from_torch`
*   **Python**: `ttnn.from_torch` in `ttnn/ttnn/operations/core.py`
    *   Wraps `ttnn.Tensor` constructor.
*   **C++ Binding**: `ttnn::Tensor` constructor in `ttnn/cpp/ttnn-pybind/pytensor.cpp`
*   **C++ Implementation**: `ttnn::Tensor` in `ttnn/core/tensor/tensor.cpp`
    *   If `device` argument is provided, calls `to_device_wrapper` in `ttnn/core/tensor/tensor_impl.cpp`.
*   **tt-metal**: `ttnn::to_device` in `ttnn/core/tensor/tensor_impl.cpp`
    *   Calls `allocate_device_buffer` and `to_device_mesh_buffer`.
    *   Uses `EnqueueWriteBuffer` (via `MeshCommandQueue`) to transfer data to device memory.

### `ttnn.to_device`
*   **Python**: `ttnn.to_device`
*   **C++ Binding**: `ttnn::to_device`
*   **C++ Implementation**: `ttnn::to_device` in `ttnn/core/tensor/tensor_impl.cpp`
    *   Similar flow to `from_torch` when moving to device.

## 3. Matrix Multiplication (`ttnn.linear` / `ttnn.matmul`)

### `ttnn.linear`
*   **Python**: `ttnn.linear` in `ttnn/ttnn/operations/matmul.py`
*   **C++ Binding**: `ttnn::linear` in `ttnn/cpp/ttnn-pybind/matmul.cpp`
*   **C++ Implementation**: `ttnn::operations::matmul::LinearOperation::invoke` in `ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp`
    *   Calls `bound_matmul`.

### `ttnn.matmul`
*   **Python**: `ttnn.matmul` in `ttnn/ttnn/operations/matmul.py`
*   **C++ Binding**: `ttnn::matmul` in `ttnn/cpp/ttnn-pybind/matmul.cpp`
*   **C++ Implementation**: `ttnn::operations::matmul::MatmulOperation::invoke` in `ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp`
    *   Calls `bound_matmul`.

### Core Matmul Logic (`bound_matmul`)
*   **Location**: `ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp`
*   **Logic**:
    1.  **Program Config Selection**: Calls `get_program_config` to determine the optimal `MatmulProgramConfig` if not provided. This considers tensor shapes, memory config, and hardware constraints.
    2.  **Kernel Selection**: Based on `MatmulProgramConfig`, selects the specific kernel implementation:
        *   `MatmulMultiCoreReuseProgramConfig` -> `matmul_multi_core_reuse`
        *   `MatmulMultiCoreReuseMultiCastProgramConfig` -> `matmul_multi_core_reuse_mcast`
        *   `MatmulMultiCoreReuseMultiCast1DProgramConfig` -> `matmul_multi_core_reuse_mcast_1d_optimized`
        *   `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` -> `matmul_multi_core_reuse_dram_sharded_optimized`
    3.  **Execution**: Calls `operation::run` with the created `Matmul` or `SparseMatmul` device operation.
*   **tt-metal**: `Matmul::create_mesh_workload` (and `create_program`) in `ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp`
    *   Constructs the tt-metal `Program`.
    *   Allocates circular buffers (`CreateCircularBuffer`).
    *   Creates compute and data movement kernels (`CreateKernel`).
    *   Configures runtime arguments (`SetRuntimeArgs`).

## 4. Data Movement & Manipulation

### `ttnn.transpose`
*   **Python**: `ttnn.transpose` in `ttnn/ttnn/operations/data_movement.py`
*   **C++ Binding**: `ttnn::transpose` in `ttnn/cpp/ttnn-pybind/operations/data_movement/data_movement_pybind.cpp`
*   **C++ Implementation**: `ttnn::operations::data_movement::ExecuteTranspose::invoke` in `ttnn/cpp/ttnn/operations/data_movement/transpose/transpose.cpp`
    *   Calls `detail::transpose_`.
    *   May use `ttnn::prim::permute` (which uses `PermuteDeviceOperation`) or `operation::run(Transpose)` depending on the case.

### `ttnn.reshape`
*   **Python**: `ttnn.reshape` in `ttnn/ttnn/operations/data_movement.py`
*   **C++ Binding**: `ttnn::reshape` in `ttnn/cpp/ttnn-pybind/operations/data_movement/data_movement_pybind.cpp`
*   **C++ Implementation**: `ttnn::operations::data_movement::ReshapeViewOperation::invoke` in `ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp`
    *   Attempts metadata-only view update first.
    *   If data movement is needed:
        *   `detail::reshape_rm` (Row Major)
        *   `detail::reshape_tiled` (Tiled) -> `operation::run(ReshapeDeviceOperation)`

### `ttnn.slice`
*   **Python**: `ttnn.slice` in `ttnn/ttnn/operations/data_movement.py`
*   **C++ Binding**: `ttnn::slice` in `ttnn/cpp/ttnn/operations/data_movement/slice/slice_pybind.cpp`
*   **C++ Implementation**: `ttnn::operations::data_movement::SliceOperation::invoke` in `ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp`
    *   Calls `operation::run(SliceDeviceOperation)`.

### `ttnn.add`
*   **Python**: `ttnn.add` in `ttnn/ttnn/operations/binary.py`
*   **C++ Binding**: `ttnn::add` in `ttnn/cpp/ttnn/operations/eltwise/binary/binary_pybind.cpp`
*   **C++ Implementation**: `ttnn::operations::binary::BinaryOperation<ADD>::invoke` in `ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp`
    *   Calls `detail::invoke_binary_ng`.
    *   Calls `ttnn::prim::binary_ng` -> `operation::run(BinaryNgDeviceOperation)`.

## 5. MatmulProgramConfig Influence

The `MatmulProgramConfig` is the central mechanism for controlling how matrix multiplication is executed on the hardware. It is determined in `get_program_config` (in `matmul_op.cpp`) or passed explicitly by the user.

*   **`MatmulMultiCoreReuseProgramConfig`**: Basic multi-core implementation where weights are reused.
*   **`MatmulMultiCoreReuseMultiCastProgramConfig`**: Uses multicast to broadcast weights to multiple cores, reducing memory bandwidth usage.
*   **`MatmulMultiCoreReuseMultiCast1DProgramConfig`**: Optimized for 1D tensor slicing/distribution, often used when one dimension is small or for specific parallelization strategies.
*   **`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`**: Optimized for cases where input/output tensors are sharded in DRAM, minimizing data movement between DRAM and L1.

The selection affects:
*   **Grid Size**: How many cores are used.
*   **Block Sizes**: The size of data chunks processed per core (`per_core_M`, `per_core_N`, `in0_block_w`).
*   **Data Movement**: Whether to use multicast, how data is sharded, and how it flows between L1 and DRAM.
*   **Kernel Code**: Which specific compute and data movement kernels are compiled and run.

# Profiling Mechanism Documentation

## Overview
This document details the implementation of iteration-aware profiling for the `weight_loading_test.py` benchmark. The goal was to distinguish between "Large Batch" (single pass) and "Minibatch" (repeated passes) execution in the kernel profiling logs to accurately measure performance characteristics of each phase.

## Implementation Components

### 1. Host-Side Instrumentation (Python)
We modified `weight_loading_test.py` to inject a unique iteration identifier into the device's L1 memory before every forward pass.

- **Function**: `write_iter_to_l1(device, iter_idx)`
- **Mechanism**: Writes the integer `iter_idx` to L1 address `100000`.
- **Address Selection**: Address `100000` was chosen to avoid conflicts with reserved memory (Mailbox, Firmware, etc.) which reside at lower addresses.
- **Identifiers**:
    - `0...99`: Reserved for Large Batch / Warmup passes.
    - `100+`: Reserved for Minibatch sequences.

### 2. Device-Side Binding Fix (C++)
The `WriteToDeviceL1` Python binding required updates to support the `MeshDevice` architecture used in the test.

- **File**: `ttnn/cpp/ttnn-pybind/device.cpp`
- **Issue**: The original binding expected a single device and failed when passed a `MeshDevice`.
- **Fix**: Updated the binding to accept `MeshDevice*` and iterate over all underlying physical devices (chips) to perform the write operation.

```cpp
// ttnn/cpp/ttnn-pybind/device.cpp
m_device.def("WriteToDeviceL1", [](MeshDevice* mesh_device, ...) {
    for (auto* device : mesh_device->get_devices()) {
        // Write to each chip in the mesh
        tt::tt_metal::detail::WriteToDeviceL1(device, ...);
    }
});
```

### 3. Kernel-Side Instrumentation (Compute Kernel)
We updated the compute kernel to read this identifier and log it into the trace buffer.

- **File**: `bmm_large_block_zm_fused_bias_activation.cpp`
- **Mechanism**:
    1.  Read the value at L1 address `100000`.
    2.  Log it using `DeviceTimestampedData`.

```cpp
// MAIN function
volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(100000);
uint32_t iter_idx = *ptr;
DeviceTimestampedData("FORWARD_PASS", (uint64_t)iter_idx);
```

This emits a `TS_DATA` packet into the profiling log containing the iteration ID.

## Analysis Logic: Matching Indices to Zones

**Q: How does the new "FORWARD_PASS" index match to other zones (e.g., `BATCH-ITERATION`)?**

The matching is achieved through **sequential state tracking** during log parsing. The Profiler logs events linearly in time for each RISC processor.

1.  **Sequential Execution**: The kernel code executes linearly.
    -   First, it reads the L1 value and logs `TS_DATA` ("FORWARD_PASS").
    -   *Then*, it enters the compute loop and logs `ZONE_START` / `ZONE_END` for `BATCH-ITERATION`.
2.  **Log Parsing (`analyze_detailed_zones.py`)**:
    -   The script reads the CSV log file line by line (which is sorted by time/cycle).
    -   It maintains a state dictionary: `core_iter_map[(core, risc)] = current_iter_id`.
    -   **Step A (Trigger)**: When the script encounters the `TS_DATA` line, it updates the `current_iter_id` for that core.
    -   **Step B (Association)**: When the script subsequently encounters a `ZONE_END` for `BATCH-ITERATION` (or any other zone), it looks up the `current_iter_id` for that core and tags the zone duration with it.

### Visualization of Log Stream
```text
[Time T1] TS_DATA: 100          <-- Script updates state: current_iter = 100
[Time T2] ZONE_START: MATMUL
[Time T3] ZONE_END: MATMUL      <-- Script sees matching START, calculates duration, assigns to iter 100
[Time T4] TS_DATA: 101          <-- Script updates state: current_iter = 101
...
```

By ensuring the `TS_DATA` event happens *before* the workload zones in the kernel code, the analysis script can correctly attribute all subsequent zones to that specific iteration until a new `TS_DATA` event is seen.

## Results Summary
Using this mechanism, we successfully differentiated:
-   **Large Batch Phase**: ~2.71 ms (Iter ID < 100)
-   **Minibatch Phase**: ~0.16 ms (Iter ID >= 100)

This confirms the "warmup" vs "measure" phases and allows for granular latency analysis per distinct forward pass.

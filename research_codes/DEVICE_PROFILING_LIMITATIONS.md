# Device Profiling Limitations at Scale

## The Problem

Even with **ALL custom zones disabled**, you're still getting:
- Buffer overflow warnings
- Marker mismatch errors ("Start 62262 and end 1026 markers do not match")

## Root Cause

### Default Firmware Zones (Cannot Be Disabled)

The firmware automatically creates guaranteed zones that **cannot be disabled** from kernel code:

- `BRISC-FW` / `BRISC-KERNEL` - Created by firmware for every BRISC kernel
- `NCRISC-FW` / `NCRISC-KERNEL` - Created by firmware for every NCRISC kernel
- `TRISC-FW` / `TRISC-KERNEL` - Created by firmware for every TRISC kernel

These use **guaranteed marker slots** in the profiler buffer and are always active when `TT_METAL_DEVICE_PROFILER=1` is set.

### The Math

With your workload:
- **130+ cores** (worker cores)
- **3 RISC types** per core (BRISC, NCRISC, TRISC)
- **5 measurement iterations**
- **2-3 default zones** per RISC per iteration

**Total zones**: 130 cores × 3 RISC × 2 zones × 5 iterations = **~3,900 zones minimum**

Even with just default zones, this exceeds the profiler buffer capacity.

## Current State

✅ **All custom zones disabled**:
- Compute kernel: `TRISC-MATMUL-FUSED-COMPUTE` - disabled
- All reader kernels: All zones disabled

❌ **Default firmware zones still active** (cannot be disabled)

## Solutions

### Option 1: Disable Device Profiling Entirely (Recommended)

For this workload scale, **device profiling is not feasible**. Use host-side timing instead:

```python
import time

# In your weight_loading_test.py
t0 = time.perf_counter()
result = linear(x_tt)
ttnn.synchronize_device(device)
t1 = time.perf_counter()
print(f"Forward pass: {(t1-t0)*1000:.3f}ms")
```

**Pros**:
- ✅ No buffer overflow
- ✅ Reliable measurements
- ✅ Simple to implement

**Cons**:
- ❌ No per-kernel breakdown
- ❌ No RISC-level timing

### Option 2: Reduce Iterations Drastically

```bash
export TT_METAL_DEVICE_PROFILER=1
python3 research_codes/weight_loading_test.py --only-large --measure-iters 1 --warmup-iters 0
```

This reduces zones to: 130 × 3 × 2 × 1 = **~780 zones** (may still overflow)

### Option 3: Profile Only Specific Cores (Advanced)

Modify the profiler to only profile a subset of cores. This requires changes to the profiler infrastructure code (not recommended unless you're familiar with the profiler internals).

### Option 4: Use Built-in Default Zones Only

Since default zones are always active, you can:
1. **Don't add any custom zones** (already done)
2. **Use only the default zones** in analysis
3. **Accept that some data may be dropped** (buffer overflow warnings)

The default zones (`BRISC-KERNEL`, `NCRISC-KERNEL`, `TRISC-KERNEL`) will still appear in the profile log, even if corrupted.

## Recommendation

**For your use case (measuring weight loading overhead)**, use **host-side timing**:

```python
# Measure overall forward pass time
t0 = time.perf_counter()
result = linear(x_tt)
ttnn.synchronize_device(device)
t1 = time.perf_counter()
forward_time_ms = (t1 - t0) * 1000.0

# Compare large batch vs minibatch
# This gives you the overhead you need without device profiling complexity
```

This is:
- ✅ More reliable
- ✅ Simpler
- ✅ Sufficient for your benchmarking needs
- ✅ No buffer overflow issues

## Why Device Profiling Fails at This Scale

The device profiler was designed for:
- **Small-scale debugging** (few cores, few iterations)
- **Selective profiling** (specific kernels, specific cores)
- **Development/debugging** scenarios

It was **not designed** for:
- **Production workloads** with 130+ cores
- **Benchmarking** with multiple iterations
- **Large-scale performance analysis**

## Alternative: Use Built-in Performance Counters

If available, use hardware performance counters instead of the device profiler. These are designed for production workloads and don't have buffer limitations.

## Summary

**Device profiling is not suitable for your workload scale.** Use host-side timing for reliable measurements of weight loading overhead.

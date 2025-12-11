# Profiler Buffer Overflow Fix

## Problem

You're experiencing:
1. **Profiler buffer overflow**: "Profiler DRAM buffers were full, markers were dropped!"
2. **Profiler corruption**: "Start 1337 and end 24300 markers do not match"

## Root Cause

With **130+ cores** running simultaneously, even a few zones per core quickly fills the profiler buffer. Each zone generates:
- Start marker (2 uint32_t)
- End marker (2 uint32_t)
- Plus overhead

With nested zones and multiple iterations, this multiplies rapidly.

## Solution: Minimal Profiling

### Option 1: Only Main Zone (Recommended)

Keep **ONLY** the main zone in the compute kernel:

```cpp
void MAIN {
    DeviceZoneScopedMainChildN("TRISC-MATMUL-FUSED-COMPUTE");
    // ... rest of code without any nested zones
}
```

**Status**: ✅ Already done - only main zone remains active.

### Option 2: Disable Reader Kernel Zones

The reader kernels are also generating zones. You may need to comment them out too:

**Files to check**:
- `reader_bmm_tile_layout.cpp`
- `reader_bmm_tile_layout_in0_sender_padding.cpp`
- `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`
- etc.

**Quick fix**: Comment out all `DeviceZoneScopedN` calls in reader kernels, keep only main zones if needed.

### Option 3: Reduce Measurement Iterations

Instead of 5 measurement iterations, use 1-2:

```bash
python3 research_codes/weight_loading_test.py --only-large --measure-iters 1
```

### Option 4: Profile Only Specific Cores

If possible, limit profiling to a subset of cores (this may require code changes in the profiler setup).

## Current State

✅ **Compute kernel**: Only main zone (`TRISC-MATMUL-FUSED-COMPUTE`) is active
- `MM-BLOCK-INIT` - removed
- `BATCH-ITERATION` - commented out
- `CB-WAIT-FRONT` - commented out
- `CB-POP-FRONT` - commented out
- `FUSE-BIAS` - commented out

## Next Steps

1. **Test with only main zone**:
   ```bash
   rm -rf ~/.cache/tt-metal-cache/*
   export TT_METAL_DEVICE_PROFILER=1
   python3 research_codes/weight_loading_test.py --only-large --measure-iters 1
   ```

2. **If still overflowing, disable reader kernel zones**:
   - Comment out zones in reader kernels
   - Keep only main zones if you need to track which kernels run

3. **If still overflowing, reduce iterations**:
   - Use `--measure-iters 1`
   - Use `--warmup-iters 1`

## Why This Happens

The profiler buffer is shared across all cores. With:
- 130 cores
- Multiple zones per kernel
- Multiple iterations
- Nested zones

The buffer fills up quickly. The profiler tries to flush to DRAM, but if it can't keep up, markers get dropped, causing corruption.

## Alternative: Use Host-Side Profiling

If you need detailed timing, consider using Python-level timing instead of device profiling for this workload:

```python
import time
t0 = time.perf_counter()
result = linear(x_tt)
ttnn.synchronize_device(device)
t1 = time.perf_counter()
print(f"Forward pass: {(t1-t0)*1000:.3f}ms")
```

This won't give you per-kernel breakdown, but it's more reliable for large workloads.

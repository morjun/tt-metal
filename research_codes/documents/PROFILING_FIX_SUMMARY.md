# Profiling Instrumentation Fix Summary

## What Was Fixed

### 1. ✅ Uncommented Profiling in `bmm_large_block_zm_fused_bias_activation.cpp`

**File**: `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`

**Changes**:
- ✅ Uncommented `#include "tools/profiler/kernel_profiler.hpp"`
- ✅ Uncommented main zone: `DeviceZoneScopedMainChildN("TRISC-MATMUL-FUSED-COMPUTE")`
- ✅ Added `MM-BLOCK-INIT` zone (matmul initialization)
- ✅ Added `BATCH-ITERATION` zone (each batch loop)
- ✅ Added `CB-WAIT-FRONT` zone (circular buffer wait - measures idle time)
- ✅ Added `CB-POP-FRONT` zone (circular buffer pop)
- ✅ Added `FUSE-BIAS` zone (bias fusion section)

**Note**: The compute zone (`MATMUL-BLOCK`) was intentionally left out to reduce profiler buffer usage (as noted in the original comment).

### 2. ✅ Reader Kernels Already Instrumented

The following reader kernels already have profiling zones:
- ✅ `reader_bmm_tile_layout.cpp` - Has `BRISC-MATMUL-READER`, `READ-IN1-WEIGHT`, `NOC-BARRIER-WAIT`
- ✅ `reader_bmm_tile_layout_in0.cpp` - Has `BRISC-MATMUL-READER-IN0`, `NOC-BARRIER-WAIT`
- ✅ `reader_bmm_tile_layout_in0_sender_padding.cpp` - Has `BRISC-MATMUL-READER-IN0-SENDER`, `READ-IN0-DRAM-TO-SRAM`, `NOC-BARRIER-WAIT`
- ✅ `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` - Has `BRISC-MATMUL-READER-WRITER-IN1-SENDER`, `READ-WEIGHT-DRAM-TO-SRAM`, `NOC-BARRIER-WAIT`
- ✅ `reader_writer_bmm_tile_layout_in1.cpp` - Has `BRISC-MATMUL-READER-WRITER-IN1`, `READ-WEIGHT-DRAM-TO-SRAM`, `NOC-BARRIER-WAIT`

### 3. ✅ matmul.cpp Does NOT Need Zones

**Answer**: No, you should NOT add zones to `matmul.cpp`.

**Reason**: `matmul.cpp` is host-side C++ code that orchestrates kernel launches. Device profiling zones only work in device kernels (BRISC, NCRISC, TRISC) that run on the device cores. The `matmul.cpp` file runs on the host CPU and just sets up and launches kernels.

## Why Zones Weren't Appearing

### Primary Issue: PROFILE_KERNEL Not Defined

The profiling macros (`DeviceZoneScopedN`, etc.) are conditionally compiled. They only work when:
1. `TT_METAL_DEVICE_PROFILER=1` is set (defines `PROFILE_KERNEL`)
2. Kernels are recompiled with this flag
3. Old cached kernels are cleared

### Secondary Issue: Wrong Kernel File

You had instrumented `bmm_large_block_zm.cpp`, but the actual kernel used by `ttnn.linear()` with bias is `bmm_large_block_zm_fused_bias_activation.cpp`. This is now fixed.

## Next Steps

### Step 1: Clear Kernel Cache
```bash
rm -rf ~/.cache/tt-metal-cache/*
```

### Step 2: Set Environment Variable
```bash
export TT_METAL_DEVICE_PROFILER=1
```

### Step 3: Run Your Test
```bash
python3 research_codes/weight_loading_test.py --only-large
```

### Step 4: Verify Zones Appear
```bash
# Check compute kernel zones
grep -E "TRISC-MATMUL-FUSED|MM-BLOCK-INIT|BATCH-ITERATION|CB-WAIT-FRONT|FUSE-BIAS" \
  generated/profiler/.logs/profile_log_device.csv | head -20

# Check reader kernel zones
grep -E "BRISC-MATMUL-READER|READ-WEIGHT|READ-IN1-WEIGHT|NOC-BARRIER-WAIT" \
  generated/profiler/.logs/profile_log_device.csv | head -20
```

### Step 5: Verify Build Logs
```bash
# Check if kernels were compiled with PROFILE_KERNEL
find ~/.cache/tt-metal-cache/ -path "*bmm_large_block_zm_fused*/trisc*/build.log" | \
  xargs grep -i "PROFILE_KERNEL\|-DPROFILE" | head -3

# Check if zones were registered
grep -i "TRISC-MATMUL-FUSED\|MM-BLOCK-INIT\|BATCH-ITERATION" \
  generated/profiler/.logs/new_zone_src_locations.log | head -10
```

## Expected Zones in Profile

After running with `TT_METAL_DEVICE_PROFILER=1`, you should see:

### Compute Kernel (TRISC):
```
TRISC-MATMUL-FUSED-COMPUTE
  MM-BLOCK-INIT
  BATCH-ITERATION
    CB-WAIT-FRONT          ← Idle time waiting for data
    CB-POP-FRONT
    FUSE-BIAS              ← If bias is used
```

### Reader Kernels (BRISC):
```
BRISC-MATMUL-READER-IN0-SENDER (or similar)
  READ-IN0-DRAM-TO-SRAM
  NOC-BARRIER-WAIT

BRISC-MATMUL-READER-WRITER-IN1-SENDER (or similar)
  READ-WEIGHT-DRAM-TO-SRAM
  NOC-BARRIER-WAIT
```

## Troubleshooting

If zones still don't appear:

1. **Check environment variable**:
   ```bash
   echo $TT_METAL_DEVICE_PROFILER
   # Should output: 1
   ```

2. **Check build logs for errors**:
   ```bash
   find ~/.cache/tt-metal-cache/ -name "build.log" -exec grep -l "error\|Error\|ERROR" {} \; | head -3
   ```

3. **Check which matmul factory is used**:
   The matmul operation uses different program factories depending on tensor shapes and memory config. Different factories use different reader kernels. All the common ones are already instrumented.

4. **Check profiler buffer overflow**:
   If you see "dropped zones" in the profile log, you may have too many zones. We've kept the zones minimal to avoid this.

## Files Modified

1. ✅ `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`
   - Uncommented profiling include
   - Uncommented main zone
   - Added key profiling zones

2. ✅ Reader kernels (already had profiling - no changes needed)

3. ❌ `matmul.cpp` - No changes needed (host-side code)

## Additional Resources

- See `PROFILING_DEBUG_GUIDE.md` for detailed troubleshooting
- See `CUSTOM_ZONE_ADDED_SUMMARY.md` for original instrumentation plan
- See `WRONG_KERNEL_FILE.md` for explanation of kernel selection

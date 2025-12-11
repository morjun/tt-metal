# Profiling Debug Guide: Why Zones Aren't Appearing

## Problem Summary
You've instrumented custom profiling zones in kernels used by `ttnn.linear()`, but they're not appearing in `profile_log_device.log`.

## Root Causes & Solutions

### 1. **PROFILE_KERNEL Not Defined** (Most Common Issue)

**Problem**: The profiling macros (`DeviceZoneScopedN`, etc.) are only active when `PROFILE_KERNEL` is defined during kernel compilation. Without it, all macros become no-ops.

**Solution**:
```bash
# MUST set this BEFORE running your script
export TT_METAL_DEVICE_PROFILER=1

# Then run your script
python3 research_codes/weight_loading_test.py --only-large
```

**Verify it's working**:
```bash
# Check if kernels were compiled with PROFILE_KERNEL
find ~/.cache/tt-metal-cache/ -path "*bmm_large_block_zm_fused*/trisc*/build.log" | \
  xargs grep -i "PROFILE_KERNEL\|-DPROFILE" | head -5
```

### 2. **Kernel Cache Issues**

**Problem**: Old kernels without profiling are cached and reused.

**Solution**:
```bash
# Delete kernel cache
rm -rf ~/.cache/tt-metal-cache/*

# Then run with TT_METAL_DEVICE_PROFILER=1
export TT_METAL_DEVICE_PROFILER=1
python3 research_codes/weight_loading_test.py --only-large
```

### 3. **Wrong Kernel File Instrumented**

**Problem**: You might have instrumented `bmm_large_block_zm.cpp` but the actual kernel used is `bmm_large_block_zm_fused_bias_activation.cpp`.

**Solution**: ✅ **FIXED** - We've now instrumented the correct file:
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`

### 4. **Reader Kernels Not Being Used**

**Problem**: Multiple reader kernel variants exist. The one you instrumented might not be the one actually called.

**Solution**: Check which reader kernels are actually used:
```bash
# After running with profiling, check which kernels were built
find ~/.cache/tt-metal-cache/ -name "build.log" | \
  xargs grep -l "reader_bmm" | head -5

# Check which reader kernels have profiling zones
grep -r "DeviceZoneScoped" \
  ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/ | \
  grep -E "reader.*\.cpp"
```

**Reader kernels that should have profiling**:
- `reader_bmm_tile_layout.cpp` ✅ (has profiling)
- `reader_bmm_tile_layout_in0.cpp` ✅ (has profiling)
- `reader_bmm_tile_layout_in0_sender_padding.cpp` ✅ (has profiling)
- `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` ✅ (has profiling)
- `reader_writer_bmm_tile_layout_in1.cpp` ✅ (has profiling)

### 5. **matmul.cpp Doesn't Need Zones**

**Question**: Should you add zones to `matmul.cpp`?

**Answer**: **NO** - `matmul.cpp` is host-side C++ code, not device kernel code. Device profiling zones only work in device kernels (BRISC, NCRISC, TRISC). The `matmul.cpp` file orchestrates kernel launches but doesn't run on device.

### 6. **Profiling Buffer Overflow**

**Problem**: Too many zones can cause buffer overflow, causing zones to be dropped.

**Solution**: We've added strategic zones without overloading:
- Main zone: `TRISC-MATMUL-FUSED-COMPUTE`
- Key zones: `MM-BLOCK-INIT`, `BATCH-ITERATION`, `CB-WAIT-FRONT`, `CB-POP-FRONT`, `FUSE-BIAS`

**Note**: The compute zone (`MATMUL-BLOCK`) was intentionally left out to reduce buffer usage (see comment in code).

## Verification Steps

### Step 1: Check Environment Variable
```bash
echo $TT_METAL_DEVICE_PROFILER
# Should output: 1
```

### Step 2: Check Build Logs
```bash
# Find build logs for the compute kernel
BUILD_LOG=$(find ~/.cache/tt-metal-cache/ -path "*bmm_large_block_zm_fused*/trisc*/build.log" | head -1)

if [ -n "$BUILD_LOG" ]; then
  echo "Checking build log: $BUILD_LOG"
  grep -i "PROFILE_KERNEL\|-DPROFILE" "$BUILD_LOG" | head -3
  grep -i "TRISC-MATMUL-FUSED\|MM-BLOCK-INIT" "$BUILD_LOG" | head -3
else
  echo "No build log found - kernel might not have been compiled yet"
fi
```

### Step 3: Check Zone Registration
```bash
# Check if zones were registered
grep -i "TRISC-MATMUL-FUSED\|MM-BLOCK-INIT\|BATCH-ITERATION\|CB-WAIT-FRONT\|FUSE-BIAS" \
  generated/profiler/.logs/new_zone_src_locations.log | head -10
```

### Step 4: Check Profile Output
```bash
# Check if zones appear in profile log
grep -E "TRISC-MATMUL-FUSED|MM-BLOCK-INIT|BATCH-ITERATION|CB-WAIT-FRONT|FUSE-BIAS" \
  generated/profiler/.logs/profile_log_device.csv | head -20
```

### Step 5: Check Reader Kernels
```bash
# Check if reader kernel zones appear
grep -E "BRISC-MATMUL-READER|READ-IN1-WEIGHT|READ-WEIGHT|NOC-BARRIER-WAIT" \
  generated/profiler/.logs/profile_log_device.csv | head -20
```

## Expected Zones in Profile Log

After running with `TT_METAL_DEVICE_PROFILER=1`, you should see:

### Compute Kernel (TRISC):
- `TRISC-MATMUL-FUSED-COMPUTE` (main zone)
- `MM-BLOCK-INIT`
- `BATCH-ITERATION`
- `CB-WAIT-FRONT` (idle time waiting for data)
- `CB-POP-FRONT`
- `FUSE-BIAS` (if bias is used)

### Reader Kernels (BRISC):
- `BRISC-MATMUL-READER` or variants
- `READ-IN1-WEIGHT` or `READ-WEIGHT-DRAM-TO-SRAM`
- `NOC-BARRIER-WAIT`

## Troubleshooting Checklist

- [ ] `TT_METAL_DEVICE_PROFILER=1` is set before running
- [ ] Kernel cache cleared: `rm -rf ~/.cache/tt-metal-cache/*`
- [ ] Correct kernel file instrumented: `bmm_large_block_zm_fused_bias_activation.cpp`
- [ ] Build log shows `PROFILE_KERNEL` defined
- [ ] Zones appear in `new_zone_src_locations.log`
- [ ] Profile log file exists: `generated/profiler/.logs/profile_log_device.csv`

## Common Errors

### "Marking errors" when instrumenting
This usually means:
1. The include is missing: `#include "tools/profiler/kernel_profiler.hpp"`
2. The macro is used outside a function scope
3. There's a syntax error in the zone name (must be a string literal)

### Zones appear in build log but not in profile
This means:
1. The kernel was compiled with profiling, but `TT_METAL_DEVICE_PROFILER=1` wasn't set at runtime
2. The profiler buffer overflowed (too many zones)
3. The kernel wasn't actually executed (wrong execution path)

## Next Steps

1. **Clear cache and rebuild**:
   ```bash
   rm -rf ~/.cache/tt-metal-cache/*
   export TT_METAL_DEVICE_PROFILER=1
   python3 research_codes/weight_loading_test.py --only-large
   ```

2. **Verify zones appear**:
   ```bash
   grep -E "TRISC-MATMUL-FUSED|MM-BLOCK-INIT|BATCH-ITERATION" \
     generated/profiler/.logs/profile_log_device.csv | head -10
   ```

3. **If still not appearing**, check:
   - Build logs for compilation errors
   - `new_zone_src_locations.log` for zone registration
   - Which matmul program factory is being used (affects which kernels are called)

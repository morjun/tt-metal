#!/bin/bash
# Debug script to diagnose why custom zones are not appearing

set -e

echo "========================================="
echo "Custom Zone Debugging"
echo "========================================="
echo ""

# Step 1: Check environment variable
echo "1. Checking TT_METAL_DEVICE_PROFILER environment variable..."
if [ -z "$TT_METAL_DEVICE_PROFILER" ]; then
    echo "   ❌ TT_METAL_DEVICE_PROFILER is NOT set!"
    echo "   → This is the problem! Custom zones require PROFILE_KERNEL to be defined."
    echo ""
    echo "   SOLUTION: Set the environment variable:"
    echo "   export TT_METAL_DEVICE_PROFILER=1"
    echo ""
else
    echo "   ✓ TT_METAL_DEVICE_PROFILER=$TT_METAL_DEVICE_PROFILER"
fi

echo ""

# Step 2: Check source code
echo "2. Checking if custom zones are in source code..."
SOURCE_FILE="ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm.cpp"
if [ -f "$SOURCE_FILE" ]; then
    ZONE_COUNT=$(grep -c "DeviceZoneScopedN\|DeviceZoneScopedMainChildN" "$SOURCE_FILE" || echo "0")
    echo "   Found $ZONE_COUNT profiling zone macros in $SOURCE_FILE"
    if [ "$ZONE_COUNT" -gt "0" ]; then
        echo "   Sample zones:"
        grep "DeviceZoneScopedN\|DeviceZoneScopedMainChildN" "$SOURCE_FILE" | head -5 | sed 's/^/     /'
    fi
else
    echo "   ❌ Source file not found: $SOURCE_FILE"
fi

echo ""

# Step 3: Check JIT cache
echo "3. Checking JIT build cache..."
CACHE_DIR="${HOME}/.cache/tt-metal-cache"
if [ -d "$CACHE_DIR" ]; then
    echo "   Cache directory: $CACHE_DIR"
    CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
    echo "   Cache size: $CACHE_SIZE"

    # Find bmm_large_block_zm kernel
    KERNEL_CACHE=$(find "$CACHE_DIR" -path "*kernels/bmm_large_block_zm*" -type d 2>/dev/null | head -1)
    if [ -n "$KERNEL_CACHE" ]; then
        echo "   ✓ Found kernel cache: $KERNEL_CACHE"

        # Check build.log for PROFILE_KERNEL
        BUILD_LOG=$(find "$KERNEL_CACHE" -name "build.log" -type f 2>/dev/null | head -1)
        if [ -n "$BUILD_LOG" ]; then
            echo "   Checking build.log for PROFILE_KERNEL..."
            if grep -q "PROFILE_KERNEL\|-DPROFILE" "$BUILD_LOG" 2>/dev/null; then
                echo "   ✓ PROFILE_KERNEL found in build.log"
                grep "PROFILE_KERNEL\|-DPROFILE" "$BUILD_LOG" | head -3 | sed 's/^/     /'
            else
                echo "   ❌ PROFILE_KERNEL NOT found in build.log"
                echo "   → Kernel was compiled without PROFILE_KERNEL!"
                echo ""
                echo "   SOLUTION: Delete cache and run with TT_METAL_DEVICE_PROFILER=1"
            fi

            # Check for custom zone pragma messages
            echo ""
            echo "   Checking for custom zone pragma messages..."
            CUSTOM_ZONES=$(grep -i "KERNEL_PROFILER" "$BUILD_LOG" 2>/dev/null | grep -i "MM-INIT\|BATCH-ITERATION\|TRISC-MATMUL\|bmm_large" || echo "")
            if [ -n "$CUSTOM_ZONES" ]; then
                echo "   ✓ Found custom zones in build.log:"
                echo "$CUSTOM_ZONES" | head -5 | sed 's/^/     /'
            else
                echo "   ❌ No custom zones found in build.log"
                echo "   → Custom zones were not compiled!"
            fi
        else
            echo "   ⚠ No build.log found in kernel cache"
        fi
    else
        echo "   ⚠ Kernel cache not found (kernel may not have been compiled yet)"
    fi
else
    echo "   ⚠ Cache directory not found"
fi

echo ""

# Step 4: Check new_zone_src_locations.log
echo "4. Checking new_zone_src_locations.log..."
LOG_FILE="generated/profiler/.logs/new_zone_src_locations.log"
if [ -f "$LOG_FILE" ]; then
    CUSTOM_COUNT=$(grep -i "MM-INIT\|BATCH-ITERATION\|TRISC-MATMUL-COMPUTE\|bmm_large" "$LOG_FILE" 2>/dev/null | wc -l || echo "0")
    if [ "$CUSTOM_COUNT" -gt "0" ]; then
        echo "   ✓ Found $CUSTOM_COUNT custom zones in log file"
        grep -i "MM-INIT\|BATCH-ITERATION\|TRISC-MATMUL-COMPUTE\|bmm_large" "$LOG_FILE" | head -5 | sed 's/^/     /'
    else
        echo "   ❌ No custom zones found in log file"
        echo "   → Custom zones were not extracted from build log!"
    fi
else
    echo "   ⚠ Log file not found: $LOG_FILE"
fi

echo ""

# Step 5: Check profile_log_device.csv
echo "5. Checking profile_log_device.csv..."
CSV_FILE="generated/profiler/.logs/profile_log_device.csv"
if [ -f "$CSV_FILE" ]; then
    CUSTOM_COUNT=$(grep -E "MM-INIT|BATCH-ITERATION|TRISC-MATMUL-COMPUTE|BLOCK-ITERATION" "$CSV_FILE" 2>/dev/null | wc -l || echo "0")
    if [ "$CUSTOM_COUNT" -gt "0" ]; then
        echo "   ✓ Found $CUSTOM_COUNT custom zones in CSV file"
        grep -E "MM-INIT|BATCH-ITERATION|TRISC-MATMUL-COMPUTE|BLOCK-ITERATION" "$CSV_FILE" | head -3 | sed 's/^/     /'
    else
        echo "   ❌ No custom zones found in CSV file"
        echo "   → Custom zones are not being recorded during execution!"
    fi
else
    echo "   ⚠ CSV file not found: $CSV_FILE"
fi

echo ""
echo "========================================="
echo "Summary & Recommendations"
echo "========================================="
echo ""

if [ -z "$TT_METAL_DEVICE_PROFILER" ]; then
    echo "❌ CRITICAL: TT_METAL_DEVICE_PROFILER is not set!"
    echo ""
    echo "This is the root cause. Without this environment variable:"
    echo "  - PROFILE_KERNEL is not defined during kernel compilation"
    echo "  - DeviceZoneScopedN becomes an empty macro"
    echo "  - Custom zone code is removed during compilation"
    echo ""
    echo "SOLUTION:"
    echo "  1. Delete JIT cache: rm -rf ~/.cache/tt-metal-cache/*"
    echo "  2. Set environment variable: export TT_METAL_DEVICE_PROFILER=1"
    echo "  3. Run your program again"
    echo ""
elif grep -q "PROFILE_KERNEL\|-DPROFILE" "$BUILD_LOG" 2>/dev/null; then
    echo "✓ PROFILE_KERNEL is being defined"
    echo "  But custom zones are still not appearing."
    echo "  This suggests the kernel may not be recompiling or there's another issue."
    echo ""
    echo "RECOMMENDATION:"
    echo "  1. Force recompile: rm -rf ~/.cache/tt-metal-cache/*"
    echo "  2. Run with: TT_METAL_DEVICE_PROFILER=1 python3 your_program.py"
    echo "  3. Check build logs during execution for compilation messages"
else
    echo "⚠ Custom zones are not appearing."
    echo "  Check the output above for specific issues."
fi

echo ""
echo "========================================="

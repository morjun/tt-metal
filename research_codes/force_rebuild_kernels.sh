#!/bin/bash
# Force rebuild kernels by completely deleting JIT cache
# This ensures kernels are recompiled with PROFILE_KERNEL when TT_METAL_DEVICE_PROFILER=1

set -e

echo "========================================="
echo "Force Rebuild Kernels for Profiling"
echo "========================================="
echo ""

# Step 1: Delete all JIT build caches
echo "🗑️  Deleting JIT build caches..."

# Main cache location
CACHE_DIR="${HOME}/.cache/tt-metal-cache"
if [ -d "$CACHE_DIR" ]; then
    echo "   Found cache at: $CACHE_DIR"
    CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
    echo "   Cache size: $CACHE_SIZE"
    echo "   Deleting..."
    rm -rf "$CACHE_DIR"/*
    echo "   ✓ Deleted $CACHE_DIR"
fi

# Alternative cache location
TMP_CACHE="/tmp/tt-metal-cache"
if [ -d "$TMP_CACHE" ]; then
    echo "   Found cache at: $TMP_CACHE"
    rm -rf "$TMP_CACHE"/*
    echo "   ✓ Deleted $TMP_CACHE"
fi

echo ""
echo "✅ Cache deletion complete!"
echo ""

# Step 2: Verify environment variable
echo "🔍 Checking environment variables..."
if [ -z "$TT_METAL_DEVICE_PROFILER" ]; then
    echo ""
    echo "⚠️  WARNING: TT_METAL_DEVICE_PROFILER is not set!"
    echo ""
    echo "You MUST set it before running your program:"
    echo "   export TT_METAL_DEVICE_PROFILER=1"
    echo ""
    echo "Or run your program with:"
    echo "   TT_METAL_DEVICE_PROFILER=1 python3 your_program.py"
    echo ""
else
    echo "   ✓ TT_METAL_DEVICE_PROFILER=$TT_METAL_DEVICE_PROFILER"
fi

echo ""
echo "========================================="
echo "Next Steps:"
echo "========================================="
echo "1. Set environment variable (if not already set):"
echo "   export TT_METAL_DEVICE_PROFILER=1"
echo ""
echo "2. Run your program - kernels will be recompiled"
echo ""
echo "3. Verify custom zones appear:"
echo "   grep -E 'MM-INIT|BATCH-ITERATION|BLOCK-ITERATION|TRISC-MATMUL-COMPUTE' \\"
echo "     generated/profiler/.logs/profile_log_device.csv"
echo ""
echo "========================================="

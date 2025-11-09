#!/bin/bash
#
# Complete Device Profile Comparison: Large Batch vs Mini-batch
#
# This script runs both scenarios and generates detailed analysis including:
# - Weight streaming time (GDDR6 → L1 SRAM)
# - NoC communication time (inter-core data transfer)
# - Computation time (actual matmul)
# - Synchronization overhead
#

set -e

cd "$(dirname "$0")/.."

echo "================================================================================"
echo "RUNNING COMPLETE DEVICE PROFILE COMPARISON"
echo "================================================================================"
echo ""
echo "This will:"
echo "  1. Profile Large Batch scenario (B=256)"
echo "  2. Analyze device time breakdown"
echo "  3. Profile Mini-batch scenario (b=32 × 8)"
echo "  4. Compare both scenarios"
echo ""
echo "Press Ctrl+C to cancel, or wait 3 seconds to continue..."
sleep 3

echo ""
echo "================================================================================"
echo "STEP 1: Large Batch Profiling (B=256)"
echo "================================================================================"
echo ""

TT_METAL_DEVICE_PROFILER=1 python research_codes/weight_loading_test.py \
    --only-large \
    --warmup-iters 0 \
    --measure-iters 1 \
    2>&1 | grep -E "(Average forward|Total avg)"

echo ""
echo "Analyzing large batch profile..."
echo ""

python research_codes/device_profile_analysis.py > /tmp/large_batch_analysis.txt
cat /tmp/large_batch_analysis.txt

echo ""
echo "================================================================================"
echo "STEP 2: Mini-batch Profiling (b=32 × 8)"
echo "================================================================================"
echo ""

TT_METAL_DEVICE_PROFILER=1 python research_codes/weight_loading_test.py \
    --only-mini \
    --warmup-iters 0 \
    --measure-iters 1 \
    2>&1 | grep -E "(Forward sum|Sequence total)"

echo ""
echo "Analyzing mini-batch profile and comparing..."
echo ""

python research_codes/device_profile_analysis.py

echo ""
echo "================================================================================"
echo "ANALYSIS COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - Large batch analysis: /tmp/large_batch_analysis.txt"
echo "  - Device profiles: generated/profiler/.logs/profile_log_device.csv"
echo ""

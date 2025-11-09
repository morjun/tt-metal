#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# TRACY profiling을 활성화하여 weight loading test 실행

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 출력 디렉토리
OUTPUT_DIR="${SCRIPT_DIR}/tracy_output"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "TRACY Weight Loading Test Runner"
echo "=========================================="
echo "TT-Metal Root: ${TT_METAL_ROOT}"
echo "Output Directory: ${OUTPUT_DIR}"
echo ""

# Tracy 도구 경로 확인
TRACY_BIN="${TT_METAL_ROOT}/build_Release/tools/profiler/bin"
if [ ! -d "${TRACY_BIN}" ]; then
    echo "ERROR: Tracy tools not found at ${TRACY_BIN}"
    echo "Please build tt-metal first with Tracy enabled (default)"
    exit 1
fi

echo "Tracy tools found at: ${TRACY_BIN}"

# 작업 디렉토리를 tt-metal root로 변경
cd "${TT_METAL_ROOT}"

# Python 환경 확인
if [ ! -f "python_env/bin/activate" ]; then
    echo "ERROR: Python virtual environment not found"
    echo "Please run ./create_venv.sh first"
    exit 1
fi

# Virtual environment 활성화
source python_env/bin/activate

echo "Running TRACY profiling..."
echo ""

# TRACY profiling 실행
# -v: verbose
# -r: generate report
# -p: partial profiling (only enabled zones)
# -o: output folder
# --no-device: host-side only profiling (device 포함하려면 제거)
# NOTE: Using warmup-iters=1 and measure-iters=1 for detailed analysis
python3 -m tracy \
    -v \
    -r \
    -p \
    -o "${OUTPUT_DIR}" \
    -n "weight_loading_test" \
    --tracy-tools-folder "${TRACY_BIN}" \
    research_codes/weight_loading_test_tracy.py \
    --warmup-iters 1 \
    --measure-iters 1

echo ""
echo "=========================================="
echo "TRACY profiling completed!"
echo "=========================================="
echo "Output files:"
echo "  - Tracy capture: ${OUTPUT_DIR}/profiler_logs/*.tracy"
echo "  - CSV reports: ${OUTPUT_DIR}/profiler_logs/*.csv"
echo "  - Benchmark results: ${SCRIPT_DIR}/benchmark_results_tracy.csv"
echo ""
echo "To view Tracy capture file:"
echo "  1. Install Tracy profiler GUI from: https://github.com/wolfpld/tracy"
echo "  2. Open the .tracy file in Tracy GUI"
echo ""

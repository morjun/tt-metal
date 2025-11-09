#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# 간단한 TRACY 테스트 실행

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${SCRIPT_DIR}/simple_tracy_output"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Simple TRACY Test Runner"
echo "=========================================="
echo "TT-Metal Root: ${TT_METAL_ROOT}"
echo "Output Directory: ${OUTPUT_DIR}"
echo ""

# Tracy 도구 경로 확인
TRACY_BIN="${TT_METAL_ROOT}/build_Release/tools/profiler/bin"
if [ ! -d "${TRACY_BIN}" ]; then
    echo "ERROR: Tracy tools not found at ${TRACY_BIN}"
    exit 1
fi

echo "Tracy tools found at: ${TRACY_BIN}"
echo ""

# 작업 디렉토리를 tt-metal root로 변경
cd "${TT_METAL_ROOT}"

# Virtual environment 활성화
if [ -f "python_env/bin/activate" ]; then
    source python_env/bin/activate
else
    echo "WARNING: Python virtual environment not found"
fi

echo "Running simple TRACY test..."
echo ""
echo "Command: python3 -m tracy -v -r -p -o ${OUTPUT_DIR} -n simple_test --tracy-tools-folder ${TRACY_BIN} research_codes/simple_tracy_test.py"
echo ""
echo "----------------------------------------"

# TRACY profiling 실행
python3 -m tracy \
    -v \
    -r \
    -p \
    -o "${OUTPUT_DIR}" \
    -n "simple_test" \
    --tracy-tools-folder "${TRACY_BIN}" \
    research_codes/simple_tracy_test.py

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
echo ""
echo "Checking output files..."
ls -lh "${OUTPUT_DIR}/.logs/" 2>/dev/null || echo "No .logs directory found"
echo ""
echo "Tracy files:"
find "${OUTPUT_DIR}" -name "*.tracy" -exec ls -lh {} \; 2>/dev/null || echo "No .tracy files found"
echo ""
echo "CSV files:"
find "${OUTPUT_DIR}" -name "*.csv" -exec ls -lh {} \; 2>/dev/null || echo "No .csv files found"
echo ""

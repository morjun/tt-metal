#!/usr/bin/env bash
set -uo pipefail
cd /home/masterjunmo/codes/tt-metal
P=research_codes/documents/l1_kv_cache_cache/reprofile
./python_env/bin/tt-smi -r >/dev/null 2>&1
echo "=== ZONE RUN 1: l1only (no-trace) ==="
bash "$P/run_zones32.sh" l1only "" 2 || { echo "L1ONLY ZONE RUN FAILED"; exit 1; }
./python_env/bin/tt-smi -r >/dev/null 2>&1
echo "=== ZONE RUN 2: dram (no-trace) ==="
bash "$P/run_zones32.sh" dram "" 2 || { echo "DRAM ZONE RUN FAILED"; exit 1; }
echo "=== GAP ANALYSIS ==="
./python_env/bin/python "$P/analyze_gaps.py" "$P/zones2_l1only_896/profile_log_device.csv" L1-only
./python_env/bin/python "$P/analyze_gaps.py" "$P/zones2_dram_896/profile_log_device.csv" DRAM
echo "=== GAP PROBE DONE ==="

#!/usr/bin/env bash
# Parity matrix: {DRAM, L1} x {trace ON, OFF}, NRUNS each, single card, batch-1 perf.
# Trace toggled via L1KV_TRACE env (simple_text_demo.py line 931 temporarily reads it).
set -uo pipefail
cd /home/masterjunmo/codes/tt-metal
export PATH="$PWD/python_env/bin:$PATH"
P=research_codes/documents/l1_kv_cache_cache/reprofile/parity
mkdir -p "$P"
NODE='models/tt_transformers/demo/simple_text_demo.py'
NRUNS=${NRUNS:-4}
L1_FLAGS=(--l1_kv_mode interleaved --l1_kv_window_size 960 --l1_kv_only_mode)

run_one() {  # $1=label  $2=trace(0/1)  $3...=extra flags
  local lbl="$1" tr="$2"; shift 2
  for r in $(seq 1 "$NRUNS"); do
    tt-smi -r >/dev/null 2>&1
    local log="$P/${lbl}_run${r}.log"
    L1KV_TRACE="$tr" TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 TT_LOGGER_LEVEL=Info \
      timeout 1200 python -m pytest -q "$NODE" -k "performance and batch-1" "$@" \
      > "$log" 2>&1
    local ec=$?
    local res; res=$(grep -aqE "(^|::)PASSED|1 passed" "$log" && echo PASS || echo FAIL)
    echo ">>> ${lbl} run${r}: exit=${ec} ${res}  ($log)"
  done
}

echo "=== PARITY MATRIX START (NRUNS=$NRUNS) ==="
run_one dram_traceoff 0
run_one dram_traceon  1
run_one l1_traceoff   0 "${L1_FLAGS[@]}"
run_one l1_traceon    1 "${L1_FLAGS[@]}"
echo "=== PARITY MATRIX DONE ==="

#!/usr/bin/env bash
set -uo pipefail
cd /home/masterjunmo/codes/tt-metal
export PATH="$PWD/python_env/bin:$PATH"
P=research_codes/documents/l1_kv_cache_cache/reprofile/parity2
mkdir -p "$P"
NODE='models/tt_transformers/demo/simple_text_demo.py'
NRUNS=4
run_one(){ local lbl="$1"; shift
  for r in $(seq 1 $NRUNS); do
    tt-smi -r >/dev/null 2>&1
    TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 TT_LOGGER_LEVEL=Info \
      timeout 1200 python -m pytest -q "$NODE" -k "performance and batch-1" "$@" > "$P/${lbl}_run${r}.log" 2>&1
    echo ">>> ${lbl} run${r}: exit=$?"
  done; }
echo "=== PARITY2 (trace ON, hit-ratio gated) START ==="
run_one l1_traceon  --l1_kv_mode interleaved --l1_kv_window_size 960 --l1_kv_only_mode
run_one dram_traceon
echo "=== PARITY2 DONE ==="

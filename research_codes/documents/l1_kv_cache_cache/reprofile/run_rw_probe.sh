#!/usr/bin/env bash
# Read-wait probe: BOTH compute + reader zones on, no-trace, ctx 896, 2 layers, 4 tokens
# (reduced tokens to stay under per-core profiler marker buffer with both zone sets).
set -uo pipefail
cd /home/masterjunmo/codes/tt-metal
export PATH="$PWD/python_env/bin:$PATH"
P=research_codes/documents/l1_kv_cache_cache/reprofile
PROMPT="$P/prompt_ctx896.json"
run(){ local mode="$1"; shift
  local OUT="$P/zones_rw_${mode}"; mkdir -p "$OUT"
  tt-smi -r >/dev/null 2>&1
  TT_METAL_HOME="$PWD" PYTHONPATH="$PWD/tools" python -m tracy -r -p -v \
    -m pytest "models/tt_transformers/demo/simple_text_demo.py::test_demo_text[blackhole-mesh_device0-device_params0-performance-batch-1]" \
    --input_prompts "$PROMPT" --max_seq_len 2048 --max_generated_tokens 4 \
    --num_layers 1 --instruct 0 --paged_attention 0 "$@" \
    > "$OUT/run.log" 2>&1 || { echo "$mode FAILED"; tail -25 "$OUT/run.log"; return 1; }
  local RPT=$(ls -1dt generated/profiler/reports/*/ | head -1)
  cp "$RPT"profile_log_device.csv "$OUT/profile_log_device.csv"
  echo "$mode OK -> $OUT ; zones:"; awk -F',' 'NR>2{gsub(/^ /,"",$11);print $11}' "$OUT/profile_log_device.csv" | sort | uniq -c | sort -rn | grep -E "QK_MM|SM_NORM|PV_MM|RD_" | head
}
echo "=== RW PROBE: wipe cache (both macros on) ==="
rm -rf "$HOME/.cache/tt-metal-cache"
run l1only --l1_kv_mode interleaved --l1_kv_only_mode --l1_kv_window_size 960 || exit 1
run dram --l1_kv_mode dram || exit 1
echo "=== RW PROBE DONE ==="

#!/usr/bin/env bash
# Direct proof that warm-up overhead = JIT compile: sample the kernel-compiler subprocess
# count (riscv-tt-elf-g++ / cc1plus) with timestamps WHILE the demo runs, then bucket
# compiler activity into per-iteration windows. If the compiler is busy during iter0/1/2
# and idle during steady (iter3+), the warm-up time IS compilation.
set -uo pipefail
cd /home/masterjunmo/codes/tt-metal
export PATH="$PWD/python_env/bin:$PATH"
P=research_codes/documents/l1_kv_cache_cache/reprofile
SAMP="$P/compile_sampler.log"
LOG="$P/compile_probe.log"
: > "$SAMP"

# background sampler: epoch.ns  <#compiler procs>   every 0.15s
( while true; do
    printf "%s %s\n" "$(date +%s.%N)" "$(pgrep -fc 'riscv-tt-elf-g\+\+|cc1plus' 2>/dev/null || echo 0)"
    sleep 0.15
  done ) > "$SAMP" &
SAMPLER=$!

TT_METAL_HOME="$PWD" timeout 1200 python -m pytest \
  "models/tt_transformers/demo/simple_text_demo.py::test_demo_text[blackhole-mesh_device0-device_params0-performance-batch-1]" \
  --input_prompts "$P/prompt_ctx896.json" --max_seq_len 2048 --max_generated_tokens 16 --num_layers 2 \
  --instruct 0 --paged_attention 0 --l1_kv_mode interleaved --l1_kv_window_size 960 --l1_kv_only_mode \
  > "$LOG" 2>&1 || true
kill "$SAMPLER" 2>/dev/null || true

echo "=== per-iteration compiler activity (bucketed) ==="
python research_codes/documents/l1_kv_cache_cache/reprofile/compile_probe_parse.py "$LOG" "$SAMP"
echo "=== PROBE COMPLETE ==="

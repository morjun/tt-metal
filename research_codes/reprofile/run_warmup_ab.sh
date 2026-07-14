#!/usr/bin/env bash
# Prove warm-up overhead = JIT compile, via cold/warm persistent-cache A/B.
# The on-disk kernel cache affects ONLY compilation. Run the identical L1 config twice
# (cold = wiped cache, warm = reuse). Warm-up time that VANISHES on the warm run is compile;
# time that STAYS is alloc+seed. Non-tracy (tracy masks per-iter timing).
set -euo pipefail
cd /home/masterjunmo/codes/tt-metal
export PATH="$PWD/python_env/bin:$PATH"
P=research_codes/documents/l1_kv_cache_cache/reprofile
PROMPT="$P/prompt_ctx896.json"

run() {  # $1 = cold|warm
  TT_METAL_HOME="$PWD" timeout 1200 python -m pytest \
    "models/tt_transformers/demo/simple_text_demo.py::test_demo_text[blackhole-mesh_device0-device_params0-performance-batch-1]" \
    --input_prompts "$PROMPT" --max_seq_len 2048 --max_generated_tokens 12 --num_layers 2 \
    --instruct 0 --paged_attention 0 --l1_kv_mode interleaved --l1_kv_window_size 960 --l1_kv_only_mode \
    > "$P/warmup_$1.log" 2>&1 || { echo "$1 RUN FAILED"; tail -25 "$P/warmup_$1.log"; return 1; }
  echo "=== $1 run: per-iteration wall times ==="
  grep -aoE "Iteration [0-9]+: [0-9]+ms" "$P/warmup_$1.log" | head -13
  echo "--- alloc/seed markers ($1) ---"
  grep -acE "Seeded .* tokens" "$P/warmup_$1.log" | sed 's/^/seed-log-lines: /'
}

tt-smi -r >/dev/null 2>&1 && echo "device reset ok"
echo "########## COLD (wipe persistent kernel cache) ##########"
rm -rf "$HOME/.cache/tt-metal-cache"
run cold
echo "########## WARM (reuse persistent kernel cache, no wipe) ##########"
run warm
echo "=== AB COMPLETE ==="

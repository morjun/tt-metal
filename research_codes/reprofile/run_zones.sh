#!/usr/bin/env bash
# Profile SDPA-decode internal zones (FPU vs SFPU + read/compute overlap).
# Requires the kernels instrumented with SDPA_PROFILE_ZONES (sdpa_flash_decode.cpp,
# dataflow_common.hpp). Usage: run_zones.sh [wipe]
#   pass "wipe" the first time (or after editing a kernel) to clear the JIT cache so the
#   edited kernel source is recompiled (build key does not hash source content).
set -euo pipefail
cd /home/masterjunmo/codes/tt-metal
WIPE="${1:-}"
CTX=1792
LABEL="zones_dram_${CTX}"
PROMPT="research_codes/documents/l1_kv_cache_cache/reprofile/prompt_ctx${CTX}.json"
OUTDIR="research_codes/documents/l1_kv_cache_cache/reprofile/$LABEL"
mkdir -p "$OUTDIR"
export PATH="$PWD/python_env/bin:$PATH"

if [[ "$WIPE" == "wipe" ]]; then
  echo "=== wiping JIT kernel cache (~/.cache/tt-metal-cache) to force recompile ==="
  rm -rf "$HOME/.cache/tt-metal-cache"
fi

echo "=== RUN $LABEL: DRAM, ctx=$CTX, 2 layers, zones on ==="
TT_METAL_HOME="$PWD" PYTHONPATH="$PWD/tools" python -m tracy -r -p -v \
  -m pytest "models/tt_transformers/demo/simple_text_demo.py::test_demo_text[blackhole-mesh_device0-device_params0-performance-batch-1]" \
  --input_prompts "$PROMPT" \
  --max_seq_len 2048 \
  --max_generated_tokens 8 \
  --num_layers 2 \
  --instruct 0 \
  --paged_attention 0 \
  --l1_kv_mode dram \
  > "$OUTDIR/run.log" 2>&1 || { echo "PYTEST FAILED, tail:"; tail -50 "$OUTDIR/run.log"; exit 1; }

RPT=$(ls -1dt generated/profiler/reports/*/ | head -1)
cp "$RPT"profile_log_device.csv "$OUTDIR/profile_log_device.csv"
cp "$RPT"ops_perf_results_*.csv "$OUTDIR/ops_perf.csv" 2>/dev/null || true
echo "report: $RPT"
echo "--- compile / prefill / dropped-zone markers ---"
grep -iE "Compiling|sdpa_flash_decode|Prefill seq len|DROPPED|markers dropped|Iteration [3-7]:" "$OUTDIR/run.log" | tail -8 || true
echo "--- zone analysis ---"
python research_codes/documents/l1_kv_cache_cache/reprofile/analyze_zones.py "$OUTDIR/profile_log_device.csv"

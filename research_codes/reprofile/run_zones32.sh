#!/usr/bin/env bash
# 32-layer zone profiling at ctx ~900, l1_only window 960 (deployment config) vs DRAM.
# Captures: SDPA-decode latency (ops_perf), FPU/SFPU + read/compute overlap (zones),
# and warm-up iter timings (item 2: alloc/seed vs first-use compile).
# Requires SDPA_PROFILE_ZONES enabled in the kernels. Usage: run_zones32.sh <dram|l1only> [wipe]
set -euo pipefail
cd /home/masterjunmo/codes/tt-metal
MODE="$1"; WIPE="${2:-}"; LAYERS="${3:-32}"
# NOTE: custom zones overflow the per-core marker buffer at high layer counts (markers
# accumulate across op invocations: ~16 SDPA calls/core fit at 2 layers, 256 at 32 layers
# overflow -> profiler abort). Use LAYERS=2 for zone (overlap/FPU-SFPU) runs and LAYERS=32
# with zones DISABLED for latency/warm-up runs.
CTX=896   # 2^7*7 -> k_chunk 128 -> 7 chunks (large pow2 divisor). <960 window = full-resident.
LABEL="zones${LAYERS}_${MODE}_${CTX}"
PROMPT="research_codes/documents/l1_kv_cache_cache/reprofile/prompt_ctx${CTX}.json"
OUTDIR="research_codes/documents/l1_kv_cache_cache/reprofile/$LABEL"
mkdir -p "$OUTDIR"
export PATH="$PWD/python_env/bin:$PATH"

EXTRA=(--l1_kv_mode dram)
if [[ "$MODE" == "l1only" ]]; then
  EXTRA=(--l1_kv_mode interleaved --l1_kv_only_mode --l1_kv_window_size 960)
fi

if [[ "$WIPE" == "wipe" ]]; then
  echo "=== wiping JIT kernel cache to force recompile ==="
  rm -rf "$HOME/.cache/tt-metal-cache"
fi

echo "=== RUN $LABEL: 32-layer, ctx=$CTX, mode=$MODE, zones on ==="
TT_METAL_HOME="$PWD" PYTHONPATH="$PWD/tools" python -m tracy -r -p -v \
  -m pytest "models/tt_transformers/demo/simple_text_demo.py::test_demo_text[blackhole-mesh_device0-device_params0-performance-batch-1]" \
  --input_prompts "$PROMPT" \
  --max_seq_len 2048 \
  --max_generated_tokens 8 \
  --num_layers "$LAYERS" \
  --instruct 0 \
  --paged_attention 0 \
  "${EXTRA[@]}" \
  > "$OUTDIR/run.log" 2>&1 || { echo "PYTEST FAILED, tail:"; tail -50 "$OUTDIR/run.log"; exit 1; }

RPT=$(ls -1dt generated/profiler/reports/*/ | head -1)
cp "$RPT"profile_log_device.csv "$OUTDIR/profile_log_device.csv"
cp "$RPT"ops_perf_results_*.csv "$OUTDIR/ops_perf.csv" 2>/dev/null || true
echo "report: $RPT"
echo "--- warm-up / seed / compile timeline (item 2) ---"
grep -iE "Prefill seq len|Total L1 KV tokens|Seeded .* tokens|Iteration [0-7]:|Compiling|DROPPED" "$OUTDIR/run.log" | tail -20 || true
echo "--- SDPA-decode op latency (ops_perf) ---"
python research_codes/documents/l1_kv_cache_cache/reprofile/extract.py "$OUTDIR/ops_perf.csv" 2>/dev/null || echo "(extract failed)"
echo "--- zone analysis (FPU/SFPU + overlap) ---"
python research_codes/documents/l1_kv_cache_cache/reprofile/analyze_zones.py "$OUTDIR/profile_log_device.csv"

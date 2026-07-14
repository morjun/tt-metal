#!/usr/bin/env bash
# Batch-size sweep of SDPA-decode compute-boundness on the DRAM path.
# For each --batch_size N: 2-layer run with SDPA_PROFILE_ZONES enabled (read-hidden + breakdown),
# ctx896. Captures the resolved parallelization scheme per batch. Requires zones compiled in
# (uncomment SDPA_PROFILE_ZONES in sdpa_flash_decode.cpp + dataflow_common.hpp, then ./build_metal.sh).
# Usage: run_batch_sweep.sh [batch list...]   (default: 1 8 16 32)
set -euo pipefail
cd /home/masterjunmo/codes/tt-metal
export PATH="$PWD/python_env/bin:$PATH"
BATCHES=("${@:-}")
[[ "${BATCHES[*]}" == "" ]] && BATCHES=(1 8 16 32)
LAYERS=1
CTX="${CTX:-896}"
PROMPT="${PROMPT:-research_codes/documents/l1_kv_cache_cache/reprofile/prompt_ctx${CTX}.json}"
P="research_codes/documents/l1_kv_cache_cache/reprofile"

for N in "${BATCHES[@]}"; do
  LABEL="zones${LAYERS}_dram_b${N}_${CTX}"
  OUTDIR="$P/$LABEL"
  mkdir -p "$OUTDIR"
  echo "==================== batch=$N (DRAM, ${LAYERS}L, ctx=$CTX) ===================="
  # reset device between runs to avoid wedge from a prior aborted run
  tt-smi -r 2>/dev/null || true

  TT_METAL_HOME="$PWD" PYTHONPATH="$PWD/tools" python -m tracy -r -p -v \
    -m pytest "models/tt_transformers/demo/simple_text_demo.py::test_demo_text[blackhole-mesh_device0-device_params0-performance-batch-1]" \
    --batch_size "$N" \
    --input_prompts "$PROMPT" \
    --max_seq_len 2048 \
    --max_generated_tokens 1 \
    --num_layers "$LAYERS" \
    --instruct 0 \
    --paged_attention 0 \
    --l1_kv_mode dram \
    > "$OUTDIR/run.log" 2>&1 || { echo "PYTEST FAILED (batch=$N), tail:"; tail -50 "$OUTDIR/run.log"; continue; }

  RPT=$(ls -1dt generated/profiler/reports/*/ | head -1)
  cp "$RPT"profile_log_device.csv "$OUTDIR/profile_log_device.csv" 2>/dev/null || true
  cp "$RPT"ops_perf_results_*.csv "$OUTDIR/ops_perf.csv" 2>/dev/null || true
  echo "report: $RPT"
  echo "--- parallelization scheme (batch=$N) ---"
  grep -iE "num_cores_per_batch|num_cores_per_head|num_heads_per_core|num_active_cores|num_reducer_cores|Parallelization" "$OUTDIR/run.log" | tail -10 || true
  echo "--- SDPA-decode op latency (batch=$N) ---"
  python "$P/extract.py" "$OUTDIR/ops_perf.csv" 2>/dev/null || echo "(extract failed)"
  echo "--- read-hidden + FPU/SFPU + per-core breakdown (batch=$N) ---"
  python "$P/analyze_zones.py" "$OUTDIR/profile_log_device.csv" 2>/dev/null || echo "(analyze failed)"
done
echo "==================== sweep done ===================="

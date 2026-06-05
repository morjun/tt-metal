#!/usr/bin/env bash
# 1-layer long-context SDPA-decode latency sweep (no zones): L1 full-resident vs DRAM.
# Tests whether read stays hidden (L1==DRAM) at extreme context (up to ~28k @ 1 layer).
# Usage: run_sweep1l.sh LABEL MODE CTX WINDOW ONLY(0|1)   MODE: dram | interleaved
set -euo pipefail
cd /home/masterjunmo/codes/tt-metal
LABEL="$1"; MODE="$2"; CTX="$3"; WIN="${4:-0}"; ONLY="${5:-0}"; PAGED="${6:-0}"
# NOTE: ctx>4096 needs chunked prefill -> requires --paged_attention 1 (page_table). That
# DISABLES the L1 KV path, so L1 full-resident is capped at ctx<=4096. Use PAGED=1 only for
# DRAM-only long-context curve (read-hiding-via-linearity test).
MSL=32768
PROMPT="research_codes/documents/l1_kv_cache_cache/reprofile/prompt_ctx${CTX}.json"
OUTDIR="research_codes/documents/l1_kv_cache_cache/reprofile/$LABEL"
mkdir -p "$OUTDIR"
export PATH="$PWD/python_env/bin:$PATH"

EXTRA=()
if [[ "$MODE" != "dram" ]]; then
  EXTRA+=(--l1_kv_window_size "$WIN")
  [[ "$ONLY" == "1" ]] && EXTRA+=(--l1_kv_only_mode)
fi

echo "=== RUN $LABEL: 1-layer, mode=$MODE ctx=$CTX window=$WIN only=$ONLY (max_seq_len $MSL) ==="
TT_METAL_HOME="$PWD" PYTHONPATH="$PWD/tools" python -m tracy -r -p -v \
  -m pytest "models/tt_transformers/demo/simple_text_demo.py::test_demo_text[blackhole-mesh_device0-device_params0-performance-batch-1]" \
  --input_prompts "$PROMPT" \
  --max_seq_len "$MSL" \
  --max_generated_tokens 8 \
  --num_layers 1 \
  --instruct 0 \
  --paged_attention "$PAGED" \
  --l1_kv_mode "$MODE" \
  "${EXTRA[@]}" \
  > "$OUTDIR/run.log" 2>&1 || { echo "PYTEST FAILED, tail:"; tail -40 "$OUTDIR/run.log"; exit 1; }

RPT=$(ls -1dt generated/profiler/reports/*/ | head -1)
cp "$RPT"ops_perf_results_*.csv "$OUTDIR/ops_perf.csv" 2>/dev/null || true
echo "report: $RPT"
echo "--- context / clamp / decode markers ---"
grep -iE "Prefill seq len|Total L1 KV tokens|interleaved tier|Iteration [3-7]:" "$OUTDIR/run.log" | tail -8 || true
echo "--- SDPA-decode op latency ---"
python research_codes/documents/l1_kv_cache_cache/reprofile/extract.py "$OUTDIR/ops_perf.csv"

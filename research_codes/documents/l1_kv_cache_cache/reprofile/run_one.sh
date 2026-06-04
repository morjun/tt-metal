#!/usr/bin/env bash
# Re-profile SDPA-decode op latency: DRAM vs L1 (full-resident), fixed context.
# Usage: run_one.sh LABEL MODE MAX_SEQ_LEN WINDOW ONLY(0|1)
#   MODE: dram | interleaved
#   WINDOW: --l1_kv_window_size (ignored for dram)
#   ONLY: 1 -> add --l1_kv_only_mode (full prefill seeded into L1, no DRAM read)
set -euo pipefail
cd /home/masterjunmo/codes/tt-metal
# Usage: run_one.sh LABEL MODE CTX WINDOW ONLY   (CTX in {512,1024,1792}; max_seq_len fixed 2048)
LABEL="$1"; MODE="$2"; CTX="$3"; WIN="${4:-0}"; ONLY="${5:-0}"
MSL=2048
PROMPT="research_codes/documents/l1_kv_cache_cache/reprofile/prompt_ctx${CTX}.json"
OUTDIR="research_codes/documents/l1_kv_cache_cache/reprofile/$LABEL"
mkdir -p "$OUTDIR"

EXTRA=()
if [[ "$MODE" != "dram" ]]; then
  EXTRA+=(--l1_kv_window_size "$WIN")
  [[ "$ONLY" == "1" ]] && EXTRA+=(--l1_kv_only_mode)
fi

echo "=== RUN $LABEL: mode=$MODE max_seq_len=$MSL window=$WIN only=$ONLY ==="
export PATH="$PWD/python_env/bin:$PATH"
TT_METAL_HOME="$PWD" PYTHONPATH="$PWD/tools" python -m tracy -r -p -v \
  -m pytest "models/tt_transformers/demo/simple_text_demo.py::test_demo_text[blackhole-mesh_device0-device_params0-performance-batch-1]" \
  --input_prompts "$PROMPT" \
  --max_seq_len "$MSL" \
  --max_generated_tokens 8 \
  --num_layers 2 \
  --instruct 0 \
  --paged_attention 0 \
  --l1_kv_mode "$MODE" \
  "${EXTRA[@]}" \
  > "$OUTDIR/run.log" 2>&1 || { echo "PYTEST FAILED, tail:"; tail -40 "$OUTDIR/run.log"; exit 1; }

# newest report dir
RPT=$(ls -1dt generated/profiler/reports/*/ | head -1)
CSV=$(ls -1 "$RPT"ops_perf_results_*.csv 2>/dev/null | head -1)
cp "$CSV" "$OUTDIR/ops_perf.csv"
echo "report: $RPT"
echo "--- context/clamp/coherence markers ---"
grep -iE "Prefill seq len|Total L1 KV tokens|Single interleaved tier|Iteration [3-7]:" "$OUTDIR/run.log" | tail -10 || true
echo "--- SDPA + Matmul device kernel duration (PER CORE AVG/MAX ns) ---"
./python_env/bin/python research_codes/documents/l1_kv_cache_cache/reprofile/extract.py "$OUTDIR/ops_perf.csv"

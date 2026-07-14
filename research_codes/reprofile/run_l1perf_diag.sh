#!/usr/bin/env bash
# Diagnose the ~4% L1 end-to-end loss: host-side per-section dispatch timing (TT_L1_KV_PERF)
# for 32-layer L1 vs DRAM. Compare the per-layer write-path host time (decode.adaptive_l1_kv_write
# vs decode.dram_kv_write) and decode.sdpa_call. Use min_ms = steady (excludes compile outliers).
set -uo pipefail
cd /home/masterjunmo/codes/tt-metal
export PATH="$PWD/python_env/bin:$PATH"
P=research_codes/documents/l1_kv_cache_cache/reprofile
NODE="models/tt_transformers/demo/simple_text_demo.py::test_demo_text[blackhole-mesh_device0-device_params0-performance-batch-1]"

runperf() {  # $1=label  $2..=args
  local lbl="$1"; shift
  rm -f "$P/perf_${lbl}.json"
  TT_L1_KV_PERF=1 TT_L1_KV_PERF_REPORT="$PWD/$P/perf_${lbl}.json" DECODE_WARMUP_ITERS=4 TT_METAL_HOME="$PWD" \
    timeout 900 python -m pytest "$NODE" \
    --input_prompts "$P/prompt_ctx896.json" --max_seq_len 2048 --max_generated_tokens 32 --num_layers 32 \
    --instruct 0 --paged_attention 0 "$@" > "$P/perf_${lbl}.log" 2>&1 || echo "$lbl FAILED (tail: $(tail -3 "$P/perf_${lbl}.log"))"
}

tt-smi -r >/dev/null 2>&1 && echo "reset ok"
runperf dram --l1_kv_mode dram
runperf l1   --l1_kv_mode interleaved --l1_kv_window_size 960 --l1_kv_only_mode

echo "--- host per-section timing (count / avg_ms / min_ms=steady / max_ms=compile-outlier) ---"
./python_env/bin/python - "$P/perf_dram.json" "$P/perf_l1.json" <<'PY'
import json, sys
for lbl, path in (("DRAM", sys.argv[1]), ("L1", sys.argv[2])):
    print(f"=== {lbl} ===")
    try:
        r = json.load(open(path))["stats"]
    except Exception as e:
        print(f"  (no report: {e})"); continue
    for k in sorted(r):
        s = r[k]
        print(f"  {k:<32} count={s['count']:>4}  avg={s['avg_ms']:.3f}  min={s['min_ms']:.3f}  max={s['max_ms']:.1f} ms")
PY
echo "=== L1PERF DONE ==="

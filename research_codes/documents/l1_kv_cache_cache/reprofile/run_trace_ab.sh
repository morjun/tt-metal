#!/usr/bin/env bash
# Stage 0: trace ON vs OFF A/B on upstream, single card, batch-1 perf test.
# Hypothesis: trace-OFF collapses toward the old branch's ~11 tok/s.
set -uo pipefail
cd /home/masterjunmo/codes/tt-metal
export PATH="$PWD/python_env/bin:$PATH"
P=research_codes/documents/l1_kv_cache_cache/reprofile
NODE='models/tt_transformers/demo/simple_text_demo.py'
COMMON='-k "performance and batch-1"'

run() {  # $1=label  $2=extra flags
  local lbl="$1"; shift
  echo ">>> RUN $lbl (flags: $*)"
  TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 TT_LOGGER_LEVEL=Info \
    timeout 1200 python -m pytest "$NODE" -k "performance and batch-1" "$@" \
    > "$P/trace_ab_${lbl}.log" 2>&1
  echo "    exit=$? log=$P/trace_ab_${lbl}.log"
}

steady() {  # $1=log -> mean ms over Iteration 5..; also raw count
  grep -aoE "Iteration [0-9]+: [0-9]+ ?ms" "$1" \
    | sed -E 's/Iteration ([0-9]+): ([0-9]+) ?ms/\1 \2/' \
    | awk '$1>=5{s+=$2;n++} END{if(n)printf "steady_avg=%.1f ms over %d iters -> %.2f tok/s\n", s/n, n, 1000.0/(s/n); else print "no Iteration lines >=5"}'
}

./python_env/bin/tt-smi -r >/dev/null 2>&1 && echo "reset ok"
run trace_on
run trace_off --disable_trace
echo "=== RESULTS ==="
echo -n "trace_on : "; steady "$P/trace_ab_trace_on.log"
echo -n "trace_off: "; steady "$P/trace_ab_trace_off.log"
echo "--- sample lines (trace_on) ---"; grep -aoE "Iteration [0-9]+: [0-9]+ ?ms|[0-9.]+ tok/s" "$P/trace_ab_trace_on.log" | tail -5
echo "--- sample lines (trace_off) ---"; grep -aoE "Iteration [0-9]+: [0-9]+ ?ms|[0-9.]+ tok/s" "$P/trace_ab_trace_off.log" | tail -5
echo "=== TRACE_AB DONE ==="

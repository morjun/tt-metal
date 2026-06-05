#!/usr/bin/env bash
# Final eval: N runs each of DRAM and L1 (interleaved l1_only), ALTERNATING to cancel drift,
# same config (32-layer, ctx896, DECODE_WARMUP_ITERS=4 so timed loop is steady). Compare means.
set -uo pipefail
cd /home/masterjunmo/codes/tt-metal
export PATH="$PWD/python_env/bin:$PATH"
P=research_codes/documents/l1_kv_cache_cache/reprofile
NODE="models/tt_transformers/demo/simple_text_demo.py::test_demo_text[blackhole-mesh_device0-device_params0-performance-batch-1]"
N=8

steady_avg() {  # $1 = logfile -> prints mean ms over iters 5..63
  grep -aoE "Iteration ([5-9]|[1-5][0-9]|6[0-3]): [0-9]+ms" "$1" | grep -oE "[0-9]+ms" | tr -d ms | \
    awk '{s+=$1;n++}END{if(n)printf "%.1f",s/n; else print "NA"}'
}
run() {  # $1=label  $2..=extra args -> prints steady avg
  local lbl="$1"; shift
  DECODE_WARMUP_ITERS=4 TT_METAL_HOME="$PWD" timeout 900 python -m pytest "$NODE" \
    --input_prompts "$P/prompt_ctx896.json" --max_seq_len 2048 --max_generated_tokens 64 --num_layers 32 \
    --instruct 0 --paged_attention 0 "$@" > "$P/feval_${lbl}.log" 2>&1 || { echo "NA"; return; }
  steady_avg "$P/feval_${lbl}.log"
}

tt-smi -r >/dev/null 2>&1 && echo "reset ok"
DRAM=""; L1=""
for i in $(seq 1 $N); do
  d=$(run "dram_$i" --l1_kv_mode dram); echo "DRAM run $i: $d ms"; DRAM="$DRAM $d"
  l=$(run "l1_$i" --l1_kv_mode interleaved --l1_kv_window_size 960 --l1_kv_only_mode); echo "L1   run $i: $l ms"; L1="$L1 $l"
done
echo "--- aggregate ---"
./python_env/bin/python - "$DRAM" "|" "$L1" <<'PY'
import sys, statistics as st
args=sys.argv[1:]
sep=args.index("|")
def flat(xs): return [float(x) for tok in xs for x in tok.split() if x not in ("NA","")]
d=flat(args[:sep]); l=flat(args[sep+1:])
def line(name,a): print(f"{name}: {a}  mean={st.mean(a):.2f} ms  std={st.pstdev(a):.2f}  min={min(a)} max={max(a)}")
line("DRAM",d); line("L1  ",l)
diff=st.mean(l)-st.mean(d)
print(f"L1 - DRAM mean = {diff:+.2f} ms ({100*diff/st.mean(d):+.1f}%)")
# separation test: do the two sample ranges overlap?
overlap = max(d) >= min(l) and max(l) >= min(d)
print(f"DRAM [{min(d)},{max(d)}]  L1 [{min(l)},{max(l)}]  -> {'OVERLAP (tie / within noise)' if overlap else 'SEPARATED (gap real)'}")
PY
echo "=== FEVAL DONE ==="

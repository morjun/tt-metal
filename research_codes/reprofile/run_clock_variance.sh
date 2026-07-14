#!/usr/bin/env bash
# (a) Prove same-command run-to-run perf variance is AICLK (Tensix clock DVFS/thermal).
# Run the identical DRAM config N times; sample AICLK during each run's decode; report
# steady-state avg decode ms vs AICLK. If avg inversely tracks clock, the variance is clock.
set -uo pipefail
cd /home/masterjunmo/codes/tt-metal
export PATH="$PWD/python_env/bin:$PATH"
P=research_codes/documents/l1_kv_cache_cache/reprofile
NODE="models/tt_transformers/demo/simple_text_demo.py::test_demo_text[blackhole-mesh_device0-device_params0-performance-batch-1]"

aiclk_mhz() {  # read AICLK telemetry, print decimal MHz
  local h
  h=$(tt-smi -s 2>/dev/null | grep -oE '"AICLK":[[:space:]]*"0x[0-9a-fA-F]+"' | grep -oE '0x[0-9a-fA-F]+' | head -1)
  [ -n "$h" ] && echo $(( 16#${h#0x} ))
}

echo "idle AICLK (before): $(aiclk_mhz) MHz"
for i in 1 2 3 4; do
  log="$P/clkvar_run$i.log"; clk="$P/clkvar_clk$i.log"; : > "$clk"
  DECODE_WARMUP_ITERS=4 TT_METAL_HOME="$PWD" timeout 600 python -m pytest "$NODE" \
    --input_prompts "$P/prompt_ctx896.json" --max_seq_len 2048 --max_generated_tokens 64 --num_layers 32 \
    --instruct 0 --paged_attention 0 --l1_kv_mode dram > "$log" 2>&1 &
  DEMO=$!
  while kill -0 "$DEMO" 2>/dev/null; do v=$(aiclk_mhz); [ -n "$v" ] && echo "$(date +%s) $v" >> "$clk"; sleep 1; done
  avg=$(grep -aoE "Iteration [0-9]+: [0-9]+ms" "$log" | tail -50 | grep -oE "[0-9]+ms" | tr -d 'ms' | awk '{s+=$1;n++}END{if(n)printf "%.1f",s/n; else print "NA"}')
  clkmax=$(awk '{print $2}' "$clk" 2>/dev/null | sort -n | tail -1)
  clkhi=$(awk '{print $2}' "$clk" 2>/dev/null | sort -n | awk '{a[NR]=$1}END{if(NR)print a[int(NR*0.8)]}')  # 80th pct (under-load)
  nsamp=$(wc -l < "$clk" 2>/dev/null)
  printf "RUN %d: steady_avg=%s ms | AICLK max=%s p80=%s MHz | %s clk-samples\n" "$i" "${avg:-NA}" "${clkmax:-NA}" "${clkhi:-NA}" "${nsamp:-0}"
done
echo "=== CLKVAR DONE ==="

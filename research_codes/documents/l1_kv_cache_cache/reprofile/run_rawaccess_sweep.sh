#!/usr/bin/env bash
# Raw-access READ latency + bandwidth vs transfer size, for DRAM / local SRAM / remote SRAM(NoC).
# Reuses the existing perf microbenchmark test_bw_and_latency (no custom kernel).
#   -m 1 = DRAM ; -m 2 = L1 (src core via -sx/-sy: == reader => LOCAL, far core => REMOTE over NoC)
#   -p = per-read size (bytes) ; -bs = total KB ; -l = latency mode ; -i = iterations
# Output: reprofile/rawaccess/rawaccess.csv  (source,size_bytes,latency_ns,bw_gbs)
set -uo pipefail
cd /home/masterjunmo/codes/tt-metal
export ARCH_NAME=blackhole
export TT_VISIBLE_DEVICES=0
B=$(ls build*/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency 2>/dev/null | head -1)
if [ -z "$B" ]; then echo "test_bw_and_latency not built"; exit 1; fi
echo "binary: $B"
OUT=research_codes/documents/l1_kv_cache_cache/reprofile/rawaccess
mkdir -p "$OUT"; CSV="$OUT/rawaccess.csv"
echo "source,size_bytes,latency_ns,bw_gbs" > "$CSV"

RX=1; RY=1            # reader/worker core
LX=1; LY=1            # local SRAM source = reader core (loopback NoC read of own L1)
HX=7; HY=6           # remote SRAM source (~half-way on BH 13x10 logical grid)
SIZES="64 256 1024 4096 16384 65536 262144 1048576"   # 64 B .. 1 MB (single resident transfer fits L1)

run() {  # $1=label  $2=size  $3..=src flags
  local label="$1" S="$2"; shift 2
  local bs=$(( (S + 1023) / 1024 )); [ "$bs" -lt 1 ] && bs=1
  local pbw=$S; [ "$pbw" -gt 65536 ] && pbw=65536          # bandwidth read granularity capped at 64K
  local llog="$OUT/${label}_${S}_lat.log" blog="$OUT/${label}_${S}_bw.log"
  TT_METAL_HOME="$PWD" "$B" -rx $RX -ry $RY "$@" -p "$S"   -bs "$bs" -l -i 2000 > "$llog" 2>&1
  TT_METAL_HOME="$PWD" "$B" -rx $RX -ry $RY "$@" -p "$pbw" -bs "$bs"    -i 2000 > "$blog" 2>&1
  local lat=$(grep -oE "Latency: [0-9.]+ us" "$llog" | grep -oE "[0-9.]+" | head -1)
  local bw=$(grep -oE "BW: [0-9.]+ GB/s" "$blog" | grep -oE "[0-9.]+" | head -1)
  local lat_ns=""; [ -n "$lat" ] && lat_ns=$(awk "BEGIN{printf \"%.1f\", $lat*1000}")
  echo "$label,$S,${lat_ns},${bw}" >> "$CSV"
  echo "  $label S=$S  lat=${lat:-NA}us  bw=${bw:-NA}GB/s"
}

./python_env/bin/tt-smi -r >/dev/null 2>&1 && echo "reset ok"
for S in $SIZES; do
  echo "=== size $S B ==="
  run DRAM        "$S" -m 1
  run local_SRAM  "$S" -m 2 -sx $LX -sy $LY
  run remote_SRAM "$S" -m 2 -sx $HX -sy $HY
done
echo "=== RAWACCESS SWEEP DONE -> $CSV ==="
cat "$CSV"

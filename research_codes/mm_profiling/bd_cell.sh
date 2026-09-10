#!/bin/bash
# 6P.40 cell. $1=cond(mcast|ring|ringbd) $2=round $3=outdir $4=mode(audit|validate|time)
set -u
W=/home/masterjunmo/codes/tt-metal-cmdprobe
S=/tmp/claude-1001/-home-masterjunmo-codes-tt-metal-gemma4-l1w/2aa00f30-476e-4db8-9ee5-b3120cbea473/scratchpad
OUT=$3/r$2_$1; [ -f $OUT/cell.jsonl ] && { echo "SKIP $1 r$2"; exit 0; }
mkdir -p $OUT; cd $W
unset GEMMA4_DISABLE_TERMINAL_TOPDOWN GEMMA4_TERMINAL_TOPDOWN_TRACE GEMMA4_ADD_WAITS
unset GEMMA4_FORCE_SEND_TAIL GEMMA4_FORCE_SEND_TRACE GEMMA4_TRACEDUMP_DIR GEMMA4_ALLOCMAP_OUT
unset GEMMA4_BINARY_BOTH_DIRS GEMMA4_BINARY_BOTH_DIRS_TRACE
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools PATH=$W/python_env/bin:$PATH
export ARCH_NAME=blackhole TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 HF_HUB_OFFLINE=1
export GEMMA4_ASSISTANT_MODEL=google/gemma-4-E2B-it-assistant
export GEMMA4_TUNE_MATMULS=1 GEMMA4_SHARD_ACTIVATIONS=0 GEMMA4_LEDGER_K=3
export TT_METAL_CACHE=$W/.jitcache PYTHONHASHSEED=0
case "$1" in
  mcast)  export GEMMA4_GATHER_IN0=0 ;;
  ring)   export GEMMA4_GATHER_IN0=1 ;;
  ringbd) export GEMMA4_GATHER_IN0=1 GEMMA4_BINARY_BOTH_DIRS=1 GEMMA4_BINARY_BOTH_DIRS_TRACE=1 ;;
  *) echo "unknown $1"; exit 2 ;;
esac
case "$4" in
  audit)    export GEMMA4_DIAG_MODE=timing GEMMA4_DIAG_WARMUP=3 GEMMA4_DIAG_REPLAYS=20 \
                   GEMMA4_TRACEDUMP_DIR=$OUT GEMMA4_ALLOCMAP_OUT=$OUT/alloc.jsonl ;;
  validate) export GEMMA4_DIAG_MODE=validate GEMMA4_DIAG_WARMUP=3 GEMMA4_DIAG_REPLAYS=3 ;;
  time)     export GEMMA4_DIAG_MODE=timing GEMMA4_DIAG_WARMUP=3 GEMMA4_DIAG_REPLAYS=400 ;;
esac
export GEMMA4_DIAG_OUT=$OUT/cell.jsonl
$S/occupancy.sh before > $OUT/occ.jsonl
( while true; do $S/occupancy.sh during >> $OUT/occ.jsonl; sleep 5; done ) & SMP=$!
python -m pytest -s -q --timeout=2400 \
  "models/demos/gemma4/tests/unit/test_assistant_standalone_l1.py::test_gather_matched_trace" \
  > $OUT/run.log 2>&1
rc=$?
kill $SMP 2>/dev/null; wait $SMP 2>/dev/null
$S/occupancy.sh after >> $OUT/occ.jsonl
pass=$(grep -c "1 passed" $OUT/run.log 2>/dev/null); pass=${pass:-0}
echo "r$2 $1 [$4] rc=$rc passed=$pass"

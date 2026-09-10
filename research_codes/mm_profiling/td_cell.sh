#!/bin/bash
# Gate 2 timing cell for the final-use placement intervention.
# $1=K $2=arm $3=cond(base|iv) $4=round $5=outdir
# Unprofiled, 400 replays, per the protocol qualified in 6P.25/6P.26.
set -u
W=/home/masterjunmo/codes/tt-metal-cmdprobe
S=/tmp/claude-1001/-home-masterjunmo-codes-tt-metal-gemma4-l1w/2aa00f30-476e-4db8-9ee5-b3120cbea473/scratchpad
OUT=$5/r$4_k$1_a$2_$3; [ -f $OUT/cell.jsonl ] && { echo "SKIP"; exit 0; }
mkdir -p $OUT; cd $W
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools PATH=$W/python_env/bin:$PATH
export ARCH_NAME=blackhole TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 HF_HUB_OFFLINE=1
export GEMMA4_ASSISTANT_MODEL=google/gemma-4-E2B-it-assistant
export GEMMA4_TUNE_MATMULS=1 GEMMA4_SHARD_ACTIVATIONS=0
export GEMMA4_GATHER_IN0=$2 GEMMA4_LEDGER_K=$1
export GEMMA4_DIAG_MODE=timing GEMMA4_DIAG_WARMUP=3 GEMMA4_DIAG_REPLAYS=400
export GEMMA4_DIAG_OUT=$OUT/cell.jsonl TT_METAL_CACHE=$W/.jitcache PYTHONHASHSEED=0
# No trace dump and no allocmap: those write files during recording and must not
# be active while wall clock is the measured quantity.
[ "$3" = "iv" ] && export GEMMA4_DISABLE_TERMINAL_TOPDOWN=1 GEMMA4_TERMINAL_TOPDOWN_TRACE=1
$S/occupancy.sh before > $OUT/occ.jsonl
( while true; do $S/occupancy.sh during >> $OUT/occ.jsonl; sleep 5; done ) & SMP=$!
python -m pytest -s -q --timeout=2400 \
  "models/demos/gemma4/tests/unit/test_assistant_standalone_l1.py::test_gather_matched_trace" \
  > $OUT/run.log 2>&1
rc=$?
kill $SMP 2>/dev/null; wait $SMP 2>/dev/null
$S/occupancy.sh after >> $OUT/occ.jsonl
echo "r$4 K=$1 arm=$2 $3 rc=$rc"

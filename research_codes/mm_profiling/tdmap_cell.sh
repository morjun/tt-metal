#!/bin/bash
# Gate 1 capture for the final-use placement intervention.
# $1=K $2=arm $3=cond(base|iv) $4=outdir
# Host-side only: no device profiler. Timing from these cells is NOT used.
set -u
W=/home/masterjunmo/codes/tt-metal-cmdprobe
S=/tmp/claude-1001/-home-masterjunmo-codes-tt-metal-gemma4-l1w/2aa00f30-476e-4db8-9ee5-b3120cbea473/scratchpad
OUT=$4/k$1_a$2_$3; [ -f $OUT/alloc.jsonl ] && { echo "SKIP k=$1 a=$2 $3"; exit 0; }
mkdir -p $OUT; cd $W
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools PATH=$W/python_env/bin:$PATH
export ARCH_NAME=blackhole TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 HF_HUB_OFFLINE=1
export GEMMA4_ASSISTANT_MODEL=google/gemma-4-E2B-it-assistant
export GEMMA4_TUNE_MATMULS=1 GEMMA4_SHARD_ACTIVATIONS=0
export GEMMA4_GATHER_IN0=$2 GEMMA4_LEDGER_K=$1
export GEMMA4_DIAG_MODE=timing GEMMA4_DIAG_WARMUP=3 GEMMA4_DIAG_REPLAYS=20
export GEMMA4_DIAG_OUT=$OUT/cell.jsonl
export TT_METAL_CACHE=$W/.jitcache PYTHONHASHSEED=0
export GEMMA4_TRACEDUMP_DIR=$OUT GEMMA4_ALLOCMAP_OUT=$OUT/alloc.jsonl
# Scope the override to the measured pinned capture (trace 1); the harness's
# initial DRAM capture (trace 0) must be left untouched and is checked as such.
[ "$3" = "iv" ] && export GEMMA4_DISABLE_TERMINAL_TOPDOWN=1 GEMMA4_TERMINAL_TOPDOWN_TRACE=1
$S/occupancy.sh before > $OUT/occ.jsonl
python -m pytest -s -q --timeout=2400 \
  "models/demos/gemma4/tests/unit/test_assistant_standalone_l1.py::test_gather_matched_trace" \
  > $OUT/run.log 2>&1
rc=$?
$S/occupancy.sh after >> $OUT/occ.jsonl
ok=$(grep -c "PASSED\|1 passed" $OUT/run.log 2>/dev/null); ok=${ok:-0}
echo "k=$1 a=$2 $3 rc=$rc passed=$ok traces=$(ls $OUT/trace*.jsonl 2>/dev/null|wc -l)"

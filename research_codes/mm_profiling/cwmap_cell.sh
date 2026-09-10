#!/bin/bash
# Configuration-wait map capture. $1=K $2=arm $3=outdir
# Host-side only: no device profiler, no PROFILE_KERNEL. The capture happens during
# trace RECORDING, so a small replay count is sufficient; timing from this cell is
# not used for anything.
set -u
W=/home/masterjunmo/codes/tt-metal-cmdprobe
S=/tmp/claude-1001/-home-masterjunmo-codes-tt-metal-gemma4-l1w/2aa00f30-476e-4db8-9ee5-b3120cbea473/scratchpad
OUT=$3/k$1_a$2; [ -f $OUT/alloc.jsonl ] && { echo "SKIP k=$1 a=$2"; exit 0; }
mkdir -p $OUT; cd $W
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools PATH=$W/python_env/bin:$PATH
export ARCH_NAME=blackhole TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 HF_HUB_OFFLINE=1
export GEMMA4_ASSISTANT_MODEL=google/gemma-4-E2B-it-assistant
export GEMMA4_TUNE_MATMULS=1 GEMMA4_SHARD_ACTIVATIONS=0
export GEMMA4_GATHER_IN0=$2 GEMMA4_LEDGER_K=$1
export GEMMA4_DIAG_MODE=timing GEMMA4_DIAG_WARMUP=3 GEMMA4_DIAG_REPLAYS=20
export GEMMA4_DIAG_OUT=$OUT/cell.jsonl
export TT_METAL_CACHE=$W/.jitcache PYTHONHASHSEED=0
export GEMMA4_TRACEDUMP_DIR=$OUT
export GEMMA4_ALLOCMAP_OUT=$OUT/alloc.jsonl
$S/occupancy.sh before > $OUT/occ.jsonl
python -m pytest -s -q --timeout=2400 \
  "models/demos/gemma4/tests/unit/test_assistant_standalone_l1.py::test_gather_matched_trace" \
  > $OUT/run.log 2>&1
rc=$?
$S/occupancy.sh after >> $OUT/occ.jsonl
echo "k=$1 a=$2 rc=$rc traces=$(ls $OUT/trace*.jsonl 2>/dev/null|wc -l) allocrec=$(wc -l < $OUT/alloc.jsonl 2>/dev/null||echo 0)"

#!/bin/bash
S=/tmp/claude-1001/-home-masterjunmo-codes-tt-metal-gemma4-l1w/2aa00f30-476e-4db8-9ee5-b3120cbea473/scratchpad
D=$S/tdmap; mkdir -p $D
for K in 3 4 5 6 7 8; do for A in 0 1; do for C in base iv; do
  $S/tdmap_cell.sh $K $A $C $D
done; done; done
echo "GATE1 DONE $(date +%H:%M:%S)"

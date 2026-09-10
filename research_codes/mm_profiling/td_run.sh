#!/bin/bash
S=/tmp/claude-1001/-home-masterjunmo-codes-tt-metal-gemma4-l1w/2aa00f30-476e-4db8-9ee5-b3120cbea473/scratchpad
D=$S/tdtime; mkdir -p $D
# Six counterbalanced rounds: the condition order alternates by round parity so
# base and intervention are not systematically first.
for R in 1 2 3 4 5 6; do
  if [ $((R % 2)) -eq 1 ]; then ORDER="base iv"; else ORDER="iv base"; fi
  for K in 3 4 5; do for A in 0 1; do for C in $ORDER; do
    $S/td_cell.sh $K $A $C $R $D
  done; done; done
  echo "ROUND $R DONE $(date +%H:%M:%S)"
done
echo "GATE2 DONE $(date +%H:%M:%S)"

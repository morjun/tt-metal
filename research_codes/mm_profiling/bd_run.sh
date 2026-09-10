#!/bin/bash
S=/tmp/claude-1001/-home-masterjunmo-codes-tt-metal-gemma4-l1w/2aa00f30-476e-4db8-9ee5-b3120cbea473/scratchpad
D=$S/bdtime; mkdir -p $D
for R in 1 2 3 4 5 6; do
  if [ $((R % 2)) -eq 1 ]; then ORDER="mcast ring ringbd"; else ORDER="ringbd ring mcast"; fi
  for C in $ORDER; do $S/bd_cell.sh $C $R $D time; done
  echo "BD ROUND $R DONE $(date +%H:%M:%S)"
done
echo "BD DONE $(date +%H:%M:%S)"

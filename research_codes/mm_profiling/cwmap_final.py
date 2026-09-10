#!/usr/bin/env python3
"""The configuration-wait map's headline table.

For each K and arm: the number of programs in the FINAL step whose wait is set by
a memory-reuse dependency rather than by launch-message-buffer capacity. Those are
the only waits in the whole trace that reach back further than the 7-program launch
window, and they are the only decision in the trace that depends on K at all
(steps 0..K-2 are decision-identical across K).
"""
import sys, os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configwait_map import load_cell, step_slice

MEM = ["dep_nonbinary_reuse", "dep_binary_reuse", "dep_fixed_addr", "dep_alloc_reset"]
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "cwmap")

cells = {}
for k in range(3, 11):
    for a in (0, 1):
        d = os.path.join(root, f"k{k}_a{a}")
        if os.path.exists(os.path.join(d, "alloc.jsonl")):
            cells[(k, a)] = load_cell(d)


def tight(cell, k, step):
    sl = step_slice(cell, k, step)
    return [p for p in sl if p["combined"] is not None and (p["idx"] - p["combined"]) < 7]


Ks = sorted({k for k, _ in cells})
print("FINAL-STEP memory-reuse-bound waits (tighter than the 7-program launch window)")
print(
    f"{'K':>3s} {'mcast':>6s} {'ring':>6s} {'ring-mcast':>11s} | "
    f"{'mcast lag1':>10s} {'ring lag1':>9s} | whole-trace tight  m / r / r-m"
)
for k in Ks:
    if (k, 0) not in cells or (k, 1) not in cells:
        continue
    tm, tr = tight(cells[(k, 0)], k, k - 1), tight(cells[(k, 1)], k, k - 1)
    l1 = lambda t: sum(1 for p in t if p["idx"] - p["combined"] == 1)
    allt = {}
    for a in (0, 1):
        ps = cells[(k, a)][max(cells[(k, a)])]["programs"]
        allt[a] = sum(1 for p in ps if p["combined"] is not None and (p["idx"] - p["combined"]) < 7)
    print(
        f"{k:3d} {len(tm):6d} {len(tr):6d} {len(tr)-len(tm):+11d} | "
        f"{l1(tm):10d} {l1(tr):9d} | {allt[0]:6d} {allt[1]:5d} {allt[1]-allt[0]:+5d}"
    )

print("\nInterior per-step tight counts (identical across K by construction; listed once per arm)")
for a in (0, 1):
    kmax = max(k for k, aa in cells if aa == a)
    c = cells[(kmax, a)]
    print(
        f"  arm{a}: "
        + str([len(tight(c, kmax, s)) for s in range(kmax)])
        + f"   (last entry is K={kmax}'s FINAL step, not an interior one)"
    )

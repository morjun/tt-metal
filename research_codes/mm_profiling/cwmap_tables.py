#!/usr/bin/env python3
"""Emit the configuration-wait map's tables in markdown, plus every check."""
import sys, os, itertools
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configwait_map import load_cell, step_slice, decision_tuple, dep_source

root = sys.argv[1]
cells = {}
for k in range(3, 11):
    for a in (0, 1):
        d = os.path.join(root, f"k{k}_a{a}")
        if os.path.exists(os.path.join(d, "alloc.jsonl")):
            cells[(k, a)] = load_cell(d)
Ks = sorted({k for k, _ in cells if (k, 0) in cells and (k, 1) in cells})
print("Ks with both arms:", Ks)

# --- check 1: interior determinism ---
print("\n[check] interior steps identical across K (same arm, same step, both traces interior)")
bad = 0
for a in (0, 1):
    for s in range(max(Ks)):
        larger = [k for k in Ks if k > s + 1]
        for x, y in itertools.combinations(larger, 2):
            sx, sy = step_slice(cells[(x, a)], x, s), step_slice(cells[(y, a)], y, s)
            d = sum(1 for i in range(len(sx)) if decision_tuple(sx[i]) != decision_tuple(sy[i]))
            if d:
                bad += 1
                print(f"   MISMATCH arm{a} step{s} K{x} vs K{y}: {d}")
print(f"   comparisons with any difference: {bad}")

# --- check 2: no wait exceeds the launch window ---
mx = 0
for (k, a), c in cells.items():
    for p in c[max(c)]["programs"]:
        if p["combined"] is not None:
            mx = max(mx, p["idx"] - p["combined"])
print(f"\n[check] maximum wait lag over all cells: {mx} (launch window = 7)")

# --- check 3: prefetcher cache pressure ---
tot = Counter()
for (k, a), c in cells.items():
    for p in c[max(c)]["programs"]:
        tot["wrap"] += p["pc_q_wrapped"]
        tot["empt"] += p["pc_q_emptied"]
        tot["evict"] += p["pc_q_evictions"]
        tot["reset"] += p["pc_reset_here"]
print(
    f"[check] prefetcher cache over all cells: wraps={tot['wrap']} drains={tot['empt']} "
    f"evictions={tot['evict']} too-large resets={tot['reset']}"
)


def resets(k, a):
    ps = cells[(k, a)][max(cells[(k, a)])]["programs"]
    return [p for p in ps if p["alloc_reset_events"]]


def tight(k, a, step=None):
    c = cells[(k, a)]
    ps = step_slice(c, k, step) if step is not None else c[max(c)]["programs"]
    return [p for p in ps if p["combined"] is not None and (p["idx"] - p["combined"]) < 7]


print("\n| K | resets mcast | resets ring | ring-mcast | final-step tight mcast | ring | ring-mcast |")
print("|--:|--:|--:|--:|--:|--:|--:|")
for k in Ks:
    rm, rr = len(resets(k, 0)), len(resets(k, 1))
    tm, tr = len(tight(k, 0, k - 1)), len(tight(k, 1, k - 1))
    print(f"| {k} | {rm} | {rr} | {rr-rm:+d} | {tm} | {tr} | {tr-tm:+d} |")

print("\nreset step positions (step index within the trace), by arm:")
for a in (0, 1):
    for k in Ks:
        L = len(cells[(k, a)][max(cells[(k, a)])]["programs"]) // k
        print(f"  arm{a} K={k}: steps " + str(sorted(Counter(p["idx"] // L for p in resets(k, a)).items())))

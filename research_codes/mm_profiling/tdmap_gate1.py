#!/usr/bin/env python3
"""Gate 1 for the final-use placement intervention.

Answers, per (K, arm), from host-side capture only:
  A. does the harness's initial DRAM trace stay untouched?
  B. which placements, resets, binary sends and wait targets change?
  C. does the intervention actually remove or shift the arm-asymmetric
     reset schedule it was designed to target?

If C is negative the intervention is not a test of that mechanism and gate 2
must not run.
"""
import sys, os, subprocess
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configwait_map import load_cell, step_slice, dep_source, decision_tuple, decision_dict

root = sys.argv[1]
KS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "3,4,5,6,7,8".split(","))]


def cell(K, a, c):
    d = os.path.join(root, f"k{K}_a{a}_{c}")
    return (load_cell(d), d) if os.path.exists(os.path.join(d, "alloc.jsonl")) else (None, d)


def resets(ps):
    return [p for p in ps if p["alloc_reset_events"]]


def tight(ps):
    return [p for p in ps if p["combined"] is not None and (p["idx"] - p["combined"]) < 7]


print("A. initial DRAM trace (trace 0) unchanged by the intervention, and test outcome")
print(
    f"{'cell':10s} {'stream0 base==iv':>17s} {'stream1 base==iv':>17s} {'base pass':>10s} {'iv pass':>8s} {'override fired':>15s}"
)
gateA = True
missing = 0
for K in KS:
    for a in (0, 1):
        (cb, db), (ci, di) = cell(K, a, "base"), cell(K, a, "iv")
        if cb is None or ci is None:
            print(f"K{K}a{a}   NOT YET CAPTURED")
            missing += 1
            continue
        eq = {}
        for f in ("stream0.bin", "stream1.bin"):
            x, y = os.path.join(db, f), os.path.join(di, f)
            eq[f] = subprocess.call(["cmp", "-s", x, y]) == 0 if (os.path.exists(x) and os.path.exists(y)) else None
        pas = lambda d: "yes" if "1 passed" in open(os.path.join(d, "run.log")).read() else "NO"
        fired = sum(r["bin_terminal_override"] for p in ci[max(ci)]["programs"] for r in p["regions"])
        print(
            f"K{K}a{a}      {str(eq['stream0.bin']):>17s} {str(eq['stream1.bin']):>17s} "
            f"{pas(db):>10s} {pas(di):>8s} {fired:>15d}"
        )
        if eq["stream0.bin"] is not True or pas(db) != "yes" or pas(di) != "yes":
            gateA = False
print(
    f"  GATE A: {'DRAM trace untouched, both arms correct' if gateA else 'FAILED'}"
    + (f"  ({missing} cells not yet captured)" if missing else "")
)

print("\nB. what changed in the measured trace")
print(
    f"{'cell':8s} {'progs changed':>13s} {'bin_off':>8s} {'nb_off':>7s} {'sendb':>6s} {'wait':>5s} "
    f"{'stalls':>7s} {'src':>5s} {'evict':>6s} | {'sendb base/iv':>14s} {'resets base/iv':>15s} {'tight base/iv':>14s}"
)
for K in KS:
    for a in (0, 1):
        (cb, _), (ci, _) = cell(K, a, "base"), cell(K, a, "iv")
        if cb is None or ci is None:
            continue
        pb, pi = cb[max(cb)]["programs"], ci[max(ci)]["programs"]
        assert len(pb) == len(pi)
        t = Counter()
        nch = 0
        for x, y in zip(pb, pi):
            dx, dy = decision_dict(x), decision_dict(y)
            keys = [k for k in dx if str(dx[k]) != str(dy[k])]
            if keys:
                nch += 1
            for k in keys:
                t[
                    "bin_off"
                    if k.endswith("bin_off")
                    else "nb_off"
                    if k.endswith("nb_off")
                    else "evict"
                    if "evicted" in k
                    else "stalls"
                    if k.startswith("stall")
                    else "src"
                    if k == "src"
                    else "wait"
                    if k == "wait_lag"
                    else "sendb"
                    if k.startswith("send_binary")
                    else k
                ] += 1
        f = lambda g, h: f"{g}/{h}"
        print(
            f"K{K}a{a}   {nch:13d} {t['bin_off']:8d} {t['nb_off']:7d} {t['sendb']:6d} {t['wait']:5d} "
            f"{t['stalls']:7d} {t['src']:5d} {t['evict']:6d} | "
            f"{f(sum(p['send_binary'] for p in pb), sum(p['send_binary'] for p in pi)):>14s} "
            f"{f(len(resets(pb)), len(resets(pi))):>15s} "
            f"{f(len(tight(pb)), len(tight(pi))):>14s}"
        )

print("\nC. does it remove or shift the targeted reset schedule?")
print(
    f"{'K':>2s} | {'resets mcast b/iv':>18s} {'resets ring b/iv':>17s} | "
    f"{'final tight m b/iv':>19s} {'final tight r b/iv':>19s} | {'diff base':>9s} {'diff iv':>7s}"
)
for K in KS:
    (m0, _), (m1, _) = cell(K, 0, "base"), cell(K, 0, "iv")
    (r0, _), (r1, _) = cell(K, 1, "base"), cell(K, 1, "iv")
    if None in (m0, m1, r0, r1):
        continue
    g = lambda c: c[max(c)]["programs"]
    fm = lambda c: len(tight(step_slice(c, K, K - 1)))
    db, di = fm(r0) - fm(m0), fm(r1) - fm(m1)
    print(
        f"{K:2d} | {len(resets(g(m0))):8d}/{len(resets(g(m1))):<9d} "
        f"{len(resets(g(r0))):7d}/{len(resets(g(r1))):<9d} | "
        f"{fm(m0):9d}/{fm(m1):<9d} {fm(r0):9d}/{fm(r1):<9d} | {db:+9d} {di:+7d}"
    )
print("\n  A useful test requires the iv columns to differ from base. If they do not,")
print("  the override does not touch the mechanism and gate 2 must not run.")

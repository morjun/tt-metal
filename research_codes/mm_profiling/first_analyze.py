#!/usr/bin/env python3
"""K=3 first-only SDPA nonbinary relocation: results.

Contrasts are formed WITHIN each round, never from medians. Ranges are observed
min-max across rounds, not confidence intervals. Each cell is one 400-replay
aggregate, so rounds are the only replication.
"""
import json, sys, statistics, random

rows = json.load(open(sys.argv[1]))
by = {}
for r in rows:
    by[(r["round"], r["condition"])] = r
rounds = sorted({r["round"] for r in rows})
print(f"cells: {len(rows)}  rounds present: {rounds}")
bad = [r for r in rows if not r["valid_clock"] or r["foreign_device"] or r["returncode"] or r["timing_records"] != 1]
print(
    f"cells failing qualification: {len(bad)}"
    + (
        "".join(f"\n   r{b['round']} {b['condition']} clk={b['aiclk']} valid={b['valid_clock']}" for b in bad)
        if bad
        else ""
    )
)
print(f"AICLK values: {sorted({r['aiclk'] for r in rows})}")
print("Endpoint clocks agree; this does NOT establish continuous frequency stability.\n")

full = [rd for rd in rounds if all((rd, c) in by for c in ("multicast", "ring", "first"))]
print(f"complete rounds: {full}\n")
print(
    f"{'round':>5s} {'mcast':>10s} {'ring':>10s} {'first':>10s} | "
    f"{'ring-mcast':>11s} {'first-ring':>11s} {'first-mcast':>12s} {'pct recovered':>14s}"
)
pen, eff, res, pct = [], [], [], []
for rd in full:
    m, r, f = (by[(rd, c)]["trace_us"] for c in ("multicast", "ring", "first"))
    p, e, q = r - m, f - r, f - m
    pen.append(p)
    eff.append(e)
    res.append(q)
    pct.append(100 * (-e) / p)
    print(f"{rd:5d} {m:10.2f} {r:10.2f} {f:10.2f} | {p:+11.2f} {e:+11.2f} {q:+12.2f} {100*(-e)/p:13.1f}%")


def rep(name, v, unit="us"):
    print(
        f"  {name:26s} median {statistics.median(v):+8.2f} {unit}  "
        f"range [{min(v):+.2f}, {max(v):+.2f}]  n={len(v)}  "
        f"{sum(1 for x in v if x < 0)}/{len(v)} negative"
    )


def boot(v, n=20000):
    random.seed(0)
    med = sorted(statistics.median(random.choices(v, k=len(v))) for _ in range(n))
    return med[int(0.025 * n)], med[int(0.975 * n)]


if full:
    print()
    rep("ring - mcast (anchor)", pen)
    rep("first - ring (PRIMARY)", eff)
    rep("first - mcast (remaining)", res)
    print(
        f"  {'recovery % (within round)':26s} median {statistics.median(pct):8.1f}%  "
        f"range [{min(pct):.1f}%, {max(pct):.1f}%]"
    )
    lo, hi = boot(eff)
    print(f"\n  paired bootstrap median interval for first - ring: [{lo:+.2f}, {hi:+.2f}] us")
    print("  (descriptive only: six deterministic-order rounds, not a sampling model)")
    if lo <= 0 <= hi:
        print("  INTERVAL INCLUDES ZERO -> inconclusive by the pre-registered rule.")
    a = statistics.median(pen)
    width = max(pen) - min(pen)
    ok = abs(a - 29.84) <= 2.0 and width <= 3.0
    print(
        f"\n  ANCHOR GATE: median ring-mcast {a:+.2f} (need within 2 of +29.84), "
        f"range width {width:.2f} (need <= 3) -> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        print(
            "  Anchor failed: report the primary contrast WITHOUT attributing it to the "
            "qualified original regression."
        )
    print("\n  Attribution limit: first-only changes 108 command ranges, 18 stall-flag nodes and the")
    print("  binary-send inventory. The measured effect is the whole allocation intervention, not")
    print("  the deferred wait alone. No decomposition of the residual is licensed.")

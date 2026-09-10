#!/usr/bin/env python3
"""Gate 2 for the final-use placement intervention.

Reports exactly the three quantities TD_PREREGISTERED.md named, and nothing else:

  1. absolute per-arm response      = intervention - base, per (round, K, arm)
  2. paired differential response   = change in (ring - multicast), within round
  3. paired K=4 dip change          = change in [mean(K3,K5) - K4] of the
                                      (ring - multicast) penalty, within round

Brackets are the OBSERVED MIN-MAX RANGE across rounds, not confidence intervals.

Clock matching follows analyze_gather_campaign.py: `valid_clock` says the clock
held within a cell, not that two cells ran at the same clock, so every contrast
here is formed only from cells whose post-batch AICLK agrees.
"""
import json, os, sys, statistics
from collections import defaultdict

root = sys.argv[1]
STRICT_CLOCK = "--any-clock" not in sys.argv


def clock(row):
    post = row.get("post")
    if not isinstance(post, dict):
        return None
    info = post.get("device_info") or [{}]
    return (info[0].get("telemetry") or {}).get("aiclk") if isinstance(info[0], dict) else None


def read_cell(d):
    """Median trace_us over the cell's timing records, plus its clock."""
    f = os.path.join(d, "cell.jsonl")
    if not os.path.exists(f):
        return None
    times, clk, ok = [], None, True
    for line in open(f):
        r = json.loads(line)
        if r.get("event") != "timing":
            continue
        if not r.get("valid_clock"):
            ok = False
        times.append(r["trace_us"])
        clk = clock(r) or clk
    if not times:
        return None
    return {"us": statistics.median(times), "n": len(times), "clock": clk, "valid": ok}


cells = {}
for name in sorted(os.listdir(root)):
    # r<round>_k<K>_a<arm>_<cond>
    try:
        r, k, a, c = name.split("_")
        key = (int(r[1:]), int(k[1:]), int(a[1:]), c)
    except (ValueError, IndexError):
        continue
    v = read_cell(os.path.join(root, name))
    if v:
        cells[key] = v

rounds = sorted({k[0] for k in cells})
Ks = sorted({k[1] for k in cells})
print(f"rounds present: {rounds}   K present: {Ks}   cells: {len(cells)}")

invalid = [k for k, v in cells.items() if not v["valid"]]
if invalid:
    print(f"WARNING cells with a clock change inside them: {invalid}")


def get(rd, K, arm, cond):
    return cells.get((rd, K, arm, cond))


def clock_ok(*cs):
    if not STRICT_CLOCK:
        return all(c is not None for c in cs)
    cl = {c["clock"] for c in cs if c}
    return all(c is not None for c in cs) and len(cl) == 1


print("\n1. ABSOLUTE per-arm response (intervention - base), us/trace. " "Positive = intervention slower.")
print(f"{'K':>2s} {'arm':>7s} | " + " ".join(f"{'r'+str(r):>9s}" for r in rounds) + f" | {'median':>8s} {'range':>18s}")
for K in Ks:
    for a, nm in ((0, "mcast"), (1, "ring")):
        vals = []
        cellsrow = []
        for rd in rounds:
            b, i = get(rd, K, a, "base"), get(rd, K, a, "iv")
            if b and i and clock_ok(b, i):
                v = i["us"] - b["us"]
                vals.append(v)
                cellsrow.append(f"{v:+9.2f}")
            else:
                cellsrow.append(f"{'-':>9s}")
        med = f"{statistics.median(vals):+8.2f}" if vals else f"{'-':>8s}"
        rng = f"[{min(vals):+7.2f},{max(vals):+7.2f}]" if vals else f"{'-':>18s}"
        print(f"{K:2d} {nm:>7s} | " + " ".join(cellsrow) + f" | {med} {rng}")

print("\n2. PAIRED differential response: change in (ring - multicast), within round.")
print(f"{'K':>2s} | " + " ".join(f"{'r'+str(r):>9s}" for r in rounds) + f" | {'median':>8s} {'range':>18s}")
diffs = {}
for K in Ks:
    vals, row = [], []
    for rd in rounds:
        c = {(a, cond): get(rd, K, a, cond) for a in (0, 1) for cond in ("base", "iv")}
        if all(c.values()) and clock_ok(*c.values()):
            base_d = c[(1, "base")]["us"] - c[(0, "base")]["us"]
            iv_d = c[(1, "iv")]["us"] - c[(0, "iv")]["us"]
            v = iv_d - base_d
            vals.append(v)
            row.append(f"{v:+9.2f}")
            diffs[(rd, K, "base")] = base_d
            diffs[(rd, K, "iv")] = iv_d
        else:
            row.append(f"{'-':>9s}")
    med = f"{statistics.median(vals):+8.2f}" if vals else f"{'-':>8s}"
    rng = f"[{min(vals):+7.2f},{max(vals):+7.2f}]" if vals else f"{'-':>18s}"
    print(f"{K:2d} | " + " ".join(row) + f" | {med} {rng}")

print("\n   underlying penalty (ring - multicast), us/trace, by condition:")
for cond in ("base", "iv"):
    for K in Ks:
        vals = [diffs[(rd, K, cond)] for rd in rounds if (rd, K, cond) in diffs]
        if vals:
            print(
                f"     {cond:4s} K={K}: median {statistics.median(vals):+8.2f}  "
                f"range [{min(vals):+7.2f}, {max(vals):+7.2f}]  n={len(vals)}"
            )

print("\n3. PAIRED K=4 dip change. dip = mean(K3, K5) - K4 of the (ring - multicast) penalty.")
if {3, 4, 5} <= set(Ks):
    print(f"{'round':>6s} {'dip base':>9s} {'dip iv':>9s} {'change':>9s}")
    ch = []
    for rd in rounds:
        try:
            db = (diffs[(rd, 3, "base")] + diffs[(rd, 5, "base")]) / 2 - diffs[(rd, 4, "base")]
            di = (diffs[(rd, 3, "iv")] + diffs[(rd, 5, "iv")]) / 2 - diffs[(rd, 4, "iv")]
        except KeyError:
            print(f"{rd:6d} {'-':>9s} {'-':>9s} {'-':>9s}")
            continue
        ch.append(di - db)
        print(f"{rd:6d} {db:9.2f} {di:9.2f} {di-db:+9.2f}")
    if ch:
        print(
            f"\n   median dip change {statistics.median(ch):+.2f}  "
            f"observed range [{min(ch):+.2f}, {max(ch):+.2f}] over {len(ch)} rounds "
            f"(range, NOT a confidence interval)"
        )
        neg = sum(1 for v in ch if v < 0)
        print(f"   rounds with a NEGATIVE dip change (dip shrinks): {neg}/{len(ch)}")
        print("\n   Pre-registered reading: H1 (the final-step reset/wait asymmetry causes the")
        print("   dip) predicts a negative dip change in a majority of rounds AND a RISING")
        print("   K=4 penalty. H0 predicts the dip unchanged within its range, or larger.")
else:
    print("   needs K=3, 4 and 5")

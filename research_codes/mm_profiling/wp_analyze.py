#!/usr/bin/env python3
"""Wait-placement experiment (6P.39) results.

Primary outcome: absolute K=3 ring trace time per condition, and the paired
`early - late` response, which estimates sensitivity to the two wait POSITIONS
under bottom-up allocation with added-command cost held equal.

Brackets are the observed min-max range across rounds, NOT confidence intervals.
Each cell is a single 400-replay aggregate (trace_us = elapsed_ns/replays/1000),
so there is no within-cell dispersion and the rounds are the only replication.
"""
import json, os, sys, statistics

root = sys.argv[1]
COND = ["ringbase", "bottomup", "early", "late"]


def clock(row):
    post = row.get("post")
    if not isinstance(post, dict):
        return None
    info = post.get("device_info") or [{}]
    return (info[0].get("telemetry") or {}).get("aiclk") if isinstance(info[0], dict) else None


def read(d):
    f = os.path.join(d, "cell.jsonl")
    if not os.path.exists(f):
        return None
    for line in open(f):
        r = json.loads(line)
        if r.get("event") == "timing":
            cpu = []
            for l2 in open(os.path.join(d, "occ.jsonl")):
                try:
                    o = json.loads(l2)
                except json.JSONDecodeError:
                    continue
                if o.get("at") == "during":
                    cpu.append(o["foreign_cpu_pct"])
            return {
                "us": r["trace_us"],
                "clock": clock(r),
                "valid": bool(r.get("valid_clock")),
                "replays": r["replays"],
                "cpu": statistics.median(cpu) if cpu else None,
            }
    return None


cells = {}
for name in sorted(os.listdir(root)):
    try:
        rd, c = name.split("_", 1)
        key = (int(rd[1:]), c)
    except ValueError:
        continue
    v = read(os.path.join(root, name))
    if v:
        cells[key] = v
rounds = sorted({k[0] for k in cells})
print(f"rounds: {rounds}   cells: {len(cells)}")
clocks = {v["clock"] for v in cells.values()}
print(
    f"AICLK values across the campaign: {clocks}   "
    f"cells with an internal clock change: {sum(1 for v in cells.values() if not v['valid'])}"
)
print(f"replays per cell: {sorted({v['replays'] for v in cells.values()})}")

print("\nAbsolute K=3 ring trace time, us (one 400-replay aggregate per cell)")
print(f"{'cond':10s} | " + " ".join(f"{'r'+str(r):>10s}" for r in rounds) + f" | {'median':>10s} {'range':>21s}")
for c in COND:
    v = [cells[(r, c)]["us"] for r in rounds if (r, c) in cells]
    row = " ".join(f"{cells[(r,c)]['us']:10.2f}" if (r, c) in cells else f"{'-':>10s}" for r in rounds)
    if v:
        print(f"{c:10s} | {row} | {statistics.median(v):10.2f} [{min(v):9.2f},{max(v):9.2f}]")


def paired(a, b, label, note=""):
    v, row = [], []
    for r in rounds:
        if (r, a) in cells and (r, b) in cells:
            d = cells[(r, a)]["us"] - cells[(r, b)]["us"]
            v.append(d)
            row.append(f"{d:+10.2f}")
        else:
            row.append(f"{'-':>10s}")
    if not v:
        return None
    print(
        f"{label:22s} | " + " ".join(row) + f" | {statistics.median(v):+10.2f} [{min(v):+9.2f},{max(v):+9.2f}]  "
        f"{sum(1 for x in v if x>0)}/{len(v)} positive {note}"
    )
    return v


print("\nPaired responses, us (positive = the first condition is slower)")
print(f"{'contrast':22s} | " + " ".join(f"{'r'+str(r):>10s}" for r in rounds) + f" | {'median':>10s} {'range':>21s}")
anchor = paired("bottomup", "ringbase", "bottomup - ringbase", "<- ANCHOR, must reproduce -7.59")
primary = paired("early", "late", "early - late", "<- PRIMARY: wait position")
paired("early", "bottomup", "early - bottomup", "(position + 2 commands)")
paired("late", "bottomup", "late - bottomup", "(2 commands only)")
paired("early", "ringbase", "early - ringbase")

print("\nForeign host CPU during each cell (median %)")
print(f"{'cond':10s} | " + " ".join(f"{'r'+str(r):>10s}" for r in rounds))
for c in COND:
    print(
        f"{c:10s} | "
        + " ".join(
            f"{cells[(r,c)]['cpu']:10.0f}" if (r, c) in cells and cells[(r, c)]["cpu"] is not None else f"{'-':>10s}"
            for r in rounds
        )
    )

if anchor and primary:
    am = statistics.median(anchor)
    pm = statistics.median(primary)
    print(
        f"\nANCHOR bottomup - ringbase = {am:+.2f} us "
        f"(6P.38 measured -7.59 [-8.47, -7.18]); "
        f"{'reproduces' if -9.0 < am < -6.5 else 'DOES NOT REPRODUCE -- campaign void'}"
    )
    print(
        f"PRIMARY early - late = {pm:+.2f} us. Pre-registered prediction: positive if "
        f"delaying these two waits contributes to the ring speedup."
    )
    if am:
        print(
            f"As a share of the {abs(am):.2f} us bottom-up ring speedup: {100*pm/abs(am):.1f}%. "
            f"This is sensitivity to two wait positions, NOT automatically an additive "
            f"component of the ring/multicast gap."
        )

#!/usr/bin/env python3
"""6P.40: how much of the K=3 ring penalty the targeted allocation variant recovers.

Ranges are observed min-max across rounds, NOT confidence intervals. Each cell is
one 400-replay aggregate, so the rounds are the only replication.
"""
import json, os, sys, statistics

root = sys.argv[1]
COND = ["mcast", "ring", "ringbd"]


def clock(r):
    p = r.get("post")
    if not isinstance(p, dict):
        return None
    i = (p.get("device_info") or [{}])[0]
    return (i.get("telemetry") or {}).get("aiclk") if isinstance(i, dict) else None


def read(d):
    f = os.path.join(d, "cell.jsonl")
    if not os.path.exists(f):
        return None
    for l in open(f):
        r = json.loads(l)
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
                "cpu": statistics.median(cpu) if cpu else None,
            }
    return None


cells = {}
for n in sorted(os.listdir(root)):
    try:
        rd, c = n.split("_", 1)
        k = (int(rd[1:]), c)
    except ValueError:
        continue
    v = read(os.path.join(root, n))
    if v:
        cells[k] = v
rounds = sorted({k[0] for k in cells})
print(
    f"rounds: {rounds}  cells: {len(cells)}  AICLK: {sorted({v['clock'] for v in cells.values()})}  "
    f"internal clock changes: {sum(1 for v in cells.values() if not v['valid'])}"
)

print("\nAbsolute K=3 trace time, us")
print(f"{'cond':8s} | " + " ".join(f"{'r'+str(r):>9s}" for r in rounds) + f" | {'median':>9s} {'range':>21s}")
for c in COND:
    v = [cells[(r, c)]["us"] for r in rounds if (r, c) in cells]
    row = " ".join(f"{cells[(r,c)]['us']:9.2f}" if (r, c) in cells else f"{'-':>9s}" for r in rounds)
    if v:
        print(f"{c:8s} | {row} | {statistics.median(v):9.2f} [{min(v):9.2f},{max(v):9.2f}]")


def paired(a, b, label):
    v, row = [], []
    for r in rounds:
        if (r, a) in cells and (r, b) in cells:
            d = cells[(r, a)]["us"] - cells[(r, b)]["us"]
            v.append(d)
            row.append(f"{d:+9.2f}")
        else:
            row.append(f"{'-':>9s}")
    if not v:
        return None
    print(
        f"{label:22s} | " + " ".join(row) + f" | {statistics.median(v):+9.2f} [{min(v):+8.2f},{max(v):+8.2f}]  "
        f"{sum(1 for x in v if x>0)}/{len(v)} pos"
    )
    return v


print("\nPaired responses, us (positive = first condition slower)")
print(f"{'contrast':22s} | " + " ".join(f"{'r'+str(r):>9s}" for r in rounds) + f" | {'median':>9s} {'range':>19s}")
pen = paired("ring", "mcast", "ring - mcast (ANCHOR)")
res = paired("ringbd", "mcast", "ringbd - mcast")
eff = paired("ringbd", "ring", "ringbd - ring")

print("\nForeign host CPU during each cell (median %)")
for c in COND:
    print(
        f"  {c:8s} "
        + " ".join(
            f"{cells[(r,c)]['cpu']:6.0f}" if (r, c) in cells and cells[(r, c)]["cpu"] is not None else f"{'-':>6s}"
            for r in rounds
        )
    )

if pen and res and eff:
    p, q, e = statistics.median(pen), statistics.median(res), statistics.median(eff)
    print(
        f"\nANCHOR ring - mcast = {p:+.2f} us "
        f"(standing 400-replay value +29.84 / +29.88); "
        f"{'reproduces' if 27 < p < 33 else 'DOES NOT REPRODUCE -- campaign void'}"
    )
    print(
        f"PRIMARY DELIVERABLE: of the {p:.2f} us ring penalty, the targeted variant recovers "
        f"{-e:+.2f} us = {100*(-e)/p:.1f}%."
    )
    print(f"Residual penalty after the variant: {q:+.2f} us.")
    print("The audit showed this variant RELOCATES the tight waits (413/449 -> 407/443) rather than")
    print("removing them: tight-wait count 20, tight-lag sum 51 and total wait count 452 are all")
    print("unchanged. Read the recovery figure against that, not as a deferral result.")

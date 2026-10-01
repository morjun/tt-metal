#!/usr/bin/env python3
"""Paired process-round analysis for the preregistered K=3 first-only campaign."""
import argparse, csv, json, random, statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
a = p.parse_args()
rows = json.loads((a.root / "summary.json").read_text())
assert len(rows) == 18 and len({(r["round"], r["condition"]) for r in rows}) == 18
assert all(
    r["returncode"] == 0 and r["timing_records"] == 1 and r["valid_clock"] and not r["foreign_device"] for r in rows
)
assert len({r["aiclk"] for r in rows}) == 1
by = {(r["round"], r["condition"]): r["trace_us"] for r in rows}
assert set(by) == {(r, c) for r in range(1, 7) for c in ["multicast", "ring", "first"]}


def stats(v):
    rng = random.Random(3911)
    boots = sorted(statistics.median(rng.choices(v, k=len(v))) for _ in range(20000))
    return dict(
        median=statistics.median(v),
        range=[min(v), max(v)],
        bootstrap_95_median=[boots[499], boots[19499]],
        positive=sum(x > 0 for x in v),
        values=v,
    )


report = {
    "scope": "us per full K=3 trace; 400 replays per process; six paired rounds",
    "absolute": {},
    "contrasts": {},
    "occupancy": {},
}
for c in ["multicast", "ring", "first"]:
    report["absolute"][c] = stats([by[r, c] for r in range(1, 7)])
for c, d in [("first", "ring"), ("ring", "multicast"), ("first", "multicast")]:
    report["contrasts"][f"{c} - {d}"] = stats([by[r, c] - by[r, d] for r in range(1, 7)])
anchor = report["contrasts"]["ring - multicast"]
report["anchor_pass"] = abs(anchor["median"] - 29.84) <= 2 and anchor["range"][1] - anchor["range"][0] <= 3
for r in rows:
    d = a.root / f"r{r['round']}_{r['condition']}"
    o = [json.loads(l) for l in (d / "occ.jsonl").read_text().splitlines()]
    assert all(not z["foreign_holders"].strip() for z in o)
    cpu = [z["foreign_cpu_pct"] for z in o if z["at"] == "during"]
    report["occupancy"][d.name] = {
        "samples": len(o),
        "foreign_cpu_median": statistics.median(cpu),
        "foreign_cpu_max": max(cpu),
    }
report["primary_cpu_pair_imbalance"] = max(
    abs(
        report["occupancy"][f"r{r}_first"]["foreign_cpu_median"]
        - report["occupancy"][f"r{r}_ring"]["foreign_cpu_median"]
    )
    for r in range(1, 7)
)
(a.root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
with (a.root / "cells.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
lines = [
    "# K=3 first-only SDPA relocation timing",
    "",
    report["scope"]
    + ". Ranges are observed minima/maxima, not confidence intervals. Bootstrap intervals resample process rounds; six rounds provide limited uncertainty resolution.",
    "",
    "| Contrast | Median | Range | Bootstrap 95% median interval | Positive rounds |",
    "|---|---:|---|---|---:|",
]
for k, v in report["contrasts"].items():
    lines.append(
        f"| {k} | {v['median']:+.3f} | [{v['range'][0]:+.3f}, {v['range'][1]:+.3f}] | [{v['bootstrap_95_median'][0]:+.3f}, {v['bootstrap_95_median'][1]:+.3f}] | {v['positive']}/6 |"
    )
lines += [
    "",
    f"Preregistered anchor gate: {'PASS' if report['anchor_pass'] else 'FAIL'}.",
    "",
    "| Condition | Absolute median us/trace | Range |",
    "|---|---:|---|",
]
for k, v in report["absolute"].items():
    lines.append(f"| {k} | {v['median']:.3f} | [{v['range'][0]:.3f}, {v['range'][1]:.3f}] |")
lines += [
    "",
    f"18 successful timing cells; one timing aggregate per cell; recorded endpoint clock {rows[0]['aiclk']} MHz throughout; no detected foreign device holders. Largest primary-pair difference in median sampled foreign CPU: {report['primary_cpu_pair_imbalance']:.1f} percentage points. Endpoint equality does not establish uninterrupted frequency stability, and CPU summaries cover the whole cell rather than only the measured batch.",
    "",
    "The allocation change was validated separately before timing. These timing cells do not compare outputs. It changes 108 command ranges, 18 stall-flag nodes and binary-send inventory; the measured response cannot be assigned entirely to the delayed node-449 wait. No dip criterion, K sweep or Item 19 implementation is included.",
    "",
]
(a.root / "REPORT.md").write_text("\n".join(lines))
print("\n".join(lines))

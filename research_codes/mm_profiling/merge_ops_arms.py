#!/usr/bin/env python3
"""Merge per-arm `analyze_ops.py` outputs into one §6P.49-form table.

§6P.49's three-column table was assembled BY HAND from separate analyze_ops.py runs.
This does it mechanically so the result is reproducible and the collapse rule is explicit.

Usage:
    merge_ops_arms.py --arm dram=a1.csv,a2.csv --arm ring=b1.csv,b2.csv [--keep 11] [--promote 0.5]

Each --arm takes one or more CSVs (repeat passes); us/step is averaged across passes and the
observed min-max is carried so a claimed delta can be checked against between-process spread.
An op code absent from an arm is reported as n=0, 0.0 -- never dropped, never silently joined away.
"""
import argparse, re, subprocess, sys, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_one(csv):
    out = subprocess.run(
        [sys.executable, os.path.join(HERE, "analyze_ops.py"), csv], capture_output=True, text=True
    ).stdout
    m = re.search(r"(\d+) rows = (\d+) setup ops \+ (\d+) reps x (\d+) ops/step", out)
    reps, ops = (int(m.group(3)), int(m.group(4))) if m else (None, None)
    tot = re.search(r"mean ([\d.]+) us\s+spread", out)
    total = float(tot.group(1)) if tot else None
    rows, seen = {}, False
    for line in out.splitlines():
        if line.startswith("BY OP CODE"):
            seen = True
            continue
        if seen:
            if not line.strip() or re.match(r"^[A-Z ]+:?$", line.strip()) or line.startswith("HOTTEST"):
                if rows:
                    break
                continue
            mm = re.match(r"\s+(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%", line)
            if mm:
                rows[mm.group(1)] = (int(mm.group(2)), float(mm.group(3)))
            elif rows:
                break
    return {"reps": reps, "ops": ops, "total": total, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=csv[,csv...]")
    ap.add_argument("--keep", type=int, default=11)
    ap.add_argument(
        "--promote", type=float, default=0.5, help="always show an op whose |delta| exceeds this in any arm"
    )
    ap.add_argument("--base", default=None, help="baseline arm name (default: first)")
    ap.add_argument("--vs", default=None, help="second reference for the final column, e.g. l1_sharded")
    ap.add_argument("--trace-mult", type=int, default=3, help="steps per trace, for the us/trace footprint")
    a = ap.parse_args()

    arms = []
    for spec in a.arm:
        name, csvs = spec.split("=", 1)
        runs = [parse_one(c) for c in csvs.split(",")]
        rows = {}
        for op in {o for r in runs for o in r["rows"]}:
            vals = [r["rows"][op][1] for r in runs if op in r["rows"]]
            ns = {r["rows"][op][0] for r in runs if op in r["rows"]}
            rows[op] = (max(ns) if ns else 0, statistics.mean(vals), min(vals), max(vals))
        arms.append(
            {
                "name": name,
                "runs": runs,
                "rows": rows,
                "total": statistics.mean([r["total"] for r in runs]),
                "ops": runs[0]["ops"],
                "reps": runs[0]["reps"],
            }
        )

    base = next(x for x in arms if x["name"] == (a.base or arms[0]["name"]))
    print("### Gates\n")
    print("| arm | reps | ops/step | passes | STEP TOTAL us/step (mean) | between-pass spread |")
    print("|---|--:|--:|--:|--:|--:|")
    for x in arms:
        t = [r["total"] for r in x["runs"]]
        print(
            f"| `{x['name']}` | {x['reps']} | {x['ops']} | {len(x['runs'])} | {x['total']:.1f} | {max(t)-min(t):.1f} |"
        )
    print()

    allops = sorted({o for x in arms for o in x["rows"]}, key=lambda o: -base["rows"].get(o, (0, 0, 0, 0))[1])

    def d(x, op):
        return x["rows"].get(op, (0, 0.0, 0.0, 0.0))[1] - base["rows"].get(op, (0, 0.0, 0.0, 0.0))[1]

    promoted = {o for o in allops if any(abs(d(x, o)) > a.promote for x in arms)}
    keep = [o for o in allops[: a.keep]] + [o for o in allops[a.keep :] if o in promoted]
    tail = [o for o in allops if o not in keep]

    hdr = "| op | " + " | ".join(f"n | {x['name']}" + ("" if x is base else " | vs base") for x in arms)
    if a.vs:
        hdr += f" | vs {a.vs}"
    print("### By op code, us/step\n")
    print(hdr + " |")
    print("|---" * (1 + sum(3 if x is not base else 2 for x in arms) + (1 if a.vs else 0)) + "|")
    vsarm = next((x for x in arms if x["name"] == a.vs), None)

    def row(label, get_n, get_us, get_d, get_vs):
        cells = [label]
        for x in arms:
            cells.append(str(get_n(x)))
            cells.append(f"{get_us(x):.1f}")
            if x is not base:
                cells.append(f"{get_d(x):+.1f}")
        if vsarm:
            cells.append(f"{get_vs():+.1f}")
        print("| " + " | ".join(cells) + " |")

    for op in keep:
        row(
            f"`{op}`",
            lambda x, op=op: x["rows"].get(op, (0,))[0],
            lambda x, op=op: x["rows"].get(op, (0, 0.0))[1],
            lambda x, op=op: d(x, op),
            lambda op=op: (arms[-1]["rows"].get(op, (0, 0.0))[1] - vsarm["rows"].get(op, (0, 0.0))[1])
            if vsarm
            else 0.0,
        )
    if tail:
        row(
            f"{len(tail)} smaller op codes",
            lambda x: sum(x["rows"].get(o, (0,))[0] for o in tail),
            lambda x: sum(x["rows"].get(o, (0, 0.0))[1] for o in tail),
            lambda x: sum(d(x, o) for o in tail),
            lambda: sum((arms[-1]["rows"].get(o, (0, 0.0))[1] - vsarm["rows"].get(o, (0, 0.0))[1]) for o in tail)
            if vsarm
            else 0.0,
        )
    row(
        "**TOTAL**",
        lambda x: sum(v[0] for v in x["rows"].values()),
        lambda x: sum(v[1] for v in x["rows"].values()),
        lambda x: sum(v[1] for v in x["rows"].values()) - sum(v[1] for v in base["rows"].values()),
        lambda: sum(v[1] for v in arms[-1]["rows"].values()) - sum(v[1] for v in vsarm["rows"].values())
        if vsarm
        else 0.0,
    )
    print()
    for x in arms:
        s = sum(v[1] for v in x["rows"].values())
        flag = "OK" if abs(s - x["total"]) < 0.5 else f"MISMATCH (collapse dropped {x['total']-s:+.1f})"
        print(f"  cross-check `{x['name']}`: sum of rows {s:.1f} vs analyze_ops STEP TOTAL {x['total']:.1f} -> {flag}")
    if vsarm:
        print(f"\n### Ring-attributable footprint (vs `{vsarm['name']}`), x{a.trace_mult} = us/trace\n")
        print("| | us/step | us/trace |")
        print("|---|--:|--:|")
        for op in keep + [None]:
            if op is None:
                continue
            dd = arms[-1]["rows"].get(op, (0, 0.0))[1] - vsarm["rows"].get(op, (0, 0.0))[1]
            if abs(dd) > a.promote:
                print(f"| `{op}` | {dd:+.1f} | {dd*a.trace_mult:+.1f} |")
        net = sum(v[1] for v in arms[-1]["rows"].values()) - sum(v[1] for v in vsarm["rows"].values())
        print(f"| **net kernel time** | **{net:+.1f}** | **{net*a.trace_mult:+.1f}** |")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-op analysis of a Gemma-4 drafter step from a tracy ops CSV.

WHY THIS EXISTS
---------------
`DEVICE KERNEL DURATION [ns]` is unusable on this build: it is computed as
`op_first_last` over `core: ANY` (earliest start across cores, latest end across
cores) and START CYCLE splits into two timer-base clusters, so any op touching a
stale-base core reports `true + ~1764 s`.  See MEASUREMENT_RECORD.md section 7.

`DEVICE KERNEL DURATION PER CORE {MIN,MAX,AVG}` is computed by
`op_core_first_last_analysis` (tools/tracy/process_device_log.py:550), which groups
the timeseries BY CORE and runs first-last within each core.  A cross-core base
mismatch cannot leak in.  Those columns are therefore correct -- but they are only
emitted by the legacy Python post-processor, so the CSV must be produced with:

    python tools/tracy/process_ops_logs.py --force-legacy-device-logs --date -n percore

(The default C++ path bypasses the Python timerAnalysis DSL entirely --
process_ops_logs.py:1099 -- which is why every CSV on disk has those columns blank.)

CONVENTIONS
-----------
* op duration      = PER CORE MAX  (the critical-path core)
* cross-core skew  = PER CORE MAX - PER CORE MIN
* Per-core "cycles since reset" are NOT synchronised across cores.  Only same-core
  durations are meaningful; this script never compares timestamps across cores.
"""
import argparse
import collections
import csv
import re
import statistics
import sys

DUR = "DEVICE KERNEL DURATION PER CORE MAX [ns]"
DUR_MIN = "DEVICE KERNEL DURATION PER CORE MIN [ns]"
DUR_AVG = "DEVICE KERNEL DURATION PER CORE AVG [ns]"
AGG = "DEVICE KERNEL DURATION [ns]"

_MM_CFG = re.compile(r"in0_block_w=(\d+).*?per_core_M=(\d+).*?per_core_N=(\d+)")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(path, device=None):
    rows = list(csv.DictReader(open(path)))
    # A multi-device run interleaves every device's ops in one CSV, which breaks the
    # constant-stride rep detection below. Analyse one device at a time; per-core
    # durations are per device anyway, so this loses nothing.
    devs = sorted({r.get("DEVICE ID") for r in rows if r.get("DEVICE ID") is not None})
    if len(devs) > 1:
        pick = device if device is not None else devs[0]
        if pick not in devs:
            sys.exit(f"{path}: DEVICE ID {pick!r} not present; have {devs}")
        rows = [r for r in rows if r.get("DEVICE ID") == pick]
        print(f"  multi-device run: devices {devs}, analysing DEVICE ID {pick} ({len(rows)} rows)")
    if not rows:
        sys.exit(f"{path}: empty")
    if DUR not in rows[0]:
        sys.exit(f"{path}: no '{DUR}' column -- was it produced with --force-legacy-device-logs?")
    blank = sum(1 for r in rows if _f(r.get(DUR)) is None)
    if blank:
        sys.exit(f"{path}: {blank}/{len(rows)} rows have a blank {DUR}. Re-run with --force-legacy-device-logs.")
    return rows


def find_period(rows):
    """Reps are identified by an op code that occurs exactly `reps` times at a constant
    stride. Returns (start_index, period, n_reps)."""
    codes = [r["OP CODE"] for r in rows]
    counts = collections.Counter(codes)
    best = None
    for code, n in counts.items():
        if n < 2:
            continue
        idx = [i for i, c in enumerate(codes) if c == code]
        gaps = {idx[i + 1] - idx[i] for i in range(len(idx) - 1)}
        if len(gaps) == 1:
            period = gaps.pop()
            if best is None or n > best[2]:
                best = (idx[0], period, n)
    if best is None:
        sys.exit("could not find a repeating op sequence -- is this a multi-rep run?")
    _, period, reps = best
    start = len(rows) - period * reps
    if start < 0:
        sys.exit(f"period {period} x {reps} reps exceeds {len(rows)} rows")
    return start, period, reps


def label_steps(rep):
    """Attribute each op of one step to its stage in DRAFTER_WALKTHROUGH.md Part II section 7.

    Layer boundaries are taken from the SdpaDecode ops -- one per layer, and the only
    op that appears exactly once per layer -- so this does not depend on op counts that
    change between tp=1 and tp=2."""
    sdpa = [i for i, r in enumerate(rep) if r["OP CODE"].startswith("SdpaDecode")]
    n_layers = len(sdpa)
    labels = [None] * len(rep)

    if n_layers == 0:
        return ["?"] * len(rep), 0

    # pre-layer: everything up to and including the first matmul (pre_projection)
    first_mm = next(i for i, r in enumerate(rep) if r["OP CODE"].startswith("Matmul"))
    for i in range(first_mm + 1):
        labels[i] = "1-3 embed/concat/pre_proj"

    # Layer boundaries are structural, not midpoints: every layer emits the same op
    # sequence, so SdpaDecode sits at a constant offset from its layer's first op.
    # Derive that offset from layer 0, whose first op is right after pre_projection.
    layer0_start = first_mm + 1
    offset = sdpa[0] - layer0_start
    bounds = [s - offset for s in sdpa]
    strides = {bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)}
    if len(strides) != 1:
        print(f"  WARNING: layer strides are not uniform ({sorted(strides)}); labels may be off", file=sys.stderr)
    stride = max(strides) if strides else 0
    bounds.append(bounds[-1] + stride)  # end of the last layer
    head_start = min(bounds[-1], len(rep))

    for li in range(n_layers):
        kind = "full" if li == n_layers - 1 else "sliding"
        for i in range(bounds[li], min(bounds[li + 1], len(rep))):
            labels[i] = f"4 layer{li} ({kind})"

    for i in range(head_start, len(rep)):
        labels[i] = "5-8 norm/head/post_proj"
    for i, v in enumerate(labels):
        if v is None:
            labels[i] = "?"
    return labels, n_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--top", type=int, default=20, help="how many ops to list in the hot list")
    ap.add_argument("--device", default=None, help="DEVICE ID to analyse (multi-device runs); default: the lowest")
    args = ap.parse_args()

    rows = load(args.csv, args.device)
    start, period, reps = find_period(rows)
    setup = start
    print(f"{args.csv}")
    print(f"  {len(rows)} rows = {setup} setup ops + {reps} reps x {period} ops/step")

    bogus = sum(1 for r in rows if (_f(r.get(AGG)) or 0) > 1e9)
    print(f"  aggregate '{AGG}' bogus (>1 s): {bogus}/{len(rows)}  <- why this script uses PER CORE MAX")

    # per-position across reps
    steps = [rows[start + k * period : start + (k + 1) * period] for k in range(reps)]
    rep0 = steps[0]
    labels, n_layers = label_steps(rep0)

    tot = []
    for k in range(reps):
        tot.append(sum(_f(r[DUR]) for r in steps[k]) / 1000.0)
    print(f"  layers detected: {n_layers}")
    print(f"\nSTEP TOTAL (sum of per-op PER CORE MAX):")
    print("  " + "  ".join(f"{t:.1f}" for t in tot) + f"   us/step")
    print(f"  mean {statistics.mean(tot):.1f} us   spread {(max(tot)-min(tot))/statistics.mean(tot)*100:.2f}%")

    # ---- per stage ----
    stage = collections.OrderedDict()
    for i, lab in enumerate(labels):
        us = statistics.mean(_f(steps[k][i][DUR]) for k in range(reps)) / 1000.0
        d = stage.setdefault(lab, [0.0, 0])
        d[0] += us
        d[1] += 1
    total = sum(v[0] for v in stage.values())
    print(f"\nBY STAGE (section 7 of DRAFTER_WALKTHROUGH.md):")
    print(f"  {'stage':<28} {'ops':>4} {'us/step':>9} {'share':>7}")
    for lab, (us, n) in stage.items():
        print(f"  {lab:<28} {n:>4} {us:>9.1f} {us/total*100:>6.1f}%")
    print(f"  {'TOTAL':<28} {period:>4} {total:>9.1f}")

    # ---- per op code ----
    byop = collections.defaultdict(lambda: [0.0, 0])
    for i in range(period):
        us = statistics.mean(_f(steps[k][i][DUR]) for k in range(reps)) / 1000.0
        d = byop[rep0[i]["OP CODE"].replace("DeviceOperation", "")]
        d[0] += us
        d[1] += 1
    print(f"\nBY OP CODE:")
    print(f"  {'op':<28} {'n':>3} {'us/step':>9} {'share':>7}")
    for op, (us, n) in sorted(byop.items(), key=lambda x: -x[1][0]):
        print(f"  {op:<28} {n:>3} {us:>9.1f} {us/total*100:>6.1f}%")

    # ---- hot list, with skew ----
    print(f"\nHOTTEST {args.top} OPS (skew = PER CORE MAX - MIN, i.e. cross-core spread):")
    print(f"  {'idx':>3} {'op':<26} {'cores':>5} {'us':>8} {'skew us':>8} {'drift':>6}  stage")
    rank = sorted(range(period), key=lambda i: -statistics.mean(_f(steps[k][i][DUR]) for k in range(reps)))
    for i in rank[: args.top]:
        vals = [_f(steps[k][i][DUR]) / 1000.0 for k in range(reps)]
        mn = statistics.mean(_f(steps[k][i][DUR_MIN]) / 1000.0 for k in range(reps))
        us = statistics.mean(vals)
        drift = (max(vals) - min(vals)) / us * 100 if us else 0
        print(
            f"  {i:>3} {rep0[i]['OP CODE'].replace('DeviceOperation',''):<26} "
            f"{rep0[i]['CORE COUNT']:>5} {us:>8.2f} {us-mn:>8.2f} {drift:>5.1f}%  {labels[i]}"
        )

    # ---- matmuls, with program config ----
    mm = [i for i in range(period) if rep0[i]["OP CODE"].startswith("Matmul")]
    print(f"\nMATMULS ({len(mm)}), with resolved program config:")
    print(f"  {'idx':>3} {'cores':>5} {'blk_w':>5} {'pcN':>4} {'us':>8} {'skew':>7}  stage")
    mm_tot = 0.0
    for i in mm:
        us = statistics.mean(_f(steps[k][i][DUR]) for k in range(reps)) / 1000.0
        mn = statistics.mean(_f(steps[k][i][DUR_MIN]) for k in range(reps)) / 1000.0
        mm_tot += us
        m = _MM_CFG.search(rep0[i].get("ATTRIBUTES", "") or "")
        blk = m.group(1) if m else "?"
        pcn = m.group(3) if m else "?"
        print(f"  {i:>3} {rep0[i]['CORE COUNT']:>5} {blk:>5} {pcn:>4} {us:>8.2f} {us-mn:>7.2f}  {labels[i]}")
    print(f"  matmul total: {mm_tot:.1f} us/step = {mm_tot/total*100:.1f}% of the step")


if __name__ == "__main__":
    main()

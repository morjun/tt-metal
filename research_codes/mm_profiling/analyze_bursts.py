#!/usr/bin/env python3
"""SEND_GO_SIGNAL analysis with explicit boundary and identity gates.

Usage: analyze_bursts.py <profile_log_device.csv> <metadata.json> [--expect N]
Metadata must contain verified_replay_ranges ([start, stop) launch indices)
and boundary_evidence. Without these, no replay measurement is emitted.
Per-op output additionally requires verified_launch_names and identity_evidence.
Legacy ops lists alone never establish either boundary or program identity.
"""
import collections
import csv
import json
import statistics
import sys

ZONE = "CQ-DISPATCH-SUBORDINATE:CQ_DISPATCH_CMD_SEND_GO_SIGNAL"


def launches(path):
    fh = open(path)
    mhz = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rd = csv.reader(fh)
    hdr = [h.strip() for h in next(rd)]
    iz, it, ip = hdr.index("zone name"), hdr.index("time[cycles since reset]"), hdr.index("type")
    rows = [r for r in rd if len(r) > ip and r[iz].strip() == ZONE and r[ip].strip() == "ZONE_START"]
    identity = [hdr.index(k) for k in ("PCIe slot", "core_x", "core_y", "RISC processor type")]
    domains = {tuple(r[i].strip() for i in identity) for r in rows}
    if len(domains) > 1:
        raise ValueError(f"multiple dispatcher clock domains: {domains}; split capture explicitly")
    ts = [int(r[it]) for r in rows]
    fh.close()
    return sorted(ts), mhz


def bursts(ts, mhz, n_expect, replay_ranges=None):
    """Only externally verified [start, stop) launch ranges define replays.

    Gap sizes and telescoping sums cannot establish a replay boundary. No
    metadata means no measurement, rather than guessing an offset from TopK.
    """
    if not replay_ranges:
        return [], []
    if n_expect < 1 or mhz <= 0 or any(b <= a for a, b in zip(ts, ts[1:])):
        raise ValueError("invalid launch timeline")
    out = []
    last_stop = 0
    for start, stop in replay_ranges:
        if not (0 <= last_stop <= start < stop <= len(ts)) or stop - start != n_expect:
            raise ValueError("incomplete, overlapping, or wrong-length replay range")
        out.append(ts[start:stop])
        last_stop = stop
    return out, [len(b) for b in out]


def main(csv_path, order_path, expect=None):
    meta = json.load(open(order_path))
    ops = meta["ops"]
    n = expect or len(ops)
    ts, mhz = launches(csv_path)
    # The producer of these ranges must provide its evidence; op order alone
    # does not establish launch identity or replay boundaries.
    ranges = meta.get("verified_replay_ranges")
    if ranges and not meta.get("boundary_evidence"):
        raise ValueError("replay ranges require boundary_evidence")
    good, all_len = bursts(ts, mhz, n, ranges)

    print(f"{csv_path}")
    print(f"  arm={meta['arm']} gather={meta['gather']} k={meta['k']}  ops/step={len(ops)}  chip {mhz:.0f} MHz")
    print(f"  launches={len(ts)}  burst lengths={all_len}  complete(n={n})={len(good)}")
    if not good:
        print("  !! NO COMPLETE BURST -- capture truncated or op count wrong. Nothing is quotable.")
        return None

    per_pos = collections.defaultdict(list)
    spans = []
    for b in good:
        iv = [(y - x) / mhz for x, y in zip(b, b[1:])]
        span = (b[-1] - b[0]) / mhz
        # GATE 1: the intervals must telescope to the span exactly.
        assert abs(sum(iv) - span) < 1e-6, f"interval sum {sum(iv)} != span {span}"
        spans.append(span)
        for i, v in enumerate(iv):
            per_pos[i].append(v)

    med_pos = {i: statistics.median(v) for i, v in per_pos.items()}
    span_med = statistics.median(spans)
    # GATE 2: medians are taken per position, so their sum need not equal the median
    # span; report the discrepancy rather than assuming it away.
    print(f"  span: median {span_med:.1f} us  all {[round(s,1) for s in spans]}")
    print(
        f"  sum(per-position medians) = {sum(med_pos.values()):.1f} us  (delta {sum(med_pos.values())-span_med:+.2f})"
    )

    # interval[i] is the time from launch i to launch i+1, i.e. what op i occupies.
    if not meta.get("verified_launch_names") or not meta.get("identity_evidence"):
        print("  Aggregate only: no verified launch identities; no per-op attribution.")
        return {"span": span_med, "n_bursts": len(good), "by_op": None}
    ops = meta["verified_launch_names"]
    if len(ops) != n:
        raise ValueError("verified launch-name count differs from replay length")
    by_op = collections.defaultdict(list)
    for i, v in med_pos.items():
        by_op[ops[i].replace("ttnn::prim::", "")].append(v)
    print(f"\n  {'op':<44}{'n':>5}{'median us':>11}{'total us':>11}")
    rows = sorted(by_op.items(), key=lambda kv: -sum(kv[1]))
    for name, vs in rows:
        print(f"  {name[:42]:<44}{len(vs):>5}{statistics.median(vs):>11.2f}{sum(vs):>11.1f}")
    return {"span": span_med, "by_op": {k: v for k, v in by_op.items()}, "n_bursts": len(good), "ops": ops}


if __name__ == "__main__":
    a = sys.argv[1:]
    exp = int(a[a.index("--expect") + 1]) if "--expect" in a else None
    main(a[0], a[1], exp)

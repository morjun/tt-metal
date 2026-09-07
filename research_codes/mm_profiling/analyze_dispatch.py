#!/usr/bin/env python3
"""Sum dispatch-core busy time from a profile_log_device.csv produced with
--profile-dispatch-cores.

Answers: how much time do the DISPATCH cores spend, as opposed to the compute
kernels that analyze_ops.py measures?  MEASUREMENT_RECORD.md 6P.10/6.24 record a
kernel-vs-step divergence that no per-op kernel profile can see; this is the
instrument named for it (11.2 item 13).

Cycles -> us uses the CHIP_FREQ[MHz] on the file's first line.
"""
import collections
import csv
import sys

DISPATCH = ("CQ-PREFETCH", "CQ-DISPATCH", "process_cmd_d")


def main(path):
    fh = open(path)
    arch = fh.readline()
    mhz = float(arch.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rd = csv.reader(fh)
    hdr = [h.strip() for h in next(rd)]
    ix = {k: hdr.index(k) for k in ("core_x", "core_y", "RISC processor type", "timer_id", "zone name")}
    it = hdr.index("time[cycles since reset]")
    ph = hdr.index("type")  # ZONE_START / ZONE_END

    open_ = {}
    dur = collections.defaultdict(float)
    cnt = collections.Counter()
    cores = collections.defaultdict(set)
    for r in rd:
        if len(r) <= ph:
            continue
        z = r[ix["zone name"]].strip()
        phase = r[ph].strip()
        key = (r[ix["core_x"]], r[ix["core_y"]], r[ix["RISC processor type"]], z)
        t = int(r[it])
        if phase == "ZONE_START":
            open_[key] = t
        elif phase == "ZONE_END" and key in open_:
            d = (t - open_.pop(key)) / mhz  # cycles / MHz = us
            if d >= 0:
                dur[z] += d
                cnt[z] += 1
                cores[z].add((r[ix["core_x"]], r[ix["core_y"]]))

    print(f"{path}\n  chip {mhz:.0f} MHz\n")
    print(f"  {'zone':<52}{'n':>7}{'total us':>12}{'us/call':>10}{'cores':>7}")
    tot = 0.0
    for z, d in sorted(dur.items(), key=lambda kv: -kv[1]):
        tag = "DISPATCH" if z.startswith(DISPATCH) else ""
        if tag:
            tot += d
        print(f"  {z[:50]:<52}{cnt[z]:>7}{d:>12.1f}{d/cnt[z]:>10.3f}{len(cores[z]):>7}  {tag}")
    print(f"\n  DISPATCH-core total: {tot:.1f} us over the whole run")
    return tot


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv")

#!/usr/bin/env python3
"""TRISC inter-zone gap analysis for SDPA-decode.

For each TRISC core, order the compute zones (QK_MM, SM_NORM, PV_MM, SM_RESCALE) by
start cycle and measure the idle gap between consecutive zones, bucketed by the
(prev_zone -> next_zone) transition. Then test the read-wait hypothesis: for the
dominant gap, how much of it is covered by an NCRISC read zone (RD_CHUNK/RD_K/RD_V)
on the SAME core (same cycle counter).

Usage: analyze_gaps.py <profile_log_device.csv> [label]
"""
import csv, sys, statistics
from collections import defaultdict

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else path
lines = open(path).read().splitlines()
freq_mhz = 1350.0
for tok in lines[0].split(","):
    if "CHIP_FREQ" in tok:
        freq_mhz = float(tok.split(":")[1])
hdr = [c.strip() for c in lines[1].split(",")]
idx = {name: i for i, name in enumerate(hdr)}
CX, CY, RISC, CYC, ZNAME, ZTYPE = (
    idx["core_x"],
    idx["core_y"],
    idx["RISC processor type"],
    idx["time[cycles since reset]"],
    idx["zone name"],
    idx["type"],
)
COMPUTE = ["QK_MM", "SM_NORM", "PV_MM", "SM_RESCALE"]
READ = {"RD_CHUNK", "RD_K", "RD_V", "RD_KBAR"}
WANT = set(COMPUTE) | READ

stacks = defaultdict(list)
zones = []
dropped = False
for ln in lines[2:]:
    if not ln.strip():
        continue
    if "DROPPED" in ln.upper():
        dropped = True
    f = ln.split(",")
    zname = f[ZNAME].strip()
    if zname not in WANT:
        continue
    core = (f[CX].strip(), f[CY].strip())
    risc = f[RISC].strip()
    cyc = int(f[CYC].strip())
    key = (core, risc)
    if f[ZTYPE].strip() == "ZONE_START":
        stacks[key].append((zname, cyc))
    elif f[ZTYPE].strip() == "ZONE_END":
        st = stacks[key]
        for j in range(len(st) - 1, -1, -1):
            if st[j][0] == zname:
                _, s = st.pop(j)
                zones.append({"core": core, "risc": risc, "zone": zname, "start": s, "end": cyc})
                break

c2ns = 1000.0 / freq_mhz
# compute zones per TRISC core
trisc = defaultdict(list)
reads = defaultdict(list)  # core -> [(start,end,zname)] across NCRISC/any read risc
for z in zones:
    if z["zone"] in COMPUTE and z["risc"].startswith("TRISC"):
        trisc[(z["core"], z["risc"])].append(z)
    if z["zone"] in READ:
        reads[z["core"]].append((z["start"], z["end"], z["zone"]))

gaps = defaultdict(list)  # (prev->next) -> [gap_cycles]
gap_read_cov = defaultdict(list)  # transition -> [covered_fraction]
for (core, risc), zs in trisc.items():
    zs.sort(key=lambda z: z["start"])
    rds = sorted(reads.get(core, []))
    for a, b in zip(zs, zs[1:]):
        g = b["start"] - a["end"]
        if g < 0:
            continue
        key = f"{a['zone']}->{b['zone']}"
        gaps[key].append(g)
        # fraction of [a.end, b.start] covered by any read interval on this core
        if g > 0:
            cov = 0
            for rs, re, _ in rds:
                lo, hi = max(a["end"], rs), min(b["start"], re)
                if hi > lo:
                    cov += hi - lo
            gap_read_cov[key].append(min(1.0, cov / g))

print(f"\n=== TRISC inter-zone gaps: {label} (freq {freq_mhz:.0f} MHz){'  [DROPPED!]' if dropped else ''} ===")
print(f"{'transition':<22} {'n':>5} {'mean_ns':>9} {'med_ns':>9} {'total_us':>9} {'read_cov%':>9}")
for key in sorted(gaps, key=lambda k: -sum(gaps[k])):
    g = gaps[key]
    covs = gap_read_cov.get(key, [])
    covpct = 100 * statistics.mean(covs) if covs else 0.0
    print(
        f"{key:<22} {len(g):>5} {statistics.mean(g)*c2ns:>9.1f} {statistics.median(g)*c2ns:>9.1f} "
        f"{sum(g)*c2ns/1000:>9.2f} {covpct:>9.1f}"
    )
# zone durations for context
print("  -- zone durations (TRISC, mean ns) --")
zd = defaultdict(list)
for (core, risc), zs in trisc.items():
    for z in zs:
        zd[z["zone"]].append(z["end"] - z["start"])
for zn in COMPUTE:
    if zd.get(zn):
        print(f"     {zn:<12} n={len(zd[zn]):>4} mean={statistics.mean(zd[zn])*c2ns:>7.1f}ns")

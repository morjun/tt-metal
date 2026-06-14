#!/usr/bin/env python3
"""Intra-op-invocation TRISC inter-zone gap analysis for SDPA-decode.

Scopes gaps to within a single SDPA op invocation using 'run host ID' (rhid), so the
inter-chunk PV_MM(ci)->QK_MM(ci+1) read-wait gap is isolated from the cross-op interval
(rest-of-model time between two SDPA calls). For each (core, risc, rhid) it orders the
compute zones by start and measures gaps, bucketed by (prev->next). Then tests read-wait:
fraction of each gap covered by an NCRISC read zone on the SAME core within the gap window.

Usage: analyze_gaps2.py <profile_log_device.csv> [label]
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
idx = {n: i for i, n in enumerate(hdr)}
CX, CY, RISC, CYC, RHID, ZNAME, ZTYPE = (
    idx["core_x"],
    idx["core_y"],
    idx["RISC processor type"],
    idx["time[cycles since reset]"],
    idx["run host ID"],
    idx["zone name"],
    idx["type"],
)
COMPUTE = ["QK_MM", "SM_NORM", "PV_MM", "SM_RESCALE"]
READ = {"RD_CHUNK", "RD_K", "RD_V", "RD_KBAR"}
WANT = set(COMPUTE) | READ

stacks = defaultdict(list)
zones = []
for ln in lines[2:]:
    if not ln.strip():
        continue
    f = ln.split(",")
    zname = f[ZNAME].strip()
    if zname not in WANT:
        continue
    key = ((f[CX].strip(), f[CY].strip()), f[RISC].strip(), f[RHID].strip())
    cyc = int(f[CYC].strip())
    if f[ZTYPE].strip() == "ZONE_START":
        stacks[key].append((zname, cyc))
    elif f[ZTYPE].strip() == "ZONE_END":
        st = stacks[key]
        for j in range(len(st) - 1, -1, -1):
            if st[j][0] == zname:
                _, s = st.pop(j)
                zones.append({"core": key[0], "risc": key[1], "rhid": key[2], "zone": zname, "start": s, "end": cyc})
                break

c2ns = 1000.0 / freq_mhz
# group compute zones per (core, risc, rhid); reads per (core, rhid)
trisc = defaultdict(list)
reads = defaultdict(list)
for z in zones:
    if z["zone"] in COMPUTE and z["risc"].startswith("TRISC"):
        trisc[(z["core"], z["risc"], z["rhid"])].append(z)
    if z["zone"] in READ:
        reads[(z["core"], z["rhid"])].append((z["start"], z["end"]))

gaps = defaultdict(list)
covs = defaultdict(list)
for (core, risc, rhid), zs in trisc.items():
    zs.sort(key=lambda z: z["start"])
    rds = sorted(reads.get((core, rhid), []))
    for a, b in zip(zs, zs[1:]):
        g = b["start"] - a["end"]
        if g < 0 or g > 10_000_000:  # drop cross-call / counter-wrap outliers (>~7.4ms)
            continue
        k = f"{a['zone']}->{b['zone']}"
        gaps[k].append(g)
        if g > 0:
            c = 0
            for rs, re in rds:
                lo, hi = max(a["end"], rs), min(b["start"], re)
                if hi > lo:
                    c += hi - lo
            covs[k].append(min(1.0, c / g))

print(f"\n=== Intra-op TRISC inter-zone gaps: {label} (freq {freq_mhz:.0f} MHz) ===")
print(f"{'transition':<20} {'n':>6} {'mean_ns':>9} {'med_ns':>9} {'p95_ns':>9} {'total_us':>10} {'read_cov%':>9}")
for k in sorted(gaps, key=lambda x: -sum(gaps[x])):
    g = sorted(gaps[k])
    cv = covs.get(k, [])
    p95 = g[int(0.95 * (len(g) - 1))] if g else 0
    print(
        f"{k:<20} {len(g):>6} {statistics.mean(g)*c2ns:>9.1f} {statistics.median(g)*c2ns:>9.1f} "
        f"{p95*c2ns:>9.1f} {sum(g)*c2ns/1000:>10.2f} {100*statistics.mean(cv) if cv else 0:>9.1f}"
    )

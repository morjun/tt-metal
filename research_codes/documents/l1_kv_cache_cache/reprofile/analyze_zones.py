#!/usr/bin/env python3
"""Analyze SDPA-decode custom device zones from profile_log_device.csv.

Goal A (FPU vs SFPU): aggregate per-(RISC, zone) durations.
  FPU zones  = QK_MM, PV_MM         (matmul)
  SFPU zones = SM_NORM, SM_RESCALE  (row-max/exp/row-sum, online rescale)
  CMP_CHUNK  = per-chunk compute envelope (TRISC), used for the overlap.
Goal B (read/compute overlap): per core, overlap of NCRISC read zones (RD_K, RD_V)
  with the union of TRISC CMP_CHUNK intervals (same core shares the cycle counter).
"""
import csv, sys, statistics
from collections import defaultdict

path = sys.argv[1]
lines = open(path).read().splitlines()
# line 0 = "ARCH: blackhole, CHIP_FREQ[MHz]: 1350"; line 1 = header; rest = data
freq_mhz = 1350.0
for tok in lines[0].split(","):
    if "CHIP_FREQ" in tok:
        freq_mhz = float(tok.split(":")[1])
hdr = [c.strip() for c in lines[1].split(",")]
idx = {name: i for i, name in enumerate(hdr)}
CX, CY, RISC, CYC, RHID, ZNAME, ZTYPE = (
    idx["core_x"],
    idx["core_y"],
    idx["RISC processor type"],
    idx["time[cycles since reset]"],
    idx["run host ID"],
    idx["zone name"],
    idx["type"],
)

CUSTOM = {"CMP_CHUNK", "QK_MM", "SM_NORM", "PV_MM", "SM_RESCALE", "RD_K", "RD_V", "RD_CHUNK", "RD_KBAR", "RD_LAT"}
READ_ZONES = ("RD_CHUNK",)  # whole-chunk read for overlap; RD_K/RD_V/RD_KBAR are nested sub-zones

# (core, risc) -> stack of (zone_name, start_cycle); closed zones collected as dicts
stacks = defaultdict(list)
zones = []  # {core,risc,zone,start,end,dur,rhid}
dropped = False
for ln in lines[2:]:
    if not ln.strip():
        continue
    f = ln.split(",")
    if "DROPPED" in ln.upper():
        dropped = True
    zname = f[ZNAME].strip()
    if zname not in CUSTOM:
        continue
    core = (f[CX].strip(), f[CY].strip())
    risc = f[RISC].strip()
    cyc = int(f[CYC].strip())
    ztype = f[ZTYPE].strip()
    key = (core, risc)
    if ztype == "ZONE_START":
        stacks[key].append((zname, cyc))
    elif ztype == "ZONE_END":
        # pop matching (RAII strictly nested -> top of stack)
        st = stacks[key]
        for j in range(len(st) - 1, -1, -1):
            if st[j][0] == zname:
                _, start = st.pop(j)
                zones.append(
                    {
                        "core": core,
                        "risc": risc,
                        "zone": zname,
                        "start": start,
                        "end": cyc,
                        "dur": cyc - start,
                        "rhid": f[RHID].strip(),
                    }
                )
                break

c2ns = 1000.0 / freq_mhz

# ---------- Goal A: per-(risc, zone) ----------
agg = defaultdict(list)
for z in zones:
    agg[(z["risc"], z["zone"])].append(z["dur"])

print(f"\n=== Goal A: FPU vs SFPU per RISC (freq {freq_mhz:.0f} MHz) ===")
print(f"{'RISC':<8} {'zone':<11} {'n':>5} {'total_us':>10} {'avg_us':>9}")
riscs = sorted({r for (r, _) in agg})
for r in riscs:
    tot_fpu = tot_sfpu = 0.0
    for zn in ("QK_MM", "SM_NORM", "PV_MM", "SM_RESCALE", "CMP_CHUNK"):
        durs = agg.get((r, zn))
        if not durs:
            continue
        tns = sum(durs) * c2ns
        print(f"{r:<8} {zn:<11} {len(durs):>5} {tns/1000:>10.2f} {statistics.mean(durs)*c2ns/1000:>9.3f}")
        if zn in ("QK_MM", "PV_MM"):
            tot_fpu += tns
        elif zn in ("SM_NORM", "SM_RESCALE"):
            tot_sfpu += tns
    if tot_fpu or tot_sfpu:
        tot = tot_fpu + tot_sfpu
        print(
            f"  -> {r}: FPU(matmul) {tot_fpu/1000:.2f}us ({100*tot_fpu/tot:.0f}%) | "
            f"SFPU(softmax) {tot_sfpu/1000:.2f}us ({100*tot_sfpu/tot:.0f}%)"
        )


# ---------- Goal B: read/compute overlap per core ----------
def merge(intervals):
    """merge overlapping intervals into a disjoint union"""
    if not intervals:
        return []
    ivs = sorted(intervals)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def covered_by(union, probe):
    """length of probe intervals covered by the disjoint union (no double counting)"""
    covered = 0
    for ps, pe in probe:
        for s, e in union:
            lo, hi = max(ps, s), min(pe, e)
            if hi > lo:
                covered += hi - lo
    return covered


by_core = defaultdict(lambda: {"rd": [], "cmp": []})
for z in zones:
    if z["zone"] in READ_ZONES:
        by_core[z["core"]]["rd"].append((z["start"], z["end"]))
    elif z["zone"] == "CMP_CHUNK":
        by_core[z["core"]]["cmp"].append((z["start"], z["end"]))

fracs, rd_tot_all, cmp_tot_all = [], 0.0, 0.0
ncores = 0
for core, d in by_core.items():
    rd, cmp = d["rd"], d["cmp"]
    if not rd or not cmp:
        continue
    rd_union = merge(rd)  # NCRISC read busy (disjoint)
    cmp_union = merge(cmp)  # TRISC compute busy across all 3 TRISCs (disjoint)
    rd_total = sum(e - s for s, e in rd_union)
    cmp_total = sum(e - s for s, e in cmp_union)
    cov = covered_by(cmp_union, rd_union)
    frac = cov / rd_total if rd_total else 0.0
    fracs.append(frac)
    rd_tot_all += rd_total * c2ns
    cmp_tot_all += cmp_total * c2ns
    ncores += 1

# read-zone breakdown by source (RD_CHUNK = L1 reader, RD_K/RD_V = DRAM reader)
print("\n=== Read zones (NCRISC) by source ===")
for zn in ("RD_LAT", "RD_KBAR", "RD_CHUNK", "RD_K", "RD_V"):
    durs = agg.get(("NCRISC", zn))
    if durs:
        print(
            f"  {zn:<9}: n={len(durs):4d}  total={sum(durs)*c2ns/1000:.2f}us  "
            f"avg={statistics.mean(durs)*c2ns/1000:.3f}us"
        )
print(
    "  (RD_CHUNK=whole-chunk read incl mask; RD_KBAR=final K barrier=memory-wait; "
    "issue ~= RD_CHUNK - barriers. Compare RD_CHUNK and RD_KBAR across the L1 vs DRAM runs.)"
)

print(f"\n=== Goal B: read/compute overlap ({ncores} cores with both NCRISC read + TRISC compute) ===")
if fracs:
    print(
        f"  read-hidden fraction (RD covered by CMP_CHUNK): mean {statistics.mean(fracs)*100:.1f}%  "
        f"min {min(fracs)*100:.1f}%  max {max(fracs)*100:.1f}%"
    )
    print(
        f"  totals across those cores: read {rd_tot_all/1000:.1f}us  compute {cmp_tot_all/1000:.1f}us  "
        f"margin(compute-read) {(cmp_tot_all-rd_tot_all)/1000:.1f}us"
    )
    print(f"  -> read {'IS' if statistics.mean(fracs) > 0.95 else 'is NOT fully'} hidden behind compute")
else:
    print("  (no cores had both NCRISC read and TRISC compute zones — check zone names / RISC attribution)")

print(f"\nDROPPED_ZONES seen: {dropped}")
print(f"distinct custom zones present: {sorted({z['zone'] for z in zones})}")
print(f"zones on RISCs: {sorted({z['risc'] for z in zones})}")

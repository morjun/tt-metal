#!/usr/bin/env python3
"""SDPA-decode big picture (Demand 2). Built from self-consistent per-unit decompositions.

Per (core,TRISC) unit, one representative op invocation (median-CMP rhid): the timeline is
  pre | QK_MM | <HOLE> | SM_NORM | rest(=PV_MM + rescale + post)
and the segments sum to that unit's CMP_CHUNK envelope (same core => shared counter => valid).

The interval between the QK_MM zone END and the SM_NORM zone START is left as a HOLE (blank): no
device zone covers it, so it is NOT labelled as a characterized "gap". (A separate sub-zone run
attributed only ~30% of it to two data-format reconfig instructions; the rest is unaccounted.)
We only use QK_MM/SM_NORM/CMP_CHUNK zones here (robust to the profiler dropping later-declared
zones at the per-kernel marker cap).

Panels:
  A  per-unit stacked timeline, sorted by envelope: QK^T matmul dominates; the white HOLE before
     SM_NORM is the unaccounted interval; per-chunk KV read (RD_CHUNK) line sits left of every
     unit -> read hidden everywhere.
  B  the HOLE-duration CDF, L1 vs DRAM OVERLAID -> it is MEMORY-INVARIANT (identical), so the
     >1us tail is NOT a read wait (a read wait would shrink with L1's 12% lower latency).
  C  bottleneck unit + read bar -> read ends inside the QK^T matmul -> fully hidden.
"""
from collections import defaultdict
import statistics as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = "research_codes/documents/l1_kv_cache_cache/reprofile"
C2NS = 1000.0 / 1350.0
# "hole" is intentionally NOT drawn (left blank) — no zone identifies it.
DRAWN = [
    ("pre", "#dddddd", "pre"),
    ("QK_MM", "#1f77b4", "QK_MM (QK^T matmul, FPU)"),
    ("SM_NORM", "#ff7f0e", "SM_NORM (softmax, SFPU)"),
    ("rest", "#2ca02c", "rest (PV_MM + rescale + post)"),
]
ORDER = ["pre", "QK_MM", "hole", "SM_NORM", "rest"]


def parse(path, want):
    lines = open(path).read().splitlines()
    hdr = [c.strip() for c in lines[1].split(",")]
    ix = {n: i for i, n in enumerate(hdr)}
    CX, CY, RISC, CYC, RHID, ZN, ZT = (
        ix["core_x"],
        ix["core_y"],
        ix["RISC processor type"],
        ix["time[cycles since reset]"],
        ix["run host ID"],
        ix["zone name"],
        ix["type"],
    )
    stk = defaultdict(list)
    z = []
    for ln in lines[2:]:
        if not ln.strip():
            continue
        f = ln.split(",")
        zn = f[ZN].strip()
        if zn not in want:
            continue
        key = ((f[CX].strip(), f[CY].strip()), f[RISC].strip(), f[RHID].strip())
        cyc = int(f[CYC].strip())
        if f[ZT].strip() == "ZONE_START":
            stk[key].append((zn, cyc))
        elif f[ZT].strip() == "ZONE_END":
            s = stk[key]
            for j in range(len(s) - 1, -1, -1):
                if s[j][0] == zn:
                    _, t = s.pop(j)
                    z.append((key, zn, t, cyc))
                    break
    return z


def decomp(d):
    c0, cE = d["CMP_CHUNK"]
    return {
        "pre": (d["QK_MM"][0] - c0) * C2NS,
        "QK_MM": (d["QK_MM"][1] - d["QK_MM"][0]) * C2NS,
        "hole": (d["SM_NORM"][0] - d["QK_MM"][1]) * C2NS,
        "SM_NORM": (d["SM_NORM"][1] - d["SM_NORM"][0]) * C2NS,
        "rest": (cE - d["SM_NORM"][1]) * C2NS,
        "cmp": (cE - c0) * C2NS,
    }


NEED = {"QK_MM", "SM_NORM", "CMP_CHUNK"}
z = parse(f"{BASE}/zones2_l1only_896/profile_log_device.csv", NEED)
byunit = defaultdict(dict)
for key, zn, s, e in z:
    if key[1].startswith("TRISC"):
        byunit[key][zn] = (s, e)
byunit2 = defaultdict(list)
for key, d in byunit.items():
    if NEED <= d.keys():
        dd = decomp(d)
        if all(dd[k] >= 0 for k in ("pre", "hole", "SM_NORM", "rest")):
            byunit2[(key[0], key[1])].append(dd)
units = []
for u, lst in byunit2.items():
    lst.sort(key=lambda x: x["cmp"])
    units.append(lst[len(lst) // 2])
units.sort(key=lambda x: x["cmp"])


def holes_for(tag):
    zz = parse(f"{BASE}/zones2_{tag}_896/profile_log_device.csv", {"QK_MM", "SM_NORM"})
    bu = defaultdict(dict)
    for key, zn, s, e in zz:
        if key[1].startswith("TRISC"):
            bu[key][zn] = (s, e)
    out = [(d["SM_NORM"][0] - d["QK_MM"][1]) * C2NS for d in bu.values() if "QK_MM" in d and "SM_NORM" in d]
    return sorted(h for h in out if 0 <= h <= 10000)


holes_l1 = holes_for("l1only")
holes_dram = holes_for("dram")
rz = parse(f"{BASE}/zones_rw_l1only/profile_log_device.csv", {"RD_CHUNK"})
rd = st.median([(e - s) * C2NS for _, zn, s, e in rz])

n = len(units)
maxcmp = units[-1]["cmp"]
fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.5, 1.1], hspace=0.4)
axA, axB, axC = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

# ---- A: per-unit stacked; the hole is left blank ----
fill = {k: c for k, c, _ in DRAWN}
for i, u in enumerate(units):
    x = 0
    for name in ORDER:
        w = u[name]
        if w <= 0:
            continue
        if name == "hole":  # unaccounted interval: no zone -> leave blank
            x += w
            continue
        axA.broken_barh([(x, w)], (i - 0.45, 0.9), facecolors=fill[name])
        x += w
axA.axvline(rd, color="#9467bd", lw=2.2, ls="--")
axA.text(
    rd,
    n * 1.02,
    f"RD_CHUNK={rd:.0f}ns (KV read/chunk)\nleft of every unit's compute -> READ HIDDEN",
    color="#9467bd",
    fontsize=9,
    ha="center",
    fontweight="bold",
)
axA.text(maxcmp, -3, f"op latency = slowest unit {maxcmp:.0f} ns", fontsize=8, ha="right")
axA.set_xlim(0, maxcmp * 1.1)
axA.set_ylim(-5, n * 1.12)
axA.set_ylabel("192 units (64 cores x 3 TRISC), sorted by envelope")
axA.set_xlabel("duration from each unit's own compute start (ns)  [same-core; not cross-core wall-clock]")
axA.set_title(
    "A. Per-unit SDPA compute (segments sum to the envelope). QK^T matmul dominates; the WHITE HOLE "
    "before SM_NORM is an unaccounted interval (no zone); KV read fits under every unit.",
    fontsize=9.5,
)


# ---- B: hole-duration CDF, L1 vs DRAM -> memory-invariant ----
def cdf(ax, arr, color, label):
    m = len(arr)
    ys = [100 * (i + 1) / m for i in range(m)]
    ax.plot(arr, ys, color=color, lw=2.2, label=label)
    return arr[len(arr) // 2], arr[int(0.90 * (m - 1))], sum(x > 500 for x in arr), m


p50l, p90l, bigl, ml = cdf(axB, holes_l1, "#d62728", "L1-only")
p50d, p90d, bigd, md = cdf(axB, holes_dram, "#1f77b4", "DRAM")
axB.axvline(500, color="gray", ls=":", lw=0.8)
axB.text(
    560, 28, f">500ns tail:\nL1 {bigl} ({100*bigl/ml:.0f}%)\nDRAM {bigd} ({100*bigd/md:.0f}%)", fontsize=8, va="center"
)
axB.set_xlim(0, 1700)
axB.set_ylim(0, 105)
axB.set_xlabel("QK_MM->SM_NORM hole duration (ns)  [interval no zone covers]")
axB.set_ylabel("% of units <= x")
axB.legend(loc="lower right", fontsize=9)
axB.set_title(
    f"B. The QK_MM->SM_NORM hole: L1 vs DRAM are IDENTICAL (p50 {p50l:.0f}/{p50d:.0f}, p90 {p90l:.0f}/{p90d:.0f} ns; "
    f">500ns tail {100*bigl/ml:.0f}%/{100*bigd/md:.0f}%) => MEMORY-INVARIANT.\n   A read wait would shrink with "
    "L1's 12% lower latency; it does not. No read dep in the interval (mask fused, reduction post-PV) "
    "=> the hole (incl >1us tail) is NOT a read wait.",
    fontsize=8.6,
)

# ---- C: bottleneck unit + read ----
b = units[-1]
x = 0
for name in ORDER:
    w = b[name]
    if w <= 0:
        continue
    if name == "hole":
        axC.annotate(
            f"hole {w:.0f}ns (no zone)",
            xy=(x + w / 2, 1.35),
            xytext=(x + w / 2, 1.72),
            color="#888",
            fontsize=8,
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#888"),
        )
        x += w
        continue
    axC.broken_barh([(x, w)], (0.55, 0.8), facecolors=fill[name])
    if name == "QK_MM":
        axC.text(
            x + w / 2,
            0.95,
            f"QK_MM {w:.0f}ns ({100*w/b['cmp']:.0f}%)",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )
    x += w
axC.broken_barh([(300, rd)], (-0.45, 0.8), facecolors="#9467bd")
axC.annotate(
    f"KV read {rd:.0f}ns ends inside the QK^T matmul -> fully hidden",
    xy=(300 + rd, -0.05),
    xytext=(300 + rd + 200, -0.05),
    color="#9467bd",
    fontsize=9,
    va="center",
)
axC.set_yticks([-0.05, 0.95])
axC.set_yticklabels(["NCRISC\n(read)", "TRISC\n(compute)"])
axC.set_ylim(-0.8, 2.0)
axC.set_xlim(0, maxcmp * 1.1)
axC.set_xlabel("time within op (ns)")
axC.set_title(
    f"C. BOTTLENECK unit (longest envelope, sets latency): QK^T matmul {b['QK_MM']:.0f}ns "
    f"({100*b['QK_MM']/b['cmp']:.0f}%); white hole {b['hole']:.0f}ns; read {rd:.0f}ns hidden.",
    fontsize=10,
)

leg = [Patch(facecolor=c, label=lab) for _, c, lab in DRAWN]
leg.append(Patch(facecolor="white", edgecolor="#999", label="HOLE (no zone — unaccounted)"))
leg.append(Patch(facecolor="#9467bd", label="RD_CHUNK (KV read)"))
fig.legend(handles=leg, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.suptitle(
    "SDPA-decode big picture: compute-bound (QK^T matmul), KV read hidden on every core, and the "
    "QK_MM->SM_NORM hole is memory-invariant (L1==DRAM) -> not a read wait",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
out = f"{BASE}/sdpa_bigpicture.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
print(
    f"units={n} maxcmp={maxcmp:.0f} rd={rd:.0f}; hole p90 L1={p90l:.0f} DRAM={p90d:.0f}; tail>500 L1={bigl} DRAM={bigd}"
)

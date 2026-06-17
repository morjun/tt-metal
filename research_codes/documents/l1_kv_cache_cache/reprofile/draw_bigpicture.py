#!/usr/bin/env python3
"""Per-backend SDPA-decode big-picture charts (DRAM and SRAM/L1) + the L1-vs-DRAM hole CDF.

Data-only figures — short titles/labels only. All explanation lives in COMPUTE_BOUND_PROOF.md
(the "Figures" section). Outputs: dram_bigpicture.png, sram_bigpicture.png, hole_cdf_l1_vs_dram.png.

Per (core,TRISC) unit, one representative invocation (median-CMP rhid); the timeline
  pre | QK_MM | <HOLE> | SM_NORM | rest(=PV_MM+rescale+post)  sums to the CMP_CHUNK envelope.
The QK_MM->SM_NORM interval is left BLANK (a hole: no zone covers it).
"""
from collections import defaultdict
import statistics as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = "research_codes/documents/l1_kv_cache_cache/reprofile"
C2NS = 1000.0 / 1350.0
DRAWN = [
    ("pre", "#dddddd", "pre"),
    ("QK_MM", "#1f77b4", "QK_MM (QK^T matmul)"),
    ("SM_NORM", "#ff7f0e", "SM_NORM (softmax)"),
    ("rest", "#2ca02c", "rest (PV_MM+rescale+post)"),
]
ORDER = ["pre", "QK_MM", "hole", "SM_NORM", "rest"]
FILL = {k: c for k, c, _ in DRAWN}


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
    # CMP_CHUNK (the envelope) gives pre/rest; the profiler drops it on the L1 path, so when it is
    # absent fall back to op-start = QK_MM start, op-end = SM_NORM end (PV/rest tail omitted).
    if "CMP_CHUNK" in d:
        c0, cE = d["CMP_CHUNK"]
    else:
        c0, cE = d["QK_MM"][0], d["SM_NORM"][1]
    return {
        "pre": (d["QK_MM"][0] - c0) * C2NS,
        "QK_MM": (d["QK_MM"][1] - d["QK_MM"][0]) * C2NS,
        "hole": (d["SM_NORM"][0] - d["QK_MM"][1]) * C2NS,
        "SM_NORM": (d["SM_NORM"][1] - d["SM_NORM"][0]) * C2NS,
        "rest": (cE - d["SM_NORM"][1]) * C2NS,
        "cmp": (cE - c0) * C2NS,
    }


NEED = {"QK_MM", "SM_NORM"}  # CMP_CHUNK used if present (DRAM); optional (L1 path drops it)


def load_units(tag):
    z = parse(f"{BASE}/zones2_{tag}_896/profile_log_device.csv", NEED | {"CMP_CHUNK"})
    byunit = defaultdict(dict)
    for key, zn, s, e in z:
        if key[1].startswith("TRISC"):
            byunit[key][zn] = (s, e)
    per = defaultdict(list)
    for key, d in byunit.items():
        if NEED <= d.keys():
            dd = decomp(d)
            if all(dd[k] >= 0 for k in ("pre", "hole", "SM_NORM", "rest")):
                per[(key[0], key[1])].append(dd)
    units = [sorted(lst, key=lambda x: x["cmp"])[len(lst) // 2] for lst in per.values()]
    units.sort(key=lambda x: x["cmp"])
    return units


def rd_median(readtag):
    z = parse(f"{BASE}/zones_rw_{readtag}/profile_log_device.csv", {"RD_CHUNK"})
    d = [(e - s) * C2NS for _, zn, s, e in z]
    return st.median(d) if d else 0.0


def holes(tag):
    z = parse(f"{BASE}/zones2_{tag}_896/profile_log_device.csv", {"QK_MM", "SM_NORM"})
    bu = defaultdict(dict)
    for key, zn, s, e in z:
        if key[1].startswith("TRISC"):
            bu[key][zn] = (s, e)
    out = [(d["SM_NORM"][0] - d["QK_MM"][1]) * C2NS for d in bu.values() if "QK_MM" in d and "SM_NORM" in d]
    return sorted(h for h in out if 0 <= h <= 10000)


def draw_backend(tag, readtag, label, out):
    units = load_units(tag)
    rd = rd_median(readtag)
    n = len(units)
    maxcmp = units[-1]["cmp"]
    fig, (axA, axC) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1.1]})
    # Panel A: per-unit stacked; hole blank
    for i, u in enumerate(units):
        x = 0
        for name in ORDER:
            w = u[name]
            if w <= 0:
                continue
            if name == "hole":
                x += w
                continue
            axA.broken_barh([(x, w)], (i - 0.45, 0.9), facecolors=FILL[name])
            x += w
    axA.axvline(rd, color="#9467bd", lw=2.0, ls="--")
    axA.set_xlim(0, maxcmp * 1.08)
    axA.set_ylim(-2, n + 2)
    axA.set_ylabel("units (64 cores x 3 TRISC), sorted by envelope")
    axA.set_xlabel("time from unit's own compute start (ns)")
    axA.set_title(f"A. Per-unit SDPA compute — {label}", fontsize=11)
    # Panel C: bottleneck unit + read bar
    b = units[-1]
    x = 0
    for name in ORDER:
        w = b[name]
        if w <= 0:
            continue
        if name == "hole":
            x += w
            continue
        axC.broken_barh([(x, w)], (0.55, 0.8), facecolors=FILL[name])
        x += w
    axC.broken_barh([(300, rd)], (-0.45, 0.8), facecolors="#9467bd")
    axC.set_yticks([-0.05, 0.95])
    axC.set_yticklabels(["NCRISC\nread", "TRISC\ncompute"])
    axC.set_ylim(-0.8, 1.6)
    axC.set_xlim(0, maxcmp * 1.08)
    axC.set_xlabel("time within op (ns)")
    axC.set_title(f"C. Bottleneck unit — {label}", fontsize=11)
    leg = [Patch(facecolor=c, label=lab) for _, c, lab in DRAWN]
    leg.append(Patch(facecolor="white", edgecolor="#999", label="hole (no zone)"))
    leg.append(Patch(facecolor="#9467bd", label="RD_CHUNK (KV read)"))
    fig.legend(handles=leg, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out, f"(units={n}, maxcmp={maxcmp:.0f}, rd={rd:.0f})")


def draw_hole_cdf(out):
    hl, hd = holes("l1only"), holes("dram")
    fig, ax = plt.subplots(figsize=(8, 4))
    for arr, color, lab in [(hl, "#d62728", "SRAM (L1)"), (hd, "#1f77b4", "DRAM")]:
        m = len(arr)
        ax.plot(arr, [100 * (i + 1) / m for i in range(m)], color=color, lw=2.2, label=lab)
    ax.axvline(500, color="gray", ls=":", lw=0.8)
    ax.set_xlim(0, 1700)
    ax.set_ylim(0, 105)
    ax.set_xlabel("QK_MM->SM_NORM hole duration (ns)")
    ax.set_ylabel("% of units <= x")
    ax.legend(loc="lower right")
    ax.set_title("QK_MM->SM_NORM hole CDF: SRAM vs DRAM", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    p90l = hl[int(0.9 * (len(hl) - 1))]
    p90d = hd[int(0.9 * (len(hd) - 1))]
    print("wrote", out, f"(p90 SRAM={p90l:.0f} DRAM={p90d:.0f})")


draw_backend("dram", "dram", "DRAM", f"{BASE}/dram_bigpicture.png")
draw_backend("l1only", "l1only", "SRAM (L1)", f"{BASE}/sram_bigpicture.png")
draw_hole_cdf(f"{BASE}/hole_cdf_l1_vs_dram.png")

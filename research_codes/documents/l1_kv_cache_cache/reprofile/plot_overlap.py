#!/usr/bin/env python3
"""Focused read/compute Gantt for ONE SDPA-decode invocation on ONE attention core.
Plots NCRISC read zones (RD_K/RD_V) against TRISC compute zones (QK_MM/SM_*/PV_MM, and the
CMP_CHUNK envelope) on a shared device-cycle axis, so read-hiding is visible directly.
Usage: plot_overlap.py <profile_log_device.csv> [out.png]
"""
import csv, sys
from collections import defaultdict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "overlap_gantt.png"
lines = open(path).read().splitlines()
freq_mhz = 1350.0
for tok in lines[0].split(","):
    if "CHIP_FREQ" in tok:
        freq_mhz = float(tok.split(":")[1])
hdr = [c.strip() for c in lines[1].split(",")]
idx = {n: i for i, n in enumerate(hdr)}
CX, CY, RISC, CYC, RHID, ZN, ZT = (
    idx["core_x"],
    idx["core_y"],
    idx["RISC processor type"],
    idx["time[cycles since reset]"],
    idx["run host ID"],
    idx["zone name"],
    idx["type"],
)
CUSTOM = {"CMP_CHUNK", "QK_MM", "SM_NORM", "PV_MM", "SM_RESCALE", "RD_K", "RD_V", "RD_CHUNK"}
READ_ZONES = ("RD_K", "RD_V", "RD_CHUNK")

stacks = defaultdict(list)
zones = []
for ln in lines[2:]:
    if not ln.strip():
        continue
    f = ln.split(",")
    zn = f[ZN].strip()
    if zn not in CUSTOM:
        continue
    key = ((f[CX].strip(), f[CY].strip()), f[RISC].strip(), f[RHID].strip())
    cyc = int(f[CYC].strip())
    if f[ZT].strip() == "ZONE_START":
        stacks[key].append((zn, cyc))
    elif f[ZT].strip() == "ZONE_END":
        st = stacks[key]
        for j in range(len(st) - 1, -1, -1):
            if st[j][0] == zn:
                _, s = st.pop(j)
                zones.append({"core": key[0], "risc": key[1], "rhid": key[2], "zone": zn, "s": s, "e": cyc})
                break

# pick the (core, rhid) with the most zones AND has both read + compute
group = defaultdict(list)
for z in zones:
    group[(z["core"], z["rhid"])].append(z)


def score(items):
    has_l1_rd = any(i["zone"] == "RD_CHUNK" for i in items)
    has_rd = any(i["zone"] in READ_ZONES for i in items)
    has_cmp = any(i["zone"] == "CMP_CHUNK" for i in items)
    # prefer an invocation that has the L1 reader (RD_CHUNK) so l1_only renders L1 reads
    return (has_l1_rd, has_rd and has_cmp, len(items))


best = max(group, key=lambda k: score(group[k]))
items = group[best]
core, rhid = best
t0 = min(i["s"] for i in items)
c2ns = 1000.0 / freq_mhz

# lanes: NCRISC read on top, then each TRISC compute
lane_order = [("NCRISC", "read"), ("TRISC_0", "cmp"), ("TRISC_1", "cmp"), ("TRISC_2", "cmp")]
zone_color = {
    "RD_K": "#1f77b4",
    "RD_V": "#4a90d9",
    "RD_CHUNK": "#1f77b4",
    "QK_MM": "#d62728",
    "PV_MM": "#ff7f0e",  # FPU = red/orange
    "SM_NORM": "#2ca02c",
    "SM_RESCALE": "#17becf",  # SFPU = green/cyan
    "CMP_CHUNK": "#cccccc",
}
# both readers now emit RD_CHUNK, so infer source from arg or path (not zone name)
read_src = sys.argv[3] if len(sys.argv) > 3 else ("L1" if "l1only" in path else "DRAM")
has_rdchunk = any(z["zone"] == "RD_CHUNK" for z in items)
lanes = []
for risc, kind in lane_order:
    if any(i["risc"] == risc for i in items):
        lanes.append((risc, kind))

fig, ax = plt.subplots(figsize=(13, 4.2))
ylabels = []
for li, (risc, kind) in enumerate(lanes):
    ylabels.append(risc + ("  (read)" if kind == "read" else "  (compute)"))
    for z in items:
        if z["risc"] != risc:
            continue
        if kind == "cmp" and z["zone"] == "CMP_CHUNK":
            ax.broken_barh(
                [((z["s"] - t0) * c2ns, (z["e"] - z["s"]) * c2ns)],
                (li - 0.42, 0.84),
                facecolors="none",
                edgecolors="#999999",
                linewidths=0.6,
                zorder=1,
            )
            continue
        if kind == "read" and z["zone"] not in READ_ZONES:
            continue
        if kind == "read" and has_rdchunk and z["zone"] != "RD_CHUNK":
            continue  # draw only the whole-chunk read band, not nested RD_K/RD_V
        if kind == "cmp" and z["zone"] == "CMP_CHUNK":
            continue
        ax.broken_barh(
            [((z["s"] - t0) * c2ns, (z["e"] - z["s"]) * c2ns)],
            (li - 0.30, 0.60),
            facecolors=zone_color.get(z["zone"], "#888"),
            edgecolors="black",
            linewidths=0.3,
            zorder=2,
        )
ax.set_yticks(range(len(lanes)))
ax.set_yticklabels(ylabels)
ax.invert_yaxis()
ax.set_xlabel("time within SDPA-decode op (ns)  [device cycles @ %.0f MHz]" % freq_mhz)
ax.set_title(
    f"SDPA-decode read/compute overlap — {read_src} read — core {core}, op rhid {rhid}\n"
    f"NCRISC {read_src} read sits inside TRISC compute (read hidden)"
)
legend = [
    Patch(facecolor="#1f77b4", label=f"K/V read ({read_src})"),
    Patch(facecolor="#d62728", label="QK_MM (FPU)"),
    Patch(facecolor="#ff7f0e", label="PV_MM (FPU)"),
    Patch(facecolor="#2ca02c", label="SM_NORM (SFPU)"),
    Patch(facecolor="#17becf", label="SM_RESCALE (SFPU)"),
    Patch(facecolor="none", edgecolor="#999999", label="CMP_CHUNK envelope"),
]
ax.legend(handles=legend, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"core={core} rhid={rhid} zones={len(items)} -> {out}")

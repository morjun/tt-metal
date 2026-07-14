#!/usr/bin/env python3
"""Plot raw-access READ latency and bandwidth vs transfer size, for DRAM / local SRAM / remote SRAM.
Input: rawaccess/rawaccess.csv (from run_rawaccess_sweep.sh). Outputs two PNGs."""
import csv
from collections import defaultdict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "research_codes/documents/l1_kv_cache_cache/reprofile"
CSV = f"{BASE}/rawaccess/rawaccess.csv"
SERIES = [("DRAM", "#1f77b4", "o"), ("local_SRAM", "#2ca02c", "s"), ("remote_SRAM", "#d62728", "^")]
LABEL = {"DRAM": "DRAM", "local_SRAM": "local SRAM (own L1)", "remote_SRAM": "remote SRAM (NoC)"}

lat = defaultdict(list)
bw = defaultdict(list)
for r in csv.DictReader(open(CSV)):
    s = int(r["size_bytes"])
    if r.get("latency_ns"):
        lat[r["source"]].append((s, float(r["latency_ns"])))
    if r.get("bw_gbs"):
        bw[r["source"]].append((s, float(r["bw_gbs"])))


def fmt_size(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 1024, 16384, 262144, 1048576])
    ax.set_xticklabels(["64B", "1KB", "16KB", "256KB", "1MB"])
    ax.grid(True, which="both", alpha=0.3)


# latency
fig, ax = plt.subplots(figsize=(8, 5))
for src, color, mk in SERIES:
    d = sorted(lat.get(src, []))
    if d:
        ax.plot([x for x, _ in d], [y for _, y in d], color=color, marker=mk, label=LABEL[src])
fmt_size(ax)
ax.set_yscale("log")
ax.set_xlabel("transfer size (per read)")
ax.set_ylabel("latency (ns, log)")
ax.set_title("Raw-access READ latency vs transfer size\n(small-size floor: local SRAM < remote SRAM < DRAM)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{BASE}/rawaccess_latency.png", dpi=130, bbox_inches="tight")
print("wrote rawaccess_latency.png")

# bandwidth
fig, ax = plt.subplots(figsize=(8, 5))
for src, color, mk in SERIES:
    d = sorted(bw.get(src, []))
    if d:
        ax.plot([x for x, _ in d], [y for _, y in d], color=color, marker=mk, label=LABEL[src])
fmt_size(ax)
ax.set_xlabel("transfer size (total)")
ax.set_ylabel("bandwidth (GB/s)")
ax.set_title("Raw-access READ bandwidth vs transfer size")
ax.legend()
fig.tight_layout()
fig.savefig(f"{BASE}/rawaccess_bandwidth.png", dpi=130, bbox_inches="tight")
print("wrote rawaccess_bandwidth.png")

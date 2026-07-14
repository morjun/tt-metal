#!/usr/bin/env python3
"""(a) Bandwidth-saturation crossover analysis: SDPA DRAM read DEMAND vs the DRAM aggregate CEILING.

DEMAND(ctx) = KV_bytes_per_token * ctx / SDPA_op_time(ctx), with the MEASURED SDPA-op compute model
op_time ~= 3.9 + 0.0116*ctx us (512/1024/1792-token points). As ctx grows the demand asymptotes at
KV_bytes_per_token / 0.0116 (GB/s). KV per token (Llama-8B, 8 kv-heads x 128 x {K,V}):
  bf16  -> 8*128*2*2 = 4096 B/tok -> asymptote 353 GB/s
  bfp8  -> ~8*128*2*1.0625 = 2176 B/tok -> asymptote ~188 GB/s
DRAM aggregate ceiling ~= 512 GB/s (P150 GDDR6 spec; single-reader measured ~63 GB/s).
Batch/context distribute across cores preserving the compute:read ratio, so batch does NOT raise the
asymptote. Demand stays below the ceiling at every ctx/batch => no single-chip crossover.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLOPE = 0.0116  # us/token (measured SDPA-op latency slope)
INTERCEPT = 3.9  # us
CEIL = 512.0  # GB/s DRAM aggregate (GDDR6 spec)
ctx = list(range(64, 32001, 64))


def demand(kv_bytes):
    return [kv_bytes * n / (INTERCEPT + SLOPE * n) for n in ctx]  # B/us == GB/s


fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(ctx, demand(4096), color="#1f77b4", lw=2, label="SDPA DRAM read demand (KV bf16, asym 353 GB/s)")
ax.plot(ctx, demand(2176), color="#2ca02c", lw=2, ls="-", label="SDPA DRAM read demand (KV bfp8, asym 188 GB/s)")
ax.axhline(CEIL, color="#d62728", lw=2, ls="--", label="DRAM aggregate ceiling ~512 GB/s (GDDR6 spec)")
ax.axhline(353, color="#1f77b4", lw=0.8, ls=":")
ax.text(32000, 360, "353", color="#1f77b4", fontsize=8, va="bottom", ha="right")
ax.fill_between(ctx, demand(4096), CEIL, color="#d62728", alpha=0.05)
ax.text(
    16000,
    440,
    "DRAM headroom (~1.45x at the bf16 asymptote)\n=> read stays hidden => no crossover",
    fontsize=9,
    ha="center",
    color="#555",
)
ax.set_xlabel("context length (tokens)")
ax.set_ylabel("read bandwidth (GB/s)")
ax.set_ylim(0, 600)
ax.set_title("(a) Single-chip bandwidth-saturation crossover: SDPA read demand never reaches the DRAM ceiling")
ax.legend(loc="center right", fontsize=8)
ax.grid(alpha=0.3)
out = "research_codes/documents/l1_kv_cache_cache/reprofile/crossover_demand_vs_ceiling.png"
fig.tight_layout()
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
print(f"bf16 asymptote = {4096/SLOPE/1000:.0f} GB/s; bfp8 asymptote = {2176/SLOPE/1000:.0f} GB/s; ceiling {CEIL:.0f}")

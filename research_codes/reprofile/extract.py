#!/usr/bin/env python3
"""Extract per-op DEVICE KERNEL DURATION PER CORE AVG/MAX for SDPA-decode and Matmul.
Aggregated 'DEVICE KERNEL DURATION' column is unreliable (timestamp overflow), so we
use the PER CORE AVG/MAX columns and aggregate across op invocations."""
import csv, sys, statistics

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
if not rows:
    print("  (empty CSV)")
    sys.exit(0)


def col(r, name):
    v = r.get(name, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


targets = {"ScaledDotProductAttentionDecode": "SDPA-decode", "Matmul": "Matmul"}
buckets = {}
for r in rows:
    code = (r.get("OP CODE") or "").strip()
    key = next((lbl for t, lbl in targets.items() if t in code), None)
    if key is None:
        continue
    avg = col(r, "DEVICE KERNEL DURATION PER CORE AVG [ns]")
    mx = col(r, "DEVICE KERNEL DURATION PER CORE MAX [ns]")
    if avg is None:
        continue
    buckets.setdefault(key, []).append((avg, mx))

for key in ("SDPA-decode", "Matmul"):
    b = buckets.get(key)
    if not b:
        print(f"  {key:14s}: (none)")
        continue
    avgs = [a for a, _ in b]
    maxs = [m for _, m in b if m is not None]
    print(
        f"  {key:14s}: n={len(b):3d}  avg(per-core avg)={statistics.mean(avgs)/1000:.2f}us  "
        f"max(per-core avg)={max(avgs)/1000:.2f}us  max(per-core max)={(max(maxs)/1000 if maxs else 0):.2f}us"
    )

#!/usr/bin/env python3
"""Exact byte-granularity overlap audit; no elapsed time inferred."""
import argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("snapshot", type=Path)
a = p.parse_args()
results = []
for r in map(json.loads, a.snapshot.read_text().splitlines()):
    idx = r["idx"]
    rs = r["regions"]
    size = r["size"]
    cap = r["capacity"]

    def overlaps(start):
        return [x for x in rs if start < x["offset"] + x["size"] and x["offset"] < start + size]

    def best_without(excluded):
        best = idx
        count = 0
        example = None
        for start in range(cap - size + 1):
            hit = [x for x in overlaps(start) if (x["program_id"], x["type"]) not in excluded]
            if any(x["last_use"] == idx for x in hit):
                continue
            pred = max([x["last_use"] for x in hit], default=-1)
            if pred < best:
                best = pred
                example = start
                count = 1
            elif pred == best:
                count += 1
        return dict(earliest_possible_predecessor=best, lag=idx - best, example_offset=example, byte_positions=count)

    base = best_without(set())
    # Removing occupancy is a counterfactual only: no safe relocation is asserted.
    counter = {str(t): best_without({(47, t)}) for t in [0, 1]}
    results.append(
        dict(
            idx=idx,
            capacity=cap,
            request=size,
            live_regions=rs,
            all_byte_candidates=base,
            remove_sdpa_region_counterfactual=counter,
        )
    )
a.snapshot.with_suffix(".analysis.json").write_text(json.dumps(results, indent=2) + "\n")
for r in results:
    print(json.dumps({k: v for k, v in r.items() if k != "live_regions"}, indent=2))

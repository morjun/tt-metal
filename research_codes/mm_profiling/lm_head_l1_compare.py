#!/usr/bin/env python3
"""Compare resident L1 allocations at setup, prefill and trace boundaries."""
import argparse
import json
from pathlib import Path


def load(root, stage):
    path = root / f"{stage}.l1.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def entries(snapshot):
    if snapshot is None:
        return {}
    meta = snapshot.get("buffers", {})
    owners = snapshot.get("owners", {})
    base = snapshot["base"]
    return {
        base
        + int(b["address"]): {
            "address": base + int(b["address"]),
            "size_per_bank": int(b["size"]),
            "layout": meta.get(str(base + int(b["address"])), {}).get("layout", "unknown"),
            "owners": owners.get(str(base + int(b["address"])), []),
            "physical_cores": len(snapshot.get("physical_pages", {}).get(str(base + int(b["address"])), {})),
        }
        for b in snapshot["blocks"]
        if b.get("allocated") == "yes"
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture_dir", type=Path)
    args = p.parse_args()
    stages = [
        "target_loaded",
        "prefill_warmup",
        "prefill_done",
        "assistant_loaded",
        "before_iteration",
        "after_iteration",
    ]
    prev = None
    report = []
    for stage in stages:
        shot = load(args.capture_dir, stage)
        if shot is None:
            continue
        now = entries(shot)
        added = [b for a, b in now.items() if a not in (prev or {})]
        removed = [b for a, b in (prev or {}).items() if a not in now]
        report.append(
            {
                "stage": stage,
                "allocated_per_bank": shot["allocated_bytes_per_bank"],
                "lowest": min(now) if now else None,
                "blocks": len(now),
                "added": added,
                "removed": removed,
                "size_65536": [b for b in now.values() if b["size_per_bank"] == 65536],
            }
        )
        prev = now
    out = args.capture_dir / "analysis"
    out.mkdir(exist_ok=True)
    (out / "lifetime.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in report:
        print(
            row["stage"],
            "allocated/bank",
            row["allocated_per_bank"],
            "added",
            [(b["address"], b["size_per_bank"], b["owners"][:2]) for b in row["added"]],
            "removed",
            [(b["address"], b["size_per_bank"]) for b in row["removed"]],
        )


if __name__ == "__main__":
    main()

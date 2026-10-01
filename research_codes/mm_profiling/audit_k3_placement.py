#!/usr/bin/env python3
"""Audit K=3 placement captures; no timing is inferred from command counts."""
import argparse
import hashlib
import json
from pathlib import Path
from configwait_map import load_cell, dep_source

p = argparse.ArgumentParser()
p.add_argument("capture_root", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
report = {"scope": "K=3; pinned trace 1; base versus terminal bottom-up", "sources": {}, "arms": {}}
for arm, name, n in [(0, "multicast", 170), (1, "ring", 178)]:
    cells = {}
    for cond in ["base", "iv"]:
        d = a.capture_root / f"k3_a{arm}_{cond}"
        cells[cond] = load_cell(d)
        for f in ["alloc.jsonl", "trace0.jsonl", "trace1.jsonl", "stream0.bin", "stream1.bin"]:
            path = d / f
            report["sources"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert (a.capture_root / f"k3_a{arm}_base/stream0.bin").read_bytes() == (
        a.capture_root / f"k3_a{arm}_iv/stream0.bin"
    ).read_bytes()
    b, v = [cells[c][1]["programs"] for c in ["base", "iv"]]
    assert len(b) == len(v) == 3 * n
    for x, y in zip(b, v):
        assert (x["idx"], x["program_id"], x["runtime_id"], x["num_workers"]) == (
            y["idx"],
            y["program_id"],
            y["runtime_id"],
            y["num_workers"],
        )

    def wait(x):
        return dict(
            predecessor=x["combined"],
            target=x["wait_target"],
            stall_first=x["stall_first"],
            stall_before_program=x["stall_before_program"],
            source=dep_source(x),
        )

    def summary(ps):
        return dict(
            resets=[
                dict(idx=x["idx"], step=x["idx"] // n, program_id=x["program_id"], wait=wait(x))
                for x in ps
                if x["alloc_reset_events"]
            ],
            send_count=sum(x["send_binary"] for x in ps),
            scheduled_binary_bytes=sum(x["send_binary"] * x["kernel_bins_sizeB"] for x in ps),
            command_bytes=sum(x["bytes"] for x in ps),
        )

    changes = []
    for x, y in zip(b, v):
        if any(x[k] != y[k] for k in ["combined", "stall_first", "stall_before_program"]):
            changes.append(dict(idx=x["idx"], program_id=x["program_id"], base=wait(x), iv=wait(y)))
    removed = []
    for x, y in zip(b, v):
        if x["combined"] is not None and y["combined"] is None:
            later = [z for z in v[x["idx"] + 1 :] if z["combined"] is not None and z["combined"] >= x["combined"]]
            removed.append(
                dict(
                    idx=x["idx"],
                    program_id=x["program_id"],
                    base=wait(x),
                    iv_next_covering_wait=None if not later else dict(idx=later[0]["idx"], **wait(later[0])),
                )
            )
    report["arms"][name] = dict(base=summary(b), iv=summary(v), wait_changes=changes, removed_waits=removed)
(a.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
for name, r in report["arms"].items():
    print(name, json.dumps({k: r[k] for k in ["base", "iv", "removed_waits"]}, indent=2))

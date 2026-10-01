#!/usr/bin/env python3
"""Locate the exact live L1 blocks at the fused iteration's highest CB frontier."""
import argparse
import json
from pathlib import Path

from research_codes.mm_profiling.lm_head_l1_analyze import cores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture_dir", type=Path)
    a = p.parse_args()
    root = a.capture_dir
    graph = json.loads((root / "iteration.graph.json").read_text())
    before = json.loads((root / "before_iteration.l1.json").read_text())
    base = before["base"]
    live = {}
    for b in before["blocks"]:
        if b["allocated"] == "yes":
            addr = base + int(b["address"])
            live[addr] = {
                "address": addr,
                "size": int(b["size"]),
                "layout": before.get("buffers", {}).get(str(addr), {}).get("layout"),
                "owner": before.get("owners", {}).get(str(addr), []),
                "physical": before.get("physical_pages", {}).get(str(addr), {}),
                "allocated_at": "before_iteration",
            }
    active_allocations = {}
    stack = []
    events = []
    for i, node in enumerate(graph):
        kind, params = node["node_type"], node.get("params", {})
        if kind == "function_start":
            stack.append(params.get("name", ""))
        elif kind == "function_end":
            if stack:
                stack.pop()
        elif kind == "buffer_allocate" and params.get("type") == "L1":
            addr = params["address"]
            physical = json.loads(params.get("physical_ranges") or "{}")
            owner = stack[0] if stack else "<unknown>"
            b = {
                "address": addr,
                "size": params["max_size_per_bank"],
                "layout": params["layout"],
                "num_cores": params["num_cores"],
                "owner": owner,
                "physical": physical,
                "allocated_at": i,
            }
            live[addr] = b
            active_allocations[addr] = b
        elif kind == "buffer_deallocate" and params.get("type") == "L1":
            b = live.pop(params["address"], None)
            if b is not None:
                b["deallocated_at"] = i
            active_allocations.pop(params["address"], None)
        elif kind == "circular_buffer_allocate" and params.get("globally_allocated") == "0":
            end = params["address"] + params["size"]
            owner = stack[0] if stack else "<unknown>"
            events.append(
                {
                    "graph_index": i,
                    "program_owner": owner,
                    "cb_end": end,
                    "cb_cores": params.get("core_range_set"),
                    "live": list(live.values()),
                    "lowest_tensor_address": min(live, default=None),
                }
            )
    if not events:
        raise SystemExit("capture has no CB events")
    max_end = max(e["cb_end"] for e in events)
    worst = [e for e in events if e["cb_end"] == max_end]
    unique = {}
    for e in worst:
        key = (e["program_owner"], tuple(sorted((b["address"], b["size"]) for b in e["live"])))
        unique.setdefault(key, e)
    top = base + before["total_bytes_per_bank"]
    result = {
        "base": base,
        "top": top,
        "cb_ceiling": max_end,
        "head_only_space_above_cb": top - max_end,
        "ceiling_states": list(unique.values()),
        "largest_live_tensor_charge_at_ceiling": max((sum(b["size"] for b in e["live"]) for e in worst), default=0),
    }
    for e in result["ceiling_states"]:
        cb_cores = cores(e["cb_cores"])
        e["physical_gap_by_core"] = {}
        for x, y in sorted(cb_cores):
            intervals = [span for b in e["live"] for span in b["physical"].get(f"{x},{y}", [])]
            lowest = min((span[0] for span in intervals), default=top)
            e["physical_gap_by_core"][f"{x},{y}"] = lowest - e["cb_end"]
        e["min_physical_gap"] = min(e["physical_gap_by_core"].values(), default=None)
        e["cb_core_count"] = len(cb_cores)
        e["nonuniform_tensor_count"] = sum(
            bool(b["physical"]) and len(b["physical"]) < before["num_banks"] for b in e["live"]
        )
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    (out / "pressure.json").write_text(json.dumps(result, indent=2) + "\n")
    for e in unique.values():
        print(
            "CB",
            e["program_owner"],
            "end",
            e["cb_end"],
            "graph",
            e["graph_index"],
            "lowest tensor",
            e["lowest_tensor_address"],
        )
        for b in sorted(e["live"], key=lambda b: b["address"]):
            print(
                " ",
                b["address"],
                b["size"],
                b["layout"],
                b["owner"],
                "physical cores",
                len(b["physical"]),
                "allocated at",
                b["allocated_at"],
                "deallocated at",
                b.get("deallocated_at"),
            )


if __name__ == "__main__":
    main()

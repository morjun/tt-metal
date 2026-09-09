#!/usr/bin/env python3
"""Inventory capture evidence without calling host graph nodes device launches."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(graph):
    functions = []
    allocations = []
    for node in graph:
        params = node.get("params", {})
        kind = node.get("node_type", "")
        if kind == "function_start":
            functions.append(
                {
                    "counter": node.get("counter"),
                    "name": params.get("name"),
                    "arguments": node.get("arguments", []),
                    "input_tensors": node.get("input_tensors", []),
                    "stacking_level": node.get("stacking_level"),
                }
            )
        if "allocate" in kind or "buffer" in kind:
            allocations.append(node)
    return {
        "node_types": dict(Counter(n.get("node_type") for n in graph)),
        "function_names_not_launch_counts": dict(Counter(n["name"] for n in functions)),
        "functions": functions,
        "allocation_events": allocations,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("graph", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(summarize(json.loads(args.graph.read_text())), indent=2))

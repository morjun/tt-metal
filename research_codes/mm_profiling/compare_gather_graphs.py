#!/usr/bin/env python3
"""Compare recorded graph arguments and tensor addresses, not dispatch payloads."""
import argparse
import hashlib
import json
from pathlib import Path


def device_ops(path):
    graph = json.loads(path.read_text())
    nodes = {n["counter"]: n for n in graph}
    rows = []
    for node in graph:
        name = (node.get("params") or {}).get("name", "")
        if node["node_type"] != "function_start" or not name.endswith("DeviceOperation"):
            continue
        inputs = []
        for ref in node["input_tensors"]:
            params = nodes[ref]["params"]
            devices = params.get("device_tensors")
            inputs.append(
                dict(
                    shape=params.get("shape"),
                    memory_config=params.get("memory_config"),
                    devices=json.loads(devices) if devices else None,
                    address=params.get("address"),
                )
            )
        rows.append(dict(name=name, arguments=node["arguments"], inputs=inputs))
    return rows


def compare(left, right):
    # Index matching is allowed only for identical complete name prefixes.
    assert len(left) <= len(right)
    assert [x["name"] for x in left] == [x["name"] for x in right[: len(left)]], "Different operation sequences"
    differences = []
    for i, (a, b) in enumerate(zip(left, right)):
        fields = [key for key in ("arguments", "inputs") if a[key] != b[key]]
        if fields:
            differences.append(
                dict(
                    op_index=i,
                    name=a["name"],
                    fields=fields,
                    left={f: a[f] for f in fields},
                    right={f: b[f] for f in fields},
                )
            )
    return dict(
        compared_entries=len(left),
        argument_differences=sum("arguments" in d["fields"] for d in differences),
        input_metadata_differences=sum("inputs" in d["fields"] for d in differences),
        differences=differences,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["graph", "capture_graph"], default="graph")
    p.add_argument("--round", type=int, default=0)
    args = p.parse_args()
    result = {
        "scope": "Untimed graph calls; not actual captured command bytes or compiled runtime arguments",
        "stage": args.stage,
        "round": args.round,
        "sources": {},
        "comparisons": {},
    }
    for arm in (0, 1):
        rows = {}
        for k in (3, 4, 5):
            paths = list(args.root.glob(f"{args.stage}_r{args.round}_k{k}_a{arm}_*.jsonl.graph.json"))
            assert len(paths) == 1, paths
            path = paths[0]
            result["sources"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows[k] = device_ops(path)
        for k in (4, 5):
            result["comparisons"][f"arm{arm}_K3_prefix_vs_K{k}"] = compare(rows[3], rows[k])
    args.output.write_text(json.dumps(result, indent=2))
    for name, comparison in result["comparisons"].items():
        print(name, {k: v for k, v in comparison.items() if k != "differences"})


if __name__ == "__main__":
    main()

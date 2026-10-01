#!/usr/bin/env python3
"""Rank every device op in a capture by its CB high-water, then map the ceiling ops.

Reads ``iteration.graph.json`` and ``before_iteration.l1.json`` from a capture that
``lm_head_l1_analyze.py`` has already processed. Writes only ``analysis/cb_ceiling``:
``ranking.json`` (every op invocation's program-local CB end and core count), and one
11x10 Tensix map per selected invocation in the same form as ``lm_head_l1_tensix_grid.py``.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages

from lm_head_l1_analyze import cores
from lm_head_l1_tensix_grid import GRID_X, GRID_Y, render


def invocations(graph):
    """Outermost *DeviceOperation spans, with their CB events and matmul position."""
    stack, result, current, matmuls = [], [], None, 0
    for index, node in enumerate(graph):
        kind, params = node["node_type"], node.get("params", {})
        if kind == "function_start":
            name = params.get("name", "")
            stack.append(name)
            if current is None and name.endswith("DeviceOperation"):
                current = {"op": name, "start": index, "depth": len(stack), "matmuls_before": matmuls, "cbs": []}
        elif kind == "function_end":
            if stack:
                stack.pop()
            if current is not None and len(stack) < current["depth"]:
                current["end_index"] = index
                matmuls += current["op"] == "MatmulDeviceOperation"
                result.append(current)
                current = None
        elif kind == "circular_buffer_allocate" and current is not None:
            current["cbs"].append(
                {
                    "address": int(params["address"]),
                    "size": int(params["size"]),
                    "global": params.get("globally_allocated") == "1",
                    "cores": sorted(cores(params.get("core_range_set", ""))),
                }
            )
    return result, matmuls


def identity(inv, total_matmuls):
    split = total_matmuls in (72, 313)
    draft_calls, per_step = (72, 24) if split else (69, 23)
    m = inv["matmuls_before"]
    if m < draft_calls:
        step, pos = divmod(m, per_step)
        layer = (pos - 1) // 5 if 1 <= pos <= 20 else None
        return {"model": "draft", "draft_step": step, "layer": layer}
    layer = (m - draft_calls) // 5
    return {"model": "target", "layer": layer if layer < 48 else None}


def live_tensors_at(graph, before, head_addr, stop):
    """L1 tensors live when graph node ``stop`` is reached, with their physical pages."""
    base = before["base"]
    live = {}
    for block in before["blocks"]:
        if block.get("allocated") == "yes":
            address = base + int(block["address"])
            live[address] = {
                "address": address,
                "size": int(block["size"]),
                "layout": "",
                "physical": before.get("physical_pages", {}).get(str(address), {}),
            }
    for node in graph[:stop]:
        kind, params = node["node_type"], node.get("params", {})
        if kind == "buffer_allocate" and params.get("type") == "L1":
            address = int(params["address"])
            live[address] = {
                "address": address,
                "size": int(params["max_size_per_bank"]),
                "layout": params.get("layout", ""),
                "physical": json.loads(params.get("physical_ranges") or "{}"),
            }
        elif kind == "buffer_deallocate" and params.get("type") == "L1":
            live.pop(int(params["address"]), None)
    for tensor in live.values():
        if tensor["address"] == head_addr:
            tensor["layout"] = "WIDTH_SHARDED_HEAD"
    return list(live.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--top", type=int, default=12, help="op names listed in the printed ranking")
    parser.add_argument("--op", help="map this op's own high-water invocations instead of the lifetime ceiling's")
    args = parser.parse_args()
    root = args.capture_dir
    graph = json.loads((root / "iteration.graph.json").read_text())
    before = json.loads((root / "before_iteration.l1.json").read_text())
    report = json.loads((root / "analysis/report.json").read_text())
    base, top = report["base"], report["top"]
    invs, total_matmuls = invocations(graph)

    rows = []
    for inv in invs:
        local = [cb for cb in inv["cbs"] if not cb["global"]]
        inv["cb_end"] = max((cb["address"] + cb["size"] for cb in local), default=base)
        inv["cb_cores"] = len({tuple(c) for cb in local for c in cb["cores"]})
        inv["local_cb_bytes"] = sum(cb["size"] for cb in local)
        inv.update(identity(inv, total_matmuls))
        rows.append(
            {k: inv[k] for k in ("op", "start", "end_index", "model", "layer", "cb_end", "cb_cores", "local_cb_bytes")}
            | {"draft_step": inv.get("draft_step")}
        )
    ceiling = max(r["cb_end"] for r in rows)
    by_op = defaultdict(list)
    for r in rows:
        by_op[r["op"]].append(r["cb_end"])
    ranking = sorted(
        (
            {
                "op": op,
                "invocations": len(v),
                "max_cb_end": max(v),
                "min_cb_end": min(v),
                "at_max": sum(e == max(v) for e in v),
            }
            for op, v in by_op.items()
        ),
        key=lambda r: -r["max_cb_end"],
    )
    at_ceiling = [r for r in rows if r["cb_end"] == ceiling]
    mapped_op = args.op
    op_max = max(r["cb_end"] for r in rows if r["op"] == mapped_op) if mapped_op else ceiling
    to_map = [r for r in rows if r["cb_end"] == op_max and (mapped_op is None or r["op"] == mapped_op)]

    out = root / "analysis/cb_ceiling"
    out.mkdir(parents=True, exist_ok=True)
    head = report["head_addr"]
    summary = {
        "capture": str(root),
        "base": base,
        "top": top,
        "lifetime_cb_ceiling": ceiling,
        "head_addr": head,
        "head_only_bytes_above_ceiling": top - ceiling,
        "headroom_below_head": (head - ceiling) if head is not None else None,
        "ranking_by_op": ranking,
        "invocations_at_ceiling": at_ceiling,
        "all_invocations": rows,
    }
    (out / "ranking.json").write_text(json.dumps(summary, indent=2) + "\n")

    # One map per distinct (model, core count) at the ceiling: its first invocation.
    chosen = {}
    for r in to_map:
        chosen.setdefault((r["model"], r["cb_cores"]), r)
    maps = []
    suffix = f"_{mapped_op}" if mapped_op else ""
    with PdfPages(out / f"ceiling_maps{suffix}.pdf") as pdf:
        for r in chosen.values():
            inv = next(i for i in invs if i["start"] == r["start"])
            tensors = live_tensors_at(graph, before, head, inv["end_index"])
            records = {}
            for y in range(GRID_Y):
                for x in range(GRID_X):
                    records[(x, y)] = {
                        "core": [x, y],
                        "tensors": tensors,
                        "cbs": [cb for cb in inv["cbs"] if not cb["global"] and (x, y) in cb["cores"]],
                    }
            ident = {
                "model": r["model"],
                "layer": r["layer"],
                "draft_step": r["draft_step"],
                "role": f'{r["op"].replace("DeviceOperation", "")}_cb_ceiling_{report["cols"]}',
            }
            layer = "head" if r["layer"] is None else f'layer{r["layer"]:02d}'
            stem = out / f'{r["model"]}_{layer}_{r["op"]}_{r["start"]:05d}'
            cores_summary = render(r["start"], ident, records, base, top, stem, pdf, title_kind="graph op")
            maps.append(
                {
                    "start": r["start"],
                    "model": r["model"],
                    "layer": r["layer"],
                    "svg": stem.with_suffix(".svg").name,
                    "png": stem.with_suffix(".png").name,
                    "cores": cores_summary,
                }
            )
    (out / f"maps{suffix}.json").write_text(json.dumps(maps, indent=2) + "\n")

    print(f"lifetime CB ceiling {ceiling:,}  (head at {head}, top {top:,})")
    for r in ranking[: args.top]:
        print(f'{r["op"]:42s} n={r["invocations"]:5d} max_end={r["max_cb_end"]:9,d} at_max={r["at_max"]}')
    print("at ceiling:", [(r["model"], r["layer"], r.get("draft_step"), r["start"], r["cb_cores"]) for r in at_ceiling])
    print("maps:", [m["png"] for m in maps])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a conservative per-core address ledger from opt-in Gemma4 captures.

Fails closed on missing cached-CB events, call/core counts, or unknown tensor
placement. The JSONL retains unresolved physical mapping rather than charging a
LOCKSTEP reservation as physical bytes.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

GRID = [(x, y) for y in range(10) for x in range(11)]
COLORS = {"reserved": "#999999", "cb": "#2878b5", "tensor": "#e69f00", "head": "#b44bb6"}


def cores(s):
    result = set()
    for x1, y1, x2, y2 in re.findall(r"\[(\d+)-(\d+)\s*-\s*(\d+)-(\d+)\]", s or ""):
        result.update((x, y) for y in range(int(y1), int(y2) + 1) for x in range(int(x1), int(x2) + 1))
    return result


def integer(v):
    return int(v, 0) if isinstance(v, str) and v.startswith("0x") else int(v)


def intervals_union(parts):
    merged = []
    for start, end in sorted(parts):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def shapes(node):
    arg = " ".join(node.get("arguments", []))
    return [list(map(int, s.split(","))) for s in re.findall(r"logical_shape=Shape\(\[([\d, ]+)\]\)", arg)[:2]]


def call_identity(index, count):
    """Current 12B K=3 order, checked against the total count below."""
    split = count in (72, 313)
    draft_calls = 72 if split else 69
    if count not in (69, 72, 310, 313):
        return {"model": "unknown", "layer": None, "role": "unknown"}
    if index < draft_calls:
        per_step = 24 if split else 23
        step, pos = divmod(index, per_step)
        if pos == 0:
            layer, role = None, "pre_projection"
        elif pos <= 20:
            layer = (pos - 1) // 5
            role = ("qkv", "o_proj", "gate_proj", "up_proj", "down_proj")[(pos - 1) % 5]
        elif split and pos in (21, 22):
            layer, role = None, ("lm_head_l1", "lm_head_dram_tail")[pos - 21]
        elif not split and pos == 21:
            layer, role = None, "lm_head"
        else:
            layer, role = None, "post_projection"
        return {"model": "draft", "draft_step": step, "layer": layer, "role": role}
    if count in (69, 72):
        return {"model": "unknown", "layer": None, "role": "unknown"}
    pos = index - draft_calls
    if pos == 240:
        return {"model": "target", "layer": None, "role": "lm_head"}
    layer, slot = divmod(pos, 5)
    return {"model": "target", "layer": layer, "role": ("qkv", "o_proj", "gate_proj", "up_proj", "down_proj")[slot]}


def analyze(root, expected):
    graph = json.loads((root / "iteration.graph.json").read_text())
    calls_path = root / "iteration.calls.json"
    calls = json.loads(calls_path.read_text()) if calls_path.exists() else []
    before = json.loads((root / "before_iteration.l1.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    base = before["base"]
    top = base + before["total_bytes_per_bank"]
    cols = manifest["cols"]
    shard_bytes = cols // 3520 * 65536 if cols else 0
    live = {}
    for b in before["blocks"]:
        if b.get("allocated") == "yes":
            # MemoryView's block table is relative to the allocator base;
            # BufferInfo, buffer pages and CB events use absolute L1 addresses.
            addr = base + integer(b["address"])
            info = before.get("buffers", {}).get(str(addr), {})
            live[addr] = {
                "address": addr,
                "size": integer(b["size"]),
                "layout": info.get("layout", "unknown"),
                "num_cores": None,
                "physical": before.get("physical_pages", {}).get(str(addr)),
                "source": "snapshot",
            }
    # The pinned uniform shard is the lowest block of the known size. Record
    # ambiguity instead of silently labelling a different allocation as head.
    head_candidates = [a for a, b in live.items() if b["size"] == shard_bytes] if shard_bytes else []
    head_addr = min(head_candidates) if head_candidates else None
    errors = []
    if shard_bytes and len(head_candidates) != 1:
        errors.append(f"expected one pinned head block of {shard_bytes} B/bank, found {len(head_candidates)}")
    if head_addr is not None:
        live[head_addr]["layout"] = "WIDTH_SHARDED_HEAD"
        live[head_addr]["num_cores"] = 110
    stack = []
    active = None
    matmuls, other_frontiers = [], []
    call_index = 0
    for node in graph:
        kind = node.get("node_type")
        p = node.get("params", {})
        if kind == "function_start":
            name = p.get("name", "")
            stack.append(name)
            if name == "MatmulDeviceOperation":
                active = {"index": call_index, "shape": shapes(node), "name": name, "cbs": {}, "buffers": None}
                call_index += 1
        elif kind == "function_end":
            if stack:
                name = stack.pop()
                if name == "MatmulDeviceOperation" and active is not None:
                    active["buffers"] = list(live.values())
                    matmuls.append(active)
                    active = None
        elif kind == "circular_buffer_allocate":
            if p.get("globally_allocated") == "1":
                continue
            addr, size = integer(p["address"]), integer(p["size"])
            cs = cores(p.get("core_range_set", ""))
            key = (addr, size, tuple(sorted(cs)))
            if active is not None:
                active["cbs"][key] = {"address": addr, "size": size, "cores": sorted(cs)}
            else:
                other_frontiers.append(
                    {"op": stack[0] if stack else "<unknown>", "end": addr + size, "cores": sorted(cs)}
                )
        elif kind == "buffer_allocate" and p.get("type") == "L1":
            addr = integer(p["address"])
            live[addr] = {
                "address": addr,
                "size": integer(p["max_size_per_bank"]),
                "layout": p.get("layout", "unknown"),
                "num_cores": integer(p.get("num_cores", 0)),
                "physical": json.loads(p["physical_ranges"]) if p.get("physical_ranges") else None,
                "source": "graph",
            }
        elif kind == "buffer_deallocate" and p.get("type") == "L1":
            live.pop(integer(p["address"]), None)
    if expected is not None and len(matmuls) != expected:
        errors.append(f"expected {expected} matmuls, observed {len(matmuls)}")
    if calls and len(calls) != len(matmuls):
        errors.append(f"Python calls {len(calls)} differ from device matmuls {len(matmuls)}")
    if not matmuls:
        errors.append("no matmuls in graph")
    if any(not m["cbs"] for m in matmuls):
        errors.append(f'{sum(not m["cbs"] for m in matmuls)} matmuls have no resolved CB events; rebuild graph hook')
    records = []
    for m in matmuls:
        m["cbs"] = list(m["cbs"].values())
        m["call"] = calls[m["index"]] if m["index"] < len(calls) else None
        active_cores = {tuple(c) for cb in m["cbs"] for c in cb["cores"]}
        xs = [c[0] for c in active_cores]
        ys = [c[1] for c in active_cores]
        resolved_grid = [max(xs) + 1, max(ys) + 1] if active_cores else None
        identity = call_identity(m["index"], len(matmuls))
        for x, y in GRID:
            c = (x, y)
            physical = []
            lockstep = []
            unknown = []
            for cb in m["cbs"]:
                part = (cb["address"], cb["address"] + cb["size"])
                if c in map(tuple, cb["cores"]):
                    physical.append(part)
                else:
                    # CB ranges are program local and do not reserve idle cores.
                    pass
            for b in m["buffers"]:
                part = (b["address"], b["address"] + b["size"])
                lockstep.append(part)
                page_map = b.get("physical")
                if page_map is not None:
                    physical.extend(map(tuple, page_map.get(f"{x},{y}", [])))
                elif b["layout"] == "WIDTH_SHARDED_HEAD" or b["num_cores"] == 110:
                    physical.append(part)
                else:
                    unknown.append(part)
            occupied = intervals_union(physical)
            charged = intervals_union(lockstep + physical)
            cbs = [cb for cb in m["cbs"] if c in map(tuple, cb["cores"])]
            cb_upper = max((cb["address"] + cb["size"] for cb in cbs), default=base)
            tensor_low = min((b["address"] for b in m["buffers"]), default=top)
            record = {
                "index": m["index"],
                "core": [x, y],
                "shape": m["shape"],
                "call": m["call"],
                "program": m["name"],
                "resolved_grid": resolved_grid,
                "active_core_count": len(active_cores),
                **identity,
                "cbs": cbs,
                "tensors": m["buffers"],
                "cb_upper": cb_upper,
                "tensor_low": tensor_low,
                "known_physical_bytes": sum(e - s for s, e in occupied),
                "lockstep_charged_bytes": sum(e - s for s, e in charged),
                "physical_mapping_unresolved": bool(unknown),
                "free_gap_bytes": max(0, tensor_low - cb_upper),
                "unreserved_minus_head_bytes": top - base - shard_bytes,
            }
            records.append(record)
    if len(records) != len(matmuls) * 110:
        errors.append("missing per-core records")
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    with (out / "core_ledger.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    max_other = max(other_frontiers, key=lambda x: x["end"], default=None)
    max_cb = max(
        ({"index": m["index"], "cb": cb} for m in matmuls for cb in m["cbs"]),
        key=lambda x: x["cb"]["address"] + x["cb"]["size"],
        default=None,
    )
    ceiling = max(
        [x for x in (max_other, max_cb) if x],
        key=lambda x: x.get("end", x.get("cb", {}).get("address", 0) + x.get("cb", {}).get("size", 0)),
        default=None,
    )
    per_core_ceiling = {f"{x},{y}": base for x, y in GRID}
    for event in other_frontiers:
        for x, y in event["cores"]:
            key = f"{x},{y}"
            per_core_ceiling[key] = max(per_core_ceiling[key], event["end"])
    for m in matmuls:
        for cb in m["cbs"]:
            for x, y in cb["cores"]:
                key = f"{x},{y}"
                per_core_ceiling[key] = max(per_core_ceiling[key], cb["address"] + cb["size"])
    report = {
        "matmul_calls": len(matmuls),
        "core_records": len(records),
        "expected": expected,
        "cols": cols,
        "base": base,
        "top": top,
        "head_addr": head_addr,
        "head_shard_bytes": shard_bytes,
        "lifetime_cb_ceiling": ceiling,
        "per_core_cb_ceiling": per_core_ceiling,
        "per_core_head_only_bytes": {core: top - end for core, end in per_core_ceiling.items()},
        "per_core_current_headroom_bytes": {
            core: (head_addr if head_addr is not None else top) - end for core, end in per_core_ceiling.items()
        },
        "other_cb_events": len(other_frontiers),
        "errors": errors,
        "physical_mapping_unresolved_records": sum(r["physical_mapping_unresolved"] for r in records),
    }
    figures = out / "figures"
    if figures.exists():
        report["plot_count"] = len(list(figures.glob("*.svg")))
        if report["plot_count"] != len(matmuls):
            errors.append("incomplete plot export")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return records, report


def plots(root, records, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch, Rectangle

    out = root / "analysis"
    by_call = defaultdict(list)
    for r in records:
        by_call[r["index"]].append(r)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    links = []
    with PdfPages(out / "all_matmuls.pdf") as pdf:
        for index, rs in by_call.items():
            fig, axes = plt.subplots(10, 1, figsize=(16, 25), sharex=True, sharey=True)
            for y, ax in enumerate(axes):
                row = sorted((r for r in rs if r["core"][1] == y), key=lambda r: r["core"][0])
                for r in row:
                    x = r["core"][0]
                    ax.add_patch(Rectangle((x - 0.4, 0), 0.8, report["base"], facecolor=COLORS["reserved"]))
                    for b in r["tensors"]:
                        color = COLORS["head"] if b["layout"] == "WIDTH_SHARDED_HEAD" else COLORS["tensor"]
                        page_map = b.get("physical")
                        actual = (
                            page_map.get(f"{x},{y}", [])
                            if page_map is not None
                            else ([[b["address"], b["address"] + b["size"]]] if b["num_cores"] == 110 else [])
                        )
                        ax.add_patch(
                            Rectangle(
                                (x - 0.4, b["address"]),
                                0.8,
                                b["size"],
                                facecolor="white",
                                hatch="///",
                                edgecolor=color,
                                linewidth=0.1,
                            )
                        )
                        for lo, hi in actual:
                            ax.add_patch(
                                Rectangle(
                                    (x - 0.4, lo), 0.8, hi - lo, facecolor=color, edgecolor="black", linewidth=0.15
                                )
                            )
                    for cb in r["cbs"]:
                        ax.add_patch(
                            Rectangle(
                                (x - 0.4, cb["address"]),
                                0.8,
                                cb["size"],
                                facecolor=COLORS["cb"],
                                edgecolor="black",
                                linewidth=0.15,
                            )
                        )
                    ax.plot(x, r["cb_upper"], "v", color="navy", ms=3)
                    ax.plot(x, r["tensor_low"], "^", color="darkred", ms=3)
                ax.set_xlim(-0.5, 10.5)
                ax.set_ylim(0, report["top"])
                ax.set_ylabel(f"y={y}\nL1 B")
                ax.grid(axis="y", alpha=0.2)
            axes[-1].set_xticks(range(11))
            axes[-1].set_xlabel("core x")
            fig.suptitle(f'Matmul {index:03d}: {rs[0]["shape"]}  {rs[0]["call"]}', fontsize=10)
            fig.legend(
                handles=[Patch(color=COLORS[k], label=k) for k in COLORS]
                + [Patch(facecolor="white", hatch="///", label="LOCKSTEP only, no physical data")],
                loc="upper center",
                ncol=5,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            pdf.savefig(fig)
            svg = figures / f"{index:03d}.svg"
            fig.savefig(svg)
            plt.close(fig)
            links.append(svg.name)
    options = "\n".join(f'<option value="figures/{name}">matmul {i:03d}</option>' for i, name in enumerate(links))
    (out / "index.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>12B L1 address plots</title>
<select id="operation" onchange="document.getElementById('figure').src=this.value">"""
        + options
        + """</select>
<iframe id="figure" src="figures/000.svg" style="width:100%;height:90vh;border:0"></iframe>"""
    )
    if len(links) != report["matmul_calls"]:
        report["errors"].append("incomplete plot export")
    report["plot_count"] = len(links)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture_dir", type=Path)
    p.add_argument("--expected", type=int, choices=[69, 72, 310, 313, 397, 400])
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()
    records, report = analyze(args.capture_dir, args.expected)
    if args.plots:
        plots(args.capture_dir, records, report)
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

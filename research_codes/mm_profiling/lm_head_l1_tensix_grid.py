#!/usr/bin/env python3
"""Curated 11x10 Tensix SRAM maps from an existing lm_head core ledger.

This writes only ``analysis/tensix_grid``. The per-call address plots and their
selector remain untouched. Each core square is a miniature absolute L1 address
bar: low addresses at its bottom, high addresses at its top.
"""

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

from lm_head_l1_analyze import COLORS, call_identity, cores, intervals_union

GRID_X, GRID_Y = 11, 10
SHOWN_LAYERS = {"draft": {0}, "target": {0, 5}}


def chosen(identity):
    model, layer = identity["model"], identity["layer"]
    if model not in SHOWN_LAYERS:
        return False
    if model == "draft" and identity.get("draft_step") != 0:
        return False
    return layer is None or layer in SHOWN_LAYERS[model]


def select_records(ledger, count):
    selected = {i: call_identity(i, count) for i in range(count) if chosen(call_identity(i, count))}
    if not selected:
        raise ValueError("no matching drafter/target call identities")
    rows = defaultdict(dict)
    with ledger.open() as source:
        for line_number, line in enumerate(source):
            index = line_number // (GRID_X * GRID_Y)
            if index not in selected:
                continue
            record = json.loads(line)
            if record["index"] != index:
                raise ValueError(f"ledger call order differs at row {line_number}")
            if record["physical_mapping_unresolved"]:
                raise ValueError(f"unresolved physical placement in call {index}")
            core = tuple(record["core"])
            if core in rows[index]:
                raise ValueError(f"duplicate core {core} for call {index}")
            rows[index][core] = record
    expected = {(x, y) for y in range(GRID_Y) for x in range(GRID_X)}
    for index in selected:
        if set(rows[index]) != expected:
            raise ValueError(f"call {index}: expected all 110 cores, got {len(rows[index])}")
    return selected, rows


def sdpa_ceiling_records(root, layer=5):
    """The full-grid target SDPA CB event, with exact live tensors and CB ranges."""
    graph = json.loads((root / "iteration.graph.json").read_text())
    before = json.loads((root / "before_iteration.l1.json").read_text())
    report = json.loads((root / "analysis/report.json").read_text())
    base = before["base"]
    live = {}
    for block in before["blocks"]:
        if block.get("allocated") != "yes":
            continue
        address = base + int(block["address"])
        live[address] = {
            "address": address,
            "size": int(block["size"]),
            "layout": before.get("buffers", {}).get(str(address), {}).get("layout", ""),
            "physical": before.get("physical_pages", {}).get(str(address), {}),
        }
    stack = []
    sdpa_cbs = []
    matmuls_seen = 0
    draft_calls = 72 if report["matmul_calls"] == 313 else 69
    candidates = []
    for graph_index, node in enumerate(graph):
        kind, params = node["node_type"], node.get("params", {})
        if kind == "function_start":
            name = params.get("name", "")
            stack.append(name)
            if name == "MatmulDeviceOperation":
                matmuls_seen += 1
            elif name == "SdpaDecodeDeviceOperation":
                sdpa_cbs = []
        elif kind == "function_end":
            if stack:
                stack.pop()
        elif kind == "buffer_allocate" and params.get("type") == "L1":
            address = int(params["address"])
            live[address] = {
                "address": address,
                "size": int(params["max_size_per_bank"]),
                "layout": params.get("layout", ""),
                "physical": json.loads(params.get("physical_ranges") or "{}"),
            }
        elif kind == "buffer_deallocate" and params.get("type") == "L1":
            live.pop(int(params["address"]), None)
        elif kind == "circular_buffer_allocate" and params.get("globally_allocated") == "0":
            if "SdpaDecodeDeviceOperation" not in stack:
                continue
            address, size = int(params["address"]), int(params["size"])
            active_cores = cores(params.get("core_range_set", ""))
            sdpa_cbs.append({"address": address, "size": size, "cores": active_cores})
            target_layer = (matmuls_seen - draft_calls) // 5
            if target_layer == layer and len(active_cores) == GRID_X * GRID_Y:
                candidates.append((graph_index, address + size, list(live.values()), list(sdpa_cbs)))
    if not candidates:
        raise ValueError(f"no full-grid target layer {layer} SDPA ceiling event in {root}")
    graph_index, _, tensors, cbs = max(candidates, key=lambda item: (item[1], sum(t["size"] for t in item[2])))
    # The snapshot's layout string differs from the graph string; identify the
    # pinned head by the allocator address already validated by the analyzer.
    for tensor in tensors:
        if tensor["address"] == report["head_addr"]:
            tensor["layout"] = "WIDTH_SHARDED_HEAD"
    records = {}
    for y in range(GRID_Y):
        for x in range(GRID_X):
            core = (x, y)
            records[core] = {
                "core": [x, y],
                "tensors": tensors,
                "cbs": [{"address": cb["address"], "size": cb["size"]} for cb in cbs if core in cb["cores"]],
            }
    return graph_index, records


def core_spans(record, base, top):
    x, y = record["core"]
    cbs = [(int(cb["address"]), int(cb["address"]) + int(cb["size"])) for cb in record["cbs"]]
    tensors = []
    physical = []
    charged = []
    for tensor in record["tensors"]:
        address, size = int(tensor["address"]), int(tensor["size"])
        if address < base or address + size > top:
            raise ValueError(f"tensor [{address}, {address + size}) is outside allocatable L1")
        color = "head" if tensor["layout"] == "WIDTH_SHARDED_HEAD" else "tensor"
        placement = tensor.get("physical")
        actual = (
            placement.get(f"{x},{y}", [])
            if placement is not None
            else ([[address, address + size]] if tensor.get("num_cores") == 110 else [])
        )
        actual = [(int(lo), int(hi)) for lo, hi in actual]
        tensors.append({"address": address, "size": size, "color": color, "physical": actual})
        physical += actual
        charged.append((address, address + size))
    for lo, hi in cbs:
        if lo < base or hi > top:
            raise ValueError(f"CB [{lo}, {hi}) is outside allocatable L1")
    charged_union = intervals_union(cbs + charged)
    physical_union = intervals_union(cbs + physical)
    charged_bytes = sum(hi - lo for lo, hi in charged_union)
    physical_bytes = sum(hi - lo for lo, hi in physical_union)
    return {
        "cbs": cbs,
        "tensors": tensors,
        "charged_bytes": charged_bytes,
        "physical_bytes": physical_bytes,
        "charged_percent": 100 * charged_bytes / (top - base),
        "physical_percent": 100 * physical_bytes / (top - base),
    }


def group_name(identity):
    return f'layer_{identity["layer"]:02d}' if identity["layer"] is not None else "head_and_other"


def label(identity):
    prefix = "Drafter" if identity["model"] == "draft" else "Target"
    if identity["layer"] is None:
        where = "head / other"
    else:
        where = f'layer {identity["layer"]}'
    step = f' · step {identity["draft_step"]}' if identity["model"] == "draft" else ""
    return f'{prefix}{step} · {where} · {identity["role"]}'


def render(index, identity, records, base, top, destination, pdf, title_kind="matmul"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 12.3))
    fig.subplots_adjust(left=0.065, right=0.93, top=0.87, bottom=0.09)
    h = 0.88

    def stripe(x, y, lo, hi, **style):
        if hi <= lo:
            return
        bottom = y + h / 2 - h * hi / top
        ax.add_patch(Rectangle((x - h / 2, bottom), h, h * (hi - lo) / top, **style))

    summary = {}
    for (x, y), record in sorted(records.items(), key=lambda item: (item[0][1], item[0][0])):
        spans = core_spans(record, base, top)
        # White is free L1. The gray reserved region is at the bottom of every
        # square. Charged-only tensor addresses are hatched; physical data then
        # paints over the corresponding ranges at their actual top-down address.
        ax.add_patch(Rectangle((x - h / 2, y - h / 2), h, h, facecolor="white", edgecolor="none"))
        stripe(x, y, 0, base, facecolor=COLORS["reserved"], edgecolor="none")
        for tensor in spans["tensors"]:
            stripe(
                x,
                y,
                tensor["address"],
                tensor["address"] + tensor["size"],
                facecolor="white",
                edgecolor=COLORS[tensor["color"]],
                hatch="////",
                linewidth=0.15,
            )
            for lo, hi in tensor["physical"]:
                stripe(x, y, lo, hi, facecolor=COLORS[tensor["color"]], edgecolor="none")
        for lo, hi in spans["cbs"]:
            stripe(x, y, lo, hi, facecolor=COLORS["cb"], edgecolor="none")
        ax.add_patch(Rectangle((x - h / 2, y - h / 2), h, h, fill=False, edgecolor="#303030", linewidth=0.5))
        ax.text(
            x,
            y,
            f'{spans["physical_percent"]:.0f}%',
            ha="center",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color="#111111",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.83, "pad": 0.9},
        )
        summary[f"{x},{y}"] = {
            "charged_bytes": spans["charged_bytes"],
            "physical_bytes": spans["physical_bytes"],
            "charged_percent": round(spans["charged_percent"], 3),
            "physical_percent": round(spans["physical_percent"], 3),
            "cb_ranges": spans["cbs"],
        }

    ax.set_xlim(-0.55, 10.55)
    ax.set_ylim(9.55, -0.55)  # device row 0 is at the top; L1 low address stays at each square's bottom
    ax.set_aspect("equal")
    ax.set_xticks(range(GRID_X))
    ax.set_yticks(range(GRID_Y))
    ax.set_xlabel("Tensix x")
    ax.set_ylabel("Tensix y")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(f"{label(identity)} · {title_kind} {index:03d}", fontsize=15, fontweight="bold", y=0.975)
    fig.text(
        0.5,
        0.937,
        f"Each square: bottom = address 0, top = {top:,} B. "
        f"Utilization = physical CB + tensor data / {top - base:,} allocatable B; hatch = LOCKSTEP-only charge.",
        ha="center",
        fontsize=9,
    )
    fig.legend(
        handles=[
            Patch(facecolor=COLORS["reserved"], label="fixed reservation"),
            Patch(facecolor=COLORS["cb"], label="program CB (bottom-up)"),
            Patch(facecolor=COLORS["tensor"], label="L1 tensor (top-down)"),
            Patch(facecolor=COLORS["head"], label="pinned head (top-down)"),
            Patch(facecolor="white", edgecolor=COLORS["tensor"], hatch="////", label="LOCKSTEP only, no physical data"),
            Patch(facecolor="white", edgecolor="#777777", label="free"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    svg = destination.with_suffix(".svg")
    png = destination.with_suffix(".png")
    fig.savefig(svg)
    fig.savefig(png, dpi=150)
    pdf.savefig(fig)
    plt.close(fig)
    return summary


def write_index(output, entries, capture_name):
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        grouped[entry["model"]][entry["group"]].append(entry)
    navigation = []
    for model in ("draft", "target"):
        if model not in grouped:
            continue
        layers = []
        for group, items in grouped[model].items():
            links = "".join(
                f'<button data-image="{html.escape(item["svg"])}">'
                f'{html.escape(item["role"])} <small>#{item["index"]:03d}</small></button>'
                for item in items
            )
            layers.append(f'<details open><summary>{html.escape(group.replace("_", " "))}</summary>{links}</details>')
        navigation.append(
            f'<details open><summary>{"Drafter" if model == "draft" else "Target"}</summary>'
            + "".join(layers)
            + "</details>"
        )
    first = entries[0]["svg"]
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>12B Tensix SRAM representative maps</title>
<style>body{{font:14px system-ui;margin:0;background:#f7f8fa;color:#222}}header{{padding:12px 18px;background:white;border-bottom:1px solid #ddd}}
main{{display:grid;grid-template-columns:270px 1fr;height:calc(100vh - 70px)}}nav{{overflow:auto;padding:12px;border-right:1px solid #ddd}}
details{{margin:5px 0 8px 8px}}summary{{cursor:pointer;font-weight:650;padding:4px}}button{{display:block;width:100%;text-align:left;
border:0;border-radius:4px;background:transparent;padding:6px 8px;cursor:pointer}}button:hover,button.selected{{background:#dfe9f8}}
small{{float:right;color:#666}}iframe{{width:100%;height:100%;border:0;background:white}}</style>
<header><strong>12B Tensix SRAM maps</strong> · {html.escape(capture_name)} · {len(entries)} representatives
<span style="float:right"><a href="representatives.pdf">Multipage PDF</a> · <a href="manifest.json">Selection and utilization data</a></span></header>
<main><nav>{''.join(navigation)}</nav><iframe id="figure" src="{html.escape(first)}" title="Tensix SRAM map"></iframe></main>
<script>for(const b of document.querySelectorAll('button[data-image]'))b.onclick=()=>{{
document.querySelectorAll('button.selected').forEach(x=>x.classList.remove('selected'));b.classList.add('selected');
document.getElementById('figure').src=b.dataset.image;}};document.querySelector('button[data-image]').classList.add('selected');</script></html>"""
    (output / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument(
        "--pressure-before",
        type=Path,
        help="35,200-column pre-reclamation capture for two SDPA ceiling comparison maps",
    )
    parser.add_argument(
        "--include-pressure", action="store_true", help="include this capture's target layer 5 SDPA ceiling map"
    )
    args = parser.parse_args()
    analysis = args.capture_dir / "analysis"
    report = json.loads((analysis / "report.json").read_text())
    if report["errors"] or report["physical_mapping_unresolved_records"]:
        raise SystemExit("capture ledger has unresolved errors")
    count, base, top = report["matmul_calls"], report["base"], report["top"]
    selected, rows = select_records(analysis / "core_ledger.jsonl", count)
    output = analysis / "tensix_grid"
    output.mkdir(exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    entries = []
    with PdfPages(output / "representatives.pdf") as pdf:
        for index, identity in selected.items():
            model, group = identity["model"], group_name(identity)
            stem = output / model / group / f'{identity["role"]}_{index:03d}'
            cores = rows[index]
            summary = render(index, identity, cores, base, top, stem, pdf)
            entries.append(
                {
                    "index": index,
                    "model": model,
                    "group": group,
                    "layer": identity["layer"],
                    "role": identity["role"],
                    "draft_step": identity.get("draft_step"),
                    "svg": stem.with_suffix(".svg").relative_to(output).as_posix(),
                    "png": stem.with_suffix(".png").relative_to(output).as_posix(),
                    "cores": summary,
                }
            )
        pressure_sources = []
        if args.pressure_before:
            pressure_sources = [("before", args.pressure_before), ("after", args.capture_dir)]
        elif args.include_pressure:
            pressure_sources = [("current", args.capture_dir)]
        for state, source in pressure_sources:
            graph_index, cores_at_sdpa = sdpa_ceiling_records(source)
            source_report = json.loads((source / "analysis/report.json").read_text())
            role = f'sdpa_ceiling_{source_report["cols"]}_{state}'
            identity = {"model": "target", "layer": 5, "role": role}
            stem = output / "target" / "layer_05" / f"{role}_{graph_index:05d}"
            summary = render(
                graph_index,
                identity,
                cores_at_sdpa,
                source_report["base"],
                source_report["top"],
                stem,
                pdf,
                title_kind="graph CB event",
            )
            entries.append(
                {
                    "index": graph_index,
                    "model": "target",
                    "group": "layer_05",
                    "layer": 5,
                    "role": role,
                    "draft_step": None,
                    "capture": str(source),
                    "svg": stem.with_suffix(".svg").relative_to(output).as_posix(),
                    "png": stem.with_suffix(".png").relative_to(output).as_posix(),
                    "cores": summary,
                }
            )
    manifest = {
        "capture": str(args.capture_dir),
        "base": base,
        "top": top,
        "allocatable_bytes_per_bank": top - base,
        "displayed_utilization": "union(physical program-local CB ranges + physical L1 tensor pages) / allocatable L1; fixed reservation excluded",
        "lockstep_charged_percent": "separately recorded per core as union(CB + LOCKSTEP tensor reservation) / allocatable L1",
        "physical_data": "solid color; LOCKSTEP-only tensor reservations are hatched and excluded from displayed utilization",
        "selection": "draft step 0, layer 0 and unlayered calls; target layers 0 and 5 and target head; optional target layer 5 SDPA ceiling",
        "total_calls": count,
        "representative_plots": len(entries),
        "entries": entries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_index(output, entries, args.capture_dir.name)
    print(f"{len(entries)} representative 11x10 maps -> {output}")
    print("draft", sum(e["model"] == "draft" for e in entries), "target", sum(e["model"] == "target" for e in entries))


if __name__ == "__main__":
    main()

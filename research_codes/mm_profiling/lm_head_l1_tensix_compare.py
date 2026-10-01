#!/usr/bin/env python3
"""Side-by-side browser for matched unpinned and pinned Tensix grid maps."""

import argparse
import html
import json
import os
from collections import defaultdict
from pathlib import Path


def image_url(output, grid, entry):
    return Path(os.path.relpath(grid / entry["svg"], output)).as_posix()


def matches(unpinned, pinned):
    index = {(entry["model"], entry["group"], entry["role"]): entry for entry in pinned}
    paired = []
    for left in unpinned:
        roles = [left["role"]]
        if left["model"] == "draft" and left["role"] == "lm_head":
            roles = ["lm_head_l1", "lm_head_dram_tail"]
        elif left["role"] == "sdpa_ceiling_0_current":
            roles = ["sdpa_ceiling_38720_after"]
        for role in roles:
            right = index.get((left["model"], left["group"], role))
            if right is None:
                raise ValueError(f'no pinned match for {left["model"]}/{left["group"]}/{role}')
            paired.append((left, right))
    return paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("unpinned_capture", type=Path)
    parser.add_argument("pinned_capture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    left_grid = args.unpinned_capture / "analysis/tensix_grid"
    right_grid = args.pinned_capture / "analysis/tensix_grid"
    output = args.output or args.pinned_capture.parent / "tensix_grid_compare"
    output.mkdir(parents=True, exist_ok=True)
    left_manifest = json.loads((left_grid / "manifest.json").read_text())
    right_manifest = json.loads((right_grid / "manifest.json").read_text())
    if left_manifest["base"] != right_manifest["base"] or left_manifest["top"] != right_manifest["top"]:
        raise ValueError("the two arms have different L1 address bounds")
    pairs = matches(left_manifest["entries"], right_manifest["entries"])
    rows = []
    for left, right in pairs:
        rows.append(
            {
                "model": left["model"],
                "group": left["group"],
                "unpinned_role": left["role"],
                "pinned_role": right["role"],
                "unpinned_index": left["index"],
                "pinned_index": right["index"],
                "unpinned_svg": image_url(output, left_grid, left),
                "pinned_svg": image_url(output, right_grid, right),
            }
        )
    if not rows:
        raise ValueError("no comparison entries")
    for row in rows:
        for key in ("unpinned_svg", "pinned_svg"):
            if not (output / row[key]).is_file():
                raise ValueError(f"missing {key}: {row[key]}")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "unpinned_capture": str(args.unpinned_capture),
                "pinned_capture": str(args.pinned_capture),
                "same_absolute_l1_bounds": [left_manifest["base"], left_manifest["top"]],
                "matching_rule": "model, layer group, role; unsplit drafter head pairs with both split head matmuls",
                "entries": rows,
            },
            indent=2,
        )
        + "\n"
    )

    grouped = defaultdict(lambda: defaultdict(list))
    for number, row in enumerate(rows):
        grouped[row["model"]][row["group"]].append((number, row))
    nav = []
    for model in ("draft", "target"):
        groups = []
        for group, items in grouped[model].items():
            buttons = "".join(
                f'<button data-entry="{number}">{html.escape(row["unpinned_role"])}'
                + (f' → {html.escape(row["pinned_role"])}' if row["unpinned_role"] != row["pinned_role"] else "")
                + f'<small>#{row["unpinned_index"]:03d} / #{row["pinned_index"]:03d}</small></button>'
                for number, row in items
            )
            groups.append(f'<details open><summary>{html.escape(group.replace("_", " "))}</summary>{buttons}</details>')
        nav.append(
            f'<details open><summary>{"Drafter" if model == "draft" else "Target"}</summary>'
            + "".join(groups)
            + "</details>"
        )
    payload = json.dumps(rows).replace("<", "\\u003c")
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>12B Tensix SRAM: DRAM vs pinned</title>
<style>body{{font:14px system-ui;margin:0;background:#f7f8fa;color:#222}}header{{padding:12px 18px;background:white;border-bottom:1px solid #ddd}}
main{{display:grid;grid-template-columns:300px 1fr;height:calc(100vh - 72px)}}nav{{overflow:auto;padding:12px;border-right:1px solid #ddd}}
details{{margin:5px 0 8px 8px}}summary{{cursor:pointer;font-weight:650;padding:4px}}button{{display:block;width:100%;text-align:left;
border:0;border-radius:4px;background:transparent;padding:6px 8px;cursor:pointer}}button:hover,button.selected{{background:#dfe9f8}}
small{{float:right;color:#666}}section{{display:grid;grid-template-columns:1fr 1fr;min-width:0}}article{{display:flex;flex-direction:column;min-width:0;
border-right:1px solid #ddd}}article h2{{font-size:13px;margin:0;padding:8px;background:white;text-align:center}}iframe{{flex:1;width:100%;border:0;background:white}}
article a{{padding:5px;text-align:center;background:white}}</style>
<header><strong>12B Tensix SRAM comparison</strong> · 0-column DRAM head versus 38,720-column pinned head · {len(rows)} matched views
<span style="float:right"><a href="manifest.json">Pairing manifest</a></span></header>
<main><nav>{''.join(nav)}</nav><section><article><h2 id="left-title">Unpinned</h2><iframe id="left" title="Unpinned Tensix grid"></iframe>
<a id="left-link" target="_blank">Open SVG</a></article><article><h2 id="right-title">Pinned</h2>
<iframe id="right" title="Pinned Tensix grid"></iframe><a id="right-link" target="_blank">Open SVG</a></article></section></main>
<script>const entries={payload};function show(i){{const e=entries[i];document.getElementById('left').src=e.unpinned_svg;
document.getElementById('right').src=e.pinned_svg;document.getElementById('left-link').href=e.unpinned_svg;
document.getElementById('right-link').href=e.pinned_svg;document.getElementById('left-title').textContent=`Unpinned · ${{e.unpinned_role}} · call ${{e.unpinned_index}}`;
document.getElementById('right-title').textContent=`Pinned · ${{e.pinned_role}} · call ${{e.pinned_index}}`;
document.querySelectorAll('button.selected').forEach(b=>b.classList.remove('selected'));
document.querySelector(`button[data-entry="${{i}}"]`).classList.add('selected');}}
for(const b of document.querySelectorAll('button[data-entry]'))b.onclick=()=>show(Number(b.dataset.entry));show(0);</script></html>"""
    (output / "index.html").write_text(page)
    print(f'{len(rows)} matched views -> {output / "index.html"}')


if __name__ == "__main__":
    main()

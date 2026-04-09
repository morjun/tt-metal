#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Parse TT-Metal L1_CB_MAP log lines (from program.cpp when TT_METAL_LOG_L1_CB_MAP is set).

Usage:
  TT_METAL_LOG_L1_CB_MAP=1 TT_LOGGER_LEVEL=Info pytest ... 2>&1 | tee run.log
  python3 parse_l1_cb_map_log.py run.log --out-dir ./l1_cb_artifacts

=== What cb_region_end actually measures ===

  cb_region_end is the highest byte address (exclusive) of the bottom-up LOCAL circular
  buffer region for a single program on a single core.

  IMPORTANT: globally-allocated CBs (those backed by a persistent tensor buffer via
  set_globally_allocated_address()) are SKIPPED in allocate_circular_buffers() and do NOT
  contribute to cb_region_end. See program.cpp:885-887:
      if (circular_buffer->globally_allocated()) { continue; }

  This means cb_region_end only captures locally-allocated (scratch) CBs, NOT the
  top-down tensor buffer allocations.

=== L1 memory layout ===

  low addr  [l1_unreserved_base]
            ┌────────────────────────────────────────┐
            │  local CBs (bottom-up)                  │ ← cb_region_end tracks this top
            │  (per-program scratch space, reused      │
            │   across programs at the same addresses) │
            ├────────────────────────────────────────┤ ← cb_region_end
            │                                         │
            │           FREE HEADROOM                 │ ← gap = lowest_top_down_addr
            │    (gap_bytes = lowest_top_down_addr    │             - cb_region_end
            │               - cb_region_end)          │
            │                                         │
            ├────────────────────────────────────────┤ ← lowest_top_down_addr
            │  globally-allocated tensor buffers       │ ← NOT in cb_region_end
            │  (top-down, L1 buffers default           │
            │   bottom_up=False, so they grow down)    │
            │  top_down_size = max_l1 - lowest_top_down│
            └────────────────────────────────────────┘
  high addr [max_l1_size]

=== True simultaneous L1 occupancy per core ===

  total_occupied = cb_region_end                      # local CB stack from bottom
                 + (max_l1_size - lowest_top_down_addr)  # top-down tensor buffers from top
  free_headroom  = lowest_top_down_addr - cb_region_end  # gap between the two

  The heatmap "pct" column shows cb_region_end as % of max_l1.
  The "true_occupied_pct" column shows total_occupied as % of max_l1.
  The "gap_bytes" column is the actual crash-free headroom for KV mirror tiles.

=== Cumulative / per-program semantics ===

  Per-program: one cb_region_end per (program, core) — always fixed once compiled
               (program caching means the same program reuses the same L1 addresses).
  Cumulative:  max over all programs for each core — approximates worst-case local CB
               pressure at any one time (valid because different programs' local CBs
               share the same bottom-up address range and don't stack).
  NOTE: cumulative does NOT sum cb_region_end values across programs. It takes the MAX.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Actual log format (loguru prefix + trailing source location stripped by search):
# 2026-04-06 03:58:07.920 | info     |           Metal | L1_CB_MAP program_id=1 runtime_id=1 phase=allocate core_range=[(x=0,y=0) - (x=7,y=7)] core=(3, 4) cb_region_end=627200 max_l1_size=1572864 lowest_top_down_addr=none (program.cpp:133)
LINE_RE = re.compile(
    r"L1_CB_MAP program_id=(\d+) runtime_id=(\d+) phase=(\w+) "
    r"core_range=\[([^\]]+)\] core=\(\s*(\d+)\s*,\s*(\d+)\s*\) "
    r"cb_region_end=(\d+) max_l1_size=(\d+) lowest_top_down_addr=(\d+|none)"
)


def parse_lines(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        m = LINE_RE.search(line.strip())
        if not m:
            continue
        low = m.group(9)
        rows.append(
            {
                "program_id": int(m.group(1)),
                "runtime_id": int(m.group(2)),
                "phase": m.group(3),
                "core_range": m.group(4),
                "x": int(m.group(5)),
                "y": int(m.group(6)),
                "cb_region_end": int(m.group(7)),
                "max_l1_size": int(m.group(8)),
                "lowest_top_down_addr": None if low == "none" else int(low),
            }
        )
    return rows


def core_key(x: int, y: int) -> str:
    return f"({x},{y})"


def build_maps(rows: list[dict]) -> tuple[dict, dict, dict]:
    """
    Build per-program, cumulative, and headroom maps.

    cb_region_end = end of LOCAL (bottom-up) CB region for that program on that core.
                    Globally-allocated CBs are excluded from this value by the runtime.

    cumulative    = MAX of cb_region_end across all programs for each core.

    headroom      = The worst-case simultaneous execution moment for a given core.
                    We define worst-case as the specific (program, core) execution
                    that produces the smallest `gap_bytes` (lowest_top_down_addr - cb_region_end)
                    where both regions are alive concurrently.
    """
    validate = [r for r in rows if r["phase"] == "validate"]
    per_program: dict[int, dict[str, dict]] = defaultdict(dict)
    cumulative: dict[str, int] = {}
    headroom: dict[str, dict] = {}

    core_events = defaultdict(list)
    for r in validate:
        pid = r["program_id"]
        k = core_key(r["x"], r["y"])
        core_events[k].append(r)

        cur = per_program[pid].get(k)
        if cur is None or r["cb_region_end"] > cur["cb_region_end"]:
            per_program[pid][k] = {
                "cb_region_end": r["cb_region_end"],
                "max_l1_size": r["max_l1_size"],
                "lowest_top_down_addr": r["lowest_top_down_addr"],
            }

    for k, events in core_events.items():
        mls = events[0]["max_l1_size"]

        worst_gap = float("inf")
        worst_event = None
        max_cb = 0

        for e in events:
            cb = e["cb_region_end"]
            low = e["lowest_top_down_addr"]
            max_cb = max(max_cb, cb)

            if low is not None:
                gap = low - cb
                # Tie-break: if gap is the same, use the one with higher local CB usage
                if gap < worst_gap or (gap == worst_gap and worst_event and cb > worst_event["cb_region_end"]):
                    worst_gap = gap
                    worst_event = e

        cumulative[k] = max_cb

        if worst_event is not None:
            end = worst_event["cb_region_end"]
            low = worst_event["lowest_top_down_addr"]
            pid = worst_event["program_id"]

            local_cb_pct = round(100.0 * end / mls, 2)
            top_down_bytes = mls - low
            top_down_pct = round(100.0 * top_down_bytes / mls, 2)
            total_occupied = end + top_down_bytes
            total_pct = round(100.0 * total_occupied / mls, 2)
            gap = low - end

            headroom[k] = {
                "cb_region_end_bytes": end,
                "local_cb_pct_of_l1": local_cb_pct,
                "lowest_top_down_addr": low,
                "top_down_size_bytes": top_down_bytes,
                "top_down_pct_of_l1": top_down_pct,
                "total_occupied_bytes": total_occupied,
                "total_occupied_pct_of_l1": total_pct,
                "gap_bytes_free_headroom": gap,
                "worst_case_pid": pid,
                "max_l1_size": mls,
            }
        else:
            headroom[k] = {
                "cb_region_end_bytes": max_cb,
                "local_cb_pct_of_l1": round(100.0 * max_cb / mls, 2) if mls else None,
                "lowest_top_down_addr": None,
                "top_down_size_bytes": None,
                "top_down_pct_of_l1": None,
                "total_occupied_bytes": None,
                "total_occupied_pct_of_l1": None,
                "gap_bytes_free_headroom": None,
                "worst_case_pid": None,
                "max_l1_size": mls,
            }

    return dict(per_program), cumulative, headroom


def heatmap_md(
    cumulative: dict[str, int],
    headroom: dict[str, dict],
    max_l1: int,
    width: int = 0,
    height: int = 0,
) -> str:
    """
    Generate a markdown heatmap showing per-core L1 pressure.

    Each cell shows:
      local_cb (pct%) | gap_bytes free
    where:
      local_cb  = cb_region_end  = top of locally-allocated CB stack (bottom-up).
                  Globally-allocated tensor CBs are NOT included here.
      gap_bytes = lowest_top_down_addr - cb_region_end
                = true free headroom between local CBs and top-down tensor buffers.
    """
    # Auto-detect grid dimensions from actual data if not specified
    if cumulative:
        all_x = [int(k.strip("()").split(",")[0]) for k in cumulative]
        all_y = [int(k.strip("()").split(",")[1]) for k in cumulative]
        actual_width = max(all_x) + 1
        actual_height = max(all_y) + 1
    else:
        actual_width, actual_height = 8, 8
    width = width or actual_width
    height = height or actual_height

    lines = [
        "# L1 CB Heatmap — local CB stack (validate-phase peak per core)",
        "",
        "**What is shown**: `cb_region_end` = end of the bottom-up, locally-allocated CB region.",
        "Globally-allocated tensor buffers (top-down) are **excluded** from this number;",
        "they appear in `lowest_top_down_addr` and contribute to `top_down_size`.",
        "",
        "**True headroom** = `lowest_top_down_addr − cb_region_end` = gap between the two regions.",
        "**True occupied** = `cb_region_end + (max_l1 − lowest_top_down_addr)`.",
        "",
        f"Grid: x=0..{width-1}, y=0..{height-1}. `max_l1_size`: **{max_l1}** bytes.",
        "",
    ]

    header = "| y \\ x |" + "".join(f" {x} |" for x in range(width))
    sep = "|" + "|".join(["---"] * (width + 1)) + "|"
    lines.extend([header, sep])

    for y in range(height):
        row = f"| **{y}** |"
        for x in range(width):
            k = core_key(x, y)
            v = cumulative.get(k)
            if v is None:
                cell = " — "
            else:
                hr = headroom.get(k, {})
                end_bytes = hr.get("cb_region_end_bytes", v)
                local_pct = hr.get("local_cb_pct_of_l1", 0.0)
                gap = hr.get("gap_bytes_free_headroom")
                total_pct = hr.get("total_occupied_pct_of_l1")
                pid = hr.get("worst_case_pid")

                if gap is not None and total_pct is not None:
                    cell = f"{end_bytes} ({local_pct:.1f}%)<br>gap={gap//1024}KiB<br>pid={pid}"
                else:
                    cell = f"{end_bytes} ({local_pct:.1f}%)"
            row += f" {cell} |"
        lines.append(row)

    lines.append("")
    lines.append("**Legend**: `cb_region_end (local_cb_pct%)` / `gap=free_headroom` / `total=true_occupied_pct`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", type=Path, help="Captured log file (stdout+stderr)")
    ap.add_argument("--out-dir", type=Path, default=Path("l1_cb_artifacts"))
    args = ap.parse_args()

    text = args.log_path.read_text(errors="replace")
    rows = parse_lines(text)
    if not rows:
        raise SystemExit(
            f"No L1_CB_MAP lines found in {args.log_path}. " "Enable TT_METAL_LOG_L1_CB_MAP=1 and TT_LOGGER_LEVEL=Info."
        )

    per_program, cumulative, headroom = build_maps(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_program_str = {str(k): v for k, v in per_program.items()}
    (args.out_dir / "per_program_core_map.json").write_text(json.dumps(per_program_str, indent=2))
    (args.out_dir / "cumulative_core_map.json").write_text(json.dumps(dict(sorted(cumulative.items())), indent=2))
    (args.out_dir / "headroom_map.json").write_text(json.dumps(dict(sorted(headroom.items())), indent=2))

    max_l1 = max(r["max_l1_size"] for r in rows)
    (args.out_dir / "l1_heatmap.md").write_text(heatmap_md(cumulative, headroom, max_l1))

    # Print summary table to stdout
    print(f"Parsed {len(rows)} L1_CB_MAP lines -> {args.out_dir}")
    print(f"  programs seen (validate): {sorted(per_program.keys())}")
    print(f"  cores in cumulative map : {len(cumulative)}")
    print()
    print("  Per-core summary (worst-case execution moment, sorted by tightest gap):")
    print(f"  {'core':<10} {'pid':>5} {'local_cb':>10} {'local%':>7} {'top_down':>10} {'gap':>10} {'total%':>8}")
    print(f"  {'-'*10} {'-'*5} {'-'*10} {'-'*7} {'-'*10} {'-'*10} {'-'*8}")
    sorted_cores = sorted(
        headroom.items(),
        key=lambda kv: kv[1].get("gap_bytes_free_headroom")
        if kv[1].get("gap_bytes_free_headroom") is not None
        else 99999999,
        reverse=False,
    )
    for k, hr in sorted_cores:
        local_cb = hr.get("cb_region_end_bytes")
        pid = hr.get("worst_case_pid")
        local_pct = f"{hr.get('local_cb_pct_of_l1', 0):.1f}%"
        top_down = hr.get("top_down_size_bytes")
        gap = hr.get("gap_bytes_free_headroom")
        total_pct = f"{hr.get('total_occupied_pct_of_l1', 0):.2f}%"

        td_str = f"{top_down:>10}" if top_down is not None else f"{'n/a':>10}"
        gap_str = f"{gap:>10}" if gap is not None else f"{'n/a':>10}"
        print(
            f"  {k:<10} {pid if pid is not None else 'n/a':>5} {local_cb:>10} {local_pct:>7} {td_str} {gap_str} {total_pct:>8}"
        )


if __name__ == "__main__":
    main()

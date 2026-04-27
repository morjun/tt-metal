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

# Matches lines emitted by program factories:
#   log_info(tt::LogOp, ">>> <op_name> program id={}", program.get_id());
OP_RE = re.compile(r">>> (\S+) program id=(\d+)")

# Matches wide-bbox diagnostic lines emitted by matmul_dram_sharded factory when
# the bounding box exceeds 8 columns (triggered by TT_METAL_LOG_L1_CB_MAP):
#   >>> matmul_dram_sharded wide-bbox: program_id=91 M=1 K=128 N=128 per_core_M=1
#       per_core_N_storage=4 bbox=[(0,0)..(12,8)] storage_cores=32
WIDE_BBOX_RE = re.compile(
    r">>> matmul_dram_sharded wide-bbox: program_id=(\d+) "
    r"M=(\d+) K=(\d+) N=(\d+) per_core_M=(\d+) per_core_N_storage=(\d+) "
    r"bbox=\[\((\d+),(\d+)\)\.\.\((\d+),(\d+)\)\] storage_cores=(\d+)"
)


def parse_op_map(text: str) -> dict[int, str]:
    """Parse '>>> <op_name> program id=N' lines into pid→op_name dict."""
    pid_to_op: dict[int, str] = {}
    for line in text.splitlines():
        m = OP_RE.search(line)
        if m:
            pid_to_op[int(m.group(2))] = m.group(1)
    return pid_to_op


def parse_wide_bbox_map(text: str) -> dict[int, dict]:
    """Parse wide-bbox diagnostic lines into pid → matrix dimension info dict."""
    result: dict[int, dict] = {}
    for line in text.splitlines():
        m = WIDE_BBOX_RE.search(line)
        if m:
            pid = int(m.group(1))
            result[pid] = {
                "M_tiles": int(m.group(2)),
                "K_tiles": int(m.group(3)),
                "N_tiles": int(m.group(4)),
                "per_core_M": int(m.group(5)),
                "per_core_N_storage": int(m.group(6)),
                "bbox_start_xy": [int(m.group(7)), int(m.group(8))],
                "bbox_end_xy": [int(m.group(9)), int(m.group(10))],
                "storage_cores": int(m.group(11)),
            }
    return result


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


def pid_table_section(pid_to_op: dict[int, str], per_program: dict[int, dict]) -> str:
    """Render a markdown table mapping every PID → op name and core count."""
    lines = [
        "",
        "## PID → Program Name",
        "",
        "| PID | Op Name | # Cores |",
        "|----:|---------|--------:|",
    ]
    for pid in sorted(per_program.keys()):
        op = pid_to_op.get(pid, "unknown")
        n_cores = len(per_program[pid])
        lines.append(f"| {pid} | {op} | {n_cores} |")
    lines.extend(
        [
            "",
            "> Multiple PIDs with the same op name are distinct compilations (different shapes/configs).",
            "> PIDs showing `unknown` come from factories not yet instrumented with logging.",
            "",
        ]
    )
    return "\n".join(lines)


def core_programs_section(per_program: dict[int, dict], pid_to_op: dict[int, str]) -> str:
    """Render a markdown table listing every program that runs on each core."""
    core_to_pids: dict[str, list[int]] = defaultdict(list)
    for pid, core_map in per_program.items():
        for core_k in core_map:
            core_to_pids[core_k].append(pid)

    lines = [
        "",
        "## Per-Core Program Inventory",
        "",
        "| Core | # Programs | Programs (pid: op_name) |",
        "|------|----------:|-------------------------|",
    ]
    for core_k in sorted(core_to_pids.keys(), key=lambda k: tuple(int(v) for v in k.strip("()").split(","))):
        pids = sorted(core_to_pids[core_k])
        prog_strs = ", ".join(f"{p}:{pid_to_op.get(p, 'unknown')}" for p in pids)
        lines.append(f"| {core_k} | {len(pids)} | {prog_strs} |")
    lines.append("")
    return "\n".join(lines)


def heatmap_md(
    cumulative: dict[str, int],
    headroom: dict[str, dict],
    max_l1: int,
    pid_to_op: dict[int, str] | None = None,
    width: int = 0,
    height: int = 0,
    per_program: dict[int, dict] | None = None,
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
                    td_pct = hr.get("top_down_pct_of_l1", 0.0)
                    op = (pid_to_op or {}).get(pid, "")
                    op_str = f"({op})" if op else ""
                    cell = f"L:{local_pct:.1f}% T:{td_pct:.1f}%<br>gap={gap//1024}KiB<br>pid={pid}{op_str}"
                else:
                    cell = f"L:{local_pct:.1f}%"
            row += f" {cell} |"
        lines.append(row)

    lines.append("")
    lines.append("**Legend**: `L=local_cb% T=top_down%` | `gap=free_headroom` | `pid=worst_case_pid`")
    lines.append("")
    if per_program is not None:
        lines.append(pid_table_section(pid_to_op or {}, per_program))
        lines.append(core_programs_section(per_program, pid_to_op or {}))
    return "\n".join(lines)


def plot_heatmaps(
    cumulative: dict[str, int],
    headroom: dict[str, dict],
    per_program: dict[int, dict],
    out_dir: Path,
    max_l1: int,
) -> None:
    """
    Generate and save two PNG heatmaps:
      1. l1_sram_usage_heatmap.png   — worst-case local CB usage % per core
      2. l1_program_count_heatmap.png — number of programs that use each core
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [warn] matplotlib not available; skipping heatmap images")
        return

    # Infer grid dimensions from observed cores
    all_x = [int(k.strip("()").split(",")[0]) for k in cumulative]
    all_y = [int(k.strip("()").split(",")[1]) for k in cumulative]
    width = max(all_x) + 1  # number of x-values (columns)
    height = max(all_y) + 1  # number of y-values (rows)

    cell_w = max(0.65, 7.0 / width)
    cell_h = max(0.65, 5.0 / height)
    fig_w = width * cell_w + 2.5
    fig_h = height * cell_h + 1.8

    # ── Plot 1: true occupied SRAM % (local CBs + top-down tensor buffers) ──
    # true_occupied = cb_region_end + (max_l1 - lowest_top_down_addr)
    # Falls back to cb_region_end/max_l1 for cores where top-down addr is unavailable.
    usage = np.full((height, width), np.nan)
    for k, info in headroom.items():
        x, y = (int(v) for v in k.strip("()").split(","))
        total_pct = info.get("total_occupied_pct_of_l1")
        if total_pct is not None:
            usage[y, x] = total_pct
        elif max_l1:
            cb = info.get("cb_region_end_bytes", 0) or 0
            usage[y, x] = 100.0 * cb / max_l1

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.cm.RdYlGn_r  # green=low, yellow=mid, red=high
    masked = np.ma.masked_invalid(usage)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for y in range(height):
        for x in range(width):
            v = usage[y, x]
            if not np.isnan(v):
                color = "white" if v > 55 else "black"
                ax.text(
                    x, y, f"{v:.0f}%", ha="center", va="center", fontsize=max(5, min(9, int(cell_w * 10))), color=color
                )

    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.set_xticklabels([str(x) for x in range(width)], fontsize=8)
    ax.set_yticklabels([str(y) for y in range(height)], fontsize=8)
    ax.set_xlabel("Core x (column)", fontsize=9)
    ax.set_ylabel("Core y (row)", fontsize=9)
    ax.set_title(
        "True occupied SRAM (% of L1) per core\n" "(cb_region_end + top-down tensor buffers) / max_l1", fontsize=10
    )
    cb1 = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb1.set_label("% of L1", fontsize=8)
    plt.tight_layout()
    out1 = out_dir / "l1_sram_usage_heatmap.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"  Saved {out1}")

    # ── Plot 2: number of programs per core ─────────────────────────────────
    n_progs = np.zeros((height, width), dtype=int)
    core_to_pids: dict[str, set] = defaultdict(set)
    for pid, core_map in per_program.items():
        for core_k in core_map:
            core_to_pids[core_k].add(pid)
    for k, pids in core_to_pids.items():
        x, y = (int(v) for v in k.strip("()").split(","))
        if 0 <= x < width and 0 <= y < height:
            n_progs[y, x] = len(pids)

    vmax = int(n_progs.max()) if n_progs.max() > 0 else 1

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap2 = plt.cm.YlOrRd
    im2 = ax.imshow(n_progs, cmap=cmap2, vmin=0, vmax=vmax, aspect="auto")

    for y in range(height):
        for x in range(width):
            v = int(n_progs[y, x])
            if v > 0:
                color = "white" if v > vmax * 0.6 else "black"
                ax.text(x, y, str(v), ha="center", va="center", fontsize=max(5, min(9, int(cell_w * 10))), color=color)

    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.set_xticklabels([str(x) for x in range(width)], fontsize=8)
    ax.set_yticklabels([str(y) for y in range(height)], fontsize=8)
    ax.set_xlabel("Core x (column)", fontsize=9)
    ax.set_ylabel("Core y (row)", fontsize=9)
    ax.set_title("Number of programs using each core", fontsize=10)
    cb2 = plt.colorbar(im2, ax=ax, fraction=0.03, pad=0.02)
    cb2.set_label("# programs", fontsize=8)
    plt.tight_layout()
    out2 = out_dir / "l1_program_count_heatmap.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"  Saved {out2}")


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
    pid_to_op = parse_op_map(text)
    wide_bbox_map = parse_wide_bbox_map(text)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_program_annotated = {
        str(pid): {
            "op_name": pid_to_op.get(pid, "unknown"),
            "cores": cores,
            **({"wide_bbox_info": wide_bbox_map[pid]} if pid in wide_bbox_map else {}),
        }
        for pid, cores in per_program.items()
    }
    (args.out_dir / "per_program_core_map.json").write_text(json.dumps(per_program_annotated, indent=2))
    (args.out_dir / "cumulative_core_map.json").write_text(json.dumps(dict(sorted(cumulative.items())), indent=2))
    (args.out_dir / "headroom_map.json").write_text(json.dumps(dict(sorted(headroom.items())), indent=2))

    max_l1 = max(r["max_l1_size"] for r in rows)
    (args.out_dir / "l1_heatmap.md").write_text(
        heatmap_md(cumulative, headroom, max_l1, pid_to_op, per_program=per_program)
    )

    plot_heatmaps(cumulative, headroom, per_program, args.out_dir, max_l1)

    # Print summary table to stdout
    print(f"Parsed {len(rows)} L1_CB_MAP lines -> {args.out_dir}")
    print(f"  programs seen (validate): {sorted(per_program.keys())}")
    print(f"  cores in cumulative map : {len(cumulative)}")
    print()

    if pid_to_op:
        print(f"  pid→op mapping ({len(pid_to_op)} entries):")
        for pid_key, op_name in sorted(pid_to_op.items()):
            print(f"    pid={pid_key:>4}  {op_name}")
        print()

    if wide_bbox_map:
        print(f"  wide-bbox programs ({len(wide_bbox_map)} entries):")
        for pid_key, info in sorted(wide_bbox_map.items()):
            op = pid_to_op.get(pid_key, "unknown")
            bs, be = info["bbox_start_xy"], info["bbox_end_xy"]
            print(
                f"    pid={pid_key:>4}  {op}  "
                f"M={info['M_tiles']} K={info['K_tiles']} N={info['N_tiles']}  "
                f"storage_cores={info['storage_cores']}  "
                f"bbox=[({bs[0]},{bs[1]})..({be[0]},{be[1]})]"
            )
        print()

    print("  Per-core summary (worst-case execution moment, sorted by tightest gap):")
    print(
        f"  {'core':<10} {'pid':>5}  {'op_name':<40} {'local_cb':>10} {'local%':>7} {'top_down':>10} {'gap':>10} {'total%':>8}"
    )
    print(f"  {'-'*10} {'-'*5}  {'-'*40} {'-'*10} {'-'*7} {'-'*10} {'-'*10} {'-'*8}")
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
        op = pid_to_op.get(pid, "unknown") if pid is not None else ""
        local_pct = f"{hr.get('local_cb_pct_of_l1', 0):.1f}%"
        top_down = hr.get("top_down_size_bytes")
        gap = hr.get("gap_bytes_free_headroom")
        total_pct = f"{hr.get('total_occupied_pct_of_l1', 0):.2f}%"

        td_str = f"{top_down:>10}" if top_down is not None else f"{'n/a':>10}"
        gap_str = f"{gap:>10}" if gap is not None else f"{'n/a':>10}"
        pid_str = f"{pid:>5}" if pid is not None else f"{'n/a':>5}"
        print(f"  {k:<10} {pid_str}  {op:<40} {local_cb:>10} {local_pct:>7} {td_str} {gap_str} {total_pct:>8}")


if __name__ == "__main__":
    main()

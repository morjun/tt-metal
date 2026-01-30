#!/usr/bin/env python3
"""
Visualize RISC-V core timeline from profile_log_device.csv.
Generates a visual timeline of the 5 RISC-V processors for a specific core.
"""

import csv
import argparse
import sys
from collections import defaultdict
from pathlib import Path
import bisect
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import copy

# Default frequency if not found in header
DEFAULT_FREQ_MHZ = 1350
global_iter_ids = set()


def compress_timeline(events, gap_threshold_cycles=None, freq_mhz=1200):
    """
    이벤트가 없는 긴 공백(Gap)을 제거하여 타임라인을 압축합니다.

    Args:
        gap_threshold_cycles: 이 값보다 긴 공백은 잘라냅니다. (기본값: 1us에 해당하는 사이클)
    """
    time_ms = 0.001
    if gap_threshold_cycles is None:
        # 기본값: 0.1ms (1ms = freq * 1000 cycles)
        # 예: 1200MHz -> 120,000 cycles
        gap_threshold_cycles = int(freq_mhz * 1000 * time_ms)

    # 1. 모든 이벤트를 수집하여 '유효 구간(Interval)'을 만듭니다.
    intervals = []
    for core_id, riscs in events.items():
        for risc, zones in riscs.items():
            for z in zones:
                intervals.append((z["start"], z["end"]))

    if not intervals:
        return events

    # 2. 구간 병합 (Merge Intervals)
    intervals.sort()
    merged = []
    if intervals:
        curr_start, curr_end = intervals[0]
        for start, end in intervals[1:]:
            # 공백이 임계값보다 작으면 그냥 하나의 덩어리로 합침
            if start < curr_end + gap_threshold_cycles:
                curr_end = max(curr_end, end)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = start, end
        merged.append((curr_start, curr_end))

    # 3. 좌표 변환 (Shift Calculation)
    # 각 유효 구간 사이의 긴 공백을 제거하고, 데이터를 앞으로 당깁니다.
    # 시각적 구분을 위해 약간의 패딩(Gap Padding)을 둡니다.

    # padding = gap_threshold_cycles // 5
    padding = 0

    valid_windows = []
    current_new_start = 0

    for start, end in merged:
        duration = end - start
        valid_windows.append(
            {"orig_start": start, "orig_end": end, "new_start": current_new_start, "shift": start - current_new_start}
        )
        current_new_start += duration + padding

    # 4. 이벤트 좌표 업데이트
    new_events = copy.deepcopy(events)

    for core_id, riscs in new_events.items():
        for risc, zones in riscs.items():
            for z in zones:
                # 현재 Zone이 속한 유효 구간을 찾아서 shift 만큼 뺍니다.
                for w in valid_windows:
                    if w["orig_start"] <= z["start"] and z["end"] <= w["orig_end"]:
                        z["start"] -= w["shift"]
                        z["end"] -= w["shift"]
                        break

    print(f"[Info] Timeline compressed. Total duration reduced to {current_new_start / (freq_mhz*1000):.2f} ms")
    return new_events


def parse_profile_events(csv_path):
    """
    Parse profile_log_device.csv.

    Strategy:
    1. Collect ALL zone events (start, end, name, core, risc).
    2. Collect ALL iteration markers (TS_DATA) for each RISC (independently).
    3. Post-process: Assign iteration ID to each event based on its start time
       and the specific markers for that RISC.

    Returns:
        events[core_id][risc_type] = list of {start, end, name, iter_id}
        freq_mhz (float)
    """

    # Raw storage
    # events_raw[core_id][risc_type] = list of (start, end, name)
    events_raw = defaultdict(lambda: defaultdict(list))

    # risc_markers[(core_id, risc_type)] = list of (timestamp, iter_id)
    risc_markers = defaultdict(list)

    # Stack for nested zones
    zone_stack = {}  # key: (core_id, risc_type)

    freq_mhz = DEFAULT_FREQ_MHZ

    print(f"Parsing {csv_path}...")

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            # Read header
            header_info = f.readline()
            if "CHIP_FREQ[MHz]" in header_info:
                try:
                    part = header_info.split("CHIP_FREQ[MHz]:")[1].split(",")[0].strip()
                    freq_mhz = float(part)
                    print(f"Detected Frequency: {freq_mhz} MHz")
                except:
                    pass

            f.readline()  # Skip columns

            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 12:
                    continue

                try:
                    core_x, core_y = int(parts[1]), int(parts[2])
                    risc_type = parts[3]  # e.g. BRISC, TRISC_0
                    cycles = int(parts[5])
                    zone_name = parts[10]
                    zone_type = parts[11]
                except:
                    continue

                core_id = (core_x, core_y)

                # 1. Collect Markers
                if zone_type == "TS_DATA":
                    try:
                        val = int(parts[6])
                        # Filter trash values: Valid IDs are usually small (e.g. < 10000)
                        # if 0 <= val < 10000:
                        risc_markers[(core_id, risc_type)].append((cycles, val))
                        global_iter_ids.add(val)
                    except:
                        pass
                    continue

                # 2. Collect Zones
                stack_key = (core_id, risc_type)

                if zone_type == "ZONE_START":
                    if stack_key not in zone_stack:
                        zone_stack[stack_key] = []
                    zone_stack[stack_key].append((zone_name, cycles))

                elif zone_type == "ZONE_END":
                    if stack_key in zone_stack and zone_stack[stack_key]:
                        top_name, start_cycle = zone_stack[stack_key][-1]
                        if top_name == zone_name:
                            zone_stack[stack_key].pop()
                            # Store raw event
                            events_raw[core_id][risc_type].append(
                                {"start": start_cycle, "end": cycles, "name": zone_name}
                            )

    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        sys.exit(1)

    print("Assigning Iteration IDs based on timestamps (Per RISC)...")

    # Post-process: Assign IDs
    for key in risc_markers:
        risc_markers[key].sort(key=lambda x: x[0])

    final_events = defaultdict(lambda: defaultdict(list))

    for core_id, riscs in events_raw.items():
        for risc_type, zone_list in riscs.items():
            # Retrieve markers specifically for this RISC
            markers = risc_markers.get((core_id, risc_type), [])

            marker_times = [m[0] for m in markers]
            marker_ids = [m[1] for m in markers]

            for z in zone_list:
                # Find the marker that started BEFORE (or at) z['start']
                idx = bisect.bisect_right(marker_times, z["start"])
                if idx > 0:
                    iter_id = marker_ids[idx - 1]
                else:
                    # Before first marker -> Assume 'Warmup' or 'Setup' (-1)
                    iter_id = -1

                z["iter_id"] = iter_id
                final_events[core_id][risc_type].append(z)

    return final_events, freq_mhz


def get_busiest_core(events):
    max_events = 0
    busiest_core = None
    for core_id, riscs in events.items():
        count = sum(len(x) for x in riscs.values())
        if count > max_events:
            max_events = count
            busiest_core = core_id
    return busiest_core


def generate_colors(iter_ids):
    sorted_ids = sorted(list(iter_ids))
    # Use tab20 for more contrast (instead of tab20c)
    cmap = plt.get_cmap("tab20")

    id_to_color = {}

    # We want consistent colors for groups.
    # For IDs < 100 (Large Batch), distinct color per ID.
    # For IDs >= 100 (Mini Batch), group by tens (100-109 -> Group 10, 110-119 -> Group 11).

    # Helper to get color index
    def get_color_key(iid):
        if iid < 100:
            return iid
        else:
            return iid // 10

    # Collect unique keys to assign stable colors
    unique_keys = sorted(list(set(get_color_key(iid) for iid in sorted_ids if iid > 0)))
    key_to_idx = {k: i for i, k in enumerate(unique_keys)}

    # tab20 layout: 14-15 are Gray. We exclude them.
    safe_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19]
    n_safe = len(safe_indices)

    for iid in sorted_ids:
        if iid == 0:
            id_to_color[iid] = "#DDDDDD"  # Gray for Warmup
        elif iid < 0 or iid >= 1000:
            id_to_color[iid] = "#AAAAAA"  # Darker Gray for Unknown
        else:
            key = get_color_key(iid)
            idx = key_to_idx[key]
            # Use a prime stride (e.g., 7) to jump around the safe list
            mapped_idx = (idx * 7) % n_safe
            real_idx = safe_indices[mapped_idx]
            id_to_color[iid] = cmap(real_idx)

    return id_to_color


def plot_timeline(events, core_id, freq_mhz, output_path):
    # [NEW] 압축 로직 적용
    # 0.1ms (100us) 이상의 공백은 잘라냅니다.
    print("Applying timeline compression...")
    events = compress_timeline(events, gap_threshold_cycles=1, freq_mhz=freq_mhz)
    if core_id not in events:
        print(f"No events for Core {core_id}")
        return

    risc_data = events[core_id]
    risc_order = ["NCRISC", "BRISC", "TRISC_0", "TRISC_1", "TRISC_2"]

    RISC_ROLE_MAP = {
        "NCRISC": "WEIGHT READER",
        "BRISC": "DATA READER & DISPATCHER",
        "TRISC_0": "UNPACKER",
        "TRISC_1": "MATH",
        "TRISC_2": "PACKER",
    }

    # Filter out basic overlapping zones AFTER compression
    # This ensures that gaps created by large FW/Kernel blocks are preserved as real time
    IGNORED_ZONES = {"BRISC-FW", "BRISC-KERNEL", "NCRISC-FW", "NCRISC-KERNEL", "TRISC-FW", "TRISC-KERNEL"}

    # Collect distinct (risc, zone_name) pairs and their events
    # We want to maintain risc_order, and then maybe sort zones alphabetically or by appearance?
    # Let's sort zones alphabetically for consistency.

    zone_map = defaultdict(list)  # Key: (risc, zone_name) -> list of events

    # Filter valid
    all_starts = []
    all_ends = []
    all_iter_ids = set()

    for r in risc_order:
        # Loop over ALL zones first
        raw_zones = risc_data.get(r, [])
        filtered_zones_for_plot = []

        for z in raw_zones:
            # Always collect ID
            all_iter_ids.add(z["iter_id"])

            if z["name"] in IGNORED_ZONES:
                # Log usage
                # print(f"[Info] Ignored zone: {z['name']} (Iter: {z['iter_id']})")
                pass
            else:
                filtered_zones_for_plot.append(z)

        # Update the list in place or usage
        risc_data[r] = filtered_zones_for_plot

        for ev in filtered_zones_for_plot:
            if ev["end"] > ev["start"]:
                all_starts.append(ev["start"])
                all_ends.append(ev["end"])
            else:
                print(f"Invalid zone: {ev}")

            zone_map[(r, ev["name"])].append(ev)

    if not all_starts:
        print("No valid time range.")
        return

    global_min = min(all_starts)
    global_max = max(all_ends)

    print(f"Time Range: {global_min} - {global_max} ({global_max - global_min} cycles)")
    print(f"Iterations: {sorted(all_iter_ids)}")
    print(f"Global Iterations: {sorted(global_iter_ids)}, Count: {len(global_iter_ids)}")

    colors = generate_colors(all_iter_ids)

    # Calculate min start time for each zone to order them by execution flow
    zone_min_start = {}
    for (r, zname), ev_list in zone_map.items():
        if ev_list:
            zone_min_start[(r, zname)] = min(ev["start"] for ev in ev_list)
        else:
            zone_min_start[(r, zname)] = float("inf")

    # Prepare Y-axis rows
    # Rows should be ordered by RISC type, then by execution order (Timestamp)
    row_keys = []
    for r in risc_order:
        # Find all zones for this RISC
        zones_for_risc = list(set(k[1] for k in zone_map.keys() if k[0] == r))
        # Sort by min start time
        zones_for_risc.sort(key=lambda z: zone_min_start.get((r, z), float("inf")))

        for zname in zones_for_risc:
            row_keys.append((r, zname))

    # Calculate figure size dynamically?
    # Approx 0.8 inches per row
    row_height = 0.8
    total_height = max(8, len(row_keys) * row_height)

    fig, ax = plt.subplots(figsize=(24, total_height))

    y_ticks = []
    y_labels = []

    # Plot rows
    # We plot from top to bottom, so index 0 is at top? No, matplotlib 0 is bottom.
    # So we iterate reversed if we want RISC order top-down.

    for i, (risc, zone_name) in enumerate(reversed(row_keys)):
        y_center = i
        y_ticks.append(y_center)
        # Label: "RISC\n(Role)\nZone"
        role = RISC_ROLE_MAP.get(risc, risc)
        dname = zone_name.replace("KERNEL_", "").replace("ZONE_", "")
        y_labels.append(f"{risc}\n({role})\n{dname}")

        events_list = zone_map[(risc, zone_name)]

        for ev in events_list:
            s_cyc = ev["start"]
            e_cyc = ev["end"]

            # Skip noise < 2000 cycles (~1-2 us)
            # if (e_cyc - s_cyc) < 2000:
            #     continue

            start_ms = (s_cyc - global_min) / (freq_mhz * 1000.0)
            dur_ms = (e_cyc - s_cyc) / (freq_mhz * 1000.0)

            iid = ev["iter_id"]
            c = colors.get(iid, "black")

            # Draw
            ax.broken_barh(
                [(start_ms, dur_ms)], (y_center - 0.4, 0.8), facecolors=c, edgecolor="none", linewidth=0, alpha=1.0
            )

            # Label inside? User said "Leave the labels on it".
            # We already have row labels, but let's keep iteration info or name if huge.
            # if dur_ms > 0.1:
            #     ax.text(
            #         start_ms + dur_ms / 2,
            #         y_center,
            #         f"{dname}\n({iid})",
            #         ha="center",
            #         va="center",
            #         fontsize=8,
            #         color="white" if iid > 0 else "black",
            #         clip_on=True,
            #         fontweight="bold",
            #     )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"RISC-V Timeline - Core {core_id}")
    ax.grid(True, axis="x", which="both", linestyle="--", alpha=0.5)
    # Add horizontal grid to separate rows
    ax.grid(True, axis="y", which="major", linestyle="-", alpha=0.3)
    # ---------------------------------------------------------
    # LEGEND GENERATION
    # ---------------------------------------------------------

    # 1. Warmup
    warmup_handles = []
    if any(i == 0 for i in all_iter_ids):
        c = colors.get(0, "#DDDDDD")
        warmup_handles.append(mpatches.Patch(color=c, label="Warmup"))

    # 2. Unknown
    unknown_handles = []
    if any(i < 0 or i >= 1000 for i in all_iter_ids):
        c = colors.get(-1, "#DDDDDD")
        unknown_handles.append(mpatches.Patch(color=c, label="Unknown"))

    # 2. Large Batches (IDs < 100, > 0)
    lb_ids = sorted([i for i in all_iter_ids if 0 < i < 100])
    lb_handles = []
    for idx, iid in enumerate(lb_ids):
        # Index from 1
        label = f"Iter {idx + 1}"
        lb_handles.append(mpatches.Patch(color=colors[iid], label=label))

    # 3. Mini Batches (IDs >= 100)
    mb_ids = sorted([i for i in all_iter_ids if i >= 100 and i < 1000])
    # Group by tens
    mb_groups = sorted(list(set(i // 10 for i in mb_ids)))
    mb_handles = []
    for idx, group_key in enumerate(mb_groups):
        # Find representative ID
        rep_id = next(i for i in mb_ids if i // 10 == group_key)
        # Index from 1
        label = f"Seq {idx + 1}"
        mb_handles.append(mpatches.Patch(color=colors[rep_id], label=label))

    # Add legends
    # We place them vertically on the right side
    extra_artists = []

    # Warmup (Top)
    if warmup_handles:
        l1 = ax.legend(handles=warmup_handles, title="Status", bbox_to_anchor=(1.01, 1.0), loc="upper left")
        ax.add_artist(l1)
        extra_artists.append(l1)

    if unknown_handles:
        l2 = ax.legend(handles=unknown_handles, title="Status", bbox_to_anchor=(1.01, 0.95), loc="upper left")
        ax.add_artist(l2)
        extra_artists.append(l2)

    # Large Batches (Below Warmup - est Y=0.9)
    if lb_handles:
        l2 = ax.legend(handles=lb_handles, title="Large Batches", bbox_to_anchor=(1.01, 0.90), loc="upper left")
        ax.add_artist(l2)
        extra_artists.append(l2)

    # Mini Batches (Below Large Batches - est Y=0.7)
    if mb_handles:
        l3 = ax.legend(
            handles=mb_handles, title="Mini Batches (Grouped)", bbox_to_anchor=(1.01, 0.60), loc="upper left"
        )
        ax.add_artist(l3)
        extra_artists.append(l3)

    # Reserve space on the right for legends
    plt.subplots_adjust(right=0.85)

    # Use bbox_extra_artists to ensure legends are included
    plt.savefig(output_path, dpi=150, bbox_extra_artists=extra_artists, bbox_inches="tight")
    print(f"Saved to {output_path}")
    # plt.show() # caused issues in non-interactive envs sometimes, can comment out or leave
    plt.close(fig)  # Close to free memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="research_codes/profile_log_device_minimized_gemm_sharding_w0m1_minionly.csv",
    )
    parser.add_argument("--output", "-o", default="riscv_timeline.png")
    parser.add_argument("--core", nargs=2, type=int)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        # Fallback
        csv_path = Path.cwd() / args.csv_path

    events, freq = parse_profile_events(csv_path)

    target = None
    if args.core:
        target = tuple(args.core)
    else:
        target = get_busiest_core(events)
        print(f"Auto-selected Core: {target}")

    if target:
        plot_timeline(events, target, freq, args.output)
    else:
        print("No data.")


if __name__ == "__main__":
    main()

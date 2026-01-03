#!/usr/bin/env python3
"""
Analyze detailed profiling zones from profile_log_device.csv.

Features:
- Calculates timestamp-based durations.
- Aggregates metrics "Per Core" (Average across active cores) to reflect wall-clock contribution.
- Categorizes zones into Weight Streaming, NoC/Sync, and Compute.
- Groups by Source File to distinguish Matmul/Transpose/Add kernels.
- Handles nested zones (Sub-zones vs Top-level).

Usage:
    python3 analyze_detailed_zones.py [csv_path]

Default: generated/profiler/.logs/profile_log_device.csv
"""

import csv
import sys
import argparse
from collections import defaultdict
from pathlib import Path

# Frequency in MHz (Blackhole = 1350, Grayskull = 1200, Wormhole = 1000)
# Ideally read from CSV header.
DEFAULT_FREQ_MHZ = 1350


def parse_detailed_profile(csv_path):
    """Parse profile_log_device.csv with custom zones."""

    # Structure: cores[core_id][risc_type][(zone_name, src_file, iter_id)] = [duration, duration, ...]
    cores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    zone_stack = {}  # Key: (core_id, risc_type) -> list of (zone_name, start_cycle)

    # Track current iteration ID per core/RISC
    core_iter_map = {}  # Key: (core_id, risc_type) -> current_iter_id

    freq_mhz = DEFAULT_FREQ_MHZ

    print(f"Parsing {csv_path}...")

    with open(csv_path, "r", encoding="utf-8") as f:
        # Read header to find frequency
        header_info = f.readline()
        if "CHIP_FREQ[MHz]" in header_info:
            try:
                part = header_info.split("CHIP_FREQ[MHz]:")[1].split(",")[0].strip()
                freq_mhz = float(part)
                print(f"Detected Frequency: {freq_mhz} MHz")
            except:
                pass

        # Skip column names
        f.readline()

        line_count = 0
        for line in f:
            line_count += 1
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 14:
                continue

            try:
                # CSV Format:
                # 0:PCIe, 1:CoreX, 2:CoreY, 3:RISC, 4:TimerID, 5:Time, 6:Data, ...
                # 10: ZoneName, 11: ZoneType, 12: SourceLine, 13: SourceFile
                core_x, core_y = int(parts[1]), int(parts[2])
                risc_type = parts[3]
                cycles = int(parts[5])
                zone_name = parts[10]
                zone_type = parts[11]
                src_file = parts[13]
            except (ValueError, IndexError):
                continue

            if not zone_name or not zone_type:
                continue

            core_id = (core_x, core_y)
            stack_key = (core_id, risc_type)

            src_file_base = Path(src_file).name

            if zone_type == "ZONE_START":
                if stack_key not in zone_stack:
                    zone_stack[stack_key] = []
                zone_stack[stack_key].append((zone_name, cycles, src_file_base))

            elif zone_type == "ZONE_END":
                if stack_key in zone_stack and zone_stack[stack_key]:
                    top_name, start_cycle, top_src = zone_stack[stack_key][-1]

                    if top_name == zone_name:
                        zone_stack[stack_key].pop()
                        duration = cycles - start_cycle

                        # Get current iteration ID
                        iter_id = core_iter_map.get(stack_key, 0)  # Default to 0 if not set

                        # Store by (ZoneName, SrcFile, IterID)
                        cores[core_id][risc_type][(zone_name, top_src, iter_id)].append(duration)
                    else:
                        pass

            elif zone_type == "TS_DATA":
                try:
                    data_val = int(parts[6])
                    core_iter_map[stack_key] = data_val
                    # print(f"DEBUG: Found TS_DATA {data_val} on Core {core_x},{core_y} {risc_type} at {cycles}")
                except:
                    pass

        print(f"Processed {line_count} lines")

    return cores, freq_mhz


def classify_zone(zone_name, src_file):
    """Classify zone into categories based on name and source file."""
    zn = zone_name.upper()
    sf = src_file.upper()

    # KERNEL Type Classification based on Source File
    kernel_type = "OTHER"
    if "MATMUL" in sf or "BMM" in sf:
        kernel_type = "MATMUL"
    elif "TRANSPOSE" in sf:
        kernel_type = "TRANSPOSE"
    elif "ADD" in sf or "BINARY" in sf:
        kernel_type = "ADD"

    if "READ-WEIGHT" in zn:
        return f"WEIGHT_STREAM_{kernel_type}"
    if "READ-IN0" in zn:
        return f"ACT_STREAM_{kernel_type}"
    if "NOC" in zn or "BARRIER" in zn or "WAIT" in zn or "SYNC" in zn or "SEM-" in zn:
        # Check specific wait types
        if "CB-WAIT" in zn:
            return f"DATA_WAIT_{kernel_type}"  # Waiting for data in CB
        return f"NOC_WAIT_{kernel_type}"

    if "MATMUL" in zn or "MATH" in zn or "UNPACK" in zn or "PACK" in zn:
        return f"COMPUTE_{kernel_type}"

    # Generic "KERNEL" top level
    if "BRISC-KERNEL" in zn or "NCRISC-KERNEL" in zn or "TRISC-KERNEL" in zn:
        return f"KERNEL_{kernel_type}"

    if "FW" in zn and "KERNEL" not in zn:
        return "FIRMWARE"

    return f"SUBZONE_{kernel_type}"


def analyze_breakdown(cores, freq_mhz):
    """Aggregate and print metrics."""

    if not cores:
        print("No data found.")
        return

    # Structure: duration_per_iter[iter_id][core_id][risc][(zone, src)] = sum(durations)
    duration_per_iter = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))

    for core_id, riscs in cores.items():
        for risc, zones in riscs.items():
            for (z_name, src_file, iter_id), durations in zones.items():
                total_dur = sum(durations)
                duration_per_iter[iter_id][core_id][risc][(z_name, src_file)] += total_dur

    # Group IDs
    groups = {"LARGE_BATCH (IDs < 100)": [], "MINIBATCH (IDs >= 100)": [], "IGNORED/WARMUP": []}

    for iter_id in sorted(duration_per_iter.keys()):
        if iter_id <= 0 or iter_id > 1000000:
            groups["IGNORED/WARMUP"].append(iter_id)
        elif iter_id < 100:
            groups["LARGE_BATCH (IDs < 100)"].append(iter_id)
        else:
            groups["MINIBATCH (IDs >= 100)"].append(iter_id)

    print(f"\n{'-'*130}")
    print(f"ITERATION ANALYSIS (Freq: {freq_mhz} MHz) - Grouped by Iteration ID")
    print(f"Metrics are averaged across iterations in each group.")
    print(f"Bottleneck analysis: Max duration across all cores.")
    print(f"{'-'*130}")

    for group_name, iter_ids in groups.items():
        if not iter_ids:
            continue

        print(f"\n>>> Group: {group_name} (Count: {len(iter_ids)}, IDs: {sorted(iter_ids)})")

        # Collect all (zone, src, risc) keys present in this group
        all_keys = set()
        for iid in iter_ids:
            for core in duration_per_iter[iid]:
                for risc in duration_per_iter[iid][core]:
                    for k in duration_per_iter[iid][core][risc]:
                        all_keys.add((k, risc))

        if not all_keys:
            print("  No data recorded.")
            continue

        print(f"{'RISC':<15} | {'ZONE':<45} | {'AVG TIME (ms)':<15} | {'MAX TIME (ms)':<15}")
        print(f"{'-'*100}")

        # Calculate stats for each zone type
        results = []
        for (z_name, src), risc in all_keys:
            total_time_group_max = 0.0  # Sum of max-core-times across iterations
            global_max_single_iter = 0.0

            for iid in iter_ids:
                # Find max duration across cores for this iid and zone
                max_core_dur = 0.0
                for core in duration_per_iter[iid]:
                    if risc in duration_per_iter[iid][core]:
                        val = duration_per_iter[iid][core][risc].get((z_name, src), 0)
                        if val > max_core_dur:
                            max_core_dur = val

                total_time_group_max += max_core_dur
                if max_core_dur > global_max_single_iter:
                    global_max_single_iter = max_core_dur

            avg_time_ms = (total_time_group_max / len(iter_ids)) / (freq_mhz * 1000)
            max_time_ms = global_max_single_iter / (freq_mhz * 1000)

            results.append((risc, z_name, avg_time_ms, max_time_ms))

        # Sort by avg time desc
        results.sort(key=lambda x: x[2], reverse=True)

        for risc, z_name, avg_t, max_t in results:
            print(f"{risc:<15} | {z_name:<45} | {avg_t:<15.4f} | {max_t:<15.4f}")

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default="generated/profiler/.logs/profile_log_device.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        # Try default location if relative path fails
        alt_path = Path(f"/home/masterjunmo/codes/tt-metal/{args.csv_path}")
        if alt_path.exists():
            csv_path = alt_path
        else:
            print(f"Error: {csv_path} not found.")
            return 1

    cores, freq = parse_detailed_profile(csv_path)
    analyze_breakdown(cores, freq)
    return 0


if __name__ == "__main__":
    main()

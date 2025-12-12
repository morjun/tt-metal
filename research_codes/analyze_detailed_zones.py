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

    # Structure: cores[core_id][risc_type][(zone_name, src_file)] = [duration, duration, ...]
    cores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    zone_stack = {}  # Key: (core_id, risc_type) -> list of (zone_name, start_cycle)

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
                # Some lines might be truncated or old format
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
                        # Store by (ZoneName, SrcFile) to distinguish same named zones in diff files
                        cores[core_id][risc_type][(zone_name, top_src)].append(duration)
                    else:
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

    # Structure: zone_data[risc][(zone_name, src_file)] = list of lists
    zone_data = defaultdict(lambda: defaultdict(list))

    for core_id, riscs in cores.items():
        for risc, zones in riscs.items():
            for (z_name, src_file), durations in zones.items():
                zone_data[risc][(z_name, src_file)].append(durations)

    print(f"\n{'-'*130}")
    print(f"ANALYSIS REPORT (Freq: {freq_mhz} MHz) - Grouped by Source File")
    print(f"Max Time = Latency of the slowest core (Bottleneck Analysis)")
    print(f"{'-'*130}")

    # Track metrics for final summary
    metrics = {
        "compute_loop": 0.0,
        "compute_stall": 0.0,
        "weight_read_issue": 0.0,  # Active BW usage (BRISC)
        "weight_read_wait": 0.0,  # Latency tail (BRISC)
        "act_read_issue": 0.0,  # Active BW usage (NCRISC)
        "act_read_wait": 0.0,  # Latency tail (NCRISC)
    }

    for risc in sorted(zone_data.keys()):
        print(f"\n[{risc}]")
        print(f"{'Category':<25} {'Zone Name':<45} {'Source File':<35} {'Max(ms)':<10} {'Avg(ms)':<10}")
        print(f"{'-'*130}")

        items = []
        for (z_name, src_file), core_dur_lists in zone_data[risc].items():
            # core_dur_lists is a list of lists (one per core)
            # Calculate total time per core
            total_time_per_core = [sum(d) for d in core_dur_lists]

            # Global stats
            max_cycles = max(total_time_per_core)
            avg_cycles = sum(total_time_per_core) / len(total_time_per_core)

            max_ms = max_cycles / (freq_mhz * 1000)
            avg_ms = avg_cycles / (freq_mhz * 1000)

            cat = classify_zone(z_name, src_file)

            items.append({"name": z_name, "src": src_file, "cat": cat, "max_ms": max_ms, "avg_ms": avg_ms})

            # Capture key metrics for inferred breakdown
            # TRISC Side
            if "BATCH-ITERATION" in z_name:
                metrics["compute_loop"] = max(metrics["compute_loop"], max_ms)
            if "CB-WAIT" in z_name:
                metrics["compute_stall"] = max(metrics["compute_stall"], max_ms)

            # BRISC Side (Weights)
            if "READ-WEIGHT" in z_name:
                metrics["weight_read_issue"] = max(metrics["weight_read_issue"], max_ms)
            if "NOC-BARRIER-WAIT-IN1" in z_name:  # Specific to IN1 (Weights)
                metrics["weight_read_wait"] = max(metrics["weight_read_wait"], max_ms)

            # NCRISC Side (Activations)
            if "READ-IN0" in z_name:
                metrics["act_read_issue"] = max(metrics["act_read_issue"], max_ms)
            if "NOC-BARRIER-WAIT-IN0" in z_name:  # Specific to IN0 (Activations)
                metrics["act_read_wait"] = max(metrics["act_read_wait"], max_ms)

        # Sort by Category
        items.sort(key=lambda x: (x["cat"], -x["max_ms"]))

        for item in items:
            print(
                f"{item['cat']:<25} {item['name']:<45} {item['src']:<35} {item['max_ms']:<10.4f} {item['avg_ms']:<10.4f}"
            )

    # Component Breakdown
    # Present raw metrics for Producer (BRISC) and Consumer (TRISC)
    # Avoid inferring overlap efficiency or hidden overheads.

    if metrics["compute_loop"] > 0:
        print(f"\n{'-'*130}")
        print(f"COMPONENT BREAKDOWN (Trace Analysis)")
        print(f"Note: Producer (BRISC) and Consumer (TRISC) run in parallel.")
        print(f"      Metrics are max latency across all cores.")
        print(f"{'-'*130}")

        total = metrics["compute_loop"]
        stall = metrics["compute_stall"]

        # 1. Pure Compute
        pure_compute = max(0, total - stall)

        # 2. DRAM Data Fetch (Weights)
        weight_issue = metrics["weight_read_issue"]  # Active BW
        weight_latency = metrics["weight_read_wait"]  # Tail Latency

        # 3. DRAM Data Fetch (Activations)
        act_issue = metrics["act_read_issue"]
        act_latency = metrics["act_read_wait"]

        # 4. NoC Multicast
        # We need to capture the new zone if it exists, or it will be 0
        # Re-iterate to find the zone in ANY risc (it should be in BRISC)
        noc_mcast_exact = 0.0

        # Helper to find zone in data
        for risc in zone_data:
            for (z_name, src_file), core_dur_lists in zone_data[risc].items():
                if "WEIGHT-STREAM-MCAST" in z_name:
                    total_time_per_core = [sum(d) for d in core_dur_lists]
                    max_cycles = max(total_time_per_core)
                    max_ms = max_cycles / (freq_mhz * 1000)
                    noc_mcast_exact = max(noc_mcast_exact, max_ms)

        print(f"Total Forward Latency       : {total:.4f} ms")
        print(f"--------------------------------------------------")
        print(f"[CONSUMER / TRISC]")
        print(f"  > Pure Compute            : {pure_compute:.4f} ms")
        print(f"  > Data Wait Stall         : {stall:.4f} ms (Total Wait for Data)")
        print(f"--------------------------------------------------")
        print(f"[PRODUCER / BRISC] (Weights)")
        print(f"  > DRAM Read Issue         : {weight_issue:.4f} ms (Active DRAM BW)")
        print(f"  > DRAM Read Latency       : {weight_latency:.4f} ms (Wait for Return)")
        print(f"  > NoC Multicast (Stream)  : {noc_mcast_exact:.4f} ms (Active NoC BW)")
        print(f"--------------------------------------------------")
        print(f"[PRODUCER / NCRISC] (Activations)")
        print(f"  > DRAM Read Issue         : {act_issue:.4f} ms (Active DRAM BW)")
        print(f"  > DRAM Read Latency       : {act_latency:.4f} ms (Wait for Return)")
        print(f"--------------------------------------------------")


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

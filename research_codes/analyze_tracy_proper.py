#!/usr/bin/env python3
"""
Properly correlate Tracy zones with device operations using timestamps
"""
import pandas as pd
import numpy as np
from pathlib import Path

# File paths
tracy_ops_times_path = Path("/home/masterjunmo/codes/tt-metal/research_codes/tracy_output/.logs/tracy_ops_times.csv")
tracy_ops_data_path = Path("/home/masterjunmo/codes/tt-metal/research_codes/tracy_output/.logs/tracy_ops_data.csv")
device_profile_path = Path("/home/masterjunmo/codes/tt-metal/research_codes/tracy_output/.logs/profile_log_device.csv")


def parse_tracy_zones():
    """Parse Tracy zones from tracy_ops_times.csv"""
    zones = []
    with open(tracy_ops_times_path, "r") as f:
        for line in f:
            if line.startswith("LargeBatch_Forward") or line.startswith("MiniBatch_Forward"):
                parts = line.strip().split(",")
                zone_name = parts[0]
                ns_since_start = int(parts[5])
                exec_time_ns = int(parts[6])

                zones.append(
                    {
                        "name": zone_name,
                        "start_ns": ns_since_start,
                        "end_ns": ns_since_start + exec_time_ns,
                        "exec_time_ns": exec_time_ns,
                    }
                )

    # Sort by start time
    zones.sort(key=lambda x: x["start_ns"])
    return zones


def parse_device_operations():
    """Parse device operations from tracy_ops_data.csv with timestamps"""
    operations = []

    with open(tracy_ops_data_path, "r") as f:
        content = f.read()

    # Split by lines and parse Matmul operations with timestamps
    lines = content.split("\n")
    for line in lines:
        if "Matmul" in line and ";" in line:
            # Format: `TT_DNN_DEVICE_OP: "Matmul", hash, 0, run_id`;timestamp
            parts = line.split(";")
            if len(parts) == 2:
                # Extract run_id from first part
                first_part = parts[0]
                run_id_str = first_part.split(",")[-1].strip().rstrip(" ->")
                run_id = int(run_id_str)

                # Extract timestamp from second part
                timestamp_ns = int(parts[1].strip())

                operations.append({"op_type": "Matmul", "run_id": run_id, "timestamp_ns": timestamp_ns})

    # Sort by timestamp
    operations.sort(key=lambda x: x["timestamp_ns"])
    return operations


def correlate_operations_with_zones(zones, operations):
    """Map device operations to Tracy zones using timestamps"""
    zone_operations = {zone["name"]: [] for zone in zones}

    for op in operations:
        # Find which zone this operation belongs to
        for zone in zones:
            if zone["start_ns"] <= op["timestamp_ns"] <= zone["end_ns"]:
                zone_operations[zone["name"]].append(op["run_id"])
                break

    return zone_operations


def load_device_zones():
    """Load device profiler zones"""
    df = pd.read_csv(device_profile_path, header=None)

    # Columns: run_host_id (col 7), zone_name (col 10), type (col 11), cycle_count (col 12)
    df.columns = list(range(df.shape[1]))
    df["run_host_id"] = df[7]
    df["zone_name"] = df[10]
    df["type"] = df[11]
    df["cycle_count"] = df[12]

    return df[["run_host_id", "zone_name", "type", "cycle_count"]]


def calculate_zone_times(device_zones, run_ids, clock_mhz=1350):
    """Calculate time breakdown for specific run_host_ids"""
    filtered = device_zones[device_zones["run_host_id"].isin(run_ids)]

    # Filter for ZONE_START only to avoid double counting
    starts = filtered[filtered["type"] == "ZONE_START"].copy()

    # Categorize zones
    def categorize_zone(name):
        if "BRISC" in name:
            return "BRISC"
        elif "NCRISC" in name:
            return "NCRISC"
        elif "TRISC" in name:
            return "TRISC"
        return "OTHER"

    starts["category"] = starts["zone_name"].apply(categorize_zone)

    # Sum cycles by category
    cycles_by_category = starts.groupby("category")["cycle_count"].sum()

    # Convert to milliseconds
    time_ms = {}
    for category, cycles in cycles_by_category.items():
        time_ms[category] = (cycles / clock_mhz) / 1000.0  # MHz to ms

    total_time = sum(time_ms.values())

    return time_ms, total_time


def main():
    print("=" * 80)
    print("TRACY ZONE AND DEVICE OPERATION CORRELATION ANALYSIS")
    print("=" * 80)

    # Parse Tracy zones
    print("\n1. Parsing Tracy zones...")
    zones = parse_tracy_zones()
    print(f"   Found {len(zones)} Tracy zones")
    for zone in zones:
        print(
            f"   - {zone['name']}: {zone['start_ns']/1e9:.6f}s - {zone['end_ns']/1e9:.6f}s ({zone['exec_time_ns']/1e6:.3f} ms)"
        )

    # Parse device operations
    print("\n2. Parsing device operations from tracy_ops_data.csv...")
    operations = parse_device_operations()
    print(f"   Found {len(operations)} Matmul operations")
    print(f"   run_ids: {[op['run_id'] for op in operations]}")
    print(f"   Timestamp range: {operations[0]['timestamp_ns']/1e9:.6f}s - {operations[-1]['timestamp_ns']/1e9:.6f}s")

    # Correlate operations with zones
    print("\n3. Correlating device operations with Tracy zones...")
    zone_operations = correlate_operations_with_zones(zones, operations)

    print("\n4. Correlation Results:")
    for zone_name, run_ids in zone_operations.items():
        print(f"\n   {zone_name}:")
        print(f"   - Number of Matmul operations: {len(run_ids)}")
        print(f"   - run_ids: {run_ids}")

    # Load device zones
    print("\n5. Loading device profiler data...")
    device_zones = load_device_zones()
    print(f"   Loaded {len(device_zones)} device zone entries")

    # Analyze each Tracy zone
    print("\n" + "=" * 80)
    print("COMPONENT TIME BREAKDOWN BY TRACY ZONE")
    print("=" * 80)

    for zone in zones:
        zone_name = zone["name"]
        run_ids = zone_operations[zone_name]

        if not run_ids:
            print(f"\n{zone_name}: NO DEVICE OPERATIONS FOUND")
            continue

        print(f"\n{zone_name}:")
        print(f"  Tracy zone exec_time: {zone['exec_time_ns']/1e6:.3f} ms")
        print(f"  Number of Matmul operations: {len(run_ids)}")
        print(f"  Mapped run_ids: {run_ids}")

        time_ms, total_time = calculate_zone_times(device_zones, run_ids)

        print(f"\n  Component Breakdown (from device profiler):")
        for category in ["BRISC", "NCRISC", "TRISC"]:
            if category in time_ms:
                ms = time_ms[category]
                pct = (ms / total_time * 100) if total_time > 0 else 0
                print(f"    {category:8s}: {ms:8.3f} ms ({pct:5.1f}%)")

        if "OTHER" in time_ms:
            print(f"    {'OTHER':8s}: {time_ms['OTHER']:8.3f} ms")

        print(f"    {'TOTAL':8s}: {total_time:8.3f} ms")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Large batch
    large_batch_zone = [z for z in zones if "LargeBatch" in z["name"]][0]
    large_batch_ids = zone_operations[large_batch_zone["name"]]
    lb_time_ms, lb_total = calculate_zone_times(device_zones, large_batch_ids)

    print("\nLarge Batch (1 forward pass):")
    print(f"  Tracy exec_time: {large_batch_zone['exec_time_ns']/1e6:.3f} ms")
    print(f"  Number of Matmul ops: {len(large_batch_ids)}")
    for category in ["BRISC", "NCRISC", "TRISC"]:
        if category in lb_time_ms:
            pct = (lb_time_ms[category] / lb_total * 100) if lb_total > 0 else 0
            print(f"  {category}: {lb_time_ms[category]:.3f} ms ({pct:.1f}%)")

    # Mini batch - aggregate all 8 passes
    mini_batch_zones = [z for z in zones if "MiniBatch" in z["name"]]
    all_mini_ids = []
    for z in mini_batch_zones:
        all_mini_ids.extend(zone_operations[z["name"]])

    mb_time_ms, mb_total = calculate_zone_times(device_zones, all_mini_ids)
    total_tracy_time = sum(z["exec_time_ns"] for z in mini_batch_zones) / 1e6

    print("\nMini Batch (8 forward passes):")
    print(f"  Total Tracy exec_time: {total_tracy_time:.3f} ms")
    print(f"  Total number of Matmul ops: {len(all_mini_ids)}")
    print(f"  Number of forward passes: {len(mini_batch_zones)}")
    for category in ["BRISC", "NCRISC", "TRISC"]:
        if category in mb_time_ms:
            pct = (mb_time_ms[category] / mb_total * 100) if mb_total > 0 else 0
            print(f"  {category}: {mb_time_ms[category]:.3f} ms ({pct:.1f}%)")

    print("\nPer Mini Batch Forward Pass (average):")
    avg_tracy_time = total_tracy_time / len(mini_batch_zones)
    print(f"  Avg Tracy exec_time: {avg_tracy_time:.3f} ms")
    print(f"  Avg Matmul ops per pass: {len(all_mini_ids) / len(mini_batch_zones):.1f}")
    for category in ["BRISC", "NCRISC", "TRISC"]:
        if category in mb_time_ms:
            avg_ms = mb_time_ms[category] / len(mini_batch_zones)
            pct = (mb_time_ms[category] / mb_total * 100) if mb_total > 0 else 0
            print(f"  {category}: {avg_ms:.3f} ms ({pct:.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze detailed profiling zones from profile_log_device.csv.

Usage:
    python3 analyze_detailed_zones.py [csv_path]

Default: tracy_output_large/profile_log_device.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def parse_detailed_profile(csv_path):
    """Parse profile_log_device.csv with custom zones."""

    zones = defaultdict(lambda: defaultdict(list))
    zone_stack = {}

    print(f"Parsing {csv_path}...")

    with open(csv_path, "r", encoding="utf-8") as f:
        # Skip first two lines (header info and column names)
        f.readline()
        f.readline()

        line_count = 0
        for line in f:
            line_count += 1
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 12:
                continue

            try:
                core_x, core_y = int(parts[1]), int(parts[2])
                risc_type = parts[3]
                cycles = int(parts[5])
                zone_name = parts[10]
                zone_type = parts[11]
            except (ValueError, IndexError):
                continue

            if not zone_name or not zone_type:
                continue

            core_key = (core_x, core_y)
            zone_key = (risc_type, zone_name)

            if zone_type == "ZONE_START":
                zone_stack[(core_key, zone_key)] = cycles
            elif zone_type == "ZONE_END":
                stack_key = (core_key, zone_key)
                if stack_key in zone_stack:
                    duration = cycles - zone_stack[stack_key]
                    zones[risc_type][zone_name].append(duration)
                    del zone_stack[stack_key]

        print(f"Processed {line_count} lines")

    return zones


def print_breakdown(zones, freq_mhz=1350):
    """Print detailed breakdown per RISC type."""

    if not zones:
        print("No zone data found!")
        return

    for risc_type in sorted(zones.keys()):
        print(f"\n{'='*90}")
        print(f"{risc_type} OPERATION BREAKDOWN")
        print(f"{'='*90}")

        risc_zones = zones[risc_type]
        if not risc_zones:
            print("No data for this RISC type")
            continue

        total_cycles = sum(sum(durations) for durations in risc_zones.values())
        if total_cycles == 0:
            print("Total cycles = 0, skipping")
            continue

        total_ms = total_cycles / (freq_mhz * 1000)

        print(f"\nTotal time: {total_ms:.2f} ms ({total_cycles:,} cycles)")
        print(f"\n{'Operation':<40} {'Avg (cycles)':<15} {'Avg (ms)':<12} {'%':<8} {'Count':<8}")
        print("-" * 90)

        # Sort by total time
        zone_items = []
        for zone_name, durations in risc_zones.items():
            if len(durations) == 0:
                continue
            avg_cycles = sum(durations) / len(durations)
            avg_ms = avg_cycles / (freq_mhz * 1000)
            total_zone_cycles = sum(durations)
            pct = (total_zone_cycles / total_cycles) * 100
            zone_items.append((zone_name, avg_cycles, avg_ms, pct, len(durations), total_zone_cycles))

        zone_items.sort(key=lambda x: x[5], reverse=True)  # Sort by total cycles

        for zone_name, avg_cycles, avg_ms, pct, count, _ in zone_items:
            print(f"{zone_name:<40} {avg_cycles:<15,.0f} {avg_ms:<12.3f} {pct:<8.1f} {count:<8,}")

        # Identify idle vs work
        print(f"\n{'─'*90}")
        idle_keywords = ["WAIT", "BARRIER", "IDLE", "CB-WAIT", "NOC-BARRIER"]
        idle_zones = [z for z in zone_items if any(idle_word in z[0].upper() for idle_word in idle_keywords)]
        work_zones = [z for z in zone_items if z not in idle_zones]

        idle_pct = sum(z[3] for z in idle_zones)
        work_pct = sum(z[3] for z in work_zones)

        print(f"{'ACTUAL WORK (estimated)':<40} {'':<15} {'':<12} {work_pct:<8.1f}")
        print(f"{'IDLE/WAITING (estimated)':<40} {'':<15} {'':<12} {idle_pct:<8.1f}")

        # List idle zones
        if idle_zones:
            print(f"\nIdle/Waiting zones:")
            for zone_name, avg_cycles, avg_ms, pct, count, _ in idle_zones:
                print(f"  - {zone_name}: {pct:.1f}%")


def main():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = Path("tracy_output_large/profile_log_device.csv")

    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        print("\nRun the test with TT_METAL_DEVICE_PROFILER=1 first:")
        print("  cd research_codes")
        print("  export TT_METAL_DEVICE_PROFILER=1")
        print("  python3 weight_loading_test_tracy.py --only-large")
        return 1

    zones = parse_detailed_profile(csv_path)

    if not zones:
        print("\nNo custom zones found in the profile!")
        print("\nThis means either:")
        print("  1. The kernels don't have DeviceZoneScopedN() added yet")
        print("  2. The profiler wasn't enabled (TT_METAL_DEVICE_PROFILER=1)")
        print("  3. The CSV file is from a different test")
        print("\nTo add custom zones, see:")
        print("  research_codes/EXAMPLE_reader_bmm_tile_layout_PROFILED.cpp")
        print("  research_codes/ACTION_PLAN_DETAILED_PROFILING.md")
        return 1

    print_breakdown(zones)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print("\nTo see more detailed profiling:")
    print("  1. Add DeviceZoneScopedN() to kernel files")
    print("  2. See EXAMPLE_*.cpp files for reference")
    print("  3. Rebuild with ./build_metal.sh")
    print("  4. Re-run with TT_METAL_DEVICE_PROFILER=1")

    return 0


if __name__ == "__main__":
    sys.exit(main())

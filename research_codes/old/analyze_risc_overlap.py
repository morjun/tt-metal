#!/usr/bin/env python3
"""Analyze RISC overlap and parallelism in device profiler data."""

import csv
import sys
from pathlib import Path
from collections import defaultdict


def parse_device_profile_with_timeline(csv_path):
    """Parse device profile and analyze timeline overlap."""

    with open(csv_path, "r") as f:
        # Skip first header line (ARCH, FREQ)
        header1 = f.readline().strip()
        freq_mhz = 1350
        if "CHIP_FREQ" in header1:
            for part in header1.split(","):
                if "CHIP_FREQ" in part:
                    freq_mhz = int(part.split(":")[-1].strip())

        freq_hz = freq_mhz * 1_000_000

        # Skip column header
        f.readline()

        # Parse zones
        zone_starts = {}
        zones_by_risc = defaultdict(list)

        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 12:
                continue

            try:
                core_x = int(parts[1])
                core_y = int(parts[2])
                risc_type = parts[3].strip()
                time_cycles = int(parts[5])
                zone_phase = parts[11].strip()

                # Normalize RISC type
                if "BRISC" in risc_type:
                    risc_normalized = "BRISC"
                elif "NCRISC" in risc_type:
                    risc_normalized = "NCRISC"
                elif "TRISC" in risc_type:
                    risc_normalized = "TRISC"
                else:
                    continue

                key = tuple(parts[:4] + parts[7:11])

                if zone_phase == "ZONE_START":
                    zone_starts[key] = time_cycles
                elif zone_phase == "ZONE_END" and key in zone_starts:
                    start = zone_starts[key]
                    end = time_cycles
                    zones_by_risc[risc_normalized].append((start, end, core_x, core_y))
                    del zone_starts[key]

            except (ValueError, IndexError):
                continue

    # Find global timeline
    all_times = []
    for zones in zones_by_risc.values():
        for start, end, _, _ in zones:
            all_times.extend([start, end])

    global_start = min(all_times)
    global_end = max(all_times)
    wall_clock_cycles = global_end - global_start
    wall_clock_ms = (wall_clock_cycles / freq_hz) * 1000.0

    # Calculate total active time per RISC
    risc_totals = {}
    for risc_type in ["BRISC", "NCRISC", "TRISC"]:
        zones = zones_by_risc[risc_type]
        total_cycles = sum(end - start for start, end, _, _ in zones)
        total_ms = (total_cycles / freq_hz) * 1000.0
        risc_totals[risc_type] = {
            "total_cycles": total_cycles,
            "total_ms": total_ms,
            "zone_count": len(zones),
            "first_start": min((start for start, _, _, _ in zones)) if zones else 0,
            "last_end": max((end for _, end, _, _ in zones)) if zones else 0,
        }

    return risc_totals, wall_clock_cycles, wall_clock_ms, global_start, global_end, freq_hz


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_risc_overlap.py <profile_log_device.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    print(f"Analyzing: {csv_path}")
    print()

    risc_totals, wall_cycles, wall_ms, global_start, global_end, freq_hz = parse_device_profile_with_timeline(csv_path)

    print(f"Device Frequency: {freq_hz/1e6:.0f} MHz")
    print()
    print("Timeline Analysis:")
    print("-" * 80)
    print(f"Global Timeline: {global_start:,} -> {global_end:,} cycles")
    print(f"Wall Clock Time: {wall_ms:.6f} ms ({wall_cycles:,} cycles)")
    print()

    total_sum_cycles = sum(r["total_cycles"] for r in risc_totals.values())
    total_sum_ms = sum(r["total_ms"] for r in risc_totals.values())

    print("Per-RISC Activity:")
    print("-" * 80)
    for risc_type in ["BRISC", "NCRISC", "TRISC"]:
        data = risc_totals[risc_type]
        span_cycles = data["last_end"] - data["first_start"]
        span_ms = (span_cycles / freq_hz) * 1000.0
        utilization = (data["total_cycles"] / span_cycles * 100) if span_cycles > 0 else 0

        print(f"{risc_type}:")
        print(f"  Total active time:  {data['total_ms']:10.6f} ms ({data['total_cycles']:15,} cycles)")
        print(f"  Timeline span:      {span_ms:10.6f} ms ({span_cycles:15,} cycles)")
        print(f"  Utilization:        {utilization:5.1f}% (active / span)")
        print(f"  Zone count:         {data['zone_count']:,}")
        print()

    print("Parallelism Analysis:")
    print("-" * 80)
    print(f"Sum of all RISC times:  {total_sum_ms:.6f} ms ({total_sum_cycles:,} cycles)")
    print(f"Wall clock time:        {wall_ms:.6f} ms ({wall_cycles:,} cycles)")
    print(f"Parallelism factor:     {total_sum_ms / wall_ms:.2f}x")
    print()
    print("If parallelism = 1.0x: RISCs run sequentially (no overlap)")
    print("If parallelism > 1.0x: RISCs run in parallel (overlapping execution)")
    print()


if __name__ == "__main__":
    main()

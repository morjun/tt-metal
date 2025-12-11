#!/usr/bin/env python3
"""Simple device profiler analyzer to calculate BRISC/NCRISC/TRISC breakdown."""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def parse_device_profile(csv_path):
    """Parse device profile and calculate component breakdown."""
    zones_by_risc = defaultdict(list)

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

        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 12:
                continue

            try:
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
                    duration = time_cycles - zone_starts[key]
                    zones_by_risc[risc_normalized].append(duration)
                    del zone_starts[key]

            except (ValueError, IndexError):
                continue

    # Calculate totals
    totals = {}
    for risc_type in ["BRISC", "NCRISC", "TRISC"]:
        if risc_type in zones_by_risc:
            total_cycles = sum(zones_by_risc[risc_type])
            total_ms = (total_cycles / freq_hz) * 1000.0
            totals[risc_type] = {"cycles": total_cycles, "ms": total_ms, "zone_count": len(zones_by_risc[risc_type])}
        else:
            totals[risc_type] = {"cycles": 0, "ms": 0.0, "zone_count": 0}

    total_cycles = sum(t["cycles"] for t in totals.values())
    total_ms = sum(t["ms"] for t in totals.values())

    return totals, total_cycles, total_ms, freq_hz


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_device_profile_simple.py <profile_log_device.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    print(f"Analyzing: {csv_path}")
    print()

    totals, total_cycles, total_ms, freq_hz = parse_device_profile(csv_path)

    print(f"Device Frequency: {freq_hz/1e6:.0f} MHz")
    print(f"Total Time: {total_ms:.6f} ms ({total_cycles:,} cycles)")
    print()
    print("Component Breakdown:")
    print("-" * 60)

    for risc_type in ["BRISC", "NCRISC", "TRISC"]:
        data = totals[risc_type]
        pct = (data["cycles"] / total_cycles * 100) if total_cycles > 0 else 0
        print(
            f"{risc_type:8s}: {pct:5.1f}% | {data['ms']:10.6f} ms | {data['cycles']:15,} cycles | {data['zone_count']:5,} zones"
        )

    print("-" * 60)
    print(f"{'TOTAL':8s}: 100.0% | {total_ms:10.6f} ms | {total_cycles:15,} cycles")
    print()


if __name__ == "__main__":
    main()

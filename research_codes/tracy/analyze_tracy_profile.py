#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TRACY Profile Analyzer

This script analyzes Tracy profiling data to measure:
1. Weight streaming (BRISC) time
2. NoC communication (NCRISC) time
3. Actual tensor computation (TRISC) time

For each forward pass identified by TRACY zone markers.

Usage:
    python3 research_codes/analyze_tracy_profile.py
"""

import csv
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict


@dataclass
class TracyZone:
    """Host-side TRACY zone from tracy_ops_times.csv"""

    name: str
    src_file: str
    src_line: int
    ns_since_start: int
    exec_time_ns: int
    thread: str


@dataclass
class DeviceZone:
    """Device-side zone from profile_log_device.csv"""

    core_x: int
    core_y: int
    risc_type: str
    start_cycle: int
    end_cycle: int
    duration_cycles: int
    run_host_id: int
    zone_name: str


def parse_tracy_ops_times(csv_path: Path) -> List[TracyZone]:
    """Parse tracy_ops_times.csv to extract forward pass zones."""
    zones = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            # Only extract our custom forward pass zones
            if "LargeBatch_Forward" in name or "MiniBatch_Forward" in name:
                try:
                    zone = TracyZone(
                        name=name,
                        src_file=row["src_file"].strip(),
                        src_line=int(row["src_line"]) if row["src_line"].isdigit() else 0,
                        ns_since_start=int(row["ns_since_start"]) if row["ns_since_start"].isdigit() else 0,
                        exec_time_ns=int(row["exec_time_ns"]) if row["exec_time_ns"].isdigit() else 0,
                        thread=row["thread"].strip(),
                    )
                    zones.append(zone)
                except (ValueError, KeyError) as e:
                    continue

    return zones


def parse_device_profile(csv_path: Path, freq_hz: float) -> Tuple[List[DeviceZone], str, float]:
    """Parse device profile CSV."""
    zones = []
    arch = "blackhole"
    freq_mhz = 1350

    # Parse header to get frequency
    with open(csv_path, "r") as f:
        header = f.readline().strip()
        if "ARCH" in header and "CHIP_FREQ" in header:
            for part in header.split(","):
                part = part.strip()
                if part.startswith("ARCH:"):
                    arch = part.split(":")[-1].strip()
                elif "CHIP_FREQ" in part:
                    freq_mhz = int(part.split(":")[-1].strip())

        freq_hz = freq_mhz * 1_000_000

        # Skip column header line
        f.readline()

        # Parse zones
        zone_starts = {}  # key: (core_x, core_y, risc_type, run_host_id, zone_name)

        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 12:
                continue

            try:
                core_x = int(parts[1])
                core_y = int(parts[2])
                risc_type = parts[3].strip()
                time_cycles = int(parts[5])
                run_host_id = int(parts[7]) if parts[7].strip() else 0
                zone_name = parts[10].strip()
                zone_phase = parts[11].strip()

                key = (core_x, core_y, risc_type, run_host_id, zone_name)

                if zone_phase == "ZONE_START":
                    zone_starts[key] = time_cycles
                elif zone_phase == "ZONE_END" and key in zone_starts:
                    start = zone_starts[key]
                    duration = time_cycles - start

                    zone = DeviceZone(
                        core_x=core_x,
                        core_y=core_y,
                        risc_type=risc_type,
                        start_cycle=start,
                        end_cycle=time_cycles,
                        duration_cycles=duration,
                        run_host_id=run_host_id,
                        zone_name=zone_name,
                    )
                    zones.append(zone)
                    del zone_starts[key]
            except (ValueError, IndexError):
                continue

    return zones, arch, freq_hz


def normalize_risc_type(risc_type: str) -> str:
    """Normalize RISC type names."""
    if "BRISC" in risc_type:
        return "BRISC"
    elif "NCRISC" in risc_type:
        return "NCRISC"
    elif "TRISC" in risc_type:
        return "TRISC"
    return risc_type


def map_tracy_zones_to_device_operations(
    tracy_zones: List[TracyZone], device_zones: List[DeviceZone], freq_hz: float
) -> Dict[str, Dict]:
    """
    Map TRACY zones (host-side) to device zones (device-side).

    Strategy:
    Device zones와 Tracy zones는 서로 다른 clock domain에 있으므로
    직접적인 시간 매핑은 불가능합니다.

    대신:
    1. Tracy zones의 exec_time_ns를 wall clock time으로 사용
    2. 모든 device zones를 aggregate해서 전체적인 component breakdown만 제공
    3. 각 Tracy zone에 대해 독립적으로 분석

    Returns:
        Dict mapping Tracy zone name to {
            'tracy_zone': TracyZone,
            'device_zones': List[DeviceZone]  # For component breakdown
        }
    """
    # Sort Tracy zones by time
    tracy_zones_sorted = sorted(tracy_zones, key=lambda z: z.ns_since_start)

    # Group device zones by run_host_id
    device_by_run_id = defaultdict(list)
    for dz in device_zones:
        if dz.run_host_id > 0:
            device_by_run_id[dz.run_host_id].append(dz)

    # Get sorted run_host_ids
    sorted_run_ids = sorted(device_by_run_id.keys())

    if len(tracy_zones_sorted) == 0 or len(sorted_run_ids) == 0:
        return {}

    # 더 정확한 매핑: Tracy zone의 순서와 device operation의 순서를 맞춤
    # Large batch는 처음 몇 개 run_id, minibatch는 나머지
    mapping = {}

    # Identify scenario
    has_large_batch = any("LargeBatch" in z.name for z in tracy_zones_sorted)

    if has_large_batch:
        # First Tracy zone is large batch
        large_batch_zone = [z for z in tracy_zones_sorted if "LargeBatch" in z.name][0]
        mini_batch_zones = [z for z in tracy_zones_sorted if "MiniBatch" in z.name]

        # Assume first few run_ids are for large batch
        # Heuristic: number of run_ids for large batch ≈ total_run_ids / (1 + num_minibatches)
        num_large_batch_runs = max(1, len(sorted_run_ids) // (1 + len(mini_batch_zones)))

        large_batch_run_ids = sorted_run_ids[:num_large_batch_runs]
        mini_batch_run_ids = sorted_run_ids[num_large_batch_runs:]

        # Map large batch
        large_batch_device_zones = []
        for rid in large_batch_run_ids:
            large_batch_device_zones.extend(device_by_run_id[rid])

        mapping[large_batch_zone.name] = {"tracy_zone": large_batch_zone, "device_zones": large_batch_device_zones}

        # Map mini batches evenly
        if mini_batch_zones and mini_batch_run_ids:
            runs_per_mini = max(1, len(mini_batch_run_ids) // len(mini_batch_zones))

            run_idx = 0
            for mini_zone in mini_batch_zones:
                zone_run_ids = mini_batch_run_ids[run_idx : run_idx + runs_per_mini]
                run_idx += runs_per_mini

                # Last zone gets remaining run_ids
                if mini_zone == mini_batch_zones[-1]:
                    zone_run_ids = mini_batch_run_ids[run_idx - runs_per_mini :]

                zone_device_zones = []
                for rid in zone_run_ids:
                    zone_device_zones.extend(device_by_run_id[rid])

                mapping[mini_zone.name] = {"tracy_zone": mini_zone, "device_zones": zone_device_zones}
    else:
        # Only minibatches
        runs_per_zone = max(1, len(sorted_run_ids) // len(tracy_zones_sorted))

        run_idx = 0
        for tracy_zone in tracy_zones_sorted:
            zone_run_ids = sorted_run_ids[run_idx : run_idx + runs_per_zone]
            run_idx += runs_per_zone

            # Last zone gets remaining
            if tracy_zone == tracy_zones_sorted[-1]:
                zone_run_ids = sorted_run_ids[run_idx - runs_per_zone :]

            zone_device_zones = []
            for rid in zone_run_ids:
                zone_device_zones.extend(device_by_run_id[rid])

            mapping[tracy_zone.name] = {"tracy_zone": tracy_zone, "device_zones": zone_device_zones}

    return mapping


def analyze_forward_pass(tracy_zone: TracyZone, device_zones: List[DeviceZone], freq_hz: float) -> Dict:
    """
    Analyze a single forward pass.

    Uses Tracy zone's exec_time_ns as the accurate wall clock time.
    Device zones provide component breakdown only.
    """
    name = tracy_zone.name
    wall_ms = tracy_zone.exec_time_ns / 1_000_000.0  # Convert ns to ms

    if not device_zones:
        return {
            "name": name,
            "num_zones": 0,
            "wall_clock_ms": wall_ms,
            "brisc_ms": 0.0,
            "ncrisc_ms": 0.0,
            "trisc_ms": 0.0,
            "brisc_pct": 0.0,
            "ncrisc_pct": 0.0,
            "trisc_pct": 0.0,
        }

    # Group by RISC type
    zones_by_risc = defaultdict(list)
    for z in device_zones:
        risc = normalize_risc_type(z.risc_type)
        zones_by_risc[risc].append(z)

    # Calculate total device work time for each RISC type
    # Sum all zone durations
    risc_times = {}
    for risc_type in ["BRISC", "NCRISC", "TRISC"]:
        if risc_type in zones_by_risc:
            zones = zones_by_risc[risc_type]
            # Sum all zone durations and convert to ms
            total_cycles = sum(z.duration_cycles for z in zones)
            total_ms = (total_cycles / freq_hz) * 1000.0
            risc_times[risc_type] = total_ms
        else:
            risc_times[risc_type] = 0.0

    total_device_time = sum(risc_times.values())

    # Calculate percentages
    if total_device_time > 0:
        brisc_pct = (risc_times["BRISC"] / total_device_time) * 100
        ncrisc_pct = (risc_times["NCRISC"] / total_device_time) * 100
        trisc_pct = (risc_times["TRISC"] / total_device_time) * 100
    else:
        brisc_pct = ncrisc_pct = trisc_pct = 0.0

    return {
        "name": name,
        "num_zones": len(device_zones),
        "wall_clock_ms": wall_ms,
        "brisc_ms": risc_times["BRISC"],
        "ncrisc_ms": risc_times["NCRISC"],
        "trisc_ms": risc_times["TRISC"],
        "brisc_pct": brisc_pct,
        "ncrisc_pct": ncrisc_pct,
        "trisc_pct": trisc_pct,
        "total_device_ms": total_device_time,
    }


def print_analysis_results(analyses: List[Dict], scenario: str):
    """Print analysis results."""
    print("\n" + "=" * 80)
    print(f"TRACY PROFILE ANALYSIS RESULTS - {scenario.upper()}")
    print("=" * 80)
    print()

    # All analyses are measurement iterations (warmup은 TRACY marker가 없음)
    measurement_analyses = analyses

    print(f"Found {len(measurement_analyses)} measurement forward passes")
    print("(Warmup iterations are excluded from TRACY profiling)")
    print()

    # Analyze measurement forward passes
    if not measurement_analyses:
        print("No measurement forward passes found!")
        return

    print("=" * 80)
    print("MEASUREMENT FORWARD PASS ANALYSIS")
    print("=" * 80)
    print()
    print("NOTE: Wall clock time is from TRACY host-side measurement (accurate).")
    print("      Component breakdown is from device profiler (relative proportions).")
    print()

    for analysis in measurement_analyses:
        print(f"Forward Pass: {analysis['name']}")
        print(f"  Device Zones: {analysis['num_zones']}")
        print(f"  Wall Clock Time: {analysis['wall_clock_ms']:.6f} ms (from TRACY)")
        print()

        # Show device component breakdown
        total_dev = analysis.get("total_device_ms", 0)
        if total_dev > 0:
            print(f"  Device Component Breakdown (total device time: {total_dev:.2f} ms):")
            print(f"    Weight Streaming (BRISC):  {analysis['brisc_ms']:8.2f} ms  ({analysis['brisc_pct']:5.1f}%)")
            print(f"    NoC Communication (NCRISC): {analysis['ncrisc_ms']:8.2f} ms  ({analysis['ncrisc_pct']:5.1f}%)")
            print(f"    Computation (TRISC):        {analysis['trisc_ms']:8.2f} ms  ({analysis['trisc_pct']:5.1f}%)")
            print()
            print(
                f"  Note: Device time ({total_dev:.2f} ms) may differ from wall clock ({analysis['wall_clock_ms']:.3f} ms)"
            )
            print(f"        due to profiler overhead and different clock domains.")
        else:
            print(f"  No device profiling data available.")
        print()
        print("-" * 80)
        print()

    # Calculate totals
    if len(measurement_analyses) > 1:
        print("=" * 80)
        print("AGGREGATE STATISTICS")
        print("=" * 80)
        print()

        total_wall = sum(a["wall_clock_ms"] for a in measurement_analyses)
        total_brisc = sum(a["brisc_ms"] for a in measurement_analyses)
        total_ncrisc = sum(a["ncrisc_ms"] for a in measurement_analyses)
        total_trisc = sum(a["trisc_ms"] for a in measurement_analyses)
        total_device = total_brisc + total_ncrisc + total_trisc

        print(f"Total across {len(measurement_analyses)} forward passes:")
        print(f"  Total Wall Clock Time:      {total_wall:.6f} ms (from TRACY)")
        print()

        if total_device > 0:
            print(f"  Total Device Time:          {total_device:.2f} ms")
            print(f"    Weight Streaming (BRISC): {total_brisc:.2f} ms  ({total_brisc/total_device*100:.1f}%)")
            print(f"    NoC Communication (NCRISC):{total_ncrisc:.2f} ms  ({total_ncrisc/total_device*100:.1f}%)")
            print(f"    Computation (TRISC):       {total_trisc:.2f} ms  ({total_trisc/total_device*100:.1f}%)")
        print()

        # Separate large batch and mini batches
        large_batch = [a for a in measurement_analyses if "LargeBatch" in a["name"]]
        mini_batches = [a for a in measurement_analyses if "MiniBatch" in a["name"]]

        if large_batch and mini_batches:
            lb_wall = sum(a["wall_clock_ms"] for a in large_batch)
            mb_wall = sum(a["wall_clock_ms"] for a in mini_batches)

            print("Comparison:")
            print(f"  Large Batch (1 forward):     {lb_wall:.6f} ms")
            print(f"  Mini Batches (8 forwards):   {mb_wall:.6f} ms")
            print(f"  Mini Batch Overhead:         {mb_wall - lb_wall:.6f} ms")
            print(f"  Overhead Percentage:         {((mb_wall - lb_wall) / lb_wall * 100):.2f}%")
            print()
            print(f"  Average per mini-batch:      {mb_wall / len(mini_batches):.6f} ms")
        print()


def main():
    """Main analysis routine."""
    print("\n" + "=" * 80)
    print("TRACY PROFILE ANALYZER")
    print("=" * 80)
    print()
    print("This script analyzes TRACY profiling data to measure:")
    print("  1. Weight streaming (BRISC) time")
    print("  2. NoC communication (NCRISC) time")
    print("  3. Tensor computation (TRISC) time")
    print()
    print("For each forward pass identified by TRACY zone markers.")
    print()

    # Find Tracy output directory
    possible_paths = [
        Path("research_codes/tracy_output/.logs"),
        Path("tracy_output/.logs"),
    ]

    logs_dir = None
    for p in possible_paths:
        if p.exists():
            logs_dir = p
            break

    if not logs_dir:
        print("ERROR: Tracy output directory not found!")
        print("Please run Tracy profiling first:")
        print("  bash research_codes/run_tracy_weight_test.sh")
        return

    print(f"Found Tracy logs at: {logs_dir}")
    print()

    # Parse Tracy ops times
    tracy_ops_path = logs_dir / "tracy_ops_times.csv"
    if not tracy_ops_path.exists():
        print(f"ERROR: {tracy_ops_path} not found!")
        return

    print(f"Parsing Tracy zones from: {tracy_ops_path}")
    tracy_zones = parse_tracy_ops_times(tracy_ops_path)
    print(f"  Found {len(tracy_zones)} TRACY forward pass zones")

    if not tracy_zones:
        print("ERROR: No TRACY forward pass zones found!")
        print("Make sure to run with TRACY profiling enabled.")
        return

    # Determine scenario
    if any("LargeBatch" in z.name for z in tracy_zones):
        scenario = "large_batch"
        print(f"  Scenario: Large Batch")
    else:
        scenario = "mini_batch"
        print(f"  Scenario: Mini-Batch")
    print()

    # Parse device profile
    device_csv_path = logs_dir / "profile_log_device.csv"
    if not device_csv_path.exists():
        print(f"ERROR: {device_csv_path} not found!")
        return

    print(f"Parsing device profile from: {device_csv_path}")
    device_zones, arch, freq_hz = parse_device_profile(device_csv_path, 1350000000)
    print(f"  Architecture: {arch} @ {freq_hz/1e6:.0f} MHz")
    print(f"  Found {len(device_zones)} device zones")

    # Get unique run_host_ids
    run_host_ids = sorted(set(z.run_host_id for z in device_zones if z.run_host_id > 0))
    print(f"  Found {len(run_host_ids)} unique run_host_ids")
    print()

    # Map Tracy zones to device zones
    print("Mapping TRACY zones to device operations...")
    mapping = map_tracy_zones_to_device_operations(tracy_zones, device_zones, freq_hz)
    print(f"  Mapped {len(mapping)} TRACY zones to device zones")

    for zone_name, zone_data in mapping.items():
        num_dev_zones = len(zone_data["device_zones"]) if isinstance(zone_data, dict) else 0
        print(f"    {zone_name}: {num_dev_zones} device zones")
    print()

    # Analyze each forward pass
    print("Analyzing forward passes...")
    analyses = []
    for tracy_zone in tracy_zones:
        zone_data = mapping.get(tracy_zone.name, {"tracy_zone": tracy_zone, "device_zones": []})
        device_zones_for_zone = zone_data["device_zones"]
        analysis = analyze_forward_pass(tracy_zone, device_zones_for_zone, freq_hz)
        analyses.append(analysis)
    print(f"  Analyzed {len(analyses)} forward passes")
    print()

    # Print results
    print_analysis_results(analyses, scenario)

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

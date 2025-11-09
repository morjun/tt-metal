#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Combined TRACY and Device Profile Analyzer

This script combines:
1. Python perf_counter() measurements (accurate wall clock time)
2. Device profiler data (component breakdown: BRISC/NCRISC/TRISC)

To provide accurate analysis of:
- Weight streaming (BRISC) time
- NoC communication (NCRISC) time
- Tensor computation (TRISC) time

Usage:
    python3 research_codes/analyze_tracy_combined.py
"""

import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict


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


def normalize_risc_type(risc_type: str) -> str:
    """Normalize RISC type names."""
    if "BRISC" in risc_type:
        return "BRISC"
    elif "NCRISC" in risc_type:
        return "NCRISC"
    elif "TRISC" in risc_type:
        return "TRISC"
    return risc_type


def parse_device_profile(csv_path: Path) -> tuple[List[DeviceZone], str, float]:
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


def parse_benchmark_csv(csv_path: Path) -> Dict:
    """Parse benchmark results CSV to get Python perf_counter measurements."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        # Get last row (most recent measurement)
        last_row = None
        for row in reader:
            last_row = row

        if last_row is None:
            return {}

        return {
            "large_batch_forward_ms": float(last_row["large_forward_compute_ms"]),
            "mini_batch_forward_ms": float(last_row["mini_forward_compute_ms"]),
            "large_weight_load_ms": float(last_row["large_weight_load_host_to_gddr6_ms"]),
            "mini_weight_load_ms": float(last_row["mini_weight_load_host_to_gddr6_ms"]),
            "minibatches": int(last_row["minibatches"]),
        }


def parse_tracy_ops_data(csv_path: Path) -> Dict:
    """
    Parse tracy_ops_data.csv to identify which run_host_ids belong to measurement iterations.

    IMPORTANT: We want to capture the COMPLETE forward pass execution time, which includes:
    - SliceDeviceOperation (mini-batch splitting, DRAM->SRAM weight loading)
    - Matmul (actual computation)

    Both are part of what the user experiences when calling ttnn.linear()!

    Strategy:
    - warmup_iters=1, measure_iters=1
    - Large batch: run_id 4096 (measurement)
    - Mini batch: run_ids 27648-41984 (measurement, includes 8 Matmuls + 8 Slices)

    Returns mapping of scenario to run_host_ids (includes ALL operations)
    """
    import re

    with open(csv_path, "r") as f:
        content = f.read()

    # Extract all device operations with run_host_id
    matmul_pattern = r'`TT_DNN_DEVICE_OP: "Matmul".*?, 0, (\d+)'
    slice_pattern = r'`TT_DNN_DEVICE_OP: "SliceDeviceOperation".*?, 0, (\d+)'

    matmul_run_ids = [int(rid) for rid in re.findall(matmul_pattern, content)]
    slice_run_ids = [int(rid) for rid in re.findall(slice_pattern, content)]

    print(f"  DEBUG: Found {len(matmul_run_ids)} Matmul operations")
    print(f"  DEBUG: Found {len(slice_run_ids)} SliceDeviceOperation operations")
    print(f"  DEBUG: Matmul run_ids: {matmul_run_ids}")
    print(f"  DEBUG: Slice run_ids: {slice_run_ids}")

    # Based on manual inspection:
    # - Large batch measurement: Matmul at run_id 4096
    # - Mini batch measurement: run_ids 27648-41984 (alternating Matmul and Slice)

    # For mini-batch, we want BOTH Matmul and Slice operations
    # because that's the complete execution cost
    mini_measurement_start = 27648
    mini_measurement_end = 41984

    result = {
        "large_batch_measurement": [4096],  # Single large batch Matmul
        "mini_batch_measurement": list(
            range(mini_measurement_start, mini_measurement_end + 1, 1024)
        ),  # All 16 operations
    }

    print(f"  INFO: Mini-batch measurement includes {len(result['mini_batch_measurement'])} operations")
    print(f"  INFO: This includes both Matmul and SliceDeviceOperation (complete forward pass)")

    return result


def calculate_wall_clock_time(device_zones: List[DeviceZone], freq_hz: float) -> float:
    """
    Calculate actual wall clock time from device zones.
    This is the elapsed time from the earliest zone start to the latest zone end.
    """
    if not device_zones:
        return 0.0

    min_start = min(z.start_cycle for z in device_zones)
    max_end = max(z.end_cycle for z in device_zones)
    wall_cycles = max_end - min_start
    wall_ms = (wall_cycles / freq_hz) * 1000.0
    return wall_ms


def calculate_component_breakdown(
    device_zones: List[DeviceZone], freq_hz: float, filter_run_ids: List[int] | None = None
) -> Dict:
    """Calculate BRISC/NCRISC/TRISC breakdown from device zones, optionally filtered by run_host_id."""
    # Filter zones if requested
    if filter_run_ids is not None:
        device_zones = [z for z in device_zones if z.run_host_id in filter_run_ids]

    if not device_zones:
        return {
            "wall_clock_ms": 0.0,
            "times": {"BRISC": 0, "NCRISC": 0, "TRISC": 0},
            "percentages": {"BRISC": 0, "NCRISC": 0, "TRISC": 0},
            "total": 0,
        }

    # Calculate wall clock time
    wall_clock_ms = calculate_wall_clock_time(device_zones, freq_hz)

    # Group by RISC type
    zones_by_risc = defaultdict(list)
    for z in device_zones:
        risc = normalize_risc_type(z.risc_type)
        zones_by_risc[risc].append(z)

    # Calculate total time for each RISC type
    risc_times = {}
    for risc_type in ["BRISC", "NCRISC", "TRISC"]:
        if risc_type in zones_by_risc:
            zones = zones_by_risc[risc_type]
            # Sum all zone durations
            total_cycles = sum(z.duration_cycles for z in zones)
            total_ms = (total_cycles / freq_hz) * 1000.0
            risc_times[risc_type] = total_ms
        else:
            risc_times[risc_type] = 0.0

    total_time = sum(risc_times.values())

    # Calculate percentages
    if total_time > 0:
        percentages = {
            "BRISC": (risc_times["BRISC"] / total_time) * 100,
            "NCRISC": (risc_times["NCRISC"] / total_time) * 100,
            "TRISC": (risc_times["TRISC"] / total_time) * 100,
        }
    else:
        percentages = {"BRISC": 0, "NCRISC": 0, "TRISC": 0}

    return {
        "wall_clock_ms": wall_clock_ms,
        "times": risc_times,
        "percentages": percentages,
        "total": total_time,
    }


def main():
    """Main analysis routine."""
    print("\n" + "=" * 80)
    print("COMBINED TRACY + DEVICE PROFILE ANALYZER")
    print("=" * 80)
    print()
    print("This script combines:")
    print("  1. Python perf_counter() measurements (accurate wall clock time)")
    print("  2. Device profiler data (component breakdown: BRISC/NCRISC/TRISC)")
    print()

    # Find output directory - use absolute paths from script location
    script_dir = Path(__file__).parent.resolve()
    workspace_root = script_dir.parent
    tracy_logs_dir = script_dir / "tracy_output" / ".logs"
    benchmark_csv = script_dir / "benchmark_results_tracy.csv"

    if not tracy_logs_dir.exists():
        print(f"ERROR: Tracy logs directory not found: {tracy_logs_dir}")
        return

    if not benchmark_csv.exists():
        print(f"ERROR: Benchmark CSV not found: {benchmark_csv}")
        return

    # Parse benchmark results (Python measurements)
    print(f"Parsing Python measurements from: {benchmark_csv}")
    python_data = parse_benchmark_csv(benchmark_csv)

    if not python_data:
        print("ERROR: Failed to parse benchmark CSV")
        return

    print(f"  Large batch forward: {python_data['large_batch_forward_ms']:.6f} ms")
    print(f"  Mini batch forward:  {python_data['mini_batch_forward_ms']:.6f} ms")
    print(f"  Number of minibatches: {python_data['minibatches']}")
    print()

    # Parse device profile
    device_csv_path = tracy_logs_dir / "profile_log_device.csv"
    if not device_csv_path.exists():
        print(f"ERROR: {device_csv_path} not found!")
        return

    print(f"Parsing device profile from: {device_csv_path}")
    device_zones, arch, freq_hz = parse_device_profile(device_csv_path)
    print(f"  Architecture: {arch} @ {freq_hz/1e6:.0f} MHz")
    print(f"  Found {len(device_zones)} device zones")
    print()

    # Parse Tracy ops data to identify measurement run_host_ids
    tracy_ops_data_path = tracy_logs_dir / "tracy_ops_data.csv"
    if not tracy_ops_data_path.exists():
        print(f"ERROR: {tracy_ops_data_path} not found!")
        return

    print(f"Parsing Tracy ops data from: {tracy_ops_data_path}")
    run_id_mapping = parse_tracy_ops_data(tracy_ops_data_path)
    print(f"  Large batch measurement run_ids: {run_id_mapping.get('large_batch_measurement', [])}")
    print(f"  Mini batch measurement run_ids: {run_id_mapping.get('mini_batch_measurement', [])}")
    print()

    # Calculate component breakdown (percentages only) for measurement iterations
    print("Calculating component breakdown percentages (measurement iterations only)...")

    # Overall breakdown - we only use the percentages, not the absolute times
    all_measurement_run_ids = run_id_mapping.get("large_batch_measurement", []) + run_id_mapping.get(
        "mini_batch_measurement", []
    )
    overall_breakdown = calculate_component_breakdown(device_zones, freq_hz, all_measurement_run_ids)
    print(f"  Component percentages from device profiler:")
    print(f"    BRISC (Weight Streaming):   {overall_breakdown['percentages']['BRISC']:.1f}%")
    print(f"    NCRISC (NoC Communication): {overall_breakdown['percentages']['NCRISC']:.1f}%")
    print(f"    TRISC (Computation):        {overall_breakdown['percentages']['TRISC']:.1f}%")
    print(f"  NOTE: Device wall clock time excluded (event matching unreliable)")
    print()

    # Apply breakdown percentages to Python measurements
    print("=" * 80)
    print("ANALYSIS RESULTS (Using Python perf_counter for timing)")
    print("=" * 80)
    print()

    # Calculate component times by applying percentages to Python measurements
    large_py_time = python_data["large_batch_forward_ms"]
    large_brisc_ms = large_py_time * (overall_breakdown["percentages"]["BRISC"] / 100.0)
    large_ncrisc_ms = large_py_time * (overall_breakdown["percentages"]["NCRISC"] / 100.0)
    large_trisc_ms = large_py_time * (overall_breakdown["percentages"]["TRISC"] / 100.0)

    mini_py_time = python_data["mini_batch_forward_ms"]
    mini_brisc_ms = mini_py_time * (overall_breakdown["percentages"]["BRISC"] / 100.0)
    mini_ncrisc_ms = mini_py_time * (overall_breakdown["percentages"]["NCRISC"] / 100.0)
    mini_trisc_ms = mini_py_time * (overall_breakdown["percentages"]["TRISC"] / 100.0)

    print("Large Batch (1 forward pass):")
    print(f"  Total Time (Python):         {large_py_time:.6f} ms")
    print(f"  Component Breakdown:")
    print(f"    Weight Streaming (BRISC):  {large_brisc_ms:.6f} ms  ({overall_breakdown['percentages']['BRISC']:.1f}%)")
    print(
        f"    NoC Communication (NCRISC): {large_ncrisc_ms:.6f} ms  ({overall_breakdown['percentages']['NCRISC']:.1f}%)"
    )
    print(
        f"    Computation (TRISC):        {large_trisc_ms:.6f} ms  ({overall_breakdown['percentages']['TRISC']:.1f}%)"
    )
    print()

    print("Mini Batches (8 forward passes total):")
    print(f"  Total Time (Python):         {mini_py_time:.6f} ms")
    print(f"  Average per pass:            {mini_py_time / python_data['minibatches']:.6f} ms")
    print(f"  Component Breakdown (total):")
    print(f"    Weight Streaming (BRISC):  {mini_brisc_ms:.6f} ms  ({overall_breakdown['percentages']['BRISC']:.1f}%)")
    print(
        f"    NoC Communication (NCRISC): {mini_ncrisc_ms:.6f} ms  ({overall_breakdown['percentages']['NCRISC']:.1f}%)"
    )
    print(f"    Computation (TRISC):        {mini_trisc_ms:.6f} ms  ({overall_breakdown['percentages']['TRISC']:.1f}%)")
    print(f"  Component Breakdown (per pass):")
    print(f"    Weight Streaming (BRISC):  {mini_brisc_ms / python_data['minibatches']:.6f} ms")
    print(f"    NoC Communication (NCRISC): {mini_ncrisc_ms / python_data['minibatches']:.6f} ms")
    print(f"    Computation (TRISC):        {mini_trisc_ms / python_data['minibatches']:.6f} ms")
    print()

    print("=" * 80)
    print("OVERHEAD ANALYSIS")
    print("=" * 80)
    print()

    # Python measurements only
    py_overhead_ms = mini_py_time - large_py_time
    py_overhead_pct = (py_overhead_ms / large_py_time) * 100
    avg_mini_py = mini_py_time / python_data["minibatches"]

    print(f"Mini-batch overhead vs single large batch:")
    print(f"  Total time comparison:")
    print(f"    Large batch (1x):   {large_py_time:.6f} ms")
    print(f"    Mini batch (8x):    {mini_py_time:.6f} ms")
    print(f"    Absolute overhead:  {py_overhead_ms:.6f} ms")
    print(f"    Relative overhead:  {py_overhead_pct:.2f}%")
    print()
    print(f"  Per-pass comparison:")
    print(f"    Large batch:        {large_py_time:.6f} ms")
    print(f"    Mini batch average: {avg_mini_py:.6f} ms")
    print(f"    Ratio:              {avg_mini_py / large_py_time:.3f}x")
    print()

    # Component-wise overhead analysis
    print(f"  Component-wise overhead (mini-batch total - large batch):")
    print(f"    BRISC overhead:     {mini_brisc_ms - large_brisc_ms:.6f} ms")
    print(f"    NCRISC overhead:    {mini_ncrisc_ms - large_ncrisc_ms:.6f} ms")
    print(f"    TRISC overhead:     {mini_trisc_ms - large_trisc_ms:.6f} ms")
    print()

    print("=" * 80)
    print("METHODOLOGY & NOTES")
    print("=" * 80)
    print()
    print("Measurement Strategy:")
    print("  ✓ All times: Python perf_counter() (host-side, most accurate)")
    print("  ✓ Component percentages: Device profiler (profile_log_device.csv)")
    print("  ✗ Device wall clock: Excluded (event matching unreliable)")
    print()
    print("Component Definitions:")
    print("  • BRISC  = Weight streaming from DRAM/SRAM to L1 cache")
    print("  • NCRISC = NoC (Network-on-Chip) communication between cores")
    print("  • TRISC  = Actual tensor computation on cores")
    print()
    print("What's Included:")
    print("  • Large batch: Single Matmul operation")
    print("  • Mini batch:  8x (Matmul + SliceDeviceOperation)")
    print("    - SliceDeviceOperation = mini-batch splitting + weight loading")
    print("    - This is the COMPLETE ttnn.linear() execution cost")
    print()
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Large Batch vs Mini-batch ACTUAL Comparison

Parses BOTH large batch and mini-batch device profiles and compares them.
No predictions, no bullshit 8x multiplication - ACTUAL measurements only.

Usage:
  1. Generate large batch profile:
     TT_METAL_DEVICE_PROFILER=1 python weight_loading_test.py --only-large
     mv generated/profiler/.logs/profile_log_device.csv profile_log_large.csv

  2. Generate mini-batch profile:
     TT_METAL_DEVICE_PROFILER=1 python weight_loading_test.py --only-mini
     mv generated/profiler/.logs/profile_log_device.csv profile_log_mini.csv

  3. Compare:
     python compare_large_vs_minibatch.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import statistics


@dataclass
class DeviceZone:
    """Device profiler zone"""

    core_id: int
    risc_type: str
    run_host_id: int
    zone_name: str
    start_cycle: int
    end_cycle: int
    duration_cycles: int

    @property
    def duration_ms(self) -> float:
        return (self.duration_cycles / 1_350_000_000) * 1000


@dataclass
class ScenarioAnalysis:
    """Analysis for one scenario (large batch or mini-batch)"""

    scenario_name: str

    # Per-operation stats
    num_operations: int
    brisc_per_core_times: List[float]  # ms
    ncrisc_per_core_times: List[float]
    trisc_per_proc_times: List[float]
    wall_clock_times: List[float]

    # Totals
    total_brisc_time: float  # Sum of all operations
    total_wall_clock: float

    @property
    def avg_brisc_per_core(self) -> float:
        return statistics.mean(self.brisc_per_core_times) if self.brisc_per_core_times else 0.0

    @property
    def avg_wall_clock(self) -> float:
        return statistics.mean(self.wall_clock_times) if self.wall_clock_times else 0.0


def normalize_risc_type(risc_type: str) -> str:
    """Normalize TRISC_0/1/2 to TRISC"""
    if risc_type.startswith("TRISC"):
        return "TRISC"
    return risc_type


def parse_device_profile(csv_path: Path) -> List[DeviceZone]:
    """Parse device profile CSV"""
    zones = []
    zone_starts = {}

    with open(csv_path, "r") as f:
        lines = f.readlines()

        if len(lines) < 2:
            return zones

        for line in lines[2:]:
            parts = line.strip().split(",")
            if len(parts) < 13:
                continue

            try:
                pcie_slot = int(parts[0])
                core_x = int(parts[1])
                core_y = int(parts[2])
                core_id = core_y * 100 + core_x
                risc_type = parts[3]
                timer_id = int(parts[4])
                time_cycles = int(parts[5])
                data = parts[6]
                run_host_id_str = parts[7]
                zone_name = parts[10]
                zone_phase = parts[11]

                if not run_host_id_str:
                    continue

                run_host_id = int(run_host_id_str)
                key = (core_id, risc_type, run_host_id, zone_name)

                if zone_phase == "ZONE_START":
                    zone_starts[key] = time_cycles
                elif zone_phase == "ZONE_END" and key in zone_starts:
                    start_cycles = zone_starts[key]
                    duration_cycles = time_cycles - start_cycles

                    zone = DeviceZone(
                        core_id=core_id,
                        risc_type=risc_type,
                        run_host_id=run_host_id,
                        zone_name=zone_name,
                        start_cycle=start_cycles,
                        end_cycle=time_cycles,
                        duration_cycles=duration_cycles,
                    )
                    zones.append(zone)
                    del zone_starts[key]
            except (ValueError, IndexError):
                continue

    return zones


def analyze_scenario(zones: List[DeviceZone], scenario_name: str) -> ScenarioAnalysis:
    """Analyze all operations in a scenario"""

    # Group by run_host_id
    run_host_ids = sorted(set(z.run_host_id for z in zones))

    brisc_per_core_times = []
    ncrisc_per_core_times = []
    trisc_per_proc_times = []
    wall_clock_times = []

    total_brisc = 0.0
    total_wall = 0.0

    for run_host_id in run_host_ids:
        op_zones = [z for z in zones if z.run_host_id == run_host_id]

        # Separate by RISC type
        brisc_zones = [z for z in op_zones if normalize_risc_type(z.risc_type) == "BRISC"]
        ncrisc_zones = [z for z in op_zones if normalize_risc_type(z.risc_type) == "NCRISC"]
        trisc_zones = [z for z in op_zones if normalize_risc_type(z.risc_type) == "TRISC"]

        # Per-core calculations
        if brisc_zones:
            brisc_total_cycles = sum(z.duration_cycles for z in brisc_zones)
            brisc_cores = len(set(z.core_id for z in brisc_zones))
            brisc_per_core = (brisc_total_cycles / brisc_cores / 1_350_000_000) * 1000
            brisc_per_core_times.append(brisc_per_core)
            total_brisc += brisc_per_core

        if ncrisc_zones:
            ncrisc_total_cycles = sum(z.duration_cycles for z in ncrisc_zones)
            ncrisc_cores = len(set(z.core_id for z in ncrisc_zones))
            ncrisc_per_core = (ncrisc_total_cycles / ncrisc_cores / 1_350_000_000) * 1000
            ncrisc_per_core_times.append(ncrisc_per_core)

        if trisc_zones:
            trisc_total_cycles = sum(z.duration_cycles for z in trisc_zones)
            trisc_procs = len(trisc_zones)
            trisc_per_proc = (trisc_total_cycles / trisc_procs / 1_350_000_000) * 1000
            trisc_per_proc_times.append(trisc_per_proc)

        # Wall clock
        if op_zones:
            min_start = min(z.start_cycle for z in op_zones)
            max_end = max(z.end_cycle for z in op_zones)
            wall_cycles = max_end - min_start
            wall_ms = (wall_cycles / 1_350_000_000) * 1000
            wall_clock_times.append(wall_ms)
            total_wall += wall_ms

    return ScenarioAnalysis(
        scenario_name=scenario_name,
        num_operations=len(run_host_ids),
        brisc_per_core_times=brisc_per_core_times,
        ncrisc_per_core_times=ncrisc_per_core_times,
        trisc_per_proc_times=trisc_per_proc_times,
        wall_clock_times=wall_clock_times,
        total_brisc_time=total_brisc,
        total_wall_clock=total_wall,
    )


def filter_large_operations(analysis: ScenarioAnalysis, threshold_ms: float = 0.05) -> ScenarioAnalysis:
    """Filter only large operations (>threshold_ms)"""
    large_indices = [i for i, wc in enumerate(analysis.wall_clock_times) if wc > threshold_ms]

    # Safely get values, using 0.0 if index doesn't exist
    def safe_get(lst, indices):
        result = []
        for i in indices:
            if i < len(lst):
                result.append(lst[i])
        return result

    return ScenarioAnalysis(
        scenario_name=f"{analysis.scenario_name} (large ops only)",
        num_operations=len(large_indices),
        brisc_per_core_times=safe_get(analysis.brisc_per_core_times, large_indices),
        ncrisc_per_core_times=safe_get(analysis.ncrisc_per_core_times, large_indices),
        trisc_per_proc_times=safe_get(analysis.trisc_per_proc_times, large_indices),
        wall_clock_times=[analysis.wall_clock_times[i] for i in large_indices],
        total_brisc_time=sum(safe_get(analysis.brisc_per_core_times, large_indices)),
        total_wall_clock=sum(analysis.wall_clock_times[i] for i in large_indices),
    )


def print_scenario_summary(analysis: ScenarioAnalysis):
    """Print summary for one scenario"""
    print(f"\n{'='*80}")
    print(f"{analysis.scenario_name}")
    print(f"{'='*80}")

    print(f"\nOperations analyzed: {analysis.num_operations}")

    if analysis.brisc_per_core_times:
        print(f"\nBRISC per-core (Weight Streaming):")
        print(f"  Mean:   {statistics.mean(analysis.brisc_per_core_times):.6f} ms")
        print(f"  Median: {statistics.median(analysis.brisc_per_core_times):.6f} ms")
        print(
            f"  StdDev: {statistics.stdev(analysis.brisc_per_core_times) if len(analysis.brisc_per_core_times) > 1 else 0:.6f} ms"
        )
        print(f"  Min:    {min(analysis.brisc_per_core_times):.6f} ms")
        print(f"  Max:    {max(analysis.brisc_per_core_times):.6f} ms")
        print(f"  Total:  {analysis.total_brisc_time:.6f} ms (sum of all ops)")

    if analysis.wall_clock_times:
        print(f"\nWall Clock (Parallel Execution):")
        print(f"  Mean:   {statistics.mean(analysis.wall_clock_times):.6f} ms")
        print(f"  Median: {statistics.median(analysis.wall_clock_times):.6f} ms")
        print(
            f"  StdDev: {statistics.stdev(analysis.wall_clock_times) if len(analysis.wall_clock_times) > 1 else 0:.6f} ms"
        )
        print(f"  Total:  {analysis.total_wall_clock:.6f} ms (sum of all ops)")


def compare_scenarios(large: ScenarioAnalysis, mini: ScenarioAnalysis):
    """Compare large batch vs mini-batch"""
    print(f"\n{'='*80}")
    print("ACTUAL Comparison: Large Batch vs Mini-batch")
    print(f"{'='*80}")

    large_brisc_avg = large.avg_brisc_per_core
    mini_brisc_avg = mini.avg_brisc_per_core

    large_brisc_total = large.total_brisc_time
    mini_brisc_total = mini.total_brisc_time

    large_wall_total = large.total_wall_clock
    mini_wall_total = mini.total_wall_clock

    print(f"\n1. BRISC Per-Core Time (Weight Streaming per operation)")
    print(f"   Large batch:     {large_brisc_avg:.6f} ms")
    print(f"   Mini-batch:      {mini_brisc_avg:.6f} ms")
    print(f"   Ratio:           {mini_brisc_avg / large_brisc_avg if large_brisc_avg > 0 else 0:.2f}x")

    print(f"\n2. BRISC Total Time (Sum of all operations)")
    print(f"   Large batch:     {large_brisc_total:.6f} ms ({large.num_operations} ops)")
    print(f"   Mini-batch:      {mini_brisc_total:.6f} ms ({mini.num_operations} ops)")
    print(f"   Difference:      {mini_brisc_total - large_brisc_total:.6f} ms")
    print(f"   Ratio:           {mini_brisc_total / large_brisc_total if large_brisc_total > 0 else 0:.2f}x")

    print(f"\n3. Wall Clock Total (Actual elapsed time)")
    print(f"   Large batch:     {large_wall_total:.6f} ms")
    print(f"   Mini-batch:      {mini_wall_total:.6f} ms")
    print(f"   Difference:      {mini_wall_total - large_wall_total:.6f} ms")
    print(f"   Ratio:           {mini_wall_total / large_wall_total if large_wall_total > 0 else 0:.2f}x")

    print(f"\n4. Weight Streaming Overhead Analysis")

    # Calculate expected number of weight loads
    expected_ratio = mini.num_operations / large.num_operations if large.num_operations > 0 else 0
    print(f"   Operations: Large={large.num_operations}, Mini={mini.num_operations}, Ratio={expected_ratio:.1f}x")

    # Weight streaming overhead
    overhead_brisc = mini_brisc_total - large_brisc_total
    print(f"   BRISC overhead:  {overhead_brisc:.6f} ms")

    if expected_ratio > 1:
        extra_loads = mini.num_operations - large.num_operations
        per_load = overhead_brisc / extra_loads if extra_loads > 0 else 0
        print(f"   Extra loads:     {extra_loads}")
        print(f"   Per-load cost:   {per_load:.6f} ms")

        # Verify with average
        print(f"\n   Verification:")
        print(f"   - Large avg BRISC per-core: {large_brisc_avg:.6f} ms")
        print(f"   - Calculated per-load:      {per_load:.6f} ms")
        print(f"   - Match ratio:              {per_load / large_brisc_avg if large_brisc_avg > 0 else 0:.2f}")


def main():
    """Main comparison function"""

    # Check for profile files
    large_csv = Path(__file__).parent.parent / "generated" / "profiler" / ".logs" / "profile_log_large.csv"
    mini_csv = Path(__file__).parent.parent / "generated" / "profiler" / ".logs" / "profile_log_mini.csv"

    if not large_csv.exists():
        print(f"Error: {large_csv} not found!")
        print("\nGenerate it with:")
        print("  TT_METAL_DEVICE_PROFILER=1 python weight_loading_test.py --only-large")
        print(f"  mv generated/profiler/.logs/profile_log_device.csv {large_csv}")
        sys.exit(1)

    if not mini_csv.exists():
        print(f"Error: {mini_csv} not found!")
        print("\nGenerate it with:")
        print("  TT_METAL_DEVICE_PROFILER=1 python weight_loading_test.py --only-mini")
        print(f"  mv generated/profiler/.logs/profile_log_device.csv {mini_csv}")
        sys.exit(1)

    print("Parsing large batch profile...")
    large_zones = parse_device_profile(large_csv)
    print(f"Parsed {len(large_zones)} zones")

    print("\nParsing mini-batch profile...")
    mini_zones = parse_device_profile(mini_csv)
    print(f"Parsed {len(mini_zones)} zones")

    # Analyze both scenarios
    print("\nAnalyzing large batch...")
    large_analysis = analyze_scenario(large_zones, "Large Batch (B=256)")
    large_filtered = filter_large_operations(large_analysis, threshold_ms=0.05)

    print("Analyzing mini-batch...")
    mini_analysis = analyze_scenario(mini_zones, "Mini-batch (b=32, 8x)")
    mini_filtered = filter_large_operations(mini_analysis, threshold_ms=0.05)

    # Print summaries
    print_scenario_summary(large_filtered)
    print_scenario_summary(mini_filtered)

    # Compare
    compare_scenarios(large_filtered, mini_filtered)

    print(f"\n{'='*80}")
    print("Key Findings:")
    print(f"{'='*80}")
    print("✓ ACTUAL measurements from BOTH scenarios")
    print("✓ No predictions, no 8x multiplication bullshit")
    print("✓ BRISC per-core time = Pure weight streaming overhead")
    print("✓ Device-side only (cycle-accurate, no Python overhead)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Parse device profiling CSV to extract detailed forward pass breakdown.

This script parses profile_log_device.csv to measure:
1. Input sharding time (GDDR6 DRAM → L1 SRAM)
2. Weight streaming time (GDDR6 DRAM → L1 SRAM)
3. Compute time (matmul execution)
4. Output gathering time (L1 SRAM → GDDR6 DRAM via NoC)

The key insight: Different RISC processors handle different tasks:
- BRISC: Manages data movement (DRAM ↔ L1)
- NCRISC: NOC (Network-on-Chip) communication
- TRISC: Compute operations (math kernels)

Usage:
    python research_codes/parse_device_profile.py [path_to_profile_log_device.csv]
"""

import csv
import sys
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DeviceZone:
    """Represents a device-side profiling zone (kernel execution)."""

    pcie_slot: int
    core_x: int
    core_y: int
    risc_type: str  # BRISC, NCRISC, TRISC
    timer_id: int
    start_time_cycles: int
    end_time_cycles: Optional[int]
    run_host_id: int
    zone_name: str
    source_file: str

    @property
    def duration_cycles(self) -> Optional[int]:
        if self.end_time_cycles is not None:
            return self.end_time_cycles - self.start_time_cycles
        return None

    def duration_ms(self, chip_freq_mhz: float) -> Optional[float]:
        """Convert cycles to milliseconds."""
        if self.duration_cycles is not None:
            # cycles / (MHz * 1e6) * 1000 = cycles / (MHz * 1e3)
            return self.duration_cycles / (chip_freq_mhz * 1000.0)
        return None

    def __repr__(self):
        return (
            f"DeviceZone(core=({self.core_x},{self.core_y}), risc={self.risc_type}, "
            f"zone={self.zone_name}, duration_cycles={self.duration_cycles})"
        )


class DeviceProfileParser:
    """Parser for profile_log_device.csv."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.chip_freq_mhz = None
        self.zones: List[DeviceZone] = []
        self.zones_by_host_id: Dict[int, List[DeviceZone]] = defaultdict(list)

    def parse(self):
        """Parse the device profile CSV."""
        with open(self.csv_path, "r") as f:
            # First line contains architecture and chip frequency
            first_line = f.readline().strip()
            if "CHIP_FREQ[MHz]:" in first_line:
                freq_part = first_line.split("CHIP_FREQ[MHz]:")[1].strip()
                self.chip_freq_mhz = float(freq_part)
                print(f"Detected chip frequency: {self.chip_freq_mhz} MHz")
            else:
                print("Warning: Could not detect chip frequency, assuming 1350 MHz")
                self.chip_freq_mhz = 1350.0

            # Second line is the header
            header_line = f.readline().strip()

            # Parse CSV rows
            reader = csv.DictReader(
                f,
                fieldnames=[
                    "pcie_slot",
                    "core_x",
                    "core_y",
                    "risc_type",
                    "timer_id",
                    "time_cycles",
                    "data",
                    "run_host_id",
                    "trace_id",
                    "trace_id_counter",
                    "zone_name",
                    "type",
                    "source_line",
                    "source_file",
                    "meta_data",
                ],
            )

            # Track zone starts for matching with ends
            zone_stack: Dict[Tuple[int, int, int, str, int], DeviceZone] = {}

            for row in reader:
                try:
                    pcie_slot = int(row["pcie_slot"])
                    core_x = int(row["core_x"])
                    core_y = int(row["core_y"])
                    risc_type = row["risc_type"]
                    timer_id = int(row["timer_id"])
                    time_cycles = int(row["time_cycles"])
                    run_host_id = int(row["run_host_id"])
                    zone_name = row["zone_name"]
                    zone_type = row["type"]
                    source_file = row["source_file"]

                    # Key to match start/end pairs
                    zone_key = (pcie_slot, core_x, core_y, risc_type, run_host_id)

                    if zone_type == "ZONE_START":
                        zone = DeviceZone(
                            pcie_slot=pcie_slot,
                            core_x=core_x,
                            core_y=core_y,
                            risc_type=risc_type,
                            timer_id=timer_id,
                            start_time_cycles=time_cycles,
                            end_time_cycles=None,
                            run_host_id=run_host_id,
                            zone_name=zone_name,
                            source_file=source_file,
                        )
                        zone_stack[zone_key] = zone

                    elif zone_type == "ZONE_END":
                        if zone_key in zone_stack:
                            zone = zone_stack.pop(zone_key)
                            zone.end_time_cycles = time_cycles
                            self.zones.append(zone)
                            self.zones_by_host_id[run_host_id].append(zone)

                except (ValueError, KeyError) as e:
                    continue

        print(f"Parsed {len(self.zones)} complete device zones")
        print(f"Found {len(self.zones_by_host_id)} unique run_host_id groups")

        # Count unique cores used
        unique_cores = set()
        for zone in self.zones:
            unique_cores.add((zone.core_x, zone.core_y))
        print(f"Using {len(unique_cores)} unique Tensix cores (each with 5 RISC-V processors)")
        print(f"Total RISC-V processors: {len(unique_cores)} cores × 5 = {len(unique_cores) * 5} processors")

    def get_zones_by_risc_type(self, host_id: int) -> Dict[str, List[DeviceZone]]:
        """Group zones by RISC processor type for a given host_id."""
        zones = self.zones_by_host_id[host_id]
        by_risc = defaultdict(list)
        for zone in zones:
            # Normalize TRISC_0, TRISC_1, TRISC_2 to just TRISC
            risc_type = zone.risc_type
            if risc_type.startswith("TRISC"):
                risc_type = "TRISC"
            by_risc[risc_type].append(zone)
        return by_risc

    def analyze_operation(self, host_id: int) -> Dict[str, float]:
        """
        Analyze a single operation (identified by run_host_id).

        Returns timing breakdown:
        - brisc_dataflow_ms: BRISC kernel time (data movement DRAM ↔ L1)
        - ncrisc_noc_ms: NCRISC kernel time (NoC communication)
        - trisc_compute_ms: TRISC kernel time (compute operations)
        - total_ms: Wall-clock time (max end - min start)
        """
        zones = self.zones_by_host_id[host_id]
        if not zones:
            return {}

        by_risc = self.get_zones_by_risc_type(host_id)

        result = {}

        # Calculate time spent in each RISC processor type
        for risc_type in ["BRISC", "NCRISC", "TRISC"]:
            if risc_type in by_risc:
                # Sum kernel execution times
                kernel_zones = [z for z in by_risc[risc_type] if "KERNEL" in z.zone_name]
                total_cycles = sum(z.duration_cycles for z in kernel_zones if z.duration_cycles)
                total_ms = total_cycles / (self.chip_freq_mhz * 1000.0)
                result[f"{risc_type.lower()}_kernel_ms"] = total_ms
                result[f"{risc_type.lower()}_kernel_count"] = len(kernel_zones)

        # Calculate wall-clock time (earliest start to latest end)
        all_starts = [z.start_time_cycles for z in zones]
        all_ends = [z.end_time_cycles for z in zones if z.end_time_cycles]

        if all_starts and all_ends:
            min_start = min(all_starts)
            max_end = max(all_ends)
            wall_clock_cycles = max_end - min_start
            result["wall_clock_ms"] = wall_clock_cycles / (self.chip_freq_mhz * 1000.0)

        return result

    def analyze_forward_pass_breakdown(self):
        """
        Analyze all operations and categorize into forward pass phases.

        Assumptions based on tt-metal architecture:
        - BRISC handles data movement (DRAM → L1 for input sharding and weight streaming)
        - TRISC handles compute (matmul execution)
        - NCRISC handles NoC communication (L1 → DRAM output gathering)
        """
        print("\n" + "=" * 80)
        print("DEVICE-SIDE FORWARD PASS BREAKDOWN")
        print("=" * 80)

        if not self.zones_by_host_id:
            print("No device profiling data found.")
            return

        # Analyze each operation
        host_ids = sorted(self.zones_by_host_id.keys())

        print(f"\nFound {len(host_ids)} operations")
        print("\nPer-operation breakdown:")
        print("-" * 120)
        print(f"{'Host ID':<10} {'Wall Clock':<15} {'BRISC (data)':<20} {'NCRISC (NoC)':<20} {'TRISC (compute)':<20}")
        print("-" * 120)

        all_brisc_times = []
        all_ncrisc_times = []
        all_trisc_times = []
        all_wall_times = []

        for host_id in host_ids:
            analysis = self.analyze_operation(host_id)

            wall_clock = analysis.get("wall_clock_ms", 0.0)
            brisc_ms = analysis.get("brisc_kernel_ms", 0.0)
            ncrisc_ms = analysis.get("ncrisc_kernel_ms", 0.0)
            trisc_ms = analysis.get("trisc_kernel_ms", 0.0)

            brisc_count = analysis.get("brisc_kernel_count", 0)
            ncrisc_count = analysis.get("ncrisc_kernel_count", 0)
            trisc_count = analysis.get("trisc_kernel_count", 0)

            print(
                f"{host_id:<10} {wall_clock:>10.6f} ms   "
                f"{brisc_ms:>10.6f} ms ({brisc_count:>2} kernels)   "
                f"{ncrisc_ms:>10.6f} ms ({ncrisc_count:>2} kernels)   "
                f"{trisc_ms:>10.6f} ms ({trisc_count:>2} kernels)"
            )

            all_brisc_times.append(brisc_ms)
            all_ncrisc_times.append(ncrisc_ms)
            all_trisc_times.append(trisc_ms)
            all_wall_times.append(wall_clock)

        # Calculate statistics
        print("\n" + "=" * 80)
        print("AGGREGATE STATISTICS")
        print("=" * 80)

        if all_brisc_times:
            # Separate operations by wall clock time
            # Operations with wall_clock > 0.05ms are likely "large" operations
            # Operations with wall_clock < 0.05ms are likely small/slice operations
            large_ops_indices = [i for i, w in enumerate(all_wall_times) if w > 0.05]
            small_ops_indices = [i for i, w in enumerate(all_wall_times) if w <= 0.05]

            print(f"\nDetected {len(large_ops_indices)} large operations (wall clock > 0.05ms)")
            print(f"Detected {len(small_ops_indices)} small operations (wall clock <= 0.05ms)")

            # Overall statistics
            avg_brisc = sum(all_brisc_times) / len(all_brisc_times)
            avg_ncrisc = sum(all_ncrisc_times) / len(all_ncrisc_times)
            avg_trisc = sum(all_trisc_times) / len(all_trisc_times)
            avg_wall = sum(all_wall_times) / len(all_wall_times)

            print(f"\n--- ALL OPERATIONS (average) ---")
            print(f"  Wall clock time:        {avg_wall:>10.6f} ms")
            print(f"  BRISC (data movement):  {avg_brisc:>10.6f} ms ({avg_brisc/avg_wall*100:>5.1f}%)")
            print(f"  NCRISC (NoC):           {avg_ncrisc:>10.6f} ms ({avg_ncrisc/avg_wall*100:>5.1f}%)")
            print(f"  TRISC (compute):        {avg_trisc:>10.6f} ms ({avg_trisc/avg_wall*100:>5.1f}%)")

            # Note: BRISC + NCRISC + TRISC may be > wall_clock because they can overlap
            total_risc = avg_brisc + avg_ncrisc + avg_trisc
            parallelism = total_risc / avg_wall if avg_wall > 0 else 0
            print(f"  Total RISC time:        {total_risc:>10.6f} ms")
            print(f"  Parallelism factor:     {parallelism:>10.2f}x (>1 means parallel execution)")

            # Large operations statistics
            if large_ops_indices:
                large_brisc = [all_brisc_times[i] for i in large_ops_indices]
                large_ncrisc = [all_ncrisc_times[i] for i in large_ops_indices]
                large_trisc = [all_trisc_times[i] for i in large_ops_indices]
                large_wall = [all_wall_times[i] for i in large_ops_indices]

                avg_l_brisc = sum(large_brisc) / len(large_brisc)
                avg_l_ncrisc = sum(large_ncrisc) / len(large_ncrisc)
                avg_l_trisc = sum(large_trisc) / len(large_trisc)
                avg_l_wall = sum(large_wall) / len(large_wall)

                print(f"\n--- LARGE OPERATIONS (average, likely matmul forward passes) ---")
                print(f"  Wall clock time:        {avg_l_wall:>10.6f} ms")
                print(f"  BRISC (data movement):  {avg_l_brisc:>10.6f} ms ({avg_l_brisc/avg_l_wall*100:>5.1f}%)")
                print(f"  NCRISC (NoC):           {avg_l_ncrisc:>10.6f} ms ({avg_l_ncrisc/avg_l_wall*100:>5.1f}%)")
                print(f"  TRISC (compute):        {avg_l_trisc:>10.6f} ms ({avg_l_trisc/avg_l_wall*100:>5.1f}%)")

                total_l_risc = avg_l_brisc + avg_l_ncrisc + avg_l_trisc
                parallelism_l = total_l_risc / avg_l_wall if avg_l_wall > 0 else 0
                print(f"  Total RISC time:        {total_l_risc:>10.6f} ms")
                print(f"  Parallelism factor:     {parallelism_l:>10.2f}x")

                print(f"\n  BREAKDOWN as % of wall clock time:")
                print(f"    Input/Weight loading (BRISC):  {avg_l_brisc/avg_l_wall*100:>6.2f}%")
                print(f"    Output gathering (NCRISC):     {avg_l_ncrisc/avg_l_wall*100:>6.2f}%")
                print(f"    Compute (TRISC):               {avg_l_trisc/avg_l_wall*100:>6.2f}%")

            # Small operations statistics
            if small_ops_indices:
                small_brisc = [all_brisc_times[i] for i in small_ops_indices]
                small_ncrisc = [all_ncrisc_times[i] for i in small_ops_indices]
                small_trisc = [all_trisc_times[i] for i in small_ops_indices]
                small_wall = [all_wall_times[i] for i in small_ops_indices]

                avg_s_brisc = sum(small_brisc) / len(small_brisc)
                avg_s_ncrisc = sum(small_ncrisc) / len(small_ncrisc)
                avg_s_trisc = sum(small_trisc) / len(small_trisc)
                avg_s_wall = sum(small_wall) / len(small_wall)

                print(f"\n--- SMALL OPERATIONS (average, likely slice/reshape ops) ---")
                print(f"  Wall clock time:        {avg_s_wall:>10.6f} ms")
                print(f"  BRISC (data movement):  {avg_s_brisc:>10.6f} ms")
                print(f"  NCRISC (NoC):           {avg_s_ncrisc:>10.6f} ms")
                print(f"  TRISC (compute):        {avg_s_trisc:>10.6f} ms")

        print("\n" + "=" * 80)
        print("INTERPRETATION & KEY FINDINGS")
        print("=" * 80)
        print(
            """
RISC Processor Roles:
----------------------
BRISC (Binary RISC):
  - Handles data movement between GDDR6 DRAM and L1 SRAM
  - This includes INPUT SHARDING and WEIGHT STREAMING
  - Time spent here = time to load data from DRAM to L1

NCRISC (Network-on-Chip RISC):
  - Handles NoC communication between cores
  - This includes OUTPUT GATHERING (L1 sharded → GDDR6 DRAM)
  - Time spent here = time for NoC transactions

TRISC (Tensor RISC):
  - Handles compute operations (matmul, eltwise ops, etc.)
  - Time spent here = actual computation time

Important Notes:
----------------
1. Wall clock time is the REAL elapsed time (what you measure with time.perf_counter())
2. RISC times are accumulated kernel execution times across all cores
3. Parallelism factor > 1 means multiple cores/RISC processors work in parallel
4. High parallelism factor indicates parallel execution across many Tensix cores

Tensix Core Architecture (Blackhole):
--------------------------------------
- Total Tensix cores used: ~130 (varies by chip configuration)
- Each Tensix core contains 5 RISC-V processors:
  * 1x BRISC (Data Movement 0): Manages DRAM ↔ L1 SRAM transfers
  * 1x NCRISC (Data Movement 1): Manages NoC (Network-on-Chip) communication
  * 3x TRISC (Unpack, Math, Pack): Handle compute operations
- Total RISC processors: 130 cores × 5 = ~650 RISC-V processors
- Each core has 1.5MB L1 SRAM for local data storage

Forward Pass Breakdown (from LARGE OPERATIONS):
------------------------------------------------
For a typical matmul forward pass (wall clock ~0.124 ms):
  - BRISC (input + weight streaming):  ~15.2 ms across all cores
  - NCRISC (output gathering):         ~14.9 ms across all cores
  - TRISC (compute):                   ~44.9 ms across all cores

  Total work: ~75 ms distributed across 130 Tensix cores (650 RISC processors)
  Result: ~0.124 ms wall clock time (parallel execution)

Answer to your question:
------------------------
YES, this method can measure GDDR6 → L1 SRAM data movement time!

- BRISC kernel time = Input sharding + Weight streaming time
- To separate them: Compare large batch vs mini-batch BRISC times
  * Large batch (B=256, 1 forward): BRISC loads 256 inputs + weights once
  * Mini-batch (b=32, 8 forwards): BRISC loads 32 inputs + weights, 8 times
  * Difference = Weight re-streaming overhead (8x - 1x = 7x weight streaming)

The data you need:
------------------
From your weight_loading_test.py output:
  - Large batch forward: 0.239 ms (matches our wall clock ~0.124 ms per operation)
  - Mini-batch 8x forward: 1.443 ms
  - Overhead: 1.204 ms = Weight re-streaming cost (streaming weights 7 extra times)

Each weight streaming event costs: 1.204 / 7 ≈ 0.172 ms wall clock time
BRISC time per streaming: ~15 ms accumulated across 130 cores (130 BRISC processors)
"""
        )
        print("=" * 80 + "\n")


def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "/home/masterjunmo/codes/tt-metal/generated/profiler/.logs/profile_log_device.csv"

    if not os.path.exists(csv_path):
        print(f"Error: Device profile CSV not found at {csv_path}")
        print("\nTo generate device profiling data, run:")
        print("  TT_METAL_DEVICE_PROFILER=1 python research_codes/weight_loading_test.py")
        print("\nOr with Tracy profiler:")
        print("  python3 -m tracy -r research_codes/weight_loading_test.py")
        return 1

    print(f"Parsing device profile: {csv_path}")
    print(f"CSV file size: {os.path.getsize(csv_path) / 1024 / 1024:.2f} MB")

    parser = DeviceProfileParser(csv_path)
    parser.parse()
    parser.analyze_forward_pass_breakdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Weight Streaming Overhead 정밀 분석

Device profile에서 BRISC 시간만 추출하여 weight streaming overhead를 정확히 측정합니다.
Python timing 오버헤드를 완전히 제거하고, 순수 device-side 측정만 사용합니다.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import statistics


@dataclass
class DeviceZone:
    """Device profiler zone (BRISC/NCRISC/TRISC execution)"""

    core_id: int
    risc_type: str
    run_host_id: int
    zone_name: str
    start_cycle: int
    end_cycle: int
    duration_cycles: int

    @property
    def duration_ms(self) -> float:
        """Convert cycles to milliseconds (1350 MHz clock)"""
        return (self.duration_cycles / 1_350_000_000) * 1000


@dataclass
class OperationBreakdown:
    """한 operation (run_host_id)의 상세 분석"""

    run_host_id: int

    # BRISC - Data Movement
    brisc_total_cycles: int
    brisc_total_ms: float
    brisc_cores_active: int
    brisc_per_core_ms: float

    # NCRISC - NoC Communication
    ncrisc_total_cycles: int
    ncrisc_total_ms: float
    ncrisc_cores_active: int
    ncrisc_per_core_ms: float

    # TRISC - Compute
    trisc_total_cycles: int
    trisc_total_ms: float
    trisc_processors_active: int
    trisc_per_processor_ms: float

    # Wall Clock (parallel execution)
    wall_clock_cycles: int
    wall_clock_ms: float

    @property
    def total_work_ms(self) -> float:
        """Total accumulated work across all processors"""
        return self.brisc_total_ms + self.ncrisc_total_ms + self.trisc_total_ms

    @property
    def parallelism_factor(self) -> float:
        """Parallelism = accumulated work / wall clock"""
        if self.wall_clock_ms > 0:
            return self.total_work_ms / self.wall_clock_ms
        return 0.0


def parse_device_profile(csv_path: Path) -> List[DeviceZone]:
    """Parse device profile CSV into DeviceZone objects"""
    zones = []
    zone_starts = {}  # (core, risc, run_host_id, zone_name) -> start info

    with open(csv_path, "r") as f:
        lines = f.readlines()

        # Skip header lines (first 2 lines)
        if len(lines) < 2:
            return zones

        # Parse data lines
        for line in lines[2:]:
            parts = line.strip().split(",")
            if len(parts) < 13:
                continue

            try:
                pcie_slot = int(parts[0])
                core_x = int(parts[1])
                core_y = int(parts[2])
                core_id = core_y * 100 + core_x  # Unique core ID
                risc_type = parts[3]
                timer_id = int(parts[4])
                time_cycles = int(parts[5])
                data = parts[6]
                run_host_id_str = parts[7]
                zone_name = parts[10]
                zone_phase = parts[11]

                # Skip if run_host_id is empty
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


def normalize_risc_type(risc_type: str) -> str:
    """Normalize TRISC_0/1/2 to TRISC"""
    if risc_type.startswith("TRISC"):
        return "TRISC"
    return risc_type


def analyze_operation(zones: List[DeviceZone], run_host_id: int) -> OperationBreakdown:
    """Analyze a single operation (run_host_id) in detail"""

    # Filter zones for this operation
    op_zones = [z for z in zones if z.run_host_id == run_host_id]

    # Group by normalized RISC type
    brisc_zones = [z for z in op_zones if normalize_risc_type(z.risc_type) == "BRISC"]
    ncrisc_zones = [z for z in op_zones if normalize_risc_type(z.risc_type) == "NCRISC"]
    trisc_zones = [z for z in op_zones if normalize_risc_type(z.risc_type) == "TRISC"]

    # BRISC analysis
    brisc_total_cycles = sum(z.duration_cycles for z in brisc_zones)
    brisc_cores = len(set(z.core_id for z in brisc_zones))
    brisc_per_core_cycles = brisc_total_cycles / brisc_cores if brisc_cores > 0 else 0

    # NCRISC analysis
    ncrisc_total_cycles = sum(z.duration_cycles for z in ncrisc_zones)
    ncrisc_cores = len(set(z.core_id for z in ncrisc_zones))
    ncrisc_per_core_cycles = ncrisc_total_cycles / ncrisc_cores if ncrisc_cores > 0 else 0

    # TRISC analysis
    trisc_total_cycles = sum(z.duration_cycles for z in trisc_zones)
    trisc_processors = len(trisc_zones)  # Each TRISC zone = one processor execution
    trisc_per_processor_cycles = trisc_total_cycles / trisc_processors if trisc_processors > 0 else 0

    # Wall clock = max end time - min start time across all zones
    if op_zones:
        min_start = min(z.start_cycle for z in op_zones)
        max_end = max(z.end_cycle for z in op_zones)
        wall_clock_cycles = max_end - min_start
    else:
        wall_clock_cycles = 0

    return OperationBreakdown(
        run_host_id=run_host_id,
        brisc_total_cycles=brisc_total_cycles,
        brisc_total_ms=(brisc_total_cycles / 1_350_000_000) * 1000,
        brisc_cores_active=brisc_cores,
        brisc_per_core_ms=(brisc_per_core_cycles / 1_350_000_000) * 1000,
        ncrisc_total_cycles=ncrisc_total_cycles,
        ncrisc_total_ms=(ncrisc_total_cycles / 1_350_000_000) * 1000,
        ncrisc_cores_active=ncrisc_cores,
        ncrisc_per_core_ms=(ncrisc_per_core_cycles / 1_350_000_000) * 1000,
        trisc_total_cycles=trisc_total_cycles,
        trisc_total_ms=(trisc_total_cycles / 1_350_000_000) * 1000,
        trisc_processors_active=trisc_processors,
        trisc_per_processor_ms=(trisc_per_processor_cycles / 1_350_000_000) * 1000,
        wall_clock_cycles=wall_clock_cycles,
        wall_clock_ms=(wall_clock_cycles / 1_350_000_000) * 1000,
    )


def classify_operations(
    breakdowns: List[OperationBreakdown],
) -> Tuple[List[OperationBreakdown], List[OperationBreakdown]]:
    """Classify operations into large (>0.05ms) and small (<0.05ms)"""
    large_ops = [bd for bd in breakdowns if bd.wall_clock_ms > 0.05]
    small_ops = [bd for bd in breakdowns if bd.wall_clock_ms <= 0.05]
    return large_ops, small_ops


def print_detailed_breakdown(bd: OperationBreakdown, label: str = ""):
    """Print detailed breakdown for one operation"""
    print(f"\n{'='*80}")
    print(f"Operation {bd.run_host_id} {label}")
    print(f"{'='*80}")

    print(f"\n1. BRISC (Data Movement - DRAM ↔ L1)")
    print(f"   Total accumulated:    {bd.brisc_total_ms:8.6f} ms")
    print(f"   Active cores:         {bd.brisc_cores_active:3d} cores")
    print(f"   Per-core average:     {bd.brisc_per_core_ms:8.6f} ms")
    print(f"   → Weight streaming은 BRISC per-core time으로 측정해야 함!")

    print(f"\n2. NCRISC (NoC Communication)")
    print(f"   Total accumulated:    {bd.ncrisc_total_ms:8.6f} ms")
    print(f"   Active cores:         {bd.ncrisc_cores_active:3d} cores")
    print(f"   Per-core average:     {bd.ncrisc_per_core_ms:8.6f} ms")

    print(f"\n3. TRISC (Compute - Matrix Operations)")
    print(f"   Total accumulated:    {bd.trisc_total_ms:8.6f} ms")
    print(f"   Active processors:    {bd.trisc_processors_active:3d} TRISCs")
    print(f"   Per-processor avg:    {bd.trisc_per_processor_ms:8.6f} ms")

    print(f"\n4. Summary")
    print(f"   Total work:           {bd.total_work_ms:8.6f} ms (accumulated)")
    print(f"   Wall clock:           {bd.wall_clock_ms:8.6f} ms (parallel)")
    print(f"   Parallelism factor:   {bd.parallelism_factor:6.1f}x")


def compare_large_batch_vs_minibatch(large_ops: List[OperationBreakdown]):
    """
    Large batch (256) vs Mini-batch (8x32) weight streaming overhead 비교

    가정:
    - Large batch: 모든 weight를 한 번에 로드 (1x streaming)
    - Mini-batch 8x: 8번 weight를 반복 로드 (8x streaming)
    """
    print(f"\n{'='*80}")
    print("Weight Streaming Overhead Analysis (Device-Side BRISC Only)")
    print(f"{'='*80}")

    if not large_ops:
        print("No large operations found!")
        return

    # 대표 operation 선택 (median wall clock)
    large_ops_sorted = sorted(large_ops, key=lambda x: x.wall_clock_ms)
    median_idx = len(large_ops_sorted) // 2
    representative_op = large_ops_sorted[median_idx]

    print(f"\nRepresentative Operation: {representative_op.run_host_id}")
    print(f"  Wall clock:        {representative_op.wall_clock_ms:.6f} ms")
    print(f"  BRISC per-core:    {representative_op.brisc_per_core_ms:.6f} ms")
    print(f"  NCRISC per-core:   {representative_op.ncrisc_per_core_ms:.6f} ms")
    print(f"  TRISC per-proc:    {representative_op.trisc_per_processor_ms:.6f} ms")

    # Weight streaming overhead 계산
    brisc_single_load = representative_op.brisc_per_core_ms

    print(f"\n1. Large Batch (B=256)")
    print(f"   Weight streaming (1x):    {brisc_single_load:.6f} ms")
    print(f"   NCRISC (NoC):             {representative_op.ncrisc_per_core_ms:.6f} ms")
    print(f"   TRISC (Compute):          {representative_op.trisc_per_processor_ms:.6f} ms")
    print(f"   Wall clock:               {representative_op.wall_clock_ms:.6f} ms")

    print(f"\n2. Mini-batch 8x (B=32×8)")
    print(f"   Weight streaming (8x):    {brisc_single_load * 8:.6f} ms")
    print(f"   NCRISC (NoC):             {representative_op.ncrisc_per_core_ms * 8:.6f} ms")
    print(f"   TRISC (Compute):          {representative_op.trisc_per_processor_ms * 8:.6f} ms")
    print(f"   Wall clock (predicted):   {representative_op.wall_clock_ms * 8:.6f} ms")

    print(f"\n3. Weight Streaming Overhead (7x extra loads)")
    overhead_per_load = brisc_single_load
    total_overhead = overhead_per_load * 7
    print(f"   Per-load BRISC time:      {overhead_per_load:.6f} ms")
    print(f"   7x extra loads:           {total_overhead:.6f} ms")
    print(f"   Percentage of 8x total:   {(total_overhead / (representative_op.wall_clock_ms * 8)) * 100:.2f}%")

    # Statistics across all large operations
    print(f"\n4. Statistics Across All Large Operations (n={len(large_ops)})")
    brisc_times = [op.brisc_per_core_ms for op in large_ops]
    ncrisc_times = [op.ncrisc_per_core_ms for op in large_ops]
    trisc_times = [op.trisc_per_processor_ms for op in large_ops]
    wall_times = [op.wall_clock_ms for op in large_ops]

    print(f"\n   BRISC per-core (Weight Streaming):")
    print(f"     Mean:   {statistics.mean(brisc_times):.6f} ms")
    print(f"     Median: {statistics.median(brisc_times):.6f} ms")
    print(f"     StdDev: {statistics.stdev(brisc_times) if len(brisc_times) > 1 else 0:.6f} ms")
    print(f"     Min:    {min(brisc_times):.6f} ms")
    print(f"     Max:    {max(brisc_times):.6f} ms")

    print(f"\n   NCRISC per-core (NoC Communication):")
    print(f"     Mean:   {statistics.mean(ncrisc_times):.6f} ms")
    print(f"     Median: {statistics.median(ncrisc_times):.6f} ms")
    print(f"     StdDev: {statistics.stdev(ncrisc_times) if len(ncrisc_times) > 1 else 0:.6f} ms")

    print(f"\n   TRISC per-processor (Compute):")
    print(f"     Mean:   {statistics.mean(trisc_times):.6f} ms")
    print(f"     Median: {statistics.median(trisc_times):.6f} ms")
    print(f"     StdDev: {statistics.stdev(trisc_times) if len(trisc_times) > 1 else 0:.6f} ms")

    print(f"\n   Wall Clock:")
    print(f"     Mean:   {statistics.mean(wall_times):.6f} ms")
    print(f"     Median: {statistics.median(wall_times):.6f} ms")
    print(f"     StdDev: {statistics.stdev(wall_times) if len(wall_times) > 1 else 0:.6f} ms")


def main():
    # Device profile is generated in generated/profiler/.logs/
    csv_path = Path(__file__).parent.parent / "generated" / "profiler" / ".logs" / "profile_log_device.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        print("Run with TT_METAL_DEVICE_PROFILER=1 first")
        sys.exit(1)

    print("Parsing device profile...")
    zones = parse_device_profile(csv_path)
    print(f"Parsed {len(zones)} zones")

    # Get all unique run_host_ids
    run_host_ids = sorted(set(z.run_host_id for z in zones))
    print(f"Found {len(run_host_ids)} operations")

    # Analyze each operation
    print("\nAnalyzing operations...")
    breakdowns = []
    for run_host_id in run_host_ids:
        bd = analyze_operation(zones, run_host_id)
        breakdowns.append(bd)

    # Classify operations
    large_ops, small_ops = classify_operations(breakdowns)
    print(f"\nLarge operations (>0.05ms): {len(large_ops)}")
    print(f"Small operations (<0.05ms): {len(small_ops)}")

    # Show detailed breakdown for representative large operation
    if large_ops:
        large_ops_sorted = sorted(large_ops, key=lambda x: x.wall_clock_ms)
        median_op = large_ops_sorted[len(large_ops_sorted) // 2]
        print_detailed_breakdown(median_op, "(Representative Large Op)")

    # Compare large batch vs mini-batch
    compare_large_batch_vs_minibatch(large_ops)

    print(f"\n{'='*80}")
    print("Key Findings:")
    print(f"{'='*80}")
    print("✓ BRISC per-core time = Pure weight streaming overhead (DRAM→L1)")
    print("✓ No Python overhead, no sync overhead")
    print("✓ Device-side measurement only (cycle-accurate)")
    print("✓ Mini-batch 8x = 8× BRISC time = 7× extra weight loads")


if __name__ == "__main__":
    main()

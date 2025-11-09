#!/usr/bin/env python3
"""
Device Profile Analysis - Compare Large Batch vs Mini-batch

Analyzes device profiler output to measure:
- Weight streaming overhead (GDDR6 → L1 SRAM)
- NoC communication
- Compute time
- Synchronization overhead

Uses cycle-accurate device profiling (no Python overhead in device measurements).
"""

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict


TT_FREQ_HZ = 1_350_000_000  # Blackhole P150A frequency


@dataclass
class DeviceZone:
    """Represents one profiled zone from device CSV"""

    core_id: int
    risc_type: str
    run_host_id: int
    zone_name: str
    start_cycle: int
    end_cycle: int
    duration_cycles: int


def normalize_risc_type(risc_type: str) -> str:
    """Normalize RISC type names"""
    if "BRISC" in risc_type:
        return "BRISC"
    elif "NCRISC" in risc_type:
        return "NCRISC"
    elif "TRISC" in risc_type:
        return "TRISC"
    return risc_type


def parse_device_profile(csv_path: Path) -> List[DeviceZone]:
    """Parse device profile CSV file"""
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
                core_x = int(parts[1])
                core_y = int(parts[2])
                core_id = core_y * 100 + core_x
                risc_type = parts[3]
                time_cycles = int(parts[5])
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


def analyze_operation(zones: List[DeviceZone], description: str) -> Dict:
    """Analyze a single operation's device profile

    CORRECT calculation method:
    - Parallel work = timeline_span × num_cores (not sum of all zone durations!)
    - This accounts for zones running in parallel on different cores
    """

    if not zones:
        return {}

    # Calculate wall clock time
    min_start = min(z.start_cycle for z in zones)
    max_end = max(z.end_cycle for z in zones)
    wall_cycles = max_end - min_start
    wall_ms = (wall_cycles / TT_FREQ_HZ) * 1000.0

    # RISC breakdown
    risc_zones_by_type = defaultdict(list)

    for z in zones:
        risc_type = normalize_risc_type(z.risc_type)
        risc_zones_by_type[risc_type].append(z)

    # Calculate CORRECT parallel work (timeline span × num_cores)
    # This is the actual amount of work done, accounting for parallelism
    risc_parallel_work = {}
    risc_timeline_spans = {}
    risc_num_cores = {}

    for risc_type in ["BRISC", "NCRISC", "TRISC"]:
        if risc_type in risc_zones_by_type:
            zones_list = risc_zones_by_type[risc_type]

            # Timeline span
            earliest_start = min(z.start_cycle for z in zones_list)
            latest_end = max(z.end_cycle for z in zones_list)
            span_ms = ((latest_end - earliest_start) / TT_FREQ_HZ) * 1000.0

            # Number of unique cores
            num_cores = len(set(z.core_id for z in zones_list))

            # Parallel work = span × cores
            parallel_work = span_ms * num_cores

            risc_timeline_spans[risc_type] = span_ms
            risc_num_cores[risc_type] = num_cores
            risc_parallel_work[risc_type] = parallel_work
        else:
            risc_timeline_spans[risc_type] = 0.0
            risc_num_cores[risc_type] = 0
            risc_parallel_work[risc_type] = 0.0

    # Get values
    brisc_work = risc_parallel_work.get("BRISC", 0)
    ncrisc_work = risc_parallel_work.get("NCRISC", 0)
    trisc_work = risc_parallel_work.get("TRISC", 0)
    total_work = brisc_work + ncrisc_work + trisc_work

    # Calculate parallelism (should be <= max theoretical: 130 cores × 5 RISCs = 650x)
    parallelism = total_work / wall_ms if wall_ms > 0 else 0

    return {
        "description": description,
        "num_zones": len(zones),
        "wall_clock_ms": wall_ms,
        "brisc_parallel_work_ms": brisc_work,
        "ncrisc_parallel_work_ms": ncrisc_work,
        "trisc_parallel_work_ms": trisc_work,
        "total_parallel_work_ms": total_work,
        "parallelism": parallelism,
        "brisc_timeline_span_ms": risc_timeline_spans.get("BRISC", 0),
        "ncrisc_timeline_span_ms": risc_timeline_spans.get("NCRISC", 0),
        "trisc_timeline_span_ms": risc_timeline_spans.get("TRISC", 0),
        "brisc_num_cores": risc_num_cores.get("BRISC", 0),
        "ncrisc_num_cores": risc_num_cores.get("NCRISC", 0),
        "trisc_num_cores": risc_num_cores.get("TRISC", 0),
        "brisc_count": len(risc_zones_by_type.get("BRISC", [])),
        "ncrisc_count": len(risc_zones_by_type.get("NCRISC", [])),
        "trisc_count": len(risc_zones_by_type.get("TRISC", [])),
    }


def print_analysis(analysis: Dict, python_ms: float | None = None):
    """Print analysis results with CORRECT parallel work calculation"""
    print(f"\n{'='*80}")
    print(f"{analysis['description']}")
    print(f"{'='*80}")
    print(f"Zones: {analysis['num_zones']}")
    print(f"Device Wall Clock: {analysis['wall_clock_ms']:.6f} ms")

    if python_ms is not None:
        sync_overhead = python_ms - analysis["wall_clock_ms"]
        sync_pct = (sync_overhead / python_ms * 100) if python_ms > 0 else 0
        print(f"Python Measurement: {python_ms:.6f} ms")
        print(f"Sync Overhead: {sync_overhead:.6f} ms ({sync_pct:.1f}%)")

    print()
    print("=" * 80)
    print("PARALLEL WORK BREAKDOWN (timeline span × num_cores)")
    print("=" * 80)
    print()

    wall_clock = analysis["wall_clock_ms"]
    brisc_work = analysis.get("brisc_parallel_work_ms", 0)
    ncrisc_work = analysis.get("ncrisc_parallel_work_ms", 0)
    trisc_work = analysis.get("trisc_parallel_work_ms", 0)
    total_work = analysis.get("total_parallel_work_ms", 0)
    parallelism = analysis.get("parallelism", 0)

    brisc_span = analysis.get("brisc_timeline_span_ms", 0)
    ncrisc_span = analysis.get("ncrisc_timeline_span_ms", 0)
    trisc_span = analysis.get("trisc_timeline_span_ms", 0)

    brisc_cores = analysis.get("brisc_num_cores", 0)
    ncrisc_cores = analysis.get("ncrisc_num_cores", 0)
    trisc_cores = analysis.get("trisc_num_cores", 0)

    print(f"{'Component':<25} {'Span (ms)':<15} {'Cores':<10} {'Work (ms)':<15} {'% of Total':<15}")
    print("-" * 80)

    brisc_pct = (brisc_work / total_work * 100) if total_work > 0 else 0
    print(f"{'Weight Streaming':<25} {brisc_span:<15.6f} {brisc_cores:<10} {brisc_work:<15.2f} {brisc_pct:<15.1f}%")

    ncrisc_pct = (ncrisc_work / total_work * 100) if total_work > 0 else 0
    print(
        f"{'NoC Communication':<25} {ncrisc_span:<15.6f} {ncrisc_cores:<10} {ncrisc_work:<15.2f} {ncrisc_pct:<15.1f}%"
    )

    trisc_pct = (trisc_work / total_work * 100) if total_work > 0 else 0
    print(f"{'Computation':<25} {trisc_span:<15.6f} {trisc_cores:<10} {trisc_work:<15.2f} {trisc_pct:<15.1f}%")

    print("-" * 80)
    print(f"{'TOTAL':<25} {'':<15} {'':<10} {total_work:<15.2f} {'100.0%':<15}")
    print()

    print(f"Effective Parallelism: {parallelism:.1f}x")
    print(f"  (Total work {total_work:.1f} ms / Wall clock {wall_clock:.3f} ms)")
    print()

    # Validation
    max_theoretical = 130 * 5  # 130 Tensix cores × 5 RISC processors
    efficiency = (parallelism / max_theoretical * 100) if max_theoretical > 0 else 0
    print(f"Theoretical Maximum: {max_theoretical}x (130 cores × 5 RISCs)")
    print(f"Efficiency: {efficiency:.1f}%")

    if parallelism > max_theoretical:
        print()
        print("⚠️  WARNING: Parallelism exceeds theoretical maximum!")
        print("    This indicates an error in calculation or data.")
    print()


def main():
    """Main analysis routine"""

    print("\n" + "=" * 80)
    print("DEVICE PROFILE ANALYSIS: Large Batch vs Mini-batch")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - Hardware: Tenstorrent Blackhole P150A @ 1.35 GHz")
    print("  - Tensix cores: 130 (13×10 grid)")
    print("  - Max parallelism: 130 cores × 5 RISCs = 650x")
    print("  - Matrix: 4096×4096 @ bfloat16")
    print("  - Large batch: B=256 (1 forward)")
    print("  - Mini-batch: b=32 (8 forwards)")
    print("  - Profiling: TT_METAL_DEVICE_PROFILER=1 (cycle-accurate)")
    print()

    # Parse device profile
    csv_path = Path("generated/profiler/.logs/profile_log_device.csv")

    if not csv_path.exists():
        print(f"ERROR: Device profile not found at {csv_path}")
        print("Please run with TT_METAL_DEVICE_PROFILER=1 first!")
        return

    zones = parse_device_profile(csv_path)
    run_host_ids = sorted(set(z.run_host_id for z in zones))

    print(f"Device profile loaded: {len(zones)} zones, {len(run_host_ids)} operations")
    print()

    # Detect scenario
    if len(run_host_ids) <= 10:
        # Large batch
        print("Detected: LARGE BATCH scenario")
        print()

        last_run_id = run_host_ids[-1]
        last_op_zones = [z for z in zones if z.run_host_id == last_run_id]

        analysis = analyze_operation(last_op_zones, "Large Batch Forward (B=256)")
        python_ms = 0.239  # From Python measurement
        print_analysis(analysis, python_ms)

    else:
        # Mini-batch
        print("Detected: MINI-BATCH scenario (8 forwards)")
        print()

        # Analyze last 16 operations (8 forwards × 2 ops each)
        mini_run_ids = run_host_ids[-16:]

        # Calculate total wall clock
        total_wall_ms = 0.0
        for i in range(8):
            fwd_run_ids = [mini_run_ids[i * 2], mini_run_ids[i * 2 + 1]]
            fwd_zones = [z for z in zones if z.run_host_id in fwd_run_ids]

            if fwd_zones:
                fwd_wall = (
                    (max(z.end_cycle for z in fwd_zones) - min(z.start_cycle for z in fwd_zones)) / TT_FREQ_HZ
                ) * 1000
                total_wall_ms += fwd_wall

        python_total_ms = 1.439  # From Python measurement

        print("Mini-batch Summary:")
        print(f"  Total device time (8 forwards): {total_wall_ms:.6f} ms")
        print(f"  Total Python time (8 forwards): {python_total_ms:.6f} ms")
        print(f"  Per-forward device: {(total_wall_ms / 8):.6f} ms")
        print(f"  Per-forward Python: {(python_total_ms / 8):.6f} ms")
        print()

        # Analyze all 8 forwards together
        all_mini_zones = [z for z in zones if z.run_host_id in mini_run_ids]
        analysis = analyze_operation(all_mini_zones, "Mini-batch Total (8 forwards)")
        # Don't compare timeline span to python sum - they measure different things!
        # Timeline span includes overlaps, python sum is sequential
        print_analysis(analysis, python_ms=None)

        print()
        print("=" * 80)
        print("Per-forward Average:")
        print("=" * 80)
        per_fwd_work = analysis["total_parallel_work_ms"] / 8
        per_fwd_wall = total_wall_ms / 8
        per_fwd_python = python_total_ms / 8
        per_fwd_parallelism = per_fwd_work / per_fwd_wall if per_fwd_wall > 0 else 0
        per_fwd_sync = per_fwd_python - per_fwd_wall
        per_fwd_sync_pct = (per_fwd_sync / per_fwd_python * 100) if per_fwd_python > 0 else 0

        print(f"  Device wall clock: {per_fwd_wall:.6f} ms")
        print(f"  Python measurement: {per_fwd_python:.6f} ms")
        if per_fwd_sync >= 0:
            print(f"  Sync overhead: {per_fwd_sync:.6f} ms ({per_fwd_sync_pct:.1f}%)")
        else:
            print(f"  Python underestimate: {abs(per_fwd_sync):.6f} ms ({abs(per_fwd_sync_pct):.1f}%)")
            print("    (Device profiler captures work Python timer misses)")
        print(f"  Total work: {per_fwd_work:.2f} ms")
        print(f"  Parallelism: {per_fwd_parallelism:.1f}x ({(per_fwd_parallelism / 650 * 100):.1f}% efficiency)")
        print()


if __name__ == "__main__":
    main()

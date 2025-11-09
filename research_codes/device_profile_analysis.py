#!/usr/bin/env python3
"""
Device Profile Analysis - Analyze Single Forward Pass

This script analyzes device profiler output to measure a SINGLE forward pass,
matching the structure of weight_loading_test.py exactly.

Key insight: A single forward pass = one call to linear(x_tt) + device_synchronize()
The device profile log captures zones for each forward pass execution.

Python time = device execution + device_synchronize() overhead
Device time = device execution only (should be <= Python time)
"""

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


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


def extract_device_info(csv_path: Path) -> Tuple[str, int]:
    """Extract architecture and frequency from CSV header

    Returns:
        (arch, freq_mhz) - Architecture name and frequency in MHz
    """
    with open(csv_path, "r") as f:
        line = f.readline()

    if "Chip clock is at " in line:
        # Grayskull format
        return "grayskull", 1200
    elif "ARCH" in line and "CHIP_FREQ" in line:
        # Modern format: "ARCH: blackhole, CHIP_FREQ[MHz]: 1350"
        arch_part = None
        freq_part = None
        for part in line.split(","):
            part = part.strip()
            if part.startswith("ARCH:"):
                arch_part = part.split(":")[-1].strip()
            elif "CHIP_FREQ" in part:
                freq_part = part.split(":")[-1].strip()

        if arch_part is None or freq_part is None:
            raise ValueError(f"Could not parse ARCH or CHIP_FREQ from: {line}")

        freq_mhz = int(freq_part)
        return arch_part, freq_mhz
    else:
        raise ValueError(f"Could not parse device info from CSV header: {line}")


def extract_benchmark_config() -> dict:
    """Extract benchmark configuration from CSV file

    Returns:
        Dictionary with config values, or empty dict if not found
    """
    import csv
    from pathlib import Path

    csv_paths = [
        Path("research_codes/benchmark_results.csv"),
        Path("benchmark_results.csv"),
    ]

    for csv_path in csv_paths:
        if not csv_path.exists():
            continue

        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    continue

                latest_row = rows[-1]
                config = {}
                if "minibatches" in latest_row:
                    try:
                        config["minibatches"] = int(latest_row["minibatches"])
                    except (ValueError, TypeError):
                        pass
                if "small_batch_size" in latest_row:
                    try:
                        config["small_batch_size"] = int(latest_row["small_batch_size"])
                    except (ValueError, TypeError):
                        pass
                if "large_batch_size" in latest_row:
                    try:
                        config["large_batch_size"] = int(latest_row["large_batch_size"])
                    except (ValueError, TypeError):
                        pass
                if "measure_iters" in latest_row:
                    try:
                        config["measure_iters"] = int(latest_row["measure_iters"])
                    except (ValueError, TypeError):
                        pass
                if "warmup_iters" in latest_row:
                    try:
                        config["warmup_iters"] = int(latest_row["warmup_iters"])
                    except (ValueError, TypeError):
                        pass

                return config
        except (ValueError, KeyError, IndexError):
            continue

    return {}


def extract_python_time_from_benchmark(scenario: str) -> float | None:
    """Try to extract Python measurement time from benchmark results CSV

    Args:
        scenario: "large" or "mini"

    Returns:
        Python time in ms, or None if not found
    """
    import csv
    from pathlib import Path

    csv_paths = [
        Path("research_codes/benchmark_results.csv"),
        Path("benchmark_results.csv"),
    ]

    for csv_path in csv_paths:
        if not csv_path.exists():
            continue

        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    continue

                latest_row = rows[-1]

                if scenario == "large":
                    if "large_forward_compute_ms" in latest_row:
                        return float(latest_row["large_forward_compute_ms"])
                elif scenario == "mini":
                    if "mini_forward_compute_ms" in latest_row:
                        return float(latest_row["mini_forward_compute_ms"])
        except (ValueError, KeyError, IndexError):
            continue

    return None


def normalize_risc_type(risc_type: str) -> str:
    """Normalize RISC type names"""
    if "BRISC" in risc_type:
        return "BRISC"
    elif "NCRISC" in risc_type:
        return "NCRISC"
    elif "TRISC" in risc_type:
        return "TRISC"
    return risc_type


def parse_device_profile(csv_path: Path) -> Tuple[List[DeviceZone], dict]:
    """Parse device profile CSV file

    Returns:
        (zones, unmatched_starts) - List of complete zones and dict of unmatched zone starts
    """
    zones = []
    zone_starts = {}
    unmatched_starts = {}

    with open(csv_path, "r") as f:
        lines = f.readlines()

        if len(lines) < 2:
            return zones, unmatched_starts

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

    unmatched_starts = zone_starts.copy()
    return zones, unmatched_starts


def analyze_single_forward_pass(
    zones: List[DeviceZone], description: str, freq_hz: float, unmatched_starts: Optional[dict] = None
) -> Dict:
    """Analyze a SINGLE forward pass operation

    This matches the structure of weight_loading_test.py:
    - A single forward pass = one call to linear(x_tt) + device_synchronize()
    - Python time includes device execution + sync overhead
    - Device time should be <= Python time

    Args:
        zones: List of device zones for ONE forward pass
        description: Description of the operation
        freq_hz: Device frequency in Hz
        unmatched_starts: Dict of unmatched zone starts
    """

    if not zones:
        return {}

    # Calculate wall clock time - actual device execution time
    min_start = min(z.start_cycle for z in zones)
    max_end = max(z.end_cycle for z in zones)

    if unmatched_starts:
        for key, start_cycle in unmatched_starts.items():
            core_id, risc_type, run_host_id, zone_name = key
            if any(z.run_host_id == run_host_id for z in zones):
                if start_cycle < min_start:
                    min_start = start_cycle

    wall_cycles = max_end - min_start
    wall_ms = (wall_cycles / freq_hz) * 1000.0

    # RISC breakdown - calculate parallel work correctly
    risc_zones_by_type = defaultdict(list)
    for z in zones:
        risc_type = normalize_risc_type(z.risc_type)
        risc_zones_by_type[risc_type].append(z)

    # Parallel work = timeline span × num_cores (NOT sum of durations)
    risc_parallel_work = {}
    risc_timeline_spans = {}
    risc_num_cores = {}

    for risc_type in ["BRISC", "NCRISC", "TRISC"]:
        if risc_type in risc_zones_by_type:
            zones_list = risc_zones_by_type[risc_type]
            earliest_start = min(z.start_cycle for z in zones_list)
            latest_end = max(z.end_cycle for z in zones_list)
            span_ms = ((latest_end - earliest_start) / freq_hz) * 1000.0
            num_cores = len(set(z.core_id for z in zones_list))
            parallel_work = span_ms * num_cores

            risc_timeline_spans[risc_type] = span_ms
            risc_num_cores[risc_type] = num_cores
            risc_parallel_work[risc_type] = parallel_work
        else:
            risc_timeline_spans[risc_type] = 0.0
            risc_num_cores[risc_type] = 0
            risc_parallel_work[risc_type] = 0.0

    brisc_work = risc_parallel_work.get("BRISC", 0)
    ncrisc_work = risc_parallel_work.get("NCRISC", 0)
    trisc_work = risc_parallel_work.get("TRISC", 0)
    total_work = brisc_work + ncrisc_work + trisc_work

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


def print_single_forward_analysis(analysis: Dict, python_ms: float | None = None):
    """Print analysis for a SINGLE forward pass"""
    print(f"\n{'='*80}")
    print(f"{analysis['description']}")
    print(f"{'='*80}")
    print(f"Zones: {analysis['num_zones']}")
    print(f"Device Wall Clock (device execution only): {analysis['wall_clock_ms']:.6f} ms")

    if python_ms is not None:
        sync_overhead = python_ms - analysis["wall_clock_ms"]
        sync_pct = (sync_overhead / python_ms * 100) if python_ms > 0 else 0
        print(f"Python Measurement (device + sync overhead): {python_ms:.6f} ms")
        if sync_overhead > 0:
            print(f"Sync Overhead: {sync_overhead:.6f} ms ({sync_pct:.1f}%)")
            print("  ✓ Device time < Python time (expected: sync overhead included in Python)")
        else:
            print(f"⚠️  Device time exceeds Python: {abs(sync_overhead):.6f} ms ({abs(sync_pct):.1f}%)")
            print("  This may indicate:")
            print("    - Device profiler captures overlapping work Python timer misses")
            print("    - Or measurement timing mismatch")

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

    max_theoretical = 130 * 5
    efficiency = (parallelism / max_theoretical * 100) if max_theoretical > 0 else 0
    print(f"Theoretical Maximum: {max_theoretical}x (130 cores × 5 RISCs)")
    print(f"Efficiency: {efficiency:.1f}%")

    if parallelism > max_theoretical:
        print()
        print("⚠️  WARNING: Parallelism exceeds theoretical maximum!")
        print("    This indicates an error in calculation or data.")
    print()


def identify_forward_pass_operations(
    zones: List[DeviceZone], config: dict, is_minibatch: bool
) -> Tuple[List[int], List[int]]:
    """
    Identify run_host_ids that correspond to forward pass operations.

    Following weight_loading_test.py structure:
    - Setup: weight loading, kernel compilation (few zones)
    - Warmup: warmup_iters forward passes (excluded from measurement)
    - Measurement: measure_iters forward passes (these we want to analyze)

    Returns:
        (all_forward_run_ids, measurement_forward_run_ids)
        - all_forward_run_ids: All forward pass operations (warmup + measurement)
        - measurement_forward_run_ids: Only measurement forward passes (excludes warmup)
    """
    zones_by_run_id = defaultdict(list)
    for z in zones:
        zones_by_run_id[z.run_host_id].append(z)

    run_host_ids = sorted(set(z.run_host_id for z in zones))
    zone_counts = [(rid, len(zones_by_run_id[rid])) for rid in run_host_ids]

    # Forward passes have significant zones (at least 5)
    # Setup/compilation typically has fewer zones
    all_forward_run_ids = []
    for rid, count in reversed(zone_counts):
        if count >= 5:  # Significant operation
            all_forward_run_ids.append(rid)

    # Get configuration
    warmup_iters = config.get("warmup_iters", 2)
    measure_iters = config.get("measure_iters", 5)
    minibatches = config.get("minibatches", 8)

    if is_minibatch:
        # Mini-batch structure:
        # - Each sequence has minibatches forward passes
        # - Total sequences = warmup_iters + measure_iters
        # - Measurement sequences = last measure_iters sequences
        # - Measurement forward passes = last (measure_iters × minibatches) forward passes
        total_expected_forwards = (warmup_iters + measure_iters) * minibatches
        measurement_expected_forwards = measure_iters * minibatches
    else:
        # Large batch structure:
        # - Total forward passes = warmup_iters + measure_iters
        # - Measurement forward passes = last measure_iters forward passes
        total_expected_forwards = warmup_iters + measure_iters
        measurement_expected_forwards = measure_iters

    # Validate total
    expected_min_ops = total_expected_forwards
    expected_max_ops = total_expected_forwards * 3

    if len(all_forward_run_ids) < expected_min_ops:
        print(
            f"WARNING: Found {len(all_forward_run_ids)} forward operations, expected at least {expected_min_ops} (warmup + measurement)"
        )
    elif len(all_forward_run_ids) > expected_max_ops:
        print(f"WARNING: Found {len(all_forward_run_ids)} forward operations, expected at most {expected_max_ops}")
        print(f"  Trimming to most recent {expected_max_ops} operations")
        all_forward_run_ids = all_forward_run_ids[:expected_max_ops]

    # Extract only measurement forward passes (exclude warmup)
    # Take the last measurement_expected_forwards forward passes
    if len(all_forward_run_ids) >= measurement_expected_forwards:
        measurement_forward_run_ids = all_forward_run_ids[-measurement_expected_forwards:]
        print(f"  Total forward passes: {len(all_forward_run_ids)}")
        print(f"  Warmup passes: {len(all_forward_run_ids) - len(measurement_forward_run_ids)}")
        print(f"  Measurement passes: {len(measurement_forward_run_ids)}")
    else:
        print(f"WARNING: Not enough forward passes to separate warmup from measurement")
        print(f"  Using all {len(all_forward_run_ids)} forward passes as measurement")
        measurement_forward_run_ids = all_forward_run_ids

    return all_forward_run_ids, measurement_forward_run_ids


def main():
    """Main analysis routine - analyzes SINGLE forward pass"""

    print("\n" + "=" * 80)
    print("DEVICE PROFILE ANALYSIS: Single Forward Pass")
    print("=" * 80)
    print()
    print("This script analyzes a SINGLE forward pass execution,")
    print("matching the structure of weight_loading_test.py exactly.")
    print()
    print("Key insight:")
    print("  - Python time = device execution + device_synchronize() overhead")
    print("  - Device time = device execution only (should be <= Python time)")
    print()

    csv_path = Path("generated/profiler/.logs/profile_log_device.csv")
    if not csv_path.exists():
        print(f"ERROR: Device profile not found at {csv_path}")
        print("Please run with TT_METAL_DEVICE_PROFILER=1 first!")
        return

    try:
        arch, freq_mhz = extract_device_info(csv_path)
        freq_hz = freq_mhz * 1_000_000
        print(f"Detected device: {arch} @ {freq_mhz} MHz ({freq_hz / 1_000_000_000:.2f} GHz)")
    except (ValueError, IndexError) as e:
        print(f"WARNING: Could not extract frequency from CSV header: {e}")
        print("Using default frequency: 1350 MHz (1.35 GHz)")
        arch = "blackhole"
        freq_mhz = 1350
        freq_hz = 1_350_000_000

    zones, unmatched_starts = parse_device_profile(csv_path)
    run_host_ids = sorted(set(z.run_host_id for z in zones))

    print(f"Device profile loaded: {len(zones)} complete zones, {len(unmatched_starts)} unmatched zone starts")
    print(f"Total operations (run_host_ids): {len(run_host_ids)}")
    print()

    # Get benchmark configuration
    config = extract_benchmark_config()
    warmup_iters = config.get("warmup_iters", 2)
    measure_iters = config.get("measure_iters", 5)
    minibatches = config.get("minibatches", 8)

    print(f"Benchmark configuration:")
    print(f"  Warmup iterations: {warmup_iters}")
    print(f"  Measurement iterations: {measure_iters}")
    if minibatches:
        print(f"  Minibatches per sequence: {minibatches}")
    print()

    # Detect scenario based on number of operations
    # Large batch: fewer operations (setup + warmup + measure_iters forwards)
    # Mini-batch: many operations (setup + warmup + measure_iters × minibatches forwards)
    if len(run_host_ids) <= 10:
        scenario = "large"
        print("Detected: LARGE BATCH scenario")
        print(f"  Structure: setup + {warmup_iters} warmup + {measure_iters} measurement forward passes")
    else:
        scenario = "mini"
        print("Detected: MINI-BATCH scenario")
        print(f"  Structure: setup + {warmup_iters} warmup sequences + {measure_iters} measurement sequences")
        print(f"  Each sequence has {minibatches} forward passes")
        print(f"  Total measurement forward passes: {measure_iters * minibatches}")

    print()

    # Identify forward pass operations (separate warmup from measurement)
    all_forward_run_ids, measurement_forward_run_ids = identify_forward_pass_operations(
        zones, config, scenario == "mini"
    )
    print()

    if not measurement_forward_run_ids:
        print("ERROR: No measurement forward pass operations found!")
        return

    if scenario == "large":
        # Analyze the LAST measurement forward pass (most recent, excludes warmup)
        # This matches the measurement iterations in weight_loading_test.py
        last_forward_run_id = measurement_forward_run_ids[-1]
        forward_zones = [z for z in zones if z.run_host_id == last_forward_run_id]
        forward_unmatched = {k: v for k, v in unmatched_starts.items() if k[2] == last_forward_run_id}

        print(f"Analyzing LAST measurement forward pass (run_host_id={last_forward_run_id})")
        print(f"  This excludes {warmup_iters} warmup passes")
        print(f"  Zones: {len(forward_zones)}")
        print()

        analysis = analyze_single_forward_pass(
            forward_zones, f"Large Batch Single Forward Pass (B=256) - Measurement Only", freq_hz, forward_unmatched
        )

        # Python time from CSV is already averaged over measure_iters
        python_ms = extract_python_time_from_benchmark("large")
        if python_ms is None:
            print("WARNING: Could not extract Python time from benchmark CSV")
            python_ms = None

        print_single_forward_analysis(analysis, python_ms)

    else:  # mini-batch
        # For mini-batch, analyze the LAST measurement forward pass (excludes warmup)
        # This represents one forward pass from the measurement sequences
        last_forward_run_id = measurement_forward_run_ids[-1]
        forward_zones = [z for z in zones if z.run_host_id == last_forward_run_id]
        forward_unmatched = {k: v for k, v in unmatched_starts.items() if k[2] == last_forward_run_id}

        print(f"Analyzing LAST measurement forward pass (run_host_id={last_forward_run_id})")
        print(f"  This excludes {warmup_iters} warmup sequences")
        print(f"  Zones: {len(forward_zones)}")
        print(f"  Note: This is ONE forward pass from the measurement sequences")
        print()

        analysis = analyze_single_forward_pass(
            forward_zones, f"Mini-batch Single Forward Pass (b=32) - Measurement Only", freq_hz, forward_unmatched
        )

        # For mini-batch, Python time is total for all measurement forwards
        # We need per-forward Python time
        python_total_ms = extract_python_time_from_benchmark("mini")
        if python_total_ms is not None:
            # Python time is total for measure_iters sequences × minibatches forwards
            python_ms = python_total_ms / (measure_iters * minibatches)
            print(f"Python total time (all measurement forwards): {python_total_ms:.6f} ms")
            print(f"Python per-forward time: {python_ms:.6f} ms")
        else:
            print("WARNING: Could not extract Python time from benchmark CSV")
            python_ms = None

        print_single_forward_analysis(analysis, python_ms)


if __name__ == "__main__":
    main()

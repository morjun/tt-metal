#!/usr/bin/env python3
"""
Analyze Tracy profiling results to break down forward pass timing.

This script parses the Tracy CSV output and calculates:
1. Large batch forward pass time
2. Mini-batch forward pass time (sum of 8 mini-batches)
3. Overhead between large batch and mini-batch scenarios

Usage:
    python research_codes/analyze_tracy_results.py
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List


def parse_tracy_csv(csv_path: str) -> Dict[str, List[float]]:
    """
    Parse Tracy CSV and extract timing data for our instrumented zones.

    Returns dict mapping zone_name -> list of execution times in milliseconds
    """
    timings = defaultdict(list)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            exec_time_ns = row["exec_time_ns"]

            # Filter for our instrumented zones
            if any(
                prefix in name
                for prefix in [
                    "TT_DNN_forward_large_",
                    "TT_DNN_forward_mini_",
                    "weight_host_to_gddr6",
                    "bias_host_to_gddr6",
                    "input_host_to_gddr6",
                    "slice_input_",
                    "minibatch_sequence_",
                ]
            ):
                try:
                    # Convert nanoseconds to milliseconds
                    exec_time_ms = float(exec_time_ns) / 1_000_000.0
                    timings[name].append(exec_time_ms)
                except ValueError:
                    continue

    return timings


def analyze_forward_pass(timings: Dict[str, List[float]]):
    """Analyze and report forward pass timing breakdown."""

    print("\n" + "=" * 80)
    print("TRACY PROFILER FORWARD PASS BREAKDOWN")
    print("=" * 80)

    # Large batch forward passes
    large_forward_times = []
    for key in timings.keys():
        if key.startswith("TT_DNN_forward_large_"):
            large_forward_times.extend(timings[key])

    # Mini-batch forward passes (grouped by sequence and minibatch)
    mini_forward_times_by_sequence = defaultdict(list)
    for key in timings.keys():
        if key.startswith("TT_DNN_forward_mini_"):
            # Extract sequence number (e.g., TT_DNN_forward_mini_2_0 -> sequence 2)
            parts = key.split("_")
            if len(parts) >= 5:
                try:
                    sequence_idx = int(parts[4])
                    mini_forward_times_by_sequence[sequence_idx].extend(timings[key])
                except (ValueError, IndexError):
                    pass

    # Weight/bias/input loading times
    weight_load_times = []
    bias_load_times = []
    input_load_times = []

    for key in timings.keys():
        if key == "weight_host_to_gddr6":
            weight_load_times.extend(timings[key])
        elif key == "bias_host_to_gddr6":
            bias_load_times.extend(timings[key])
        elif key == "input_host_to_gddr6":
            input_load_times.extend(timings[key])

    # Slice times (device-side tensor slicing)
    slice_times = []
    for key in timings.keys():
        if key.startswith("slice_input_"):
            slice_times.extend(timings[key])

    # Print results
    print("\n--- LARGE BATCH (B=256) ---")
    if large_forward_times:
        avg_large = sum(large_forward_times) / len(large_forward_times)
        print(f"Forward passes measured: {len(large_forward_times)}")
        print(f"Average forward time: {avg_large:.6f} ms")
        print(f"Min: {min(large_forward_times):.6f} ms")
        print(f"Max: {max(large_forward_times):.6f} ms")
    else:
        print("No large batch forward data found")
        avg_large = 0.0

    print("\n--- MINI-BATCH (b=32 x 8) ---")
    if mini_forward_times_by_sequence:
        # Calculate total time for each sequence (sum of 8 mini-batches)
        sequence_totals = []
        for seq_idx in sorted(mini_forward_times_by_sequence.keys()):
            times = mini_forward_times_by_sequence[seq_idx]
            total_time = sum(times)
            sequence_totals.append(total_time)
            print(f"Sequence {seq_idx}: {len(times)} mini-batches, total: {total_time:.6f} ms")

        avg_mini_total = sum(sequence_totals) / len(sequence_totals)
        print(f"\nAverage total time (8 mini-batches): {avg_mini_total:.6f} ms")
        print(f"Min sequence total: {min(sequence_totals):.6f} ms")
        print(f"Max sequence total: {max(sequence_totals):.6f} ms")

        # Calculate average time per mini-batch
        all_mini_times = []
        for times in mini_forward_times_by_sequence.values():
            all_mini_times.extend(times)
        avg_per_mini = sum(all_mini_times) / len(all_mini_times)
        print(f"\nAverage per mini-batch: {avg_per_mini:.6f} ms")
        print(f"Total mini-batches measured: {len(all_mini_times)}")
    else:
        print("No mini-batch forward data found")
        avg_mini_total = 0.0

    print("\n--- OVERHEAD ANALYSIS ---")
    if large_forward_times and mini_forward_times_by_sequence:
        overhead_ms = avg_mini_total - avg_large
        overhead_pct = (overhead_ms / avg_large) * 100.0
        print(f"Overhead: {overhead_ms:.6f} ms ({overhead_pct:.2f}%)")
        print(f"Speedup if large batch used: {avg_mini_total / avg_large:.2f}x")
    else:
        print("Cannot calculate overhead - missing data")

    print("\n--- DATA MOVEMENT (Host DRAM → Device GDDR6) ---")
    if weight_load_times:
        avg_weight = sum(weight_load_times) / len(weight_load_times)
        print(f"Weight load: {avg_weight:.6f} ms (measured {len(weight_load_times)} times)")
    else:
        print("Weight load: No data")

    if bias_load_times:
        avg_bias = sum(bias_load_times) / len(bias_load_times)
        print(f"Bias load: {avg_bias:.6f} ms (measured {len(bias_load_times)} times)")
    else:
        print("Bias load: No data")

    if input_load_times:
        avg_input = sum(input_load_times) / len(input_load_times)
        print(f"Input load: {avg_input:.6f} ms (measured {len(input_load_times)} times)")
    else:
        print("Input load: No data")

    print("\n--- DEVICE-SIDE OPERATIONS ---")
    if slice_times:
        avg_slice = sum(slice_times) / len(slice_times)
        print(f"Tensor slice (on device): {avg_slice:.9f} ms per slice (measured {len(slice_times)} times)")
        print(f"Note: Slice overhead is negligible ({avg_slice*1000:.3f} µs)")
    else:
        print("Tensor slice: No data")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("1. Forward pass time includes ALL of:")
    print("   - Input sharding (GDDR6 DRAM → L1 SRAM sharded)")
    print("   - Weight streaming (GDDR6 DRAM → L1 SRAM)")
    print("   - Matmul compute")
    print("   - Output gathering (L1 sharded → GDDR6 DRAM via NoC)")
    print("\n2. The overhead in mini-batch comes from weight streaming")
    print("   happening 8 times instead of 1 time (weights not cached in L1)")
    print("\n3. To decompose forward pass further, we need device-side profiling")
    print("   (use --device-trace-profiler flag with Tracy)")
    print("=" * 80 + "\n")


def main():
    # Find Tracy CSV file
    tracy_csv = "/home/masterjunmo/codes/tt-metal/generated/profiler/.logs/tracy_ops_times.csv"

    if not os.path.exists(tracy_csv):
        print(f"Error: Tracy CSV not found at {tracy_csv}")
        print("Run the instrumented script first:")
        print("  python3 -m tracy -r research_codes/weight_loading_test_tracy.py")
        return

    print(f"Parsing Tracy CSV: {tracy_csv}")
    timings = parse_tracy_csv(tracy_csv)

    if not timings:
        print("No timing data found in Tracy CSV")
        print("Make sure you ran: python3 -m tracy -r research_codes/weight_loading_test_tracy.py")
        return

    analyze_forward_pass(timings)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Analyze forward pass breakdown using Tracy profiler.

This script runs weight_loading_test.py with Tracy profiler enabled
and extracts the detailed breakdown of each forward pass into:
1. Tensor sharding (DRAM -> L1 SRAM)
2. Weight streaming (DRAM -> L1)
3. Compute operations
4. NoC communication (output gathering)

The key principle: NO additional operations are added. We simply profile
the existing forward pass and decompose it using Tracy.

Usage:
    python analyze_forward_pass_breakdown.py

This will generate Tracy .csv files which can be viewed with tracy-csvexport or tracy UI.
"""

import os
import sys
import json
import subprocess
import csv
from pathlib import Path
from datetime import datetime
import re


def run_with_tracy(batch_size, is_minibatch=False, num_minibatches=8):
    """
    Run weight_loading_test.py with Tracy profiler enabled.

    Args:
        batch_size: Batch size to test
        is_minibatch: Whether this is minibatch mode
        num_minibatches: Number of minibatches (if is_minibatch=True)

    Returns:
        Path to tracy CSV output file
    """
    env = os.environ.copy()

    # Enable Tracy profiling
    env["ENABLE_TRACY"] = "1"

    # Set Tracy output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_minibatch:
        output_file = f"tracy_minibatch_{batch_size}x{num_minibatches}_{timestamp}.csv"
    else:
        output_file = f"tracy_large_{batch_size}_{timestamp}.csv"

    env["TRACY_CSV_OUT"] = output_file

    # Build command
    cmd = [
        sys.executable,
        "research_codes/weight_loading_test.py",
        "--in-features",
        "4096",
        "--out-features",
        "4096",
        "--warmup-iters",
        "1",
        "--measure-iters",
        "1",  # Single iteration for Tracy profiling
    ]

    if is_minibatch:
        cmd.extend(
            [
                "--large-batch-size",
                "0",  # Skip large batch
                "--small-batch-size",
                str(batch_size),
                "--minibatches",
                str(num_minibatches),
            ]
        )
    else:
        cmd.extend(
            [
                "--large-batch-size",
                str(batch_size),
                "--small-batch-size",
                "0",  # Skip minibatch
            ]
        )

    print(f"\n{'='*70}")
    print(f"Running Tracy profiler for {'mini-batch' if is_minibatch else 'large batch'} (batch={batch_size})")
    print(f"Output file: {output_file}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running Tracy profiler:")
        print(result.stderr)
        print(result.stdout)
        return None

    print(result.stdout)

    # Check if output file was created
    if os.path.exists(output_file):
        return output_file
    else:
        print(f"Warning: Tracy output file not created: {output_file}")
        return None


def parse_tracy_csv(csv_file):
    """
    Parse Tracy CSV output to extract operation breakdown.

    Tracy CSV format typically has columns like:
    - name: Zone/operation name
    - time: Execution time in nanoseconds
    - src_file, src_line: Source location

    Returns:
        dict with breakdown of sharding, weight streaming, compute, NoC
    """
    if not csv_file or not os.path.exists(csv_file):
        print(f"Tracy output file not found: {csv_file}")
        return None

    breakdown = {
        "sharding_us": 0,
        "weight_streaming_us": 0,
        "compute_us": 0,
        "noc_communication_us": 0,
        "total_us": 0,
        "operations": [],
    }

    print(f"Parsing Tracy output: {csv_file}...")

    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Extract operation name and duration
                # Tracy CSV columns may vary, try different common formats
                op_name = row.get("name", row.get("zone", row.get("function", "")))

                # Duration could be in different columns and units
                duration = 0
                if "time" in row:
                    duration = float(row["time"])
                elif "duration" in row:
                    duration = float(row["duration"])
                elif "exec_time" in row:
                    duration = float(row["exec_time"])

                # Convert to microseconds (assuming nanoseconds)
                duration_us = duration / 1000 if duration > 0 else 0

                if not op_name:
                    continue

                breakdown["operations"].append({"name": op_name, "duration_us": duration_us})

                # Categorize operations based on name patterns
                op_lower = op_name.lower()

                if any(kw in op_lower for kw in ["interleaved_to_sharded", "shard", "to_sharded"]):
                    breakdown["sharding_us"] += duration_us
                elif any(kw in op_lower for kw in ["weight", "dram_to_l1", "load_weight"]):
                    breakdown["weight_streaming_us"] += duration_us
                elif any(kw in op_lower for kw in ["matmul", "bmm", "mm", "compute", "linear"]):
                    breakdown["compute_us"] += duration_us
                elif any(kw in op_lower for kw in ["sharded_to_interleaved", "gather", "noc", "to_interleaved"]):
                    breakdown["noc_communication_us"] += duration_us

                breakdown["total_us"] += duration_us

    except Exception as e:
        print(f"Error parsing Tracy CSV: {e}")
        return None

    if not breakdown["operations"]:
        print(f"No operations found in Tracy output. File may be empty or in unexpected format.")
        # Try to print first few lines for debugging
        try:
            with open(csv_file, "r") as f:
                print("First few lines of file:")
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"  {line.strip()}")
                    else:
                        break
        except:
            pass

    return breakdown


def analyze_and_compare(large_batch_file, minibatch_file):
    """
    Compare large batch vs minibatch Tracy profiling results.
    """
    print(f"\n{'='*70}")
    print("ANALYZING TRACY PROFILER RESULTS")
    print(f"{'='*70}\n")

    large_breakdown = parse_tracy_csv(large_batch_file)
    mini_breakdown = parse_tracy_csv(minibatch_file)

    if not large_breakdown or not mini_breakdown:
        print("Failed to parse profiler outputs")
        return

    print("\n--- Large Batch Breakdown ---")
    print(f"  Sharding:           {large_breakdown['sharding_us']:>10.3f} us")
    print(f"  Weight Streaming:   {large_breakdown['weight_streaming_us']:>10.3f} us")
    print(f"  Compute:            {large_breakdown['compute_us']:>10.3f} us")
    print(f"  NoC Communication:  {large_breakdown['noc_communication_us']:>10.3f} us")
    print(f"  Total:              {large_breakdown['total_us']:>10.3f} us")

    print("\n--- Mini-Batch Breakdown (8x total) ---")
    print(f"  Sharding:           {mini_breakdown['sharding_us']:>10.3f} us")
    print(f"  Weight Streaming:   {mini_breakdown['weight_streaming_us']:>10.3f} us")
    print(f"  Compute:            {mini_breakdown['compute_us']:>10.3f} us")
    print(f"  NoC Communication:  {mini_breakdown['noc_communication_us']:>10.3f} us")
    print(f"  Total:              {mini_breakdown['total_us']:>10.3f} us")

    print("\n--- Overhead Analysis ---")
    overhead_sharding = mini_breakdown["sharding_us"] - large_breakdown["sharding_us"]
    overhead_weight = mini_breakdown["weight_streaming_us"] - large_breakdown["weight_streaming_us"]
    overhead_compute = mini_breakdown["compute_us"] - large_breakdown["compute_us"]
    overhead_noc = mini_breakdown["noc_communication_us"] - large_breakdown["noc_communication_us"]
    overhead_total = mini_breakdown["total_us"] - large_breakdown["total_us"]

    print(f"  Sharding:           {overhead_sharding:>10.3f} us")
    print(f"  Weight Streaming:   {overhead_weight:>10.3f} us")
    print(f"  Compute:            {overhead_compute:>10.3f} us")
    print(f"  NoC Communication:  {overhead_noc:>10.3f} us")
    print(f"  Total:              {overhead_total:>10.3f} us ({overhead_total/large_breakdown['total_us']*100:.2f}%)")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "large_batch": large_breakdown,
        "minibatch": mini_breakdown,
        "overhead": {
            "sharding_us": overhead_sharding,
            "weight_streaming_us": overhead_weight,
            "compute_us": overhead_compute,
            "noc_communication_us": overhead_noc,
            "total_us": overhead_total,
            "percentage": overhead_total / large_breakdown["total_us"] * 100,
        },
    }

    output_file = f"forward_pass_breakdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def main():
    """Main execution."""
    print("=" * 70)
    print("FORWARD PASS BREAKDOWN ANALYSIS (Tracy)")
    print("=" * 70)
    print("\nThis tool uses Tracy profiler to decompose weight_loading_test.py")
    print("forward pass into: Sharding + Weight Streaming + Compute + NoC Communication")
    print("\nNO additional operations are added - we only profile existing code.")
    print("\nNote: Make sure tt-metal was built with Tracy support enabled.")
    print("=" * 70)

    # Run Tracy profiler for large batch
    large_file = run_with_tracy(batch_size=256, is_minibatch=False)

    # Run Tracy profiler for minibatch
    mini_file = run_with_tracy(batch_size=32, is_minibatch=True, num_minibatches=8)

    # Analyze and compare
    if large_file and mini_file:
        analyze_and_compare(large_file, mini_file)
    elif not large_file and not mini_file:
        print("\n" + "=" * 70)
        print("TRACY PROFILER NOT AVAILABLE")
        print("=" * 70)
        print("\nTracy profiling requires:")
        print("1. tt-metal built with Tracy support (check build logs for 'Enable Tracy: ON')")
        print("2. Tracy zones instrumented in the code")
        print("3. Proper Tracy environment setup")
        print("\nAlternative: Use ttnn operation-level timing instead.")
        print("=" * 70)
    else:
        print(f"\nPartial failure: large_file={large_file}, mini_file={mini_file}")


if __name__ == "__main__":
    main()

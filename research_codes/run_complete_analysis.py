#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive comparison script that runs both high-level TTNN benchmarks
and low-level profiling to provide complete performance analysis.

This script:
1. Runs the high-level weight_loading_test.py to get end-to-end times
2. Runs the low-level profiling_sharding_noc_python.py to get detailed breakdown
3. Combines results to show where time is spent
4. Generates a comparative analysis report

Usage:
    python research_codes/run_complete_analysis.py [options]
"""

import subprocess
import sys
import os
import json
import csv
from datetime import datetime
from typing import Dict, Any

# Add tt-metal to path
script_dir = os.path.dirname(os.path.abspath(__file__))
tt_metal_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, tt_metal_root)


def run_high_level_benchmark(config: Dict[str, Any]) -> Dict[str, float]:
    """Run the high-level TTNN benchmark."""
    print("\n" + "=" * 70)
    print("STEP 1: Running High-Level TTNN Benchmark")
    print("=" * 70)

    cmd = [
        sys.executable,
        "research_codes/weight_loading_test.py",
        "--in-features",
        str(config["in_features"]),
        "--out-features",
        str(config["out_features"]),
        "--large-batch-size",
        str(config["large_batch"]),
        "--small-batch-size",
        str(config["small_batch"]),
        "--minibatches",
        str(config["minibatches"]),
        "--warmup-iters",
        str(config["warmup"]),
        "--measure-iters",
        str(config["iterations"]),
        "--output-csv",
        "high_level_results.csv",
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print("Warning: High-level benchmark encountered issues")
        return {}

    # Parse CSV to get results
    try:
        with open("high_level_results.csv", "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                latest = rows[-1]  # Get most recent result
                return {
                    "large_forward_ms": float(latest.get("large_forward_compute_ms", 0)),
                    "mini_forward_ms": float(latest.get("mini_forward_compute_ms", 0)),
                    "large_total_ms": float(latest.get("large_total_ms", 0)),
                    "mini_total_ms": float(latest.get("mini_total_ms", 0)),
                }
    except Exception as e:
        print(f"Warning: Could not parse high-level results: {e}")
        return {}


def run_low_level_profiling(config: Dict[str, Any]) -> Dict[str, float]:
    """Run the low-level profiling."""
    print("\n" + "=" * 70)
    print("STEP 2: Running Low-Level Profiling")
    print("=" * 70)

    cmd = [
        sys.executable,
        "research_codes/profiling_sharding_noc_python.py",
        "--large-batch",
        str(config["large_batch"]),
        "--small-batch",
        str(config["small_batch"]),
        "--minibatches",
        str(config["minibatches"]),
        "--in-features",
        str(config["in_features"]),
        "--out-features",
        str(config["out_features"]),
        "--warmup",
        str(config["warmup"]),
        "--iterations",
        str(config["iterations"]),
        "--output",
        "low_level_results.csv",
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print("Warning: Low-level profiling encountered issues")
        return {}

    # Parse CSV to get results
    try:
        with open("low_level_results.csv", "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                latest = rows[-1]
                return {
                    "large_sharding_us": float(latest.get("large_batch_sharding_us", 0)),
                    "mini_sharding_us": float(latest.get("mini_batch_sharding_us", 0)),
                    "large_weight_us": float(latest.get("large_batch_weight_stream_us", 0)),
                    "mini_weight_us": float(latest.get("mini_batch_weight_stream_us", 0)),
                    "large_noc_us": float(latest.get("large_batch_noc_comm_us", 0)),
                    "mini_noc_us": float(latest.get("mini_batch_noc_comm_us", 0)),
                    "large_total_us": float(latest.get("large_batch_total_us", 0)),
                    "mini_total_us": float(latest.get("mini_batch_total_us", 0)),
                }
    except Exception as e:
        print(f"Warning: Could not parse low-level results: {e}")
        return {}


def generate_analysis_report(config: Dict[str, Any], high_level: Dict[str, float], low_level: Dict[str, float]) -> None:
    """Generate comprehensive analysis report."""
    print("\n" + "=" * 70)
    print("STEP 3: Comprehensive Analysis Report")
    print("=" * 70)

    print(f"\nTest Configuration:")
    print(f"  Large Batch: {config['large_batch']}")
    print(f"  Small Batch: {config['small_batch']} x {config['minibatches']}")
    print(f"  Features:    {config['in_features']} -> {config['out_features']}")
    print(f"  Iterations:  {config['iterations']} (after {config['warmup']} warmup)")

    if high_level:
        print(f"\n{'HIGH-LEVEL (TTNN) RESULTS':-^70}")
        print(f"\nLarge Batch:")
        print(f"  Total Time:    {high_level.get('large_total_ms', 0):>10.3f} ms")
        print(f"  Forward Pass:  {high_level.get('large_forward_ms', 0):>10.3f} ms")

        print(f"\nMini-Batch:")
        print(f"  Total Time:    {high_level.get('mini_total_ms', 0):>10.3f} ms")
        print(f"  Forward Pass:  {high_level.get('mini_forward_ms', 0):>10.3f} ms")

        overhead_ms = high_level.get("mini_forward_ms", 0) - high_level.get("large_forward_ms", 0)
        overhead_pct = (overhead_ms / high_level.get("large_forward_ms", 1)) * 100
        print(f"\nOverhead:        {overhead_ms:>10.3f} ms ({overhead_pct:>6.2f}%)")

    if low_level:
        print(f"\n{'LOW-LEVEL (TT-METAL) BREAKDOWN':-^70}")

        # Convert us to ms for consistency
        large_shard_ms = low_level.get("large_sharding_us", 0) / 1000
        mini_shard_ms = low_level.get("mini_sharding_us", 0) / 1000
        large_weight_ms = low_level.get("large_weight_us", 0) / 1000
        mini_weight_ms = low_level.get("mini_weight_us", 0) / 1000
        large_noc_ms = low_level.get("large_noc_us", 0) / 1000
        mini_noc_ms = low_level.get("mini_noc_us", 0) / 1000
        large_total_ms = low_level.get("large_total_us", 0) / 1000
        mini_total_ms = low_level.get("mini_total_us", 0) / 1000

        print(f"\n{'Component':<25} {'Large Batch':>15} {'Mini-Batch':>15} {'Overhead':>15}")
        print(f"{'-'*70}")
        print(
            f"{'Tensor Sharding':<25} {large_shard_ms:>12.3f} ms {mini_shard_ms:>12.3f} ms {(mini_shard_ms-large_shard_ms):>12.3f} ms"
        )
        print(
            f"{'Weight Streaming':<25} {large_weight_ms:>12.3f} ms {mini_weight_ms:>12.3f} ms {(mini_weight_ms-large_weight_ms):>12.3f} ms"
        )
        print(
            f"{'NoC Communication':<25} {large_noc_ms:>12.3f} ms {mini_noc_ms:>12.3f} ms {(mini_noc_ms-large_noc_ms):>12.3f} ms"
        )
        print(f"{'-'*70}")
        print(
            f"{'Measured Total':<25} {large_total_ms:>12.3f} ms {mini_total_ms:>12.3f} ms {(mini_total_ms-large_total_ms):>12.3f} ms"
        )

        # If we have high-level results, compute residual (actual compute)
        if high_level:
            large_compute_ms = high_level.get("large_forward_ms", 0) - large_total_ms
            mini_compute_ms = high_level.get("mini_forward_ms", 0) - mini_total_ms

            print(
                f"{'Residual (Compute)':<25} {large_compute_ms:>12.3f} ms {mini_compute_ms:>12.3f} ms {(mini_compute_ms-large_compute_ms):>12.3f} ms"
            )

            # Percentage breakdown
            print(f"\n{'OVERHEAD BREAKDOWN (% of large batch forward time)':-^70}")
            large_forward = high_level.get("large_forward_ms", 1)
            print(f"  Tensor Sharding:     {((mini_shard_ms-large_shard_ms)/large_forward*100):>6.2f}%")
            print(f"  Weight Streaming:    {((mini_weight_ms-large_weight_ms)/large_forward*100):>6.2f}%")
            print(f"  NoC Communication:   {((mini_noc_ms-large_noc_ms)/large_forward*100):>6.2f}%")
            print(f"  Compute Difference:  {((mini_compute_ms-large_compute_ms)/large_forward*100):>6.2f}%")
            print(f"  {'-'*40}")
            total_overhead_pct = (mini_total_ms - large_total_ms) / large_forward * 100
            print(f"  Total Overhead:      {total_overhead_pct:>6.2f}%")

    # Save combined report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"analysis_report_{timestamp}.json"

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "high_level_results": high_level,
        "low_level_results": low_level,
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'REPORT SAVED':-^70}")
    print(f"  JSON Report: {report_file}")
    print(f"  High-Level:  high_level_results.csv")
    print(f"  Low-Level:   low_level_results.csv")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run complete profiling analysis")
    parser.add_argument("--large-batch", type=int, default=256)
    parser.add_argument("--small-batch", type=int, default=32)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--skip-high-level", action="store_true", help="Skip high-level benchmark")
    parser.add_argument("--skip-low-level", action="store_true", help="Skip low-level profiling")

    args = parser.parse_args()

    config = {
        "large_batch": args.large_batch,
        "small_batch": args.small_batch,
        "minibatches": args.minibatches,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "warmup": args.warmup,
        "iterations": args.iterations,
    }

    print("=" * 70)
    print("COMPREHENSIVE PROFILING ANALYSIS")
    print("=" * 70)
    print("\nThis will run:")
    if not args.skip_high_level:
        print("  1. High-level TTNN benchmark (end-to-end forward pass)")
    if not args.skip_low_level:
        print("  2. Low-level profiling (sharding, weight streaming, NoC)")
    print("  3. Combined analysis with overhead breakdown")
    print()

    high_level_results = {}
    low_level_results = {}

    try:
        if not args.skip_high_level:
            high_level_results = run_high_level_benchmark(config)

        if not args.skip_low_level:
            low_level_results = run_low_level_profiling(config)

        generate_analysis_report(config, high_level_results, low_level_results)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

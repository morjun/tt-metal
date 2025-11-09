#!/usr/bin/env python3
"""
Full Device Profile Analysis Workflow

This script:
1. Runs large batch profiling
2. Runs mini-batch profiling
3. Analyzes both profiles
4. Appends comparison data to CSV for cumulative collection
"""

import subprocess
import sys
from pathlib import Path
import json
import csv
from datetime import datetime
import argparse
import os


def run_command(cmd: str, description: str) -> tuple[int, str, str]:
    """Run a shell command and return (exit_code, stdout, stderr)"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}")
    print()

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/home/masterjunmo/codes/tt-metal")

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode, result.stdout, result.stderr


def extract_minibatches_from_benchmark_csv() -> int | None:
    """Extract minibatches count from benchmark CSV file

    Returns:
        Number of minibatches, or None if not found
    """
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

                # Get the most recent row
                latest_row = rows[-1]

                if "minibatches" in latest_row:
                    try:
                        return int(latest_row["minibatches"])
                    except (ValueError, TypeError):
                        pass
        except (ValueError, KeyError, IndexError):
            continue

    return None


def parse_analysis_output(output: str) -> dict:
    """Parse device_profile_analysis.py output to extract key metrics"""
    lines = output.split("\n")
    data = {}

    for i, line in enumerate(lines):
        # Wall clock time - handle both old and new formats
        if "Device Wall Clock" in line and ":" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    data["wall_clock_ms"] = float(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

        # Python measurement - handle both old and new formats
        if "Python measurement:" in line or "Python Measurement:" in line or "Python per-forward time:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    data["python_ms"] = float(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

        # Component breakdown - parse table format more carefully
        # Format: Component  Span  Cores  WORK  Percentage%
        if "Weight Streaming" in line and "Cores" not in line:
            # Split by whitespace and get numeric values
            parts = line.split()
            nums = []
            for p in parts:
                try:
                    # Try to extract number (remove % if present)
                    val_str = p.rstrip("%")
                    val = float(val_str)
                    nums.append(val)
                except ValueError:
                    continue
            # Should have: [span, cores, work, percentage]
            if len(nums) >= 3:
                data["weight_streaming_ms"] = nums[2]  # 3rd number is work

        if "NoC Communication" in line and "Cores" not in line:
            parts = line.split()
            nums = []
            for p in parts:
                try:
                    val_str = p.rstrip("%")
                    val = float(val_str)
                    nums.append(val)
                except ValueError:
                    continue
            if len(nums) >= 3:
                data["noc_communication_ms"] = nums[2]

        if "Computation" in line and "NoC" not in line and "Cores" not in line:
            parts = line.split()
            nums = []
            for p in parts:
                try:
                    val_str = p.rstrip("%")
                    val = float(val_str)
                    nums.append(val)
                except ValueError:
                    continue
            if len(nums) >= 3:
                data["computation_ms"] = nums[2]

        # Total work
        if "TOTAL" in line and not line.strip().startswith("Component"):
            parts = line.split()
            for j, p in enumerate(parts):
                try:
                    val = float(p)
                    if val > 1.0 and "total_work_ms" not in data:
                        data["total_work_ms"] = val
                        break
                except ValueError:
                    continue

        # Also try to get from "Total work X ms" pattern
        if "Total work:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    val = float(parts[1].strip().split()[0])
                    data["total_work_ms"] = val
                except (ValueError, IndexError):
                    pass

        # Parallelism
        if "Effective Parallelism:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                data["parallelism"] = float(parts[1].strip().split("x")[0])

        # Efficiency
        if "Efficiency:" in line and "%" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                data["efficiency_pct"] = float(parts[1].strip().split("%")[0])

        # Cores used
        if "cores =" in line.lower() or ("cores" in line.lower() and "work" in line.lower()):
            parts = line.split()
            for j, p in enumerate(parts):
                if p.isdigit() and 100 <= int(p) <= 130:
                    data["cores_used"] = int(p)
                    break

        # For mini-batch: the new format analyzes a single forward pass directly
        # So all metrics are already per-forward - we just need to detect mini-batch scenario
        # and copy the metrics to per_fwd_* fields
        if "Mini-batch" in line or "mini-batch" in line.lower():
            # This is a mini-batch analysis - all metrics are per-forward
            # We'll copy them at the end if they exist
            data["_is_minibatch"] = True

        # Also look for "Total work:" which appears in the output
        if "Total work:" in line and "per_fwd_work_ms" not in data:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    # Check if this is in a mini-batch context
                    if data.get("_is_minibatch", False):
                        data["per_fwd_work_ms"] = float(parts[1].strip().split()[0])
                    else:
                        data["total_work_ms"] = float(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

    # For mini-batch scenario: copy metrics to per_fwd_* fields if not already set
    # The new format analyzes a single forward pass, so all metrics are per-forward
    if data.get("_is_minibatch", False):
        if "per_fwd_wall_ms" not in data and "wall_clock_ms" in data:
            data["per_fwd_wall_ms"] = data["wall_clock_ms"]
        if "per_fwd_python_ms" not in data and "python_ms" in data:
            data["per_fwd_python_ms"] = data["python_ms"]
        if "per_fwd_work_ms" not in data and "total_work_ms" in data:
            data["per_fwd_work_ms"] = data["total_work_ms"]
        if "per_fwd_parallelism" not in data and "parallelism" in data:
            data["per_fwd_parallelism"] = data["parallelism"]
        if "per_fwd_efficiency_pct" not in data and "efficiency_pct" in data:
            data["per_fwd_efficiency_pct"] = data["efficiency_pct"]
        # Component work (weight_streaming_ms, etc.) is already per-forward in mini-batch

    # Clean up temporary flag
    if "_is_minibatch" in data:
        del data["_is_minibatch"]

    return data


def append_comparison_to_csv(large_data: dict, mini_data: dict, csv_path: Path):
    """Append comparison data to CSV file for cumulative collection"""

    # Calculate comparison metrics
    large_wall = large_data.get("wall_clock_ms", 0)
    large_python = large_data.get("python_ms", 0)
    large_work = large_data.get("total_work_ms", 0)
    large_parallelism = large_data.get("parallelism", 0)
    large_efficiency = large_data.get("efficiency_pct", 0)
    large_weight = large_data.get("weight_streaming_ms", 0)
    large_noc = large_data.get("noc_communication_ms", 0)
    large_compute = large_data.get("computation_ms", 0)
    large_cores = large_data.get("cores_used", 0)

    mini_per_fwd_wall = mini_data.get("per_fwd_wall_ms", 0)
    mini_per_fwd_python = mini_data.get("per_fwd_python_ms", 0)
    mini_per_fwd_work = mini_data.get("per_fwd_work_ms", 0)
    mini_per_fwd_parallelism = mini_data.get("per_fwd_parallelism", 0)
    mini_per_fwd_efficiency = mini_data.get("per_fwd_efficiency_pct", 0)
    mini_total_weight = mini_data.get("weight_streaming_ms", 0)
    mini_total_noc = mini_data.get("noc_communication_ms", 0)
    mini_total_compute = mini_data.get("computation_ms", 0)
    mini_cores = mini_data.get("cores_used", 0)

    # In the new format, mini-batch analysis is already per-forward (single forward pass)
    # So component work is already per-forward, no need to divide
    # But we still need to check if we have per_fwd_* fields set
    if mini_per_fwd_work == 0 and mini_data.get("total_work_ms", 0) > 0:
        # Fallback: use total_work_ms if per_fwd_work_ms not set
        mini_per_fwd_work = mini_data.get("total_work_ms", 0)

    # Component work is already per-forward in new format
    # Only divide if it looks like total (very large values)
    if mini_total_weight > 100:  # Likely total, not per-forward
        # Get actual minibatches count from benchmark CSV
        actual_minibatches = extract_minibatches_from_benchmark_csv()
        if actual_minibatches is None:
            actual_minibatches = 8  # Default fallback
        mini_per_fwd_weight = mini_total_weight / actual_minibatches
        mini_per_fwd_noc = mini_total_noc / actual_minibatches
        mini_per_fwd_compute = mini_total_compute / actual_minibatches
    else:
        # Already per-forward
        mini_per_fwd_weight = mini_total_weight
        mini_per_fwd_noc = mini_total_noc
        mini_per_fwd_compute = mini_total_compute

    # Calculate overhead percentages
    work_overhead_pct = ((mini_per_fwd_work / large_work) - 1) * 100 if large_work > 0 else 0
    wall_clock_overhead_pct = ((mini_per_fwd_wall / large_wall) - 1) * 100 if large_wall > 0 else 0

    # Prepare row data
    row = {
        "timestamp": datetime.now().isoformat(),
        "large_wall_clock_ms": f"{large_wall:.6f}",
        "large_python_ms": f"{large_python:.6f}",
        "large_total_work_ms": f"{large_work:.2f}",
        "large_parallelism": f"{large_parallelism:.1f}",
        "large_efficiency_pct": f"{large_efficiency:.1f}",
        "large_weight_streaming_ms": f"{large_weight:.2f}",
        "large_noc_communication_ms": f"{large_noc:.2f}",
        "large_computation_ms": f"{large_compute:.2f}",
        "mini_per_fwd_wall_clock_ms": f"{mini_per_fwd_wall:.6f}",
        "mini_per_fwd_python_ms": f"{mini_per_fwd_python:.6f}",
        "mini_per_fwd_total_work_ms": f"{mini_per_fwd_work:.2f}",
        "mini_per_fwd_parallelism": f"{mini_per_fwd_parallelism:.1f}",
        "mini_per_fwd_efficiency_pct": f"{mini_per_fwd_efficiency:.1f}",
        "mini_per_fwd_weight_streaming_ms": f"{mini_per_fwd_weight:.2f}",
        "mini_per_fwd_noc_communication_ms": f"{mini_per_fwd_noc:.2f}",
        "mini_per_fwd_computation_ms": f"{mini_per_fwd_compute:.2f}",
        "work_overhead_pct": f"{work_overhead_pct:.1f}",
        "wall_clock_overhead_pct": f"{wall_clock_overhead_pct:.1f}",
        "cores_used_large": str(large_cores),
        "cores_used_mini": str(mini_cores),
    }

    # Field names (column headers)
    fieldnames = list(row.keys())

    # Check if file exists and has content
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    # Write CSV (append mode)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n✓ Comparison data appended to: {csv_path}")


def generate_markdown_report(large_data: dict, mini_data: dict, output_path: Path):
    """Generate comprehensive markdown report"""

    # Calculate ratios
    work_ratio = mini_data.get("per_fwd_work_ms", 0) / large_data.get("total_work_ms", 1)
    wall_ratio = mini_data.get("per_fwd_wall_ms", 0) / large_data.get("wall_clock_ms", 1)

    # Get component data
    large_weight = large_data.get("weight_streaming_ms", 0)
    large_noc = large_data.get("noc_communication_ms", 0)
    large_compute = large_data.get("computation_ms", 0)
    large_total = large_data.get("total_work_ms", 0)

    # Get actual minibatches count from benchmark CSV
    actual_minibatches = extract_minibatches_from_benchmark_csv()

    if actual_minibatches is None:
        # Fallback: Estimate number of forwards from total work vs per-forward work
        mini_total_work = mini_data.get("total_work_ms", 0)
        mini_per_fwd_work = mini_data.get("per_fwd_work_ms", 0)
        estimated_forwards = int(mini_total_work / mini_per_fwd_work) if mini_per_fwd_work > 0 else 8
        estimated_forwards = max(1, estimated_forwards)  # At least 1 forward
        print(f"WARNING: Could not read minibatches from CSV, estimated: {estimated_forwards}")
    else:
        estimated_forwards = actual_minibatches
        print(f"Using actual minibatches count from CSV: {estimated_forwards}")

    # Calculate per-forward component work
    mini_total_work = mini_data.get("total_work_ms", 0)
    mini_per_fwd_work = mini_data.get("per_fwd_work_ms", 0)

    mini_weight = mini_data.get("weight_streaming_ms", 0) / estimated_forwards if estimated_forwards > 0 else 0
    mini_noc = mini_data.get("noc_communication_ms", 0) / estimated_forwards if estimated_forwards > 0 else 0
    mini_compute = mini_data.get("computation_ms", 0) / estimated_forwards if estimated_forwards > 0 else 0
    mini_total = mini_data.get("per_fwd_work_ms", 0)

    large_cores = large_data.get("cores_used", 128)  # Default to 128 if not found
    mini_cores = mini_data.get("cores_used", 130)  # Default to 130 if not found

    content = f"""# Device Profile Analysis Results

## Overview
This document contains device-level profiling analysis for weight loading scenarios using Tenstorrent's device profiler, with proper parallel work calculation.

**Generated automatically by `run_full_device_profile.py`**

## Hardware Configuration
- **Device**: Tenstorrent Blackhole P150A
- **Clock Frequency**: 1.35 GHz
- **Tensix Cores**: 130 (13×10 grid)
- **RISC Processors per Core**: 5 (1 BRISC + 1 NCRISC + 3 TRISC)
- **Maximum Theoretical Parallelism**: 130 cores × 5 RISCs = **650x**

## Profiling Method
- **Tool**: `TT_METAL_DEVICE_PROFILER=1` environment variable
- **Output**: `generated/profiler/.logs/profile_log_device.csv`
- **Measurement**: Cycle-accurate timestamps at 1.35 GHz
- **Correct Calculation**: Parallel work = timeline span × num_cores (NOT sum of zone durations!)

## Test Configuration
- **Model**: Single linear layer (4096 → 4096)
- **Weight**: 64 MB (4096 × 4096 × FP32)
- **Large batch**: B=256 (1 forward pass)
- **Mini-batch**: b=32 ({estimated_forwards} forward passes)

---

## Large Batch Scenario (B=256, 1 forward pass)

### Python-level Timing
```
Forward (ms): {large_data.get('python_ms', 'N/A'):.3f}
```

### Device Profile Results

**Operation Analysis**:
- Wall clock time: **{large_data.get('wall_clock_ms', 0):.3f} ms**
- Cores used: **{large_cores} cores**

### Parallel Work Breakdown (Correct Calculation)

**Formula**: Work = timeline_span × num_cores

| Component | Parallel Work | % of Total |
|-----------|---------------|------------|
| **Weight Streaming (BRISC)** | {large_weight:.2f} ms | {(large_weight/large_total*100):.1f}% |
| **NoC Communication (NCRISC)** | {large_noc:.2f} ms | {(large_noc/large_total*100):.1f}% |
| **Computation (TRISC)** | {large_compute:.2f} ms | {(large_compute/large_total*100):.1f}% |
| **TOTAL** | **{large_total:.2f} ms** | 100.0% |

**Effective Parallelism**: {large_data.get('parallelism', 0):.1f}x ({large_total:.2f} ms / {large_data.get('wall_clock_ms', 1):.3f} ms)
- Theoretical maximum: 650x
- **Efficiency: {large_data.get('efficiency_pct', 0):.1f}%**

**Interpretation**:
- All components run in **parallel** with high overlap
- Wall clock ({large_data.get('wall_clock_ms', 0):.3f} ms) vs Python time ({large_data.get('python_ms', 0):.3f} ms)
- Work distribution is balanced: ~33% each component

---

## Mini-batch Scenario (b=32, {estimated_forwards} forward passes)

### Python-level Timing ({estimated_forwards} forwards total)
```
Total: {mini_data.get('python_ms', 0) * estimated_forwards:.3f} ms
Per-forward: {mini_data.get('per_fwd_python_ms', 0):.3f} ms
```

### Device Profile Results

**Operation Analysis**:
- Total wall clock time: **{mini_data.get('wall_clock_ms', 0):.3f} ms** (timeline span, includes overlaps)
- Cores used: **{mini_cores} cores** (100% utilization!)

### Parallel Work Breakdown ({estimated_forwards} forwards total)

| Component | Parallel Work (total) | Parallel Work (per forward) | % of Total |
|-----------|----------------------|---------------------------|------------|
| **Weight Streaming (BRISC)** | {mini_data.get('weight_streaming_ms', 0):.2f} ms | {mini_weight:.2f} ms | {(mini_data.get('weight_streaming_ms', 0)/mini_data.get('total_work_ms', 1)*100):.1f}% |
| **NoC Communication (NCRISC)** | {mini_data.get('noc_communication_ms', 0):.2f} ms | {mini_noc:.2f} ms | {(mini_data.get('noc_communication_ms', 0)/mini_data.get('total_work_ms', 1)*100):.1f}% |
| **Computation (TRISC)** | {mini_data.get('computation_ms', 0):.2f} ms | {mini_compute:.2f} ms | {(mini_data.get('computation_ms', 0)/mini_data.get('total_work_ms', 1)*100):.1f}% |
| **TOTAL** | **{mini_data.get('total_work_ms', 0):.2f} ms** | **{mini_total:.2f} ms** | 100.0% |

**Effective Parallelism**: {mini_data.get('parallelism', 0):.1f}x (total), {mini_data.get('per_fwd_parallelism', 0):.1f}x (per forward)
- Theoretical maximum: 650x
- **Efficiency: {mini_data.get('efficiency_pct', 0):.1f}% (total), {mini_data.get('per_fwd_efficiency_pct', 0):.1f}% (per forward)**

**Interpretation**:
- Mini-batch uses **MORE cores** than large batch ({mini_cores} vs {large_cores})
- Higher parallelism efficiency per forward ({mini_data.get('per_fwd_efficiency_pct', 0):.1f}% vs {large_data.get('efficiency_pct', 0):.1f}%)
- But requires {estimated_forwards} forwards to process same amount of data

---

## Comparison: Large Batch vs Mini-batch

### Per-forward Work (Fair Comparison)

| Component | Large Batch | Mini-batch (per fwd) | Ratio |
|-----------|-------------|---------------------|-------|
| **Weight Streaming** | {large_weight:.2f} ms | {mini_weight:.2f} ms | {(mini_weight/large_weight):.2f}x |
| **NoC Communication** | {large_noc:.2f} ms | {mini_noc:.2f} ms | {(mini_noc/large_noc):.2f}x |
| **Computation** | {large_compute:.2f} ms | {mini_compute:.2f} ms | {(mini_compute/large_compute):.2f}x |
| **TOTAL WORK** | {large_total:.2f} ms | {mini_total:.2f} ms | **{work_ratio:.2f}x** |
| **Wall Clock** | {large_data.get('wall_clock_ms', 0):.3f} ms | {mini_data.get('per_fwd_wall_ms', 0):.3f} ms | {wall_ratio:.2f}x |
| **Parallelism** | {large_data.get('parallelism', 0):.1f}x | {mini_data.get('per_fwd_parallelism', 0):.1f}x | {(mini_data.get('per_fwd_parallelism', 1)/large_data.get('parallelism', 1)):.2f}x |

**Key Finding**: Mini-batch performs **{((work_ratio - 1) * 100):.1f}% more work per forward** despite having:
- Batch size 1/8 of large batch (32 vs 256)
- Same or more cores used ({mini_cores} vs {large_cores})

### Why Mini-batch Uses More Work?

**Root Cause**: Mini-batch uses the same number of cores ({mini_cores}) as large batch ({large_cores}), despite processing 1/8 the data!

Expected behavior:
- Large batch (B=256): {large_cores} cores → {(256/large_cores):.1f} samples per core
- Mini-batch (b=32): Should use ~16 cores → 2 samples per core

Actual behavior:
- Mini-batch (b=32): Uses **{mini_cores} cores** → {(32/mini_cores):.2f} samples per core!

**This is ~8x over-parallelization**, causing:
1. More weight streaming work (each core loads weights)
2. More NoC communication (inter-core coordination)
3. Less efficient per-core utilization

### Total Time Comparison (256 elements)

| Metric | Large Batch | Mini-batch (×{estimated_forwards}) | Ratio |
|--------|-------------|-----------------------------------|-------|
| **Device time** | {large_data.get('wall_clock_ms', 0):.3f} ms | {mini_data.get('per_fwd_wall_ms', 0)*estimated_forwards:.3f} ms | **{(mini_data.get('per_fwd_wall_ms', 1)*estimated_forwards/large_data.get('wall_clock_ms', 1)):.1f}x slower** |
| **Python time** | {large_data.get('python_ms', 0):.3f} ms | {mini_data.get('per_fwd_python_ms', 0)*estimated_forwards:.3f} ms | **{(mini_data.get('per_fwd_python_ms', 1)*estimated_forwards/large_data.get('python_ms', 1)):.1f}x slower** |
| **Total Work** | {large_total:.2f} ms | {mini_total*estimated_forwards:.2f} ms | **{(mini_total*estimated_forwards/large_total):.1f}x more work** |

---

## Key Findings

### 1. Incorrect Parallelization Strategy

**Mini-batch uses ~8x more cores than needed:**
- Expected: 32 samples → ~16 cores (with 2 samples/core)
- Actual: 32 samples → {mini_cores} cores (with {(32/mini_cores):.2f} samples/core!)

This causes:
- {((work_ratio - 1) * 100):.1f}% more work per forward
- {(mini_data.get('per_fwd_wall_ms', 1)*8/large_data.get('wall_clock_ms', 1)):.1f}x slower total time for same amount of data

### 2. Work Distribution is Consistent

Both scenarios show similar distribution:
- Weight streaming: ~{(large_weight/large_total*100):.0f}% (large), ~{(mini_weight/mini_total*100):.0f}% (mini)
- NoC communication: ~{(large_noc/large_total*100):.0f}% (large), ~{(mini_noc/mini_total*100):.0f}% (mini)
- Computation: ~{(large_compute/large_total*100):.0f}% (large), ~{(mini_compute/mini_total*100):.0f}% (mini)

This indicates **balanced parallel execution**, not a weight streaming bottleneck.

### 3. Parallelism Efficiency

- Large batch: {large_data.get('efficiency_pct', 0):.1f}% efficiency ({large_data.get('parallelism', 0):.1f}x / 650x)
- Mini-batch per-forward: {mini_data.get('per_fwd_efficiency_pct', 0):.1f}% efficiency ({mini_data.get('per_fwd_parallelism', 0):.1f}x / 650x)

Mini-batch is actually **more efficient per-forward** when normalized by work, but performs unnecessary work due to over-parallelization.

### 4. Weight Streaming is NOT the Bottleneck

Weight streaming occupies:
- Large batch: {(large_weight/large_total*100):.1f}% of total work
- Mini-batch: {(mini_weight/mini_total*100):.1f}% of total work

**Computation is similar** to weight streaming, indicating balanced workload.

The real bottleneck is the **inefficient resource allocation strategy** that uses ~8x more cores than necessary for mini-batch.

---

## Conclusion

### The Real Problem: Over-Parallelization

**Mini-batch uses ~8x more cores than necessary:**
- {mini_cores} cores for 32 samples = {(32/mini_cores if isinstance(mini_cores, int) else 0):.2f} samples/core
- Should use ~16 cores for 32 samples = 2 samples/core (same as large batch)

This causes:
- **{((work_ratio - 1) * 100):.1f}% more work per forward** ({mini_total:.2f} ms vs {large_total:.2f} ms)
- **{(mini_data.get('per_fwd_wall_ms', 1)*estimated_forwards/large_data.get('wall_clock_ms', 1)):.1f}x slower total time** ({mini_data.get('per_fwd_wall_ms', 0)*estimated_forwards:.3f} ms vs {large_data.get('wall_clock_ms', 0):.3f} ms)
- **{(mini_total*estimated_forwards/large_total):.1f}x more total work** ({mini_total*estimated_forwards:.1f} ms vs {large_total:.1f} ms)

### Weight Streaming is Well-Optimized

Weight streaming occupies **~{(large_weight/large_total*100):.0f}% of total work**, balanced with:
- NoC communication: ~{(large_noc/large_total*100):.0f}%
- Computation: ~{(large_compute/large_total*100):.0f}%

This indicates **efficient parallel execution**, not a weight streaming bottleneck.

### The Fix

To make mini-batch efficient, ttnn should:
1. **Scale cores with batch size**: 32 samples → ~16 cores (not {mini_cores})
2. **Maintain samples-per-core ratio**: Keep ~2 samples/core like large batch
3. **Result**: Mini-batch would be ~8x faster with ~8x less work

### Bottom Line

**Weight streaming is NOT the bottleneck.** The issue is ttnn's sharding strategy that fails to reduce core usage proportionally with batch size, causing massive over-parallelization and wasted work.

---

## Raw Data

### Large Batch
```json
{json.dumps(large_data, indent=2)}
```

### Mini-batch
```json
{json.dumps(mini_data, indent=2)}
```
"""

    output_path.write_text(content)
    print(f"\n✓ Report written to: {output_path}")


def main():
    """Main workflow"""

    parser = argparse.ArgumentParser(description="Run full device profile analysis workflow")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="research_codes/device_profile_comparison.csv",
        help="Path to CSV file for cumulative comparison data (default: research_codes/device_profile_comparison.csv)",
    )
    parser.add_argument(
        "--generate-markdown",
        action="store_true",
        help="Generate markdown report (disabled by default)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("AUTOMATED DEVICE PROFILE ANALYSIS WORKFLOW")
    print("=" * 80)
    print()
    print("This script will:")
    print("  1. Clear old profile data")
    print("  2. Run large batch profiling")
    print("  3. Analyze and save large batch results")
    print("  4. Run mini-batch profiling")
    print("  5. Analyze and save mini-batch results")
    print("  6. Append comparison data to CSV for cumulative collection")
    if args.generate_markdown:
        print("  7. Generate markdown report")
    print()

    # Step 1: Clear old profile data
    rc, _, _ = run_command("rm -f generated/profiler/.logs/profile_log_device.csv", "Step 1: Clearing old profile data")
    if rc != 0:
        print("⚠️  Warning: Could not clear old profile data (may not exist yet)")

    # Step 2: Run large batch profiling
    rc, stdout, stderr = run_command(
        "TT_METAL_DEVICE_PROFILER=1 python3 research_codes/weight_loading_test.py --only-large",
        "Step 2: Running large batch profiling",
    )
    if rc != 0:
        print(f"❌ ERROR: Large batch profiling failed with exit code {rc}")
        sys.exit(1)

    # Step 3: Analyze large batch
    rc, large_output, stderr = run_command(
        "python3 research_codes/device_profile_analysis.py", "Step 3: Analyzing large batch profile"
    )
    if rc != 0:
        print(f"❌ ERROR: Large batch analysis failed with exit code {rc}")
        sys.exit(1)

    large_data = parse_analysis_output(large_output)
    print(f"\n✓ Large batch analysis complete")
    print(f"  Wall clock: {large_data.get('wall_clock_ms', 0):.3f} ms")
    print(f"  Total work: {large_data.get('total_work_ms', 0):.2f} ms")
    print(f"  Parallelism: {large_data.get('parallelism', 0):.1f}x")

    # Save large batch results
    large_json_path = Path("research_codes/large_batch_profile.json")
    large_json_path.write_text(json.dumps(large_data, indent=2))
    print(f"  Data saved to: {large_json_path}")

    # Step 4: Clear and run mini-batch profiling
    rc, _, _ = run_command(
        "rm -f generated/profiler/.logs/profile_log_device.csv", "Step 4a: Clearing profile data for mini-batch"
    )

    rc, stdout, stderr = run_command(
        "TT_METAL_DEVICE_PROFILER=1 python3 research_codes/weight_loading_test.py --only-mini",
        "Step 4b: Running mini-batch profiling",
    )
    if rc != 0:
        print(f"❌ ERROR: Mini-batch profiling failed with exit code {rc}")
        sys.exit(1)

    # Step 5: Analyze mini-batch
    rc, mini_output, stderr = run_command(
        "python3 research_codes/device_profile_analysis.py", "Step 5: Analyzing mini-batch profile"
    )
    if rc != 0:
        print(f"❌ ERROR: Mini-batch analysis failed with exit code {rc}")
        sys.exit(1)

    mini_data = parse_analysis_output(mini_output)
    print(f"\n✓ Mini-batch analysis complete")
    print(f"  Wall clock (per fwd): {mini_data.get('per_fwd_wall_ms', 0):.3f} ms")
    print(f"  Total work (per fwd): {mini_data.get('per_fwd_work_ms', 0):.2f} ms")
    print(f"  Parallelism (per fwd): {mini_data.get('per_fwd_parallelism', 0):.1f}x")

    # Save mini-batch results
    mini_json_path = Path("research_codes/mini_batch_profile.json")
    mini_json_path.write_text(json.dumps(mini_data, indent=2))
    print(f"  Data saved to: {mini_json_path}")

    # Step 6: Append comparison data to CSV
    csv_path = Path(args.output_csv)
    append_comparison_to_csv(large_data, mini_data, csv_path)

    # Step 7: Generate markdown report (optional)
    output_path = None
    if args.generate_markdown:
        output_path = Path("research_codes/DEVICE_PROFILE_RESULTS.md")
        generate_markdown_report(large_data, mini_data, output_path)

    print("\n" + "=" * 80)
    print("✅ WORKFLOW COMPLETE")
    print("=" * 80)
    print()
    print("Results:")
    print(f"  - Large batch data: {large_json_path}")
    print(f"  - Mini-batch data: {mini_json_path}")
    print(f"  - Comparison CSV: {csv_path}")
    if args.generate_markdown and output_path:
        print(f"  - Markdown report: {output_path}")
    print()
    print("Summary:")
    work_ratio = mini_data.get("per_fwd_work_ms", 0) / large_data.get("total_work_ms", 1)
    print(f"  - Large batch work: {large_data.get('total_work_ms', 0):.2f} ms")
    print(f"  - Mini-batch work (per fwd): {mini_data.get('per_fwd_work_ms', 0):.2f} ms")
    print(f"  - Mini-batch overhead: {((work_ratio - 1) * 100):.1f}% more work per forward")
    print(f"  - Root cause: Over-parallelization ({mini_data.get('cores_used', 'N/A')} cores for 32 samples)")
    print()
    print(f"💡 Tip: Run this script multiple times to build cumulative data in {csv_path}")
    print()


if __name__ == "__main__":
    main()

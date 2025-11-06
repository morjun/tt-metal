#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Python wrapper for low-level profiling of tensor sharding, weight streaming, and NoC communication.

This script uses tt-metal APIs to measure:
1. Tensor sharding time: GDDR6 DRAM -> L1 SRAM (interleaved to sharded conversion)
2. Weight streaming time: GDDR6 DRAM -> L1 SRAM (weight loading for each forward pass)
3. NoC communication time: Inter-core data transfer

Comparison: Large batch (single pass) vs Mini-batch (multiple passes)

Usage:
    # Basic usage
    python profiling_sharding_noc_python.py

    # With Tracy profiling enabled
    TT_METAL_DEVICE_PROFILER=1 python -m tracy -r profiling_sharding_noc_python.py

    # Custom configuration
    python profiling_sharding_noc_python.py --large-batch 512 --small-batch 64 --minibatches 8
"""

import os
import sys
import time
import argparse
import csv
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

import torch
import ttnn

# Ensure we're in the tt-metal directory
script_dir = os.path.dirname(os.path.abspath(__file__))
tt_metal_root = os.path.abspath(os.path.join(script_dir, ".."))
if os.path.exists(tt_metal_root) and os.path.isdir(tt_metal_root):
    os.chdir(tt_metal_root)
if tt_metal_root not in sys.path:
    sys.path.insert(0, tt_metal_root)


@dataclass
class ProfilingConfig:
    """Configuration for profiling benchmarks."""

    large_batch_size: int = 256
    small_batch_size: int = 32
    num_minibatches: int = 8
    in_features: int = 4096
    out_features: int = 4096
    warmup_iterations: int = 2
    measure_iterations: int = 5
    enable_tracy: bool = False
    output_csv: Optional[str] = "profiling_results_python.csv"


@dataclass
class ProfilingResults:
    """Results from profiling measurements."""

    # Sharding times (DRAM interleaved -> L1 sharded)
    large_batch_sharding_us: float = 0.0
    mini_batch_sharding_us: float = 0.0

    # Weight streaming times (DRAM -> L1)
    large_batch_weight_stream_us: float = 0.0
    mini_batch_weight_stream_us: float = 0.0

    # NoC communication times
    large_batch_noc_comm_us: float = 0.0
    mini_batch_noc_comm_us: float = 0.0

    # Total times
    large_batch_total_us: float = 0.0
    mini_batch_total_us: float = 0.0

    # Overheads
    sharding_overhead_us: float = 0.0
    weight_streaming_overhead_us: float = 0.0
    noc_comm_overhead_us: float = 0.0
    total_overhead_us: float = 0.0
    overhead_percent: float = 0.0

    def calculate_overheads(self):
        """Calculate overhead metrics."""
        self.sharding_overhead_us = self.mini_batch_sharding_us - self.large_batch_sharding_us
        self.weight_streaming_overhead_us = self.mini_batch_weight_stream_us - self.large_batch_weight_stream_us
        self.noc_comm_overhead_us = self.mini_batch_noc_comm_us - self.large_batch_noc_comm_us
        self.total_overhead_us = self.mini_batch_total_us - self.large_batch_total_us
        if self.large_batch_total_us > 0:
            self.overhead_percent = (self.total_overhead_us / self.large_batch_total_us) * 100.0

    def print(self):
        """Print results to console."""
        print("\n" + "=" * 60)
        print("           PROFILING RESULTS")
        print("=" * 60)

        print("\n--- Tensor Sharding (DRAM -> L1 SRAM) ---")
        print(f"  Large batch:  {self.large_batch_sharding_us:>12.3f} us")
        print(f"  Mini-batch:   {self.mini_batch_sharding_us:>12.3f} us")
        print(f"  Overhead:     {self.sharding_overhead_us:>12.3f} us")

        print("\n--- Weight Streaming (DRAM -> L1 SRAM) ---")
        print(f"  Large batch:  {self.large_batch_weight_stream_us:>12.3f} us")
        print(f"  Mini-batch:   {self.mini_batch_weight_stream_us:>12.3f} us")
        print(f"  Overhead:     {self.weight_streaming_overhead_us:>12.3f} us")

        print("\n--- NoC Communication ---")
        print(f"  Large batch:  {self.large_batch_noc_comm_us:>12.3f} us")
        print(f"  Mini-batch:   {self.mini_batch_noc_comm_us:>12.3f} us")
        print(f"  Overhead:     {self.noc_comm_overhead_us:>12.3f} us")

        print("\n--- Total Operation Time ---")
        print(f"  Large batch:  {self.large_batch_total_us:>12.3f} us")
        print(f"  Mini-batch:   {self.mini_batch_total_us:>12.3f} us")
        print(f"  Total Overhead: {self.total_overhead_us:>10.3f} us ({self.overhead_percent:>6.2f}%)")

        print("\n" + "=" * 60 + "\n")

    def save_to_csv(self, filename: str, config: ProfilingConfig):
        """Save results to CSV file."""
        file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0

        with open(filename, "a", newline="") as f:
            fieldnames = [
                "timestamp",
                "large_batch_size",
                "small_batch_size",
                "num_minibatches",
                "in_features",
                "out_features",
                "large_batch_sharding_us",
                "mini_batch_sharding_us",
                "sharding_overhead_us",
                "large_batch_weight_stream_us",
                "mini_batch_weight_stream_us",
                "weight_stream_overhead_us",
                "large_batch_noc_comm_us",
                "mini_batch_noc_comm_us",
                "noc_comm_overhead_us",
                "large_batch_total_us",
                "mini_batch_total_us",
                "total_overhead_us",
                "overhead_percent",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "large_batch_size": config.large_batch_size,
                "small_batch_size": config.small_batch_size,
                "num_minibatches": config.num_minibatches,
                "in_features": config.in_features,
                "out_features": config.out_features,
                "large_batch_sharding_us": f"{self.large_batch_sharding_us:.6f}",
                "mini_batch_sharding_us": f"{self.mini_batch_sharding_us:.6f}",
                "sharding_overhead_us": f"{self.sharding_overhead_us:.6f}",
                "large_batch_weight_stream_us": f"{self.large_batch_weight_stream_us:.6f}",
                "mini_batch_weight_stream_us": f"{self.mini_batch_weight_stream_us:.6f}",
                "weight_stream_overhead_us": f"{self.weight_streaming_overhead_us:.6f}",
                "large_batch_noc_comm_us": f"{self.large_batch_noc_comm_us:.6f}",
                "mini_batch_noc_comm_us": f"{self.mini_batch_noc_comm_us:.6f}",
                "noc_comm_overhead_us": f"{self.noc_comm_overhead_us:.6f}",
                "large_batch_total_us": f"{self.large_batch_total_us:.6f}",
                "mini_batch_total_us": f"{self.mini_batch_total_us:.6f}",
                "total_overhead_us": f"{self.total_overhead_us:.6f}",
                "overhead_percent": f"{self.overhead_percent:.2f}",
            }

            writer.writerow(row)

        print(f"Results saved to: {filename}")


def device_sync(device: ttnn.Device):
    """Synchronize device to ensure all operations complete."""
    try:
        ttnn.synchronize_device(device)
    except Exception:
        # Fallback for different builds
        try:
            device.synchronize()
        except Exception:
            pass


def measure_interleaved_to_sharded_time(
    device: ttnn.Device, batch_size: int, features: int, num_iterations: int
) -> float:
    """
    Measure time to convert interleaved tensor (DRAM) to sharded tensor (L1 SRAM).

    This simulates the tensor sharding operation where an input tensor stored in
    GDDR6 DRAM (interleaved layout) is distributed across multiple Tensix cores'
    L1 SRAM (sharded layout).
    """
    timings = []

    # Create a tensor in DRAM with interleaved layout
    shape = (1, 1, batch_size, features)
    input_torch = torch.randn(shape, dtype=torch.bfloat16)

    # Compute grid for sharding
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = compute_grid.x * compute_grid.y

    # Define sharding spec (height-sharded for linear layer input)
    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )

    # Shard height is batch dimension divided across cores
    shard_height = ((batch_size + num_cores - 1) // num_cores + 31) // 32 * 32  # Round to tile
    shard_width = ((features + 31) // 32) * 32  # Round to tile

    shard_spec = ttnn.ShardSpec(shard_grid, (shard_height, shard_width), ttnn.ShardOrientation.ROW_MAJOR)

    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    for _ in range(num_iterations):
        # Create tensor in DRAM (interleaved)
        tensor_dram = ttnn.from_torch(
            input_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Measure conversion to sharded L1
        device_sync(device)
        start = time.perf_counter()

        # This operation does: DRAM (interleaved) -> L1 (sharded)
        tensor_sharded = ttnn.to_memory_config(tensor_dram, sharded_mem_config)

        device_sync(device)
        end = time.perf_counter()

        elapsed_us = (end - start) * 1e6
        timings.append(elapsed_us)

        # Cleanup
        tensor_dram.deallocate()
        tensor_sharded.deallocate()

    return sum(timings) / len(timings)


def measure_weight_streaming_time(
    device: ttnn.Device, weight_rows: int, weight_cols: int, num_iterations: int
) -> float:
    """
    Measure time to stream weights from DRAM to L1 SRAM.

    This simulates loading weights from GDDR6 DRAM to L1 SRAM, which happens
    when a matmul kernel needs to access weight data.
    """
    timings = []

    shape = (1, 1, weight_rows, weight_cols)
    weight_torch = torch.randn(shape, dtype=torch.bfloat16)

    for _ in range(num_iterations):
        # Weights stored in DRAM
        weight_dram = ttnn.from_torch(
            weight_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Measure streaming to L1
        device_sync(device)
        start = time.perf_counter()

        # Stream to L1 (simulating weight load for compute)
        weight_l1 = ttnn.to_memory_config(weight_dram, ttnn.L1_MEMORY_CONFIG)

        device_sync(device)
        end = time.perf_counter()

        elapsed_us = (end - start) * 1e6
        timings.append(elapsed_us)

        # Cleanup
        weight_dram.deallocate()
        weight_l1.deallocate()

    return sum(timings) / len(timings)


def measure_noc_communication_time(device: ttnn.Device, data_size_elements: int, num_iterations: int) -> float:
    """
    Measure NoC communication time by creating tensors on different cores.

    This measures inter-core communication via the Network-on-Chip (NoC).
    """
    timings = []

    # Create tensor that will be distributed across cores
    shape = (1, 1, 32, data_size_elements)  # Ensure tile-aligned
    data_torch = torch.randn(shape, dtype=torch.bfloat16)

    compute_grid = device.compute_with_storage_grid_size()

    # Create sharded config that forces multi-core distribution
    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )

    num_cores = compute_grid.x * compute_grid.y
    shard_height = 32
    shard_width = ((data_size_elements + num_cores - 1) // num_cores + 31) // 32 * 32

    shard_spec = ttnn.ShardSpec(shard_grid, (shard_height, shard_width), ttnn.ShardOrientation.ROW_MAJOR)

    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    for _ in range(num_iterations):
        # Create sharded tensor (distributed across cores)
        tensor = ttnn.from_torch(
            data_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_mem_config
        )

        # Measure time to collect back (involves NoC communication)
        device_sync(device)
        start = time.perf_counter()

        # Convert back to interleaved (requires gathering from all cores via NoC)
        tensor_gathered = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)

        device_sync(device)
        end = time.perf_counter()

        elapsed_us = (end - start) * 1e6
        timings.append(elapsed_us)

        # Cleanup
        tensor.deallocate()
        tensor_gathered.deallocate()

    return sum(timings) / len(timings)


def run_profiling_benchmark(config: ProfilingConfig) -> ProfilingResults:
    """Run complete profiling benchmark."""
    results = ProfilingResults()

    # Initialize device
    device = ttnn.open_device(device_id=0)

    print(f"Device initialized: 0")
    print(f"Compute grid size: {device.compute_with_storage_grid_size()}")
    print()

    # Warmup
    print("Running warmup iterations...")
    for _ in range(config.warmup_iterations):
        measure_interleaved_to_sharded_time(device, config.large_batch_size, config.in_features, 1)

    print("Running measurements...\n")

    # === LARGE BATCH MEASUREMENTS ===
    print(f"Measuring large batch scenario (batch={config.large_batch_size})...")

    # 1. Tensor sharding
    results.large_batch_sharding_us = measure_interleaved_to_sharded_time(
        device, config.large_batch_size, config.in_features, config.measure_iterations
    )
    print(f"  Sharding time: {results.large_batch_sharding_us:.3f} us")

    # 2. Weight streaming
    results.large_batch_weight_stream_us = measure_weight_streaming_time(
        device, config.out_features, config.in_features, config.measure_iterations
    )
    print(f"  Weight streaming time: {results.large_batch_weight_stream_us:.3f} us")

    # 3. NoC communication
    results.large_batch_noc_comm_us = measure_noc_communication_time(
        device, config.in_features, config.measure_iterations
    )
    print(f"  NoC comm time: {results.large_batch_noc_comm_us:.3f} us")

    results.large_batch_total_us = (
        results.large_batch_sharding_us + results.large_batch_weight_stream_us + results.large_batch_noc_comm_us
    )

    # === MINI-BATCH MEASUREMENTS ===
    print(f"\nMeasuring mini-batch scenario (batch={config.small_batch_size} x {config.num_minibatches})...")

    total_mini_sharding = 0.0
    total_mini_weight_stream = 0.0
    total_mini_noc_comm = 0.0

    for mb in range(config.num_minibatches):
        # 1. Tensor sharding for each minibatch
        total_mini_sharding += measure_interleaved_to_sharded_time(
            device, config.small_batch_size, config.in_features, config.measure_iterations
        )

        # 2. Weight streaming (weights may be reloaded for each minibatch)
        total_mini_weight_stream += measure_weight_streaming_time(
            device, config.out_features, config.in_features, config.measure_iterations
        )

        # 3. NoC communication
        total_mini_noc_comm += measure_noc_communication_time(device, config.in_features, config.measure_iterations)

    results.mini_batch_sharding_us = total_mini_sharding
    results.mini_batch_weight_stream_us = total_mini_weight_stream
    results.mini_batch_noc_comm_us = total_mini_noc_comm
    results.mini_batch_total_us = total_mini_sharding + total_mini_weight_stream + total_mini_noc_comm

    print(f"  Total sharding time: {results.mini_batch_sharding_us:.3f} us")
    print(f"  Total weight streaming time: {results.mini_batch_weight_stream_us:.3f} us")
    print(f"  Total NoC comm time: {results.mini_batch_noc_comm_us:.3f} us")

    # Calculate overheads
    results.calculate_overheads()

    # Cleanup
    ttnn.close_device(device)

    return results


def parse_args() -> ProfilingConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile tensor sharding, weight streaming, and NoC communication at tt-metal level"
    )

    parser.add_argument("--large-batch", type=int, default=256, help="Large batch size")
    parser.add_argument("--small-batch", type=int, default=32, help="Small batch size for minibatching")
    parser.add_argument("--minibatches", type=int, default=8, help="Number of minibatches")
    parser.add_argument("--in-features", type=int, default=4096, help="Input feature dimension")
    parser.add_argument("--out-features", type=int, default=4096, help="Output feature dimension")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=5, help="Number of measurement iterations")
    parser.add_argument("--output", type=str, default="profiling_results_python.csv", help="Output CSV file")
    parser.add_argument(
        "--enable-tracy", action="store_true", help="Enable Tracy profiling (requires TT_METAL_DEVICE_PROFILER=1)"
    )

    args = parser.parse_args()

    return ProfilingConfig(
        large_batch_size=args.large_batch,
        small_batch_size=args.small_batch,
        num_minibatches=args.minibatches,
        in_features=args.in_features,
        out_features=args.out_features,
        warmup_iterations=args.warmup,
        measure_iterations=args.iterations,
        enable_tracy=args.enable_tracy,
        output_csv=args.output,
    )


def main():
    """Main entry point."""
    config = parse_args()

    print("=" * 60)
    print("     TT-Metal Low-Level Profiling (Python)")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Large batch size:  {config.large_batch_size}")
    print(f"  Small batch size:  {config.small_batch_size}")
    print(f"  Num minibatches:   {config.num_minibatches}")
    print(f"  In features:       {config.in_features}")
    print(f"  Out features:      {config.out_features}")
    print(f"  Warmup iters:      {config.warmup_iterations}")
    print(f"  Measure iters:     {config.measure_iterations}")
    print(f"  Tracy enabled:     {config.enable_tracy}")
    print(f"  Output CSV:        {config.output_csv}")
    print("=" * 60 + "\n")

    # Check if Tracy is enabled via environment
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        print("Note: TT_METAL_DEVICE_PROFILER is enabled. Tracy profiling active.")
        config.enable_tracy = True

    try:
        results = run_profiling_benchmark(config)
        results.print()
        results.save_to_csv(config.output_csv, config)

        print("\nTo visualize with Tracy, run:")
        print(f"  TT_METAL_DEVICE_PROFILER=1 python -m tracy -r {__file__}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# Ensure the tt-metal directory is in PYTHONPATH and set as working directory
# This handles cases where the script is run from different directories
script_dir = os.path.dirname(os.path.abspath(__file__))
tt_metal_root = os.path.abspath(os.path.join(script_dir, ".."))

# Set working directory to tt-metal root
if os.path.exists(tt_metal_root) and os.path.isdir(tt_metal_root):
    os.chdir(tt_metal_root)

# Add tt-metal root to PYTHONPATH if not already present
if tt_metal_root not in sys.path:
    sys.path.insert(0, tt_metal_root)

import argparse
import time
import csv
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any

import torch
import ttnn

from models.common.helper_funcs import Linear as TtLinear


@dataclass
class BenchmarkConfig:
    in_features: int = 4096
    out_features: int = 4096
    large_batch_size: int = 256
    small_batch_size: int = 32
    minibatches: int = 8
    dtype: str = "bfloat16"  # choices: bfloat16, bfloat8_b
    warmup_iters: int = 2
    measure_iters: int = 5
    seed: int = 1337
    output_csv: Optional[str] = None
    append_csv: bool = False
    only_large: bool = False  # Run only large batch scenario
    only_mini: bool = False  # Run only mini-batch scenario
    enable_weight_sharding: bool = False  # Enable weight sharding to SRAM (L1)


@dataclass
class PhaseTimings:
    """
    Fine-grained timing metrics for each phase of execution.
    """

    kernel_compilation_ms: float = 0.0  # Time to compile kernels (one-time cost, amortized)
    weight_load_host_to_gddr6_ms: float = 0.0  # Time to transfer weights from Host DRAM to Device GDDR6
    forward_compute_ms: float = 0.0  # Total forward pass time (includes compute, weight streaming, communication)
    total_ms: float = 0.0  # Total time including all phases

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for CSV export."""
        return {
            "kernel_compilation_ms": self.kernel_compilation_ms,
            "weight_load_host_to_gddr6_ms": self.weight_load_host_to_gddr6_ms,
            "forward_compute_ms": self.forward_compute_ms,
            "total_ms": self.total_ms,
        }

    def average_with(self, other: "PhaseTimings") -> "PhaseTimings":
        """Create average of two PhaseTimings."""
        return PhaseTimings(
            kernel_compilation_ms=(self.kernel_compilation_ms + other.kernel_compilation_ms) / 2.0,
            weight_load_host_to_gddr6_ms=(self.weight_load_host_to_gddr6_ms + other.weight_load_host_to_gddr6_ms) / 2.0,
            forward_compute_ms=(self.forward_compute_ms + other.forward_compute_ms) / 2.0,
            total_ms=(self.total_ms + other.total_ms) / 2.0,
        )

    @classmethod
    def average_list(cls, timings_list: List["PhaseTimings"]) -> "PhaseTimings":
        """Create average of a list of PhaseTimings."""
        if not timings_list:
            return cls()
        n = len(timings_list)
        return cls(
            kernel_compilation_ms=sum(t.kernel_compilation_ms for t in timings_list) / n,
            weight_load_host_to_gddr6_ms=sum(t.weight_load_host_to_gddr6_ms for t in timings_list) / n,
            forward_compute_ms=sum(t.forward_compute_ms for t in timings_list) / n,
            total_ms=sum(t.total_ms for t in timings_list) / n,
        )


@dataclass
class BenchmarkResult:
    """
    Container for all benchmark results from a single run.
    Holds metrics for both large batch and minibatch scenarios.
    """

    # Large batch metrics
    large_phase_timings: PhaseTimings
    large_total_ms: float
    large_weight_load_ms: float
    large_forward_ms: float

    # Minibatch metrics
    mini_phase_timings: PhaseTimings
    mini_total_ms: float
    mini_weight_load_ms: float
    mini_forward_ms: float

    # Overhead metrics
    overhead_ms: float

    @property
    def overhead_percentage(self) -> float:
        """Calculate overhead as percentage of large forward time."""
        if self.large_forward_ms > 0:
            return (self.overhead_ms / self.large_forward_ms) * 100.0
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            "large_phase_timings": self.large_phase_timings.to_dict(),
            "large_total_ms": self.large_total_ms,
            "large_weight_load_ms": self.large_weight_load_ms,
            "large_forward_ms": self.large_forward_ms,
            "mini_phase_timings": self.mini_phase_timings.to_dict(),
            "mini_total_ms": self.mini_total_ms,
            "mini_weight_load_ms": self.mini_weight_load_ms,
            "mini_forward_ms": self.mini_forward_ms,
            "overhead_ms": self.overhead_ms,
            "overhead_percentage": self.overhead_percentage,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"BenchmarkResult(\n"
            f"  Large batch: total={self.large_total_ms:.3f}ms, "
            f"weight_load={self.large_weight_load_ms:.3f}ms, "
            f"forward={self.large_forward_ms:.3f}ms\n"
            f"  Minibatch: total={self.mini_total_ms:.3f}ms, "
            f"weight_load={self.mini_weight_load_ms:.3f}ms, "
            f"forward={self.mini_forward_ms:.3f}ms\n"
            f"  Overhead: {self.overhead_ms:.3f}ms ({self.overhead_percentage:.2f}%)\n"
            f")"
        )


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Benchmark weight-loading overhead")
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--large-batch-size", type=int, default=256)
    parser.add_argument("--small-batch-size", type=int, default=32)
    parser.add_argument("--minibatches", type=int, default=8, help="Number of minibatches when using small batch")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "bfloat8_b"],
    )
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to CSV file to save results. If not provided, saves to 'benchmark_results.csv' in script directory. Use 'auto' for timestamped filename.",
    )
    parser.add_argument(
        "--append-csv",
        action="store_true",
        help="Append to existing CSV file instead of overwriting",
    )
    parser.add_argument(
        "--only-large",
        action="store_true",
        help="Run only large batch scenario (for device profiling comparison)",
    )
    parser.add_argument(
        "--only-mini",
        action="store_true",
        help="Run only mini-batch scenario (for device profiling comparison)",
    )
    parser.add_argument(
        "--enable-weight-sharding",
        action="store_true",
        help="Enable weight sharding to SRAM (L1) for improved performance",
    )

    args = parser.parse_args()
    return BenchmarkConfig(
        in_features=args.in_features,
        out_features=args.out_features,
        large_batch_size=args.large_batch_size,
        small_batch_size=args.small_batch_size,
        minibatches=args.minibatches,
        dtype=args.dtype,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        seed=args.seed,
        output_csv=args.output_csv,
        append_csv=args.append_csv,
        only_large=args.only_large,
        only_mini=args.only_mini,
        enable_weight_sharding=args.enable_weight_sharding,
    )


def resolve_dtype(dtype_str: str):
    if dtype_str == "bfloat16":
        return ttnn.bfloat16
    if dtype_str == "bfloat8_b":
        return ttnn.bfloat8_b
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def create_weight_memory_config(
    device: ttnn.Device,
    out_features: int,
    in_features: int,
    enable_sharding: bool = False,
) -> ttnn.MemoryConfig:
    """
    Create memory config for weights, optionally using L1 sharding.

    Args:
        device: Device to check grid size
        out_features: Output dimension of weight matrix
        in_features: Input dimension of weight matrix
        enable_sharding: If True, create sharded L1 config; otherwise use DRAM

    Returns:
        Memory config for weight tensor
    """
    if not enable_sharding:
        return ttnn.DRAM_MEMORY_CONFIG

    # Use WIDTH_SHARDED strategy for weights
    # Shard across available cores for parallel access
    compute_grid_size = device.compute_with_storage_grid_size()

    # Use a reasonable number of cores (up to 8x8 = 64 cores for weights)
    # Weights are [1, 1, out_features, in_features] in 4D
    max_cores = min(compute_grid_size.x * compute_grid_size.y, 64)

    # For WIDTH_SHARDED: shard the width (in_features) dimension
    # Each core gets a portion of the input features
    num_cores_x = min(compute_grid_size.x, 8)
    num_cores_y = min(max_cores // num_cores_x, compute_grid_size.y, 8)
    total_cores = num_cores_x * num_cores_y

    # Calculate shard shape: [out_features, in_features // total_cores]
    # The shard shape must tile-align (multiple of 32)
    shard_width = (in_features + total_cores - 1) // total_cores
    shard_width = ((shard_width + 31) // 32) * 32  # Round up to tile boundary
    shard_height = out_features  # Full output dimension per shard

    # Create core grid
    core_grid = ttnn.CoreGrid(y=num_cores_y, x=num_cores_x)

    # Create sharded memory config
    try:
        memory_config = ttnn.create_sharded_memory_config(
            shape=(shard_height, shard_width),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return memory_config
    except Exception as e:
        # If sharding fails, fall back to DRAM
        print(f"[WARN] Failed to create sharded memory config for weights: {e}")
        print(f"[WARN] Falling back to DRAM_MEMORY_CONFIG")
        return ttnn.DRAM_MEMORY_CONFIG


def make_tt_weight_and_bias(
    device: ttnn.Device,
    in_features: int,
    out_features: int,
    dtype,
    seed: int,
    timings: Optional[PhaseTimings] = None,
    enable_weight_sharding: bool = False,
) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor], Optional[PhaseTimings]]:
    """
    Create weight and bias tensors on device, measuring weight loading time.
    Returns: (weight_tensor, bias_tensor, updated_timings)
    """
    g = torch.Generator().manual_seed(seed)
    # weight shape expected by helper is [1, 1, out, in]; we'll build torch as (out, in)
    w_pt = torch.randn((out_features, in_features), dtype=torch.float32, generator=g)
    # Bias optional; include to stress more movement
    b_pt = torch.randn((out_features,), dtype=torch.float32, generator=g)

    # Create memory config for weights (DRAM or sharded L1)
    weight_memory_config = create_weight_memory_config(device, out_features, in_features, enable_weight_sharding)

    # Measure weight loading from Host DRAM to Device memory (DRAM or L1)
    t_weight_load_start = time.perf_counter()
    w_tt = ttnn.from_torch(
        w_pt,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weight_memory_config,
    )
    device_synchronize(device)
    t_weight_load_end = time.perf_counter()
    weight_load_time = (t_weight_load_end - t_weight_load_start) * 1000.0

    # Ensure expected 4D shape [1, 1, out, in] for helper's assertion (metadata-only)
    w_tt = ttnn.reshape(w_tt, ttnn.Shape([1, 1, out_features, in_features]))

    # Measure bias loading (bias typically stays in DRAM as it's small)
    t_bias_load_start = time.perf_counter()
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    device_synchronize(device)
    t_bias_load_end = time.perf_counter()
    bias_load_time = (t_bias_load_end - t_bias_load_start) * 1000.0

    if timings is not None:
        timings.weight_load_host_to_gddr6_ms += weight_load_time + bias_load_time

    # Ensure weight layout matches helper expectation (helper will transpose internally)
    # Convert weight to expected padded 4D automatically handled by from_torch
    return w_tt, b_tt, timings


def make_tt_input(
    device: ttnn.Device, batch_size: int, in_features: int, dtype, seed: int, timings: Optional[PhaseTimings] = None
) -> ttnn.Tensor:
    """
    Create input tensor on device, optionally measuring loading time.
    """
    g = torch.Generator().manual_seed(seed + 1)
    # Use 4D [1,1,B,in] so padded shapes match weight helper assertions
    x_pt = torch.randn((1, 1, batch_size, in_features), dtype=torch.float32, generator=g)

    if timings is not None:
        t_load_start = time.perf_counter()

    x_tt = ttnn.from_torch(
        x_pt,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if timings is not None:
        device_synchronize(device)
        t_load_end = time.perf_counter()
        timings.weight_load_host_to_gddr6_ms += (t_load_end - t_load_start) * 1000.0

    return x_tt


def make_staged_input(device: ttnn.Device, total_batch_size: int, in_features: int, dtype, seed: int) -> ttnn.Tensor:
    """Create a single large input tensor on device DRAM to be sliced per minibatch."""
    return make_tt_input(device, total_batch_size, in_features, dtype, seed)


def device_synchronize(device: ttnn.Device):
    # Best-effort sync to ensure timing accuracy; core ops enqueue work
    try:
        ttnn.synchronize_device(device)
    except Exception:
        # Fallback: some builds expose this under ttnn.device
        try:
            device.synchronize()
        except Exception:
            pass


def time_forward(
    device: ttnn.Device,
    in_features: int,
    out_features: int,
    batch_size: int,
    dtype,
    seed: int,
    warmup_iters: int,
    measure_iters: int,
    pre_measured_compile_ms: Optional[float] = None,
    enable_weight_sharding: bool = False,
) -> Tuple[PhaseTimings, float, float, float]:
    """
    Measure forward pass with fine-grained timing.
    Returns: (avg_phase_timings, avg_total_ms, avg_weight_load_ms, avg_forward_ms)
    """
    # Prepare weights once: they persist in device memory across all passes
    timings = PhaseTimings()
    t_prep_start = time.perf_counter()

    # Measure weight loading and sharding
    w_tt, b_tt, timings = make_tt_weight_and_bias(
        device, in_features, out_features, dtype, seed, timings, enable_weight_sharding
    )

    # Measure kernel compilation time (first execution compiles kernels)
    t_compile_start = time.perf_counter()
    linear = TtLinear(in_features, out_features, w_tt, b_tt, output_mem_config=ttnn.DRAM_MEMORY_CONFIG)
    device_synchronize(device)
    t_compile_end = time.perf_counter()
    timings.kernel_compilation_ms = (t_compile_end - t_compile_start) * 1000.0

    # Prepare one input tensor to reuse (same input each iteration as requested)
    x_tt = make_tt_input(device, batch_size, in_features, dtype, seed)

    # Warmup and ensure kernel is resident
    if pre_measured_compile_ms is not None:
        timings.kernel_compilation_ms = pre_measured_compile_ms
        print(f"[PROFILE] Warmup forward 0 (kernel already compiled)")
        _ = linear(x_tt)
        device_synchronize(device)
    else:
        t_warmup_compile_start = time.perf_counter()
        print(f"[PROFILE] Warmup forward 0 (compiling kernels)")
        _ = linear(x_tt)
        device_synchronize(device)
        t_warmup_compile_end = time.perf_counter()
        if timings.kernel_compilation_ms < 1.0:
            timings.kernel_compilation_ms = (t_warmup_compile_end - t_warmup_compile_start) * 1000.0

    # Continue with remaining warmup iterations (compilation already done)
    for i in range(1, warmup_iters):
        print(f"[PROFILE] Warmup forward {i}")
        _ = linear(x_tt)
        device_synchronize(device)

    # Measure with fine-grained timing
    phase_timings_list = []
    total_times = []
    weight_load_times = []
    fwd_times = []

    # One-time costs that should be included in total time and weight load time
    initial_kernel_compilation_ms = timings.kernel_compilation_ms
    initial_weight_load_ms = timings.weight_load_host_to_gddr6_ms

    for iter_idx in range(measure_iters):
        iter_timings = PhaseTimings()

        # For the first iteration, include initial weight loading time
        # For subsequent iterations, weight is already loaded (no additional cost)
        if iter_idx == 0:
            weight_ms = initial_weight_load_ms  # Include initial weight loading for first pass
        else:
            weight_ms = 0.0  # No host->device reload for subsequent passes

        # Forward pass - measure compute time
        t_f0 = time.perf_counter()
        print(f"[PROFILE] Measurement forward {iter_idx}")
        _ = linear(x_tt)
        device_synchronize(device)
        t_f1 = time.perf_counter()
        fwd_ms = (t_f1 - t_f0) * 1000.0

        iter_timings.forward_compute_ms = fwd_ms

        # Set phase timings for this iteration
        # For first iteration, include one-time costs; for others, only forward pass
        if iter_idx == 0:
            iter_timings.kernel_compilation_ms = initial_kernel_compilation_ms
            iter_timings.weight_load_host_to_gddr6_ms = initial_weight_load_ms
        else:
            iter_timings.kernel_compilation_ms = 0.0  # Already compiled
            iter_timings.weight_load_host_to_gddr6_ms = 0.0  # Already loaded

        # Total time excludes kernel compilation
        if iter_idx == 0:
            total_ms = initial_weight_load_ms + fwd_ms
        else:
            total_ms = fwd_ms

        iter_timings.total_ms = total_ms

        total_times.append(total_ms)
        weight_load_times.append(weight_ms)
        fwd_times.append(fwd_ms)
        phase_timings_list.append(iter_timings)

    # Average phase timings
    avg_phase_timings = PhaseTimings.average_list(phase_timings_list)

    avg_total = sum(total_times) / len(total_times)
    avg_w = sum(weight_load_times) / len(weight_load_times) if weight_load_times else 0.0
    avg_f = sum(fwd_times) / len(fwd_times) if fwd_times else 0.0
    return avg_phase_timings, avg_total, avg_w, avg_f


def time_minibatch_sequence(
    device: ttnn.Device,
    in_features: int,
    out_features: int,
    small_batch_size: int,
    num_minibatches: int,
    dtype,
    seed: int,
    staged_large_input: ttnn.Tensor,
    warmup_iters: int,
    measure_iters: int,
    pre_measured_compile_ms: Optional[float] = None,
    enable_weight_sharding: bool = False,
) -> Tuple[PhaseTimings, float, float, float]:
    """
    Measure minibatch sequence with fine-grained timing.
    Returns: (avg_phase_timings, avg_total_ms, avg_weight_load_ms, avg_forward_ms)
    """
    # Prepare weights once in device memory
    timings = PhaseTimings()
    w_tt, b_tt, timings = make_tt_weight_and_bias(
        device, in_features, out_features, dtype, seed, timings, enable_weight_sharding
    )

    # Measure kernel compilation
    t_compile_start = time.perf_counter()
    linear = TtLinear(in_features, out_features, w_tt, b_tt, output_mem_config=ttnn.DRAM_MEMORY_CONFIG)
    device_synchronize(device)
    t_compile_end = time.perf_counter()
    timings.kernel_compilation_ms = (t_compile_end - t_compile_start) * 1000.0

    # One-time costs that should be included in total time and weight load time
    initial_kernel_compilation_ms = timings.kernel_compilation_ms
    initial_weight_load_ms = timings.weight_load_host_to_gddr6_ms

    def run_one_sequence(sequence_idx: int, include_one_time_costs: bool) -> Tuple[PhaseTimings, float, float, float]:
        seq_timings = PhaseTimings()
        seq_fwd_ms = 0.0

        # For first sequence, include initial weight loading time
        if include_one_time_costs:
            seq_weight_load_ms = initial_weight_load_ms
        else:
            seq_weight_load_ms = 0.0

        # Slice the staged large input per minibatch without host copies
        for i in range(num_minibatches):
            start_b = i * small_batch_size
            end_b = start_b + small_batch_size
            slice_start = (0, 0, start_b, 0)
            slice_end = (1, 1, end_b, in_features)
            slice_step = (1, 1, 1, 1)

            x_slice = ttnn.slice(staged_large_input, slice_start, slice_end, slice_step)
            # Ensure expected 4D shape
            x_slice = ttnn.reshape(x_slice, ttnn.Shape([1, 1, small_batch_size, in_features]))

            # Forward pass - includes weight streaming, compute, and communication
            # (all happen during kernel execution, measured as total forward time)
            t_f0 = time.perf_counter()
            _ = linear(x_slice)
            device_synchronize(device)
            fwd_ms = (time.perf_counter() - t_f0) * 1000.0
            seq_fwd_ms += fwd_ms

        # Set phase timings
        if include_one_time_costs:
            seq_timings.kernel_compilation_ms = initial_kernel_compilation_ms
            seq_timings.weight_load_host_to_gddr6_ms = initial_weight_load_ms
            # Total time excludes kernel compilation and sharding
            seq_total_ms = initial_weight_load_ms + seq_fwd_ms
        else:
            seq_timings.kernel_compilation_ms = 0.0  # Already compiled
            seq_timings.weight_load_host_to_gddr6_ms = 0.0  # Already loaded
            # Total time excludes sharding
            seq_total_ms = seq_fwd_ms

        seq_timings.forward_compute_ms = seq_fwd_ms
        seq_timings.total_ms = seq_total_ms

        return seq_timings, seq_total_ms, seq_weight_load_ms, seq_fwd_ms

    # Warmup: run full sequence warmup_iters times (not recorded)
    # Note: Kernel compilation happens lazily on first execution, not when creating TtLinear
    # We need to measure compilation time on the first forward pass call, not the entire sequence
    # For accurate measurement, we'll do a single forward pass first to trigger compilation
    # Then run the full warmup sequences
    #
    # IMPORTANT NOTE ABOUT KERNEL COMPILATION TIME DIFFERENCE:
    # The kernel compilation time may differ between large batch and minibatch scenarios because:
    # 1. They compile DIFFERENT kernels: batch_size=256 vs batch_size=32 require different program configurations
    # 2. The kernel hash includes input tensor shapes, so different batch sizes = different kernels
    # 3. Device state may differ: if large batch runs first, device thermal/state may affect minibatch compilation
    # 4. Different program configurations (core grid layout, tile distribution) may have different compilation complexity
    #
    # To ensure fair comparison, we create the input tensor BEFORE measuring compilation time,
    # matching the large batch scenario exactly. The actual minibatches will reuse this compiled kernel.
    #
    # IMPORTANT: Create the input tensor BEFORE starting the timer to match the large batch scenario
    # where x_tt is created before measuring compilation time.
    dummy_input_tt = make_tt_input(
        device, small_batch_size, in_features, dtype, seed + 1000, timings=None
    )  # Don't track timing here

    # Ensure any tensor creation overhead is complete before measuring compilation
    device_synchronize(device)

    # Use pre-measured compile time if provided; otherwise measure once
    if pre_measured_compile_ms is not None:
        timings.kernel_compilation_ms = pre_measured_compile_ms
        _ = linear(dummy_input_tt)
        device_synchronize(device)
    else:
        t_warmup_compile_start = time.perf_counter()
        _ = linear(dummy_input_tt)
        device_synchronize(device)
        t_warmup_compile_end = time.perf_counter()
        if timings.kernel_compilation_ms < 1.0:
            timings.kernel_compilation_ms = (t_warmup_compile_end - t_warmup_compile_start) * 1000.0

    # Update initial_kernel_compilation_ms with the measured value
    initial_kernel_compilation_ms = timings.kernel_compilation_ms

    # Now run warmup sequences (compilation already done, so these are fast)
    # All minibatches in run_one_sequence will reuse the kernel compiled above
    for i in range(warmup_iters):
        _, _, _, _ = run_one_sequence(i, include_one_time_costs=False)

    # Measure: run full sequence measure_iters times and average
    phase_timings_list = []
    totals = []
    weight_loads = []
    fwds = []
    for seq_idx in range(measure_iters):
        include_one_time = seq_idx == 0  # Include one-time costs only for first sequence
        seq_timings, total_ms, weight_load_ms, fwd_ms = run_one_sequence(seq_idx, include_one_time)
        totals.append(total_ms)
        weight_loads.append(weight_load_ms)
        fwds.append(fwd_ms)
        phase_timings_list.append(seq_timings)

    # Average phase timings
    avg_phase_timings = PhaseTimings.average_list(phase_timings_list)

    avg_total = sum(totals) / len(totals)
    avg_weight_load = sum(weight_loads) / len(weight_loads)
    avg_fwd = sum(fwds) / len(fwds)
    return avg_phase_timings, avg_total, avg_weight_load, avg_fwd


def _save_results_to_csv(
    cfg: BenchmarkConfig,
    large_phase_timings: PhaseTimings,
    large_total_ms: float,
    large_w_ms: float,
    large_f_ms: float,
    mini_phase_timings: PhaseTimings,
    mini_total_ms: float,
    mini_w_ms: float,
    mini_f_ms: float,
    overhead_ms: float,
):
    """Save benchmark results to CSV file."""
    # Determine output file path
    if cfg.output_csv is None:
        csv_path = os.path.join(script_dir, "benchmark_results.csv")
    elif cfg.output_csv == "auto":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(script_dir, f"benchmark_results_{timestamp}.csv")
    else:
        csv_path = cfg.output_csv

    # Prepare data row with all phase timings
    timestamp = datetime.now().isoformat()
    large_timings_dict = large_phase_timings.to_dict()
    mini_timings_dict = mini_phase_timings.to_dict()

    row = {
        "timestamp": timestamp,
        "in_features": cfg.in_features,
        "out_features": cfg.out_features,
        "large_batch_size": cfg.large_batch_size,
        "small_batch_size": cfg.small_batch_size,
        "minibatches": cfg.minibatches,
        "dtype": cfg.dtype,
        "warmup_iters": cfg.warmup_iters,
        "measure_iters": cfg.measure_iters,
        "seed": cfg.seed,
        "enable_weight_sharding": cfg.enable_weight_sharding,
        # Timing metrics
        "large_total_ms": f"{large_total_ms:.6f}",
        "mini_total_ms": f"{mini_total_ms:.6f}",
        "overhead_ms": f"{overhead_ms:.6f}",
        "overhead_percentage": f"{(overhead_ms / large_f_ms * 100):.2f}" if large_f_ms > 0 else "0.00",
        # Fine-grained phase timings for large batch
        "large_weight_load_host_to_gddr6_ms": f"{large_timings_dict['weight_load_host_to_gddr6_ms']:.6f}",
        "large_forward_compute_ms": f"{large_timings_dict['forward_compute_ms']:.6f}",
        # Fine-grained phase timings for minibatch
        "mini_weight_load_host_to_gddr6_ms": f"{mini_timings_dict['weight_load_host_to_gddr6_ms']:.6f}",
        "mini_forward_compute_ms": f"{mini_timings_dict['forward_compute_ms']:.6f}",
    }

    # Field names (column headers)
    fieldnames = list(row.keys())

    # Check if file exists and has content
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    # Determine write mode and whether to write header
    if cfg.append_csv and file_exists:
        # Append mode: don't write header
        write_mode = "a"
        write_header = False
    else:
        # Write mode: write header (either new file or overwriting)
        write_mode = "w"
        write_header = True

    # Write CSV
    with open(csv_path, write_mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nResults saved to: {csv_path}")
    return csv_path


def run_benchmark(cfg: BenchmarkConfig) -> BenchmarkResult:
    torch.manual_seed(cfg.seed)
    device = ttnn.open_device(device_id=0)

    dtype = resolve_dtype(cfg.dtype)

    # Validate minibatch math
    if cfg.minibatches <= 0:
        raise ValueError("minibatches must be > 0")
    small_total = cfg.small_batch_size * cfg.minibatches
    if small_total != cfg.large_batch_size:
        print(
            f"[warn] small_batch_size * minibatches ({small_total}) != large_batch_size ({cfg.large_batch_size}); comparing unequal total tokens"
        )

    # Check for mutually exclusive options
    if cfg.only_large and cfg.only_mini:
        raise ValueError("Cannot use --only-large and --only-mini together")

    # Pre-stage the full logical-batch input in DRAM once
    staged_large_input = make_staged_input(device, cfg.large_batch_size, cfg.in_features, dtype, cfg.seed + 42)

    # Initialize result variables
    large_phase_timings = None
    large_total_ms = 0.0
    large_w_ms = 0.0
    large_f_ms = 0.0
    mini_phase_timings = None
    mini_total_ms = 0.0
    mini_w_ms = 0.0
    mini_f_ms = 0.0

    # Run large batch scenario
    if not cfg.only_mini:
        # Measure large batch kernel compilation
        pre_w_large, pre_b_large, _ = make_tt_weight_and_bias(
            device,
            cfg.in_features,
            cfg.out_features,
            dtype,
            cfg.seed + 9999,
            timings=None,
            enable_weight_sharding=cfg.enable_weight_sharding,
        )
        pre_linear_large = TtLinear(
            cfg.in_features, cfg.out_features, pre_w_large, pre_b_large, output_mem_config=ttnn.DRAM_MEMORY_CONFIG
        )
        pre_x_large = make_tt_input(device, cfg.large_batch_size, cfg.in_features, dtype, cfg.seed + 9998, timings=None)
        t_compile_large_start = time.perf_counter()
        _ = pre_linear_large(pre_x_large)
        device_synchronize(device)
        t_compile_large_end = time.perf_counter()
        large_batch_compile_ms = (t_compile_large_end - t_compile_large_start) * 1000.0

        # Clean up and let device settle
        pre_x_large.deallocate()
        pre_w_large.deallocate()
        if pre_b_large is not None:
            pre_b_large.deallocate()
        device_synchronize(device)
        time.sleep(0.05)

        # Run large batch benchmark
        large_phase_timings, large_total_ms, large_w_ms, large_f_ms = time_forward(
            device,
            cfg.in_features,
            cfg.out_features,
            cfg.large_batch_size,
            dtype,
            cfg.seed,
            cfg.warmup_iters,
            cfg.measure_iters,
            pre_measured_compile_ms=large_batch_compile_ms,
            enable_weight_sharding=cfg.enable_weight_sharding,
        )

    # Run mini-batch scenario
    if not cfg.only_large:
        # Measure small batch kernel compilation
        pre_w_small, pre_b_small, _ = make_tt_weight_and_bias(
            device,
            cfg.in_features,
            cfg.out_features,
            dtype,
            cfg.seed + 9997,
            timings=None,
            enable_weight_sharding=cfg.enable_weight_sharding,
        )
        pre_linear_small = TtLinear(
            cfg.in_features, cfg.out_features, pre_w_small, pre_b_small, output_mem_config=ttnn.DRAM_MEMORY_CONFIG
        )
        pre_x_small = make_tt_input(device, cfg.small_batch_size, cfg.in_features, dtype, cfg.seed + 9996, timings=None)
        t_compile_small_start = time.perf_counter()
        _ = pre_linear_small(pre_x_small)
        device_synchronize(device)
        t_compile_small_end = time.perf_counter()
        small_batch_compile_ms = (t_compile_small_end - t_compile_small_start) * 1000.0

        # Clean up
        pre_x_small.deallocate()
        pre_w_small.deallocate()
        if pre_b_small is not None:
            pre_b_small.deallocate()
        device_synchronize(device)
        time.sleep(0.1)

        # Run mini-batch benchmark
        mini_phase_timings, mini_total_ms, mini_w_ms, mini_f_ms = time_minibatch_sequence(
            device,
            cfg.in_features,
            cfg.out_features,
            cfg.small_batch_size,
            cfg.minibatches,
            dtype,
            cfg.seed,
            staged_large_input,
            cfg.warmup_iters,
            cfg.measure_iters,
            pre_measured_compile_ms=small_batch_compile_ms,
            enable_weight_sharding=cfg.enable_weight_sharding,
        )

    # Overhead assessment
    overhead_ms = mini_f_ms - large_f_ms

    ttnn.close_device(device)

    # Create default PhaseTimings if not set
    if large_phase_timings is None:
        large_phase_timings = PhaseTimings()
    if mini_phase_timings is None:
        mini_phase_timings = PhaseTimings()

    return BenchmarkResult(
        large_phase_timings=large_phase_timings,
        large_total_ms=large_total_ms,
        large_weight_load_ms=large_w_ms,
        large_forward_ms=large_f_ms,
        mini_phase_timings=mini_phase_timings,
        mini_total_ms=mini_total_ms,
        mini_weight_load_ms=mini_w_ms,
        mini_forward_ms=mini_f_ms,
        overhead_ms=overhead_ms,
    )


def report_results(cfg: BenchmarkConfig, results: BenchmarkResult):
    """Prints benchmark results to console and saves to CSV."""
    print("\n===== TTNN Linear Benchmark (single device) =====")
    print("Comparison         : Large batch (single load) vs Minibatches (same weights; DRAM→L1 re-stream)")
    print(f"DType              : {cfg.dtype}")
    print(f"Dims               : in={cfg.in_features}, out={cfg.out_features}")
    print(f"Large batch        : B={cfg.large_batch_size}")
    print(
        f"Small batch        : b={cfg.small_batch_size} x {cfg.minibatches} (total={cfg.small_batch_size * cfg.minibatches})"
    )
    print(f"Warmup/Measure     : {cfg.warmup_iters}/{cfg.measure_iters} iters per case")
    print(f"Weight Sharding    : {'Enabled (L1/SRAM)' if cfg.enable_weight_sharding else 'Disabled (DRAM)'}")
    print("\n-- Single pass: LARGE batch --")
    print(f"Total avg (ms)     : {results.large_total_ms:8.3f}")
    print(f"  Weight load (from host DRAM to device GDDR6 DRAM) (ms) : {results.large_weight_load_ms:8.3f}")
    print(f"  Forward (ms)     : {results.large_forward_ms:8.3f}")
    print("\n  Fine-grained phase breakdown (average per iteration):")
    print(
        f"    Weight load (Host DRAM -> GDDR6) (ms)      : {results.large_phase_timings.weight_load_host_to_gddr6_ms:8.3f}"
    )
    print(f"    Forward compute (ms)                     : {results.large_phase_timings.forward_compute_ms:8.3f}")
    print("\n-- Repeated passes: SMALL minibatch (same weights on device) --")
    print(f"Sequence total (ms) for {cfg.minibatches} passes : {results.mini_total_ms:8.3f}")
    print(
        f"  Weight load sum (from host DRAM to device GDDR6 DRAM) (ms)                           : {results.mini_weight_load_ms:8.3f}"
    )
    print(f"  Forward sum (ms)                               : {results.mini_forward_ms:8.3f}")
    print("\n  Fine-grained phase breakdown (average per sequence):")
    print(
        f"    Weight load (Host DRAM -> GDDR6) (ms)      : {results.mini_phase_timings.weight_load_host_to_gddr6_ms:8.3f}"
    )
    print(f"    Forward compute (ms)                     : {results.mini_phase_timings.forward_compute_ms:8.3f}")
    print("\n-- Overhead vs single large batch --")
    print(f"Estimated overhead (ms): {results.overhead_ms:8.3f}")
    overhead_pct = (results.overhead_ms / results.large_forward_ms * 100) if results.large_forward_ms > 0 else 0.0
    print(f"Overhead percentage     : {overhead_pct:6.2f}%")

    # Save results to CSV
    _save_results_to_csv(
        cfg,
        results.large_phase_timings,
        results.large_total_ms,
        results.large_weight_load_ms,
        results.large_forward_ms,
        results.mini_phase_timings,
        results.mini_total_ms,
        results.mini_weight_load_ms,
        results.mini_forward_ms,
        results.overhead_ms,
    )


def main():
    cfg = parse_args()
    results = run_benchmark(cfg)
    report_results(cfg, results)


if __name__ == "__main__":
    main()

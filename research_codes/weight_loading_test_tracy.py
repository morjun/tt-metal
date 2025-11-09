#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Forward Pass Breakdown Using Tracy Profiler

This script instruments weight_loading_test.py with Tracy zones to decompose
the forward pass into detailed phases:
1. Input Sharding (GDDR6 DRAM → L1 SRAM sharded)
2. Weight Streaming (GDDR6 DRAM → L1 SRAM per matmul)
3. Compute (actual matmul execution)
4. Output Gathering (L1 sharded → GDDR6 DRAM via NoC)

Usage:
    python3 -m tracy -r research_codes/weight_loading_test_tracy.py

The Tracy profiler will generate detailed timing reports showing the breakdown
of each forward pass component.
"""

import os
import sys

# Ensure the tt-metal directory is in PYTHONPATH and set as working directory
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
from typing import Optional, Tuple

import torch
import ttnn

from models.common.helper_funcs import Linear as TtLinear


def parse_args():
    parser = argparse.ArgumentParser(description="Tracy-instrumented forward pass breakdown")
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--large-batch-size", type=int, default=256)
    parser.add_argument("--small-batch-size", type=int, default=32)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "bfloat8_b"])
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def resolve_dtype(dtype_str: str):
    if dtype_str == "bfloat16":
        return ttnn.bfloat16
    if dtype_str == "bfloat8_b":
        return ttnn.bfloat8_b
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def device_synchronize(device: ttnn.Device):
    try:
        ttnn.synchronize_device(device)
    except Exception:
        try:
            device.synchronize()
        except Exception:
            pass


def make_tt_weight_and_bias(device, in_features, out_features, dtype, seed):
    """Create weight and bias tensors on device."""
    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "make_tt_weight_and_bias", 85, 0)

    g = torch.Generator().manual_seed(seed)
    w_pt = torch.randn((out_features, in_features), dtype=torch.float32, generator=g)
    b_pt = torch.randn((out_features,), dtype=torch.float32, generator=g)

    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "weight_host_to_gddr6", 91, 0xFF0000)
    w_tt = ttnn.from_torch(
        w_pt,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    device_synchronize(device)
    ttnn.stop_tracy_zone("weight_host_to_gddr6", 0xFF0000)

    w_tt = ttnn.reshape(w_tt, ttnn.Shape([1, 1, out_features, in_features]))

    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "bias_host_to_gddr6", 105, 0xFF0000)
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    device_synchronize(device)
    ttnn.stop_tracy_zone("bias_host_to_gddr6", 0xFF0000)

    ttnn.stop_tracy_zone("make_tt_weight_and_bias", 0)
    return w_tt, b_tt


def make_tt_input(device, batch_size, in_features, dtype, seed):
    """Create input tensor on device."""
    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "make_tt_input", 121, 0)

    g = torch.Generator().manual_seed(seed + 1)
    x_pt = torch.randn((1, 1, batch_size, in_features), dtype=torch.float32, generator=g)

    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "input_host_to_gddr6", 126, 0x00FF00)
    x_tt = ttnn.from_torch(
        x_pt,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    device_synchronize(device)
    ttnn.stop_tracy_zone("input_host_to_gddr6", 0x00FF00)

    ttnn.stop_tracy_zone("make_tt_input", 0)
    return x_tt


def run_large_batch_forward(device, in_features, out_features, batch_size, dtype, seed, warmup_iters, measure_iters):
    """Run large batch forward pass with Tracy instrumentation."""
    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "run_large_batch_forward", 143, 0)

    # Create weights once (in GDDR6 DRAM)
    w_tt, b_tt = make_tt_weight_and_bias(device, in_features, out_features, dtype, seed)

    # Create Linear layer (kernel compilation happens on first forward)
    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "create_linear_layer", 149, 0)
    linear = TtLinear(in_features, out_features, w_tt, b_tt, output_mem_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.stop_tracy_zone("create_linear_layer", 0)

    # Create input once (in GDDR6 DRAM)
    x_tt = make_tt_input(device, batch_size, in_features, dtype, seed)

    # Warmup
    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "warmup_large", 158, 0)
    for i in range(warmup_iters):
        ttnn.start_tracy_zone("weight_loading_test_tracy.py", f"warmup_iter_{i}", 160, 0)
        _ = linear(x_tt)
        device_synchronize(device)
        ttnn.stop_tracy_zone(f"warmup_iter_{i}", 0)
    ttnn.stop_tracy_zone("warmup_large", 0)

    # Measure - This is where we want detailed breakdown
    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "measure_large", 167, 0)
    for iter_idx in range(measure_iters):
        ttnn.start_tracy_zone("weight_loading_test_tracy.py", f"TT_DNN_forward_large_{iter_idx}", 169, 0x0000FF)

        # The linear() call internally does:
        # 1. Input sharding (GDDR6 DRAM → L1 SRAM sharded) - handled by ttnn internally
        # 2. Weight streaming (GDDR6 DRAM → L1 SRAM) - handled by matmul kernel
        # 3. Compute (matmul execution) - handled by matmul kernel
        # 4. Output gathering (L1 sharded → GDDR6 DRAM) - handled by ttnn internally
        #
        # Tracy will automatically instrument ttnn operations called within linear(),
        # showing the breakdown of these phases in the Tracy report.
        _ = linear(x_tt)
        device_synchronize(device)

        ttnn.stop_tracy_zone(f"TT_DNN_forward_large_{iter_idx}", 0x0000FF)
    ttnn.stop_tracy_zone("measure_large", 0)

    ttnn.stop_tracy_zone("run_large_batch_forward", 0)


def run_minibatch_forward(
    device,
    in_features,
    out_features,
    small_batch_size,
    num_minibatches,
    dtype,
    seed,
    staged_large_input,
    warmup_iters,
    measure_iters,
):
    """Run minibatch forward passes with Tracy instrumentation."""
    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "run_minibatch_forward", 194, 0)

    # Create weights once (in GDDR6 DRAM)
    w_tt, b_tt = make_tt_weight_and_bias(device, in_features, out_features, dtype, seed)

    # Create Linear layer
    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "create_linear_layer_mini", 200, 0)
    linear = TtLinear(in_features, out_features, w_tt, b_tt, output_mem_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.stop_tracy_zone("create_linear_layer_mini", 0)

    def run_one_sequence(sequence_idx: int):
        ttnn.start_tracy_zone("weight_loading_test_tracy.py", f"minibatch_sequence_{sequence_idx}", 206, 0)

        for i in range(num_minibatches):
            ttnn.start_tracy_zone("weight_loading_test_tracy.py", f"minibatch_{sequence_idx}_{i}", 209, 0)

            # Slice input (no host copy, stays on device)
            ttnn.start_tracy_zone("weight_loading_test_tracy.py", f"slice_input_{i}", 212, 0xFFFF00)
            start_b = i * small_batch_size
            end_b = start_b + small_batch_size
            slice_start = (0, 0, start_b, 0)
            slice_end = (1, 1, end_b, in_features)
            slice_step = (1, 1, 1, 1)
            x_slice = ttnn.slice(staged_large_input, slice_start, slice_end, slice_step)
            x_slice = ttnn.reshape(x_slice, ttnn.Shape([1, 1, small_batch_size, in_features]))
            device_synchronize(device)
            ttnn.stop_tracy_zone(f"slice_input_{i}", 0xFFFF00)

            # Forward pass with detailed breakdown marker
            ttnn.start_tracy_zone(
                "weight_loading_test_tracy.py", f"TT_DNN_forward_mini_{sequence_idx}_{i}", 224, 0x00FFFF
            )

            # The linear() call internally does:
            # 1. Input sharding (GDDR6 DRAM → L1 SRAM sharded)
            # 2. Weight streaming (GDDR6 DRAM → L1 SRAM) - THIS IS THE KEY OVERHEAD
            # 3. Compute (matmul execution)
            # 4. Output gathering (L1 sharded → GDDR6 DRAM)
            #
            # For minibatch, weight streaming happens EVERY iteration (not cached in L1),
            # causing the significant overhead we're measuring.
            _ = linear(x_slice)
            device_synchronize(device)

            ttnn.stop_tracy_zone(f"TT_DNN_forward_mini_{sequence_idx}_{i}", 0x00FFFF)
            ttnn.stop_tracy_zone(f"minibatch_{sequence_idx}_{i}", 0)

        ttnn.stop_tracy_zone(f"minibatch_sequence_{sequence_idx}", 0)

    # Warmup
    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "warmup_mini", 244, 0)
    for i in range(warmup_iters):
        run_one_sequence(i)
    ttnn.stop_tracy_zone("warmup_mini", 0)

    # Measure
    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "measure_mini", 250, 0)
    for seq_idx in range(measure_iters):
        run_one_sequence(seq_idx + warmup_iters)
    ttnn.stop_tracy_zone("measure_mini", 0)

    ttnn.stop_tracy_zone("run_minibatch_forward", 0)


def main():
    args = parse_args()

    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "main", 261, 0)

    torch.manual_seed(args.seed)
    device = ttnn.open_device(device_id=0)
    dtype = resolve_dtype(args.dtype)

    # Validate minibatch math
    small_total = args.small_batch_size * args.minibatches
    if small_total != args.large_batch_size:
        print(f"[warn] small_batch_size * minibatches ({small_total}) != large_batch_size ({args.large_batch_size})")

    print("\n===== Tracy-Instrumented Forward Pass Breakdown =====")
    print(f"DType              : {args.dtype}")
    print(f"Dims               : in={args.in_features}, out={args.out_features}")
    print(f"Large batch        : B={args.large_batch_size}")
    print(f"Small batch        : b={args.small_batch_size} x {args.minibatches}")
    print(f"Warmup/Measure     : {args.warmup_iters}/{args.measure_iters} iters")
    print("\nRun this with: python3 -m tracy -r research_codes/weight_loading_test_tracy.py")
    print("Tracy will generate detailed reports showing forward pass breakdown.\n")

    # Pre-stage the full logical-batch input in DRAM once
    ttnn.start_tracy_zone("weight_loading_test_tracy.py", "create_staged_input", 287, 0)
    g = torch.Generator().manual_seed(args.seed + 42)
    x_pt_large = torch.randn((1, 1, args.large_batch_size, args.in_features), dtype=torch.float32, generator=g)
    staged_large_input = ttnn.from_torch(
        x_pt_large,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    device_synchronize(device)
    ttnn.stop_tracy_zone("create_staged_input", 0)

    print("Running large batch forward...")
    run_large_batch_forward(
        device,
        args.in_features,
        args.out_features,
        args.large_batch_size,
        dtype,
        args.seed,
        args.warmup_iters,
        args.measure_iters,
    )

    print("Running minibatch forward...")
    run_minibatch_forward(
        device,
        args.in_features,
        args.out_features,
        args.small_batch_size,
        args.minibatches,
        dtype,
        args.seed,
        staged_large_input,
        args.warmup_iters,
        args.measure_iters,
    )

    ttnn.close_device(device)
    ttnn.stop_tracy_zone("main", 0)

    print("\nDone! Check Tracy report for detailed forward pass breakdown.")
    print("Look for zones prefixed with 'TT_DNN_forward_' to see the forward pass details.")


if __name__ == "__main__":
    main()

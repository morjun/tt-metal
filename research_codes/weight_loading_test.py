#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple

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
    )


def resolve_dtype(dtype_str: str):
    if dtype_str == "bfloat16":
        return ttnn.bfloat16
    if dtype_str == "bfloat8_b":
        return ttnn.bfloat8_b
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def make_tt_weight_and_bias(
    device: ttnn.Device,
    in_features: int,
    out_features: int,
    dtype,
    seed: int,
) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    # weight shape expected by helper is [1, 1, out, in]; we'll build torch as (out, in)
    w_pt = torch.randn((out_features, in_features), dtype=torch.float32, generator=g)
    # Bias optional; include to stress more movement
    b_pt = torch.randn((out_features,), dtype=torch.float32, generator=g)

    w_tt = ttnn.from_torch(
        w_pt,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Ensure expected 4D shape [1, 1, out, in] for helper's assertion
    w_tt = ttnn.reshape(w_tt, ttnn.Shape([1, 1, out_features, in_features]))
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Ensure weight layout matches helper expectation (helper will transpose internally)
    # Convert weight to expected padded 4D automatically handled by from_torch
    return w_tt, b_tt


def make_tt_input(device: ttnn.Device, batch_size: int, in_features: int, dtype, seed: int) -> ttnn.Tensor:
    g = torch.Generator().manual_seed(seed + 1)
    # Use 4D [1,1,B,in] so padded shapes match weight helper assertions
    x_pt = torch.randn((1, 1, batch_size, in_features), dtype=torch.float32, generator=g)
    x_tt = ttnn.from_torch(
        x_pt,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
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
) -> Tuple[float, float, float]:
    # Returns: (avg_total_ms, avg_weight_load_ms, avg_forward_ms)
    total_times = []
    weight_load_times = []
    fwd_times = []

    # Prepare weights once: they persist in device DRAM across all passes
    t0 = time.perf_counter()
    w_tt, b_tt = make_tt_weight_and_bias(device, in_features, out_features, dtype, seed)
    linear = TtLinear(in_features, out_features, w_tt, b_tt, output_mem_config=ttnn.DRAM_MEMORY_CONFIG)
    device_synchronize(device)
    wload_prep_ms = (time.perf_counter() - t0) * 1000.0

    # Prepare one input tensor to reuse (same input each iteration as requested)
    x_tt = make_tt_input(device, batch_size, in_features, dtype, seed)

    # Warmup
    for _ in range(warmup_iters):
        _ = linear(x_tt)
        device_synchronize(device)

    # Measure
    for _ in range(measure_iters):
        t_start = time.perf_counter()
        weight_ms = 0.0  # no host->device reload per pass; measuring only compute path

        # Forward time
        t_f0 = time.perf_counter()
        _ = linear(x_tt)
        device_synchronize(device)
        t_f1 = time.perf_counter()
        fwd_ms = (t_f1 - t_f0) * 1000.0

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0 + wload_prep_ms

        total_times.append(total_ms)
        weight_load_times.append(weight_ms)
        fwd_times.append(fwd_ms)

    avg_total = sum(total_times) / len(total_times)
    avg_w = sum(weight_load_times) / len(weight_load_times) if weight_load_times else 0.0
    avg_f = sum(fwd_times) / len(fwd_times) if fwd_times else 0.0
    return avg_total, avg_w, avg_f


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
) -> Tuple[float, float, float]:
    # Returns total_ms, total_weight_ms, total_forward_ms for a contiguous sequence of minibatch passes
    # Prepare weights once in device DRAM
    w_tt, b_tt = make_tt_weight_and_bias(device, in_features, out_features, dtype, seed)
    linear = TtLinear(in_features, out_features, w_tt, b_tt, output_mem_config=ttnn.DRAM_MEMORY_CONFIG)

    def run_one_sequence() -> Tuple[float, float]:
        seq_fwd_ms = 0.0
        t_seq0 = time.perf_counter()
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

            t_f0 = time.perf_counter()
            _ = linear(x_slice)
            device_synchronize(device)
            seq_fwd_ms += (time.perf_counter() - t_f0) * 1000.0

        seq_total_ms = (time.perf_counter() - t_seq0) * 1000.0
        return seq_total_ms, seq_fwd_ms

    # Warmup: run full sequence warmup_iters times (not recorded)
    for _ in range(warmup_iters):
        _ = run_one_sequence()

    # Measure: run full sequence measure_iters times and average
    totals = []
    fwds = []
    for _ in range(measure_iters):
        total_ms, fwd_ms = run_one_sequence()
        totals.append(total_ms)
        fwds.append(fwd_ms)

    avg_total = sum(totals) / len(totals)
    avg_fwd = sum(fwds) / len(fwds)
    total_weight_ms = 0.0  # no host reloads
    return avg_total, total_weight_ms, avg_fwd


def run_benchmark(cfg: BenchmarkConfig):
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

    # Pre-stage the full logical-batch input in DRAM once
    staged_large_input = make_staged_input(device, cfg.large_batch_size, cfg.in_features, dtype, cfg.seed + 42)

    # Goal comparison: large batch (single weight load) vs minibatches (same weights, re-streamed DRAM->L1)
    # Large batch single pass, weights created once, use full staged input
    large_total_ms, large_w_ms, large_f_ms = time_forward(
        device,
        cfg.in_features,
        cfg.out_features,
        cfg.large_batch_size,
        dtype,
        cfg.seed,
        cfg.warmup_iters,
        cfg.measure_iters,
    )

    # Mini-batch repeated passes: slice from staged_large_input to avoid host transfers
    mini_total_ms, mini_w_ms, mini_f_ms = time_minibatch_sequence(
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
    )

    # Report
    print("\n===== TTNN Linear Benchmark (single device) =====")
    print("Comparison         : Large batch (single load) vs Minibatches (same weights; DRAM→L1 re-stream)")
    print(f"DType              : {cfg.dtype}")
    print(f"Dims               : in={cfg.in_features}, out={cfg.out_features}")
    print(f"Large batch        : B={cfg.large_batch_size}")
    print(f"Small batch        : b={cfg.small_batch_size} x {cfg.minibatches} (total={cfg.small_batch_size * cfg.minibatches})")
    print(f"Warmup/Measure     : {cfg.warmup_iters}/{cfg.measure_iters} iters per case")

    print("\n-- Single pass: LARGE batch --")
    print(f"Total avg (ms)     : {large_total_ms:8.3f}")
    print(f"  Weight load (ms) : {large_w_ms:8.3f}")
    print(f"  Forward (ms)     : {large_f_ms:8.3f}")

    print("\n-- Repeated passes: SMALL minibatch (same weights on device) --")
    print(f"Sequence total (ms) for {cfg.minibatches} passes : {mini_total_ms:8.3f}")
    print(f"  Weight load sum (ms)                           : {mini_w_ms:8.3f}")
    print(f"  Forward sum (ms)                               : {mini_f_ms:8.3f}")

    # Overhead assessment
    overhead_ms = mini_total_ms - large_total_ms
    print("\n-- Overhead vs single large batch --")
    print(f"Estimated overhead (ms): {overhead_ms:8.3f}")
    # mini_w_ms remains 0.0 here since we are not performing host reloads; overhead is due to DRAM→L1 re-stream + launches

    ttnn.close_device(device)


def main():
    cfg = parse_args()
    run_benchmark(cfg)


if __name__ == "__main__":
    main()


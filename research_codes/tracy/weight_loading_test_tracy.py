#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TRACY 프로파일링이 통합된 weight_loading_test.py 개선판

각 forward pass에 TRACY zone 마커를 추가하여 device profiler CSV에서
정확한 forward pass 경계를 식별할 수 있도록 함.

Usage:
    # Tracy profiling을 활성화하여 실행 (추천)
    python3 -m tracy -v -r -p -o ./tracy_output research_codes/weight_loading_test_tracy.py

    # 또는 일반 실행 (TRACY 마커는 추가되지만 캡처되지 않음)
    python3 research_codes/weight_loading_test_tracy.py

    # Custom batch sizes
    python3 -m tracy -v -r -p -o ./tracy_output research_codes/weight_loading_test_tracy.py --large-batch-size 512 --small-batch-size 64

Output:
    - Tracy 캡처 파일: ./tracy_output/profiler_logs/*.tracy
    - CSV 리포트: ./tracy_output/profiler_logs/*.csv
    - 벤치마크 결과: ./research_codes/benchmark_results_tracy.csv
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
import csv
import inspect
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any

import torch
import ttnn

# TRACY imports
TRACY_AVAILABLE = False
try:
    # signpost must be imported from tracy module
    from tracy import signpost

    # ttnn profiler functions
    from ttnn.profiler import start_tracy_zone, stop_tracy_zone
    import ttnn.profiler

    tracy_message = ttnn.profiler.tracy_message  # Keep reference for compatibility
    TRACY_AVAILABLE = True
    print("[INFO] TRACY profiling enabled")
except ImportError as e:
    print(f"[WARNING] TRACY not available: {e}. Running without TRACY markers.")

    # Dummy functions
    def signpost(header, message=None):
        pass

    def start_tracy_zone(*args, **kwargs):
        pass

    def stop_tracy_zone(*args, **kwargs):
        return True

    def tracy_message(msg, color=0xF0F8FF):
        pass


from models.common.helper_funcs import Linear as TtLinear


@dataclass
class BenchmarkConfig:
    in_features: int = 4096
    out_features: int = 4096
    large_batch_size: int = 256
    small_batch_size: int = 32
    minibatches: int = 8
    dtype: str = "bfloat16"
    warmup_iters: int = 2
    measure_iters: int = 5
    seed: int = 1337
    output_csv: Optional[str] = None
    append_csv: bool = False
    only_large: bool = False
    only_mini: bool = False


@dataclass
class PhaseTimings:
    kernel_compilation_ms: float = 0.0
    weight_load_host_to_gddr6_ms: float = 0.0
    forward_compute_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "kernel_compilation_ms": self.kernel_compilation_ms,
            "weight_load_host_to_gddr6_ms": self.weight_load_host_to_gddr6_ms,
            "forward_compute_ms": self.forward_compute_ms,
            "total_ms": self.total_ms,
        }

    @classmethod
    def average_list(cls, timings_list: List["PhaseTimings"]) -> "PhaseTimings":
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
    large_phase_timings: PhaseTimings
    large_total_ms: float
    large_weight_load_ms: float
    large_forward_ms: float
    mini_phase_timings: PhaseTimings
    mini_total_ms: float
    mini_weight_load_ms: float
    mini_forward_ms: float
    overhead_ms: float

    @property
    def overhead_percentage(self) -> float:
        if self.large_forward_ms > 0:
            return (self.overhead_ms / self.large_forward_ms) * 100.0
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
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


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Benchmark weight-loading overhead with TRACY profiling")
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--large-batch-size", type=int, default=256)
    parser.add_argument("--small-batch-size", type=int, default=32)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "bfloat8_b"])
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--append-csv", action="store_true")
    parser.add_argument("--only-large", action="store_true")
    parser.add_argument("--only-mini", action="store_true")

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
    timings: Optional[PhaseTimings] = None,
) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor], Optional[PhaseTimings]]:
    """Create weight and bias tensors on device, measuring weight loading time."""
    g = torch.Generator().manual_seed(seed)
    w_pt = torch.randn((out_features, in_features), dtype=torch.float32, generator=g)
    b_pt = torch.randn((out_features,), dtype=torch.float32, generator=g)

    t_weight_load_start = time.perf_counter()
    w_tt = ttnn.from_torch(
        w_pt, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    device_synchronize(device)
    t_weight_load_end = time.perf_counter()
    weight_load_time = (t_weight_load_end - t_weight_load_start) * 1000.0

    w_tt = ttnn.reshape(w_tt, ttnn.Shape([1, 1, out_features, in_features]))

    t_bias_load_start = time.perf_counter()
    b_tt = ttnn.from_torch(
        b_pt, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    device_synchronize(device)
    t_bias_load_end = time.perf_counter()
    bias_load_time = (t_bias_load_end - t_bias_load_start) * 1000.0

    if timings is not None:
        timings.weight_load_host_to_gddr6_ms += weight_load_time + bias_load_time

    return w_tt, b_tt, timings


def make_tt_input(
    device: ttnn.Device, batch_size: int, in_features: int, dtype, seed: int, timings: Optional[PhaseTimings] = None
) -> ttnn.Tensor:
    """Create input tensor on device."""
    g = torch.Generator().manual_seed(seed + 1)
    x_pt = torch.randn((1, 1, batch_size, in_features), dtype=torch.float32, generator=g)

    if timings is not None:
        t_load_start = time.perf_counter()

    x_tt = ttnn.from_torch(
        x_pt, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
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
    """Synchronize device."""
    try:
        ttnn.synchronize_device(device)
    except Exception:
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
) -> Tuple[PhaseTimings, float, float, float]:
    """Measure forward pass with TRACY markers."""
    timings = PhaseTimings()
    t_prep_start = time.perf_counter()

    w_tt, b_tt, timings = make_tt_weight_and_bias(device, in_features, out_features, dtype, seed, timings)

    t_compile_start = time.perf_counter()
    linear = TtLinear(in_features, out_features, w_tt, b_tt, output_mem_config=ttnn.DRAM_MEMORY_CONFIG)
    device_synchronize(device)
    t_compile_end = time.perf_counter()
    timings.kernel_compilation_ms = (t_compile_end - t_compile_start) * 1000.0

    x_tt = make_tt_input(device, batch_size, in_features, dtype, seed)

    # Warmup
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

    # Remaining warmup iterations (no TRACY markers)
    for i in range(1, warmup_iters):
        print(f"[PROFILE] Warmup forward {i}")
        _ = linear(x_tt)
        device_synchronize(device)

    # Measurement with TRACY markers
    phase_timings_list = []
    total_times = []
    weight_load_times = []
    fwd_times = []

    initial_kernel_compilation_ms = timings.kernel_compilation_ms
    initial_weight_load_ms = timings.weight_load_host_to_gddr6_ms

    # Get current file info for TRACY zones
    current_file = __file__
    current_line = inspect.currentframe().f_lineno if hasattr(inspect, "currentframe") else 0

    for iter_idx in range(measure_iters):
        iter_timings = PhaseTimings()

        if iter_idx == 0:
            weight_ms = initial_weight_load_ms
        else:
            weight_ms = 0.0

        # TRACY zone 시작 - measurement iteration에서만
        zone_name = f"LargeBatch_Forward_{iter_idx}"
        if TRACY_AVAILABLE:
            # Get actual line number for this iteration
            actual_line = inspect.currentframe().f_lineno if hasattr(inspect, "currentframe") else 330
            start_tracy_zone(source=current_file, functName=zone_name, lineNum=actual_line, color=0xFF0000)  # Red
            try:
                signpost(header=f"LargeBatch_Forward_{iter_idx}_Start", message=f"Large batch forward pass {iter_idx}")
            except:
                pass  # signpost may not be available

        # Forward pass
        t_f0 = time.perf_counter()
        print(f"[PROFILE] Measurement forward {iter_idx}")
        _ = linear(x_tt)
        device_synchronize(device)
        t_f1 = time.perf_counter()
        fwd_ms = (t_f1 - t_f0) * 1000.0

        # TRACY zone 종료
        if TRACY_AVAILABLE:
            stop_tracy_zone(name=zone_name, color=0xFF0000)
            try:
                signpost(
                    header=f"LargeBatch_Forward_{iter_idx}_End",
                    message=f"Completed large batch forward pass {iter_idx}",
                )
            except:
                pass  # signpost may not be available

        iter_timings.forward_compute_ms = fwd_ms

        if iter_idx == 0:
            iter_timings.kernel_compilation_ms = initial_kernel_compilation_ms
            iter_timings.weight_load_host_to_gddr6_ms = initial_weight_load_ms
        else:
            iter_timings.kernel_compilation_ms = 0.0
            iter_timings.weight_load_host_to_gddr6_ms = 0.0

        if iter_idx == 0:
            total_ms = initial_weight_load_ms + fwd_ms
        else:
            total_ms = fwd_ms

        iter_timings.total_ms = total_ms

        total_times.append(total_ms)
        weight_load_times.append(weight_ms)
        fwd_times.append(fwd_ms)
        phase_timings_list.append(iter_timings)

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
) -> Tuple[PhaseTimings, float, float, float]:
    """Measure minibatch sequence with TRACY markers for each forward pass."""
    timings = PhaseTimings()
    w_tt, b_tt, timings = make_tt_weight_and_bias(device, in_features, out_features, dtype, seed, timings)

    t_compile_start = time.perf_counter()
    linear = TtLinear(in_features, out_features, w_tt, b_tt, output_mem_config=ttnn.DRAM_MEMORY_CONFIG)
    device_synchronize(device)
    t_compile_end = time.perf_counter()
    timings.kernel_compilation_ms = (t_compile_end - t_compile_start) * 1000.0

    initial_kernel_compilation_ms = timings.kernel_compilation_ms
    initial_weight_load_ms = timings.weight_load_host_to_gddr6_ms

    def run_one_sequence(
        sequence_idx: int, include_one_time_costs: bool, is_measurement: bool = False
    ) -> Tuple[PhaseTimings, float, float, float]:
        """Run one sequence of minibatches with TRACY markers (only for measurement iterations)."""
        seq_timings = PhaseTimings()
        seq_fwd_ms = 0.0

        if include_one_time_costs:
            seq_weight_load_ms = initial_weight_load_ms
        else:
            seq_weight_load_ms = 0.0

        # Get current file info for TRACY zones
        current_file = __file__
        current_line = inspect.currentframe().f_lineno if hasattr(inspect, "currentframe") else 0

        # Sequence 시작 마커 (measurement only)
        if is_measurement and TRACY_AVAILABLE:
            try:
                signpost(
                    header=f"MiniBatch_Sequence_{sequence_idx}_Start",
                    message=f"Starting minibatch sequence {sequence_idx}",
                )
            except:
                pass  # signpost may not be available

        # 각 minibatch에 대해 TRACY zone 추가
        for i in range(num_minibatches):
            start_b = i * small_batch_size
            end_b = start_b + small_batch_size
            slice_start = (0, 0, start_b, 0)
            slice_end = (1, 1, end_b, in_features)
            slice_step = (1, 1, 1, 1)

            x_slice = ttnn.slice(staged_large_input, slice_start, slice_end, slice_step)
            x_slice = ttnn.reshape(x_slice, ttnn.Shape([1, 1, small_batch_size, in_features]))

            # TRACY zone 시작 - measurement iteration에서만 찍기
            zone_name = f"MiniBatch_Forward_{sequence_idx}_{i}"
            if is_measurement and TRACY_AVAILABLE:
                # Get actual line number for this iteration
                actual_line = inspect.currentframe().f_lineno if hasattr(inspect, "currentframe") else 438
                start_tracy_zone(source=current_file, functName=zone_name, lineNum=actual_line, color=0x00FF00)  # Green
                try:
                    signpost(
                        header=f"MiniBatch_Forward_{sequence_idx}_{i}_Start",
                        message=f"Sequence {sequence_idx}, Forward pass {i}",
                    )
                except:
                    pass  # signpost may not be available

            # Forward pass
            t_f0 = time.perf_counter()
            _ = linear(x_slice)
            device_synchronize(device)
            fwd_ms = (time.perf_counter() - t_f0) * 1000.0
            seq_fwd_ms += fwd_ms

            # TRACY zone 종료 (measurement only)
            if is_measurement and TRACY_AVAILABLE:
                stop_tracy_zone(name=zone_name, color=0x00FF00)
                try:
                    signpost(
                        header=f"MiniBatch_Forward_{sequence_idx}_{i}_End",
                        message=f"Completed sequence {sequence_idx}, forward pass {i}",
                    )
                except:
                    pass  # signpost may not be available

        # Sequence 종료 마커 (measurement only)
        if is_measurement and TRACY_AVAILABLE:
            try:
                signpost(
                    header=f"MiniBatch_Sequence_{sequence_idx}_End",
                    message=f"Completed minibatch sequence {sequence_idx}",
                )
            except:
                pass  # signpost may not be available

        if include_one_time_costs:
            seq_timings.kernel_compilation_ms = initial_kernel_compilation_ms
            seq_timings.weight_load_host_to_gddr6_ms = initial_weight_load_ms
            seq_total_ms = initial_weight_load_ms + seq_fwd_ms
        else:
            seq_timings.kernel_compilation_ms = 0.0
            seq_timings.weight_load_host_to_gddr6_ms = 0.0
            seq_total_ms = seq_fwd_ms

        seq_timings.forward_compute_ms = seq_fwd_ms
        seq_timings.total_ms = seq_total_ms

        return seq_timings, seq_total_ms, seq_weight_load_ms, seq_fwd_ms

    # Warmup: compile kernel first
    dummy_input_tt = make_tt_input(device, small_batch_size, in_features, dtype, seed + 1000, timings=None)
    device_synchronize(device)

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

    initial_kernel_compilation_ms = timings.kernel_compilation_ms

    # Warmup sequences (no TRACY markers)
    for i in range(warmup_iters):
        _, _, _, _ = run_one_sequence(i, include_one_time_costs=False, is_measurement=False)

    # Measurement sequences (with TRACY markers)
    phase_timings_list = []
    totals = []
    weight_loads = []
    fwds = []
    for seq_idx in range(measure_iters):
        include_one_time = seq_idx == 0
        seq_timings, total_ms, weight_load_ms, fwd_ms = run_one_sequence(seq_idx, include_one_time, is_measurement=True)
        totals.append(total_ms)
        weight_loads.append(weight_load_ms)
        fwds.append(fwd_ms)
        phase_timings_list.append(seq_timings)

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
    if cfg.output_csv is None:
        csv_path = os.path.join(script_dir, "benchmark_results_tracy.csv")
    elif cfg.output_csv == "auto":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(script_dir, f"benchmark_results_tracy_{timestamp}.csv")
    else:
        csv_path = cfg.output_csv

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
        "large_total_ms": f"{large_total_ms:.6f}",
        "mini_total_ms": f"{mini_total_ms:.6f}",
        "overhead_ms": f"{overhead_ms:.6f}",
        "overhead_percentage": f"{(overhead_ms / large_f_ms * 100):.2f}" if large_f_ms > 0 else "0.00",
        "large_weight_load_host_to_gddr6_ms": f"{large_timings_dict['weight_load_host_to_gddr6_ms']:.6f}",
        "large_forward_compute_ms": f"{large_timings_dict['forward_compute_ms']:.6f}",
        "mini_weight_load_host_to_gddr6_ms": f"{mini_timings_dict['weight_load_host_to_gddr6_ms']:.6f}",
        "mini_forward_compute_ms": f"{mini_timings_dict['forward_compute_ms']:.6f}",
    }

    fieldnames = list(row.keys())
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    if cfg.append_csv and file_exists:
        write_mode = "a"
        write_header = False
    else:
        write_mode = "w"
        write_header = True

    with open(csv_path, write_mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nResults saved to: {csv_path}")
    return csv_path


def run_benchmark(cfg: BenchmarkConfig) -> BenchmarkResult:
    """Run benchmark with TRACY profiling."""
    if TRACY_AVAILABLE:
        print("[TRACY] Profiling enabled - zones and signposts will be recorded")
    else:
        print("[WARNING] TRACY not available - running without profiling markers")

    torch.manual_seed(cfg.seed)
    device = ttnn.open_device(device_id=0)

    dtype = resolve_dtype(cfg.dtype)

    if cfg.minibatches <= 0:
        raise ValueError("minibatches must be > 0")
    small_total = cfg.small_batch_size * cfg.minibatches
    if small_total != cfg.large_batch_size:
        print(f"[warn] small_batch_size * minibatches ({small_total}) != large_batch_size ({cfg.large_batch_size})")

    if cfg.only_large and cfg.only_mini:
        raise ValueError("Cannot use --only-large and --only-mini together")

    staged_large_input = make_staged_input(device, cfg.large_batch_size, cfg.in_features, dtype, cfg.seed + 42)

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
        pre_w_large, pre_b_large, _ = make_tt_weight_and_bias(
            device, cfg.in_features, cfg.out_features, dtype, cfg.seed + 9999, timings=None
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

        pre_x_large.deallocate()
        pre_w_large.deallocate()
        if pre_b_large is not None:
            pre_b_large.deallocate()
        device_synchronize(device)
        time.sleep(0.05)

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
        )

    # Run mini-batch scenario
    if not cfg.only_large:
        pre_w_small, pre_b_small, _ = make_tt_weight_and_bias(
            device, cfg.in_features, cfg.out_features, dtype, cfg.seed + 9997, timings=None
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

        pre_x_small.deallocate()
        pre_w_small.deallocate()
        if pre_b_small is not None:
            pre_b_small.deallocate()
        device_synchronize(device)
        time.sleep(0.1)

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
        )

    overhead_ms = mini_f_ms - large_f_ms

    ttnn.close_device(device)

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
    """Print benchmark results and save to CSV."""
    print("\n===== TTNN Linear Benchmark with TRACY Profiling =====")
    print("Comparison: Large batch (single load) vs Minibatches (same weights; DRAM→L1 re-stream)")
    print(f"DType: {cfg.dtype}")
    print(f"Dims: in={cfg.in_features}, out={cfg.out_features}")
    print(f"Large batch: B={cfg.large_batch_size}")
    print(f"Small batch: b={cfg.small_batch_size} x {cfg.minibatches} (total={cfg.small_batch_size * cfg.minibatches})")
    print(f"Warmup/Measure: {cfg.warmup_iters}/{cfg.measure_iters} iters per case")
    print("\n-- Single pass: LARGE batch --")
    print(f"Total avg (ms): {results.large_total_ms:8.3f}")
    print(f"  Weight load (ms): {results.large_weight_load_ms:8.3f}")
    print(f"  Forward (ms): {results.large_forward_ms:8.3f}")
    print("\n-- Repeated passes: SMALL minibatch --")
    print(f"Sequence total (ms) for {cfg.minibatches} passes: {results.mini_total_ms:8.3f}")
    print(f"  Weight load sum (ms): {results.mini_weight_load_ms:8.3f}")
    print(f"  Forward sum (ms): {results.mini_forward_ms:8.3f}")
    print("\n-- Overhead vs single large batch --")
    print(f"Estimated overhead (ms): {results.overhead_ms:8.3f}")
    overhead_pct = (results.overhead_ms / results.large_forward_ms * 100) if results.large_forward_ms > 0 else 0.0
    print(f"Overhead percentage: {overhead_pct:6.2f}%")

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

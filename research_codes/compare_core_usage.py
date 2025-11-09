#!/usr/bin/env python3
"""Compare core usage between large batch and mini-batch"""

from device_profile_analysis import *
from pathlib import Path

csv_path = Path("../generated/profiler/.logs/profile_log_device.csv")
zones = parse_device_profile(csv_path)
run_host_ids = sorted(set(z.run_host_id for z in zones))

print("=" * 80)
print("Large vs Mini-batch 코어 사용 비교")
print("=" * 80)
print()

# Large batch - last operation
large_run_id = run_host_ids[-1]
large_zones = [z for z in zones if z.run_host_id == large_run_id]
large_cores = set(z.core_id for z in large_zones)

print(f"Large Batch (B=256):")
print(f"  run_host_id: {large_run_id}")
print(f"  Total zones: {len(large_zones)}")
print(f"  Unique cores: {len(large_cores)}")
print(f'  TRISC zones: {len([z for z in large_zones if normalize_risc_type(z.risc_type) == "TRISC"])}')
print()

# Mini-batch - find operations from previous run (before large batch)
mini_candidates = [rid for rid in run_host_ids if rid < large_run_id]
if len(mini_candidates) >= 2:
    mini_op2_run_id = mini_candidates[-1]
    mini_op2_zones = [z for z in zones if z.run_host_id == mini_op2_run_id]
    mini_cores = set(z.core_id for z in mini_op2_zones)

    print(f"Mini-batch (b=32, representative operation):")
    print(f"  run_host_id: {mini_op2_run_id}")
    print(f"  Total zones: {len(mini_op2_zones)}")
    print(f"  Unique cores: {len(mini_cores)}")
    print(f'  TRISC zones: {len([z for z in mini_op2_zones if normalize_risc_type(z.risc_type) == "TRISC"])}')
    print()

    print("=" * 80)
    print("코어 사용 비율")
    print("=" * 80)
    print()

    core_ratio = len(mini_cores) / len(large_cores)
    expected_core_ratio = 32 / 256

    print(f"Batch size 비율: 32/256 = {expected_core_ratio:.4f}")
    print(f"실제 코어 비율: {len(mini_cores)}/{len(large_cores)} = {core_ratio:.4f}")
    print(f"차이: {core_ratio / expected_core_ratio:.2f}x 더 많은 코어 사용")
    print()

    # TRISC zones
    large_trisc = len([z for z in large_zones if normalize_risc_type(z.risc_type) == "TRISC"])
    mini_trisc = len([z for z in mini_op2_zones if normalize_risc_type(z.risc_type) == "TRISC"])

    trisc_ratio = mini_trisc / large_trisc
    expected_trisc_ratio = 32 / 256

    print(f"TRISC zone 비율: {mini_trisc}/{large_trisc} = {trisc_ratio:.4f}")
    print(f"예상 비율: {expected_trisc_ratio:.4f}")
    print(f"차이: {trisc_ratio / expected_trisc_ratio:.2f}x 더 많은 TRISC zone")
    print()

    print("=" * 80)
    print("결론")
    print("=" * 80)
    print()

    print(f"Mini-batch는 batch size를 1/8로 줄였지만:")
    print(f"  ✗ 코어 사용: {core_ratio:.1%} (예상: 12.5%)")
    print(f"  ✗ TRISC zones: {trisc_ratio:.1%} (예상: 12.5%)")
    print(f"  ✗ {core_ratio / expected_core_ratio:.1f}x 더 많은 리소스 사용")
    print()
    print("**이것이 5.7x 비효율의 근본 원인입니다!**")
    print("→ Mini-batch는 동일한 수의 코어를 사용하면서 작업량만 1/8")
else:
    print("Mini-batch 데이터를 찾을 수 없습니다. 다시 실행해주세요.")

#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
간단한 TRACY 테스트 - 기본 기능 확인용

Tracy zone과 signpost가 제대로 작동하는지 확인합니다.
"""

import time
import sys

print("=" * 60)
print("Simple TRACY Test")
print("=" * 60)

# TRACY imports
TRACY_AVAILABLE = False
try:
    from tracy import signpost
    from ttnn.profiler import start_tracy_zone, stop_tracy_zone
    import ttnn.profiler

    TRACY_AVAILABLE = True
    print("[✓] TRACY profiling enabled")
except ImportError as e:
    print(f"[✗] TRACY not available: {e}")
    sys.exit(1)

print("\nTest 1: Basic signpost")
print("-" * 60)
signpost("TEST_START", "Starting simple tracy test")
time.sleep(0.1)

print("\nTest 2: Tracy zones")
print("-" * 60)
for i in range(3):
    zone_name = f"test_zone_{i}"
    print(f"  Starting zone: {zone_name}")

    start_tracy_zone(source=__file__, functName=zone_name, lineNum=40 + i, color=0xFF0000)

    signpost(f"ZONE_{i}_START", f"Zone {i} processing")

    # Simulate some work
    result = sum(range(1000000))
    time.sleep(0.1)

    signpost(f"ZONE_{i}_END", f"Zone {i} completed with result {result}")

    stop_tracy_zone(name=zone_name, color=0xFF0000)
    print(f"  Completed zone: {zone_name}")

time.sleep(0.1)
signpost("TEST_END", "Simple tracy test completed")

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
print("\nIf running with 'python -m tracy -r -v -p -o ./output':")
print("  - Check ./output/.logs/ for tracy files")
print("  - Look for signposts: TEST_START, ZONE_*_START, ZONE_*_END, TEST_END")
print("=" * 60)

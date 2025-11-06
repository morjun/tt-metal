#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Quick test script to verify profiling setup.
Runs a minimal version of the profiling benchmark.
"""

import sys
import os

# Add tt-metal to path
script_dir = os.path.dirname(os.path.abspath(__file__))
tt_metal_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, tt_metal_root)

try:
    import ttnn

    print("✓ TTNN imported successfully")
except ImportError as e:
    print(f"✗ Failed to import TTNN: {e}")
    sys.exit(1)

try:
    # Try to open device
    device = ttnn.open_device(device_id=0)
    print(f"✓ Device opened: {device.id()}")
    print(f"  Compute grid: {device.compute_with_storage_grid_size()}")

    # Close device
    ttnn.close_device(device)
    print("✓ Device closed successfully")

    print("\n" + "=" * 60)
    print("✓ Profiling environment is ready!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the profiling benchmark:")
    print("   python research_codes/profiling_sharding_noc_python.py")
    print("\n2. Or with Tracy profiling:")
    print("   TT_METAL_DEVICE_PROFILER=1 python -m tracy -r research_codes/profiling_sharding_noc_python.py")
    print("\n3. Check the README:")
    print("   cat research_codes/PROFILING_README.md")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

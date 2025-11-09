#!/usr/bin/env python3
"""
Analyze the relationship between device profile wall clock times and
Python-level forward pass timing.
"""

import sys

# Parse device profile output
device_profile_output = """
Using 130 unique Tensix cores (each with 5 RISC-V processors)

Large operations wall clock times (from parse_device_profile.py):
  0.190624 ms, 0.164737 ms, 0.188522 ms, 0.112172 ms, 0.188364 ms,
  0.164657 ms, 0.164590 ms, 0.164618 ms, 0.164527 ms, 0.164623 ms,
  0.188755 ms, 0.112216 ms, 0.112177 ms, 0.112109 ms, 0.112338 ms,
  0.112143 ms, 0.112242 ms, 0.112187 ms, 0.112364 ms, 0.112301 ms, ...
  (52 large operations total)
  Average: 0.124 ms per operation
"""

# Python-level timing from weight_loading_test.py
python_timing = """
Large batch forward: 0.237 ms
Mini-batch 8x forward: 1.435 ms
"""

large_ops = [
    0.190624,
    0.164737,
    0.188522,
    0.112172,
    0.188364,
    0.164657,
    0.164590,
    0.164618,
    0.164527,
    0.164623,
    0.188755,
    0.112216,
]

print("=" * 80)
print("DEVICE PROFILE vs PYTHON TIMING ANALYSIS")
print("=" * 80)

print("\n1. Device Profile (individual operations):")
print(f"   Sample of {len(large_ops)} large operations:")
for i, t in enumerate(large_ops, 1):
    print(f"   Operation {i}: {t:.6f} ms")

avg_device_op = sum(large_ops) / len(large_ops)
print(f"\n   Average per operation: {avg_device_op:.6f} ms")

print("\n2. Python-level timing (ttnn.matmul call):")
print(f"   Forward pass: 0.237 ms")

print("\n3. Analysis:")
print(f"   Ratio: 0.237 / {avg_device_op:.6f} = {0.237 / avg_device_op:.2f}x")

if 0.237 / avg_device_op > 1.5:
    print("\n   *** KEY INSIGHT ***")
    print("   One Python forward pass (ttnn.matmul) = MULTIPLE device operations!")
    print("   The 0.237ms includes:")
    print("   - Input preparation operations")
    print("   - Actual matmul operation(s)")
    print("   - Output gathering operations")
    print("   - Python overhead between operations")

    num_ops = round(0.237 / avg_device_op)
    print(f"\n   Estimated: ~{num_ops} device operations per forward pass")
else:
    print("\n   Device operation time ≈ Python forward time")
    print("   One forward pass = One device operation")

print("\n4. Corrected interpretation:")
print("   Device profile wall clock (0.124ms) = Time for ONE device operation")
print("   Python forward pass (0.237ms) = Time for COMPLETE forward including:")
print("     - Host → Device sync")
print("     - Multiple device operations")
print("     - Device → Host sync")
print("     - Python overhead")

print("\n5. Weight streaming overhead (from Python timing):")
print("   Large batch: 0.237 ms (1 forward)")
print("   Mini-batch:  1.435 ms (8 forwards)")
print("   Overhead:    1.198 ms = 7 extra weight streamings")
print("   Per stream:  0.171 ms wall clock (Python-measured)")
print("\n   This 0.171ms includes:")
print("   - Actual GDDR6 → L1 transfer (measured by BRISC)")
print("   - Device synchronization")
print("   - Python overhead")

print("=" * 80)

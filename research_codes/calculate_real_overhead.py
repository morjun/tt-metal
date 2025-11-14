#!/usr/bin/env python3
"""Calculate real overhead based on wall clock time and Python measurements."""

print("=" * 80)
print("REAL OVERHEAD CALCULATION")
print("=" * 80)
print()

# Device Profiler Wall Clock Times (from analyze_risc_overlap.py)
large_device_wall_clock_ms = 1276.257
mini_device_wall_clock_ms = 2105.549

# Python perf_counter times (from benchmark)
large_python_forward_ms = 0.280  # From previous large-only run
mini_python_forward_ms = 1.722  # From previous mini-only run (8 passes total)

print("1. Device Profiler Wall Clock Times:")
print(f"   Large batch (1 pass):     {large_device_wall_clock_ms:10.3f} ms")
print(f"   Mini batch (8 passes):    {mini_device_wall_clock_ms:10.3f} ms")
print()

print("2. Python perf_counter Times:")
print(f"   Large batch (1 pass):     {large_python_forward_ms:10.3f} ms")
print(f"   Mini batch (8 passes):    {mini_python_forward_ms:10.3f} ms")
print()

print("=" * 80)
print("OVERHEAD ANALYSIS")
print("=" * 80)
print()

print("A. Device Profiler Wall Clock (most accurate for device operations):")
print("-" * 80)
device_overhead_abs = mini_device_wall_clock_ms - large_device_wall_clock_ms
device_overhead_pct = (device_overhead_abs / large_device_wall_clock_ms) * 100
mini_per_pass_device = mini_device_wall_clock_ms / 8
overhead_per_pass_device = mini_per_pass_device - large_device_wall_clock_ms

print(f"   Total overhead (8 mini vs 1 large):  {device_overhead_abs:10.3f} ms ({device_overhead_pct:6.2f}%)")
print(f"   Mini batch per pass:                 {mini_per_pass_device:10.3f} ms")
print(f"   Overhead per mini pass:              {overhead_per_pass_device:10.3f} ms")
print(f"   Efficiency (mini/large per pass):    {mini_per_pass_device/large_device_wall_clock_ms:10.3f}x")
print()

print("B. Python perf_counter (host-side wall clock):")
print("-" * 80)
python_overhead_abs = mini_python_forward_ms - large_python_forward_ms
python_overhead_pct = (python_overhead_abs / large_python_forward_ms) * 100
mini_per_pass_python = mini_python_forward_ms / 8
overhead_per_pass_python = mini_per_pass_python - large_python_forward_ms

print(f"   Total overhead (8 mini vs 1 large):  {python_overhead_abs:10.3f} ms ({python_overhead_pct:6.2f}%)")
print(f"   Mini batch per pass:                 {mini_per_pass_python:10.3f} ms")
print(f"   Overhead per mini pass:              {overhead_per_pass_python:10.3f} ms")
print(f"   Efficiency (mini/large per pass):    {mini_per_pass_python/large_python_forward_ms:10.3f}x")
print()

print("=" * 80)
print("COMPONENT-WISE OVERHEAD (Device Profiler)")
print("=" * 80)
print()

# From analyze_device_profile_simple.py output
large_brisc_ms = 192.504
large_ncrisc_ms = 186.023
large_trisc_ms = 559.582

mini_brisc_ms = 359.900
mini_ncrisc_ms = 355.031
mini_trisc_ms = 1059.634

print("Active time per component:")
print("-" * 80)
print(f"                    Large (1x)    Mini (8x)     Overhead    Ratio")
print(
    f"BRISC:           {large_brisc_ms:10.3f} ms  {mini_brisc_ms:10.3f} ms  {mini_brisc_ms - large_brisc_ms:10.3f} ms  {mini_brisc_ms/large_brisc_ms:.3f}x"
)
print(
    f"NCRISC:          {large_ncrisc_ms:10.3f} ms  {mini_ncrisc_ms:10.3f} ms  {mini_ncrisc_ms - large_ncrisc_ms:10.3f} ms  {mini_ncrisc_ms/large_ncrisc_ms:.3f}x"
)
print(
    f"TRISC:           {large_trisc_ms:10.3f} ms  {mini_trisc_ms:10.3f} ms  {mini_trisc_ms - large_trisc_ms:10.3f} ms  {mini_trisc_ms/large_trisc_ms:.3f}x"
)
print()

print("=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print()
print("1. Device Profiler Wall Clock shows:")
print(f"   - 8 mini-batches take {device_overhead_pct:.1f}% MORE time than 1 large batch")
print(f"   - Each mini-batch pass takes {mini_per_pass_device:.3f} ms")
print(f"   - That's {mini_per_pass_device/large_device_wall_clock_ms:.3f}x the time of large batch per pass")
print()
print("2. Python perf_counter shows:")
print(f"   - 8 mini-batches take {python_overhead_pct:.1f}% MORE time than 1 large batch")
print(f"   - Each mini-batch pass takes {mini_per_pass_python:.3f} ms")
print(f"   - That's {mini_per_pass_python/large_python_forward_ms:.3f}x the time of large batch per pass")
print()
print("3. Why is mini-batch LESS efficient per pass?")
print("   - SliceDeviceOperation overhead (mini-batch splitting)")
print("   - Reduced parallelism opportunities (smaller batch size)")
print("   - More frequent kernel launches and synchronization")
print()
print("4. Component overhead breakdown:")
print(f"   - BRISC increases by {(mini_brisc_ms/large_brisc_ms - 1)*100:.1f}%")
print(f"   - NCRISC increases by {(mini_ncrisc_ms/large_ncrisc_ms - 1)*100:.1f}%")
print(f"   - TRISC increases by {(mini_trisc_ms/large_trisc_ms - 1)*100:.1f}%")
print("   - All components increase proportionally (balanced overhead)")
print()
print("=" * 80)

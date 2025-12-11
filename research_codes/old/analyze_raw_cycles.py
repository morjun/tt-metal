#!/usr/bin/env python3
"""Analyze RAW cycle values from profile_log_device.csv - sum cycles directly."""

import csv
from collections import defaultdict


def analyze_raw_cycles(csv_path, label):
    """Sum up the raw 'time [cycles since reset]' values per RISC type."""

    risc_cycles = defaultdict(int)
    risc_count = defaultdict(int)

    with open(csv_path, "r", encoding="utf-8") as f:
        # Skip first two lines (header info and column names)
        f.readline()  # Skip "ARCH: blackhole..."
        header = f.readline().strip().split(",")

        # Clean up header names
        header = [h.strip() for h in header]

        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            risc_type = parts[3]  # RISC processor type
            try:
                cycles = int(parts[5])  # time[cycles since reset]
            except (ValueError, IndexError):
                continue

            # Sum raw cycle values
            risc_cycles[risc_type] += cycles
            risc_count[risc_type] += 1

    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    print()

    total_cycles = sum(risc_cycles.values())

    print(f"Raw cycle sums (직접 cycles 값을 더한 것):")
    print(f"{'-'*80}")
    print(f"{'RISC Type':<15} {'Total Cycles':<20} {'Row Count':<15} {'Avg/Row':<15} {'%':<10}")
    print(f"{'-'*80}")

    for risc_type in sorted(risc_cycles.keys()):
        cycles = risc_cycles[risc_type]
        count = risc_count[risc_type]
        avg = cycles / count if count > 0 else 0
        pct = cycles / total_cycles * 100 if total_cycles > 0 else 0

        print(f"{risc_type:<15} {cycles:<20,} {count:<15,} {avg:<15,.0f} {pct:<10.2f}")

    print(f"{'-'*80}")
    print(f"{'TOTAL':<15} {total_cycles:<20,}")
    print()

    return risc_cycles, risc_count


print("=" * 80)
print("RAW CYCLE ANALYSIS - Summing 'time [cycles since reset]' directly")
print("=" * 80)
print()
print("⚠️  WARNING: This might NOT be meaningful because:")
print("  - 'cycles since reset' is a TIMESTAMP, not a duration")
print("  - Adding timestamps gives meaningless sum")
print("  - But let's check what we get...")
print()

# Analyze both datasets
large_cycles, large_count = analyze_raw_cycles(
    "tracy_output_large/profile_log_device.csv", "LARGE BATCH (1 pass, batch_size=256)"
)

mini_cycles, mini_count = analyze_raw_cycles(
    "tracy_output_mini/profile_log_device.csv", "MINI BATCH (8 passes, batch_size=32)"
)

# Compare
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")
print()

print(f"{'RISC Type':<15} {'Large Cycles':<20} {'Mini Cycles':<20} {'Ratio':<10}")
print(f"{'-'*80}")

for risc_type in sorted(large_cycles.keys()):
    large_val = large_cycles.get(risc_type, 0)
    mini_val = mini_cycles.get(risc_type, 0)
    ratio = mini_val / large_val if large_val > 0 else 0

    print(f"{risc_type:<15} {large_val:<20,} {mini_val:<20,} {ratio:<10.2f}x")

print()
print(f"Row counts:")
print(f"{'-'*80}")
print(f"{'RISC Type':<15} {'Large Rows':<20} {'Mini Rows':<20} {'Ratio':<10}")
print(f"{'-'*80}")

for risc_type in sorted(large_count.keys()):
    large_val = large_count.get(risc_type, 0)
    mini_val = mini_count.get(risc_type, 0)
    ratio = mini_val / large_val if large_val > 0 else 0

    print(f"{risc_type:<15} {large_val:<20,} {mini_val:<20,} {ratio:<10.2f}x")

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()
print("If cycle sums are MUCH larger for mini-batch:")
print("  → This is expected (more rows × later timestamps)")
print("  → Raw sum is NOT meaningful for comparison")
print()
print("If row counts increased 4-8x:")
print("  → Confirms more zone entries (instrumentation overhead)")
print()
print("The CORRECT metric remains:")
print("  → Zone duration (ZONE_END - ZONE_START)")
print("  → Which we already calculated correctly")
print()

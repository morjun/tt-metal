import csv
import sys
from collections import defaultdict


def analyze_blocks(filename):
    print(f"--- Analyzing {filename} ---")

    # Store blocks per batch
    # key: batch_index
    # value: count of CB-WAIT-FRONT

    # We need to track the "current batch"
    # But filtering by thread makes it easier.

    events = []

    try:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 12:
                    continue
                risc = row[3]
                if "TRISC_0" not in risc:
                    continue

                name = row[10]
                ztype = row[11]
                time = int(row[5])

                if name == "BATCH-ITERATION" and ztype == "ZONE_START":
                    events.append("BATCH_START")
                elif name == "CB-WAIT-FRONT" and ztype == "ZONE_START":
                    events.append("BLOCK_START")

    except FileNotFoundError:
        print("File not found")
        return

    batch_blocks = []
    current_blocks = 0
    in_batch = False

    for e in events:
        if e == "BATCH_START":
            if in_batch:
                # Close previous batch
                batch_blocks.append(current_blocks)
            in_batch = True
            current_blocks = 0
        elif e == "BLOCK_START":
            if in_batch:
                current_blocks += 1

    if in_batch:
        batch_blocks.append(current_blocks)

    print(f"Total Batches Found: {len(batch_blocks)}")
    if not batch_blocks:
        return

    # Statistics
    avg = sum(batch_blocks) / len(batch_blocks)
    min_b = min(batch_blocks)
    max_b = max(batch_blocks)

    print(f"Blocks per Batch: Avg={avg:.2f}, Min={min_b}, Max={max_b}")

    # Print first 10
    print(f"First 10 batches: {batch_blocks[:10]}")


analyze_blocks("research_codes/profile_log_device_detailed_nosharding_w0m1.csv")
analyze_blocks("research_codes/profile_log_device_nosharding_deactivated_w0m1.csv")

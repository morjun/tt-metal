import sys
import re


def parse_hit_ratio(log_file):
    total_l1_hits = 0
    total_dram_reads = 0

    # Regex to match: K(L1:<hits> DRAM:<misses>)
    pattern = re.compile(r"K\(L1:(\d+)\s+DRAM:(\d+)\)")

    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                total_l1_hits += int(match.group(1))
                total_dram_reads += int(match.group(2))

    total_reads = total_l1_hits + total_dram_reads
    if total_reads == 0:
        print("No DPRINT lines found. Did you enable DEBUG_PRINT 1?")
        return

    hit_ratio = (total_l1_hits / total_reads) * 100
    print(f"Total KV Tiles Read: {total_reads}")
    print(f"L1 Hits:           {total_l1_hits} ({hit_ratio:.2f}%)")
    print(f"DRAM Reads:        {total_dram_reads} ({100 - hit_ratio:.2f}%)")


if __name__ == "__main__":
    parse_hit_ratio(sys.argv[1])

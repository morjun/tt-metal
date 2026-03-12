import sys
import re

total_tiles = 0
occurrences = 0

with open(sys.argv[1], "r") as f:
    for line in f:
        m = re.search(r"CHUNK (\d+) \| K\(L1:(\d+) DRAM:(\d+)\) \| V\(L1:(\d+) DRAM:(\d+)\)", line)
        if m:
            occurrences += 1
            k_l1 = int(m.group(2))
            k_dram = int(m.group(3))
            v_l1 = int(m.group(4))
            v_dram = int(m.group(5))
            tiles = k_l1 + k_dram + v_l1 + v_dram
            total_tiles += tiles

print(
    f"Total KV tiles = {total_tiles} (but wait, regex matched K(L1:x DRAM:x)? Note: your parse script only counted K hits)"
)

k_hits_total = 0
k_dram_total = 0
with open(sys.argv[1], "r") as f:
    for line in f:
        m = re.search(r"K\(L1:(\d+) DRAM:(\d+)\)", line)
        if m:
            k_hits_total += int(m.group(1))
            k_dram_total += int(m.group(2))
print(f"Total K tiles read: {k_hits_total + k_dram_total}")
print(f"Total occurrences of DPRINT: {occurrences}")
print(f"Estimated tokens generated: {occurrences / 32.0}")

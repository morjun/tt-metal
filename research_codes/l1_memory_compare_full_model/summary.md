# L1 Memory Usage Comparison

| Metric | Dual-source | DRAM-only | Delta |
| --- | ---: | ---: | ---: |
| Per-bank allocated bytes | 73984 | 4352 | +69632 |
| Per-bank free bytes | 1396096 | 1465728 | -69632 |
| Largest contiguous free bytes per bank | 1396096 | 1465728 | -69632 |
| Largest interleavable free bytes estimate | 181492480 | 190544640 | -9052160 |
| Chip-total allocated bytes | 9617920 | 565760 | +9052160 |
| Chip-total free bytes | 181492480 | 190544640 | -9052160 |
| Per-bank allocated % | 5.03% | 0.30% | 4.74% |
| Per-bank free % | 94.97% | 99.70% | -4.74% |
| Per-bank largest contiguous free % | 94.97% | 99.70% | -4.74% |

## Window Estimate

| Metric | Dual-source | DRAM-only |
| --- | ---: | ---: |
| Bytes per KV token per layer | 2048.0 | 2048.0 |
| Bytes per KV token across all layers | 65536.0 | 65536.0 |
| Max `l1_kv_window_size` from total free bytes | 2752 | 2880 |
| Max `l1_kv_window_size` from largest interleavable free | 2752 | 2880 |
| Max `l1_kv_window_size` from safe interleavable free | 2464 | 2592 |

The `largest interleavable free` estimate is usually the most realistic upper bound.
The `safe interleavable free` number applies the configured safety margin and is the best starting point for experiments.

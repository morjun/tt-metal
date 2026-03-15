# L1 Memory Usage Comparison

| Metric | Dual-source | DRAM-only | Delta |
| --- | ---: | ---: | ---: |
| Per-bank allocated bytes | 6528 | 4352 | +2176 |
| Per-bank free bytes | 1463552 | 1465728 | -2176 |
| Largest contiguous free bytes per bank | 1463552 | 1465728 | -2176 |
| Largest interleavable free bytes estimate | 190261760 | 190544640 | -282880 |
| Chip-total allocated bytes | 848640 | 565760 | +282880 |
| Chip-total free bytes | 190261760 | 190544640 | -282880 |
| Per-bank allocated % | 0.44% | 0.30% | 0.15% |
| Per-bank free % | 99.56% | 99.70% | -0.15% |
| Per-bank largest contiguous free % | 99.56% | 99.70% | -0.15% |

## Window Estimate

| Metric | Dual-source | DRAM-only |
| --- | ---: | ---: |
| Bytes per KV token | 2048.0 | 2048.0 |
| Max `l1_kv_window_size` from total free bytes | 92896 | 93024 |
| Max `l1_kv_window_size` from largest interleavable free | 92896 | 93024 |
| Max `l1_kv_window_size` from safe interleavable free | 83584 | 83712 |

The `largest interleavable free` estimate is usually the most realistic upper bound.
The `safe interleavable free` number applies the configured safety margin and is the best starting point for experiments.

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

These bounds come from allocator-visible `get_memory_view()` state only.
They can significantly overestimate the real decode-time limit because static circular buffers are typically not allocator-managed.

## Allocator Caveat

- `get_memory_view()` reports allocator-managed L1 state.
- Static circular buffers are typically not allocator-managed and are not fully reflected in these free-space numbers.
- Snapshot note: `Static circular buffers are typically not allocator-managed and are not fully reflected in get_memory_view().`

## Real Window Probe

- Max passing `l1_kv_window_size`: `512`
- Min failing `l1_kv_window_size`: `544`
- First observed failure reason: `static_cb_l1_clash`
- Static CB end address on constraining cores: `1249664`
- Inferred runtime SRAM already occupied by static CBs on constraining cores: `1249664` bytes (85.01%)
- Requested L1 buffer start address on constraining cores: `1220352`
- Real top-of-bank headroom on constraining cores: `220416` bytes (14.99%)
- Requested buffer bytes on constraining cores: `249728` bytes (16.99%)
- Observed overlap on constraining cores: `29312` bytes (1.99%)

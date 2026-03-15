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

These bounds come from allocator-visible state (from `dump_device_memory_state()` CSVs) only.
They can significantly overestimate the real decode-time limit because static circular buffers are typically not allocator-managed.

## Allocator Caveat

- L1 state is read from `dump_device_memory_state()` CSV reports (allocator-managed).
- Static circular buffers are typically not allocator-managed and are not fully reflected in these free-space numbers.
- Snapshot note: `Static circular buffers are typically not allocator-managed and are not fully reflected in get_memory_view().`

## Combined Usage (Allocator vs Real Runtime)

Single compact view of allocator-visible state vs inferred runtime bottleneck-bank usage.

| View | Source | Occupied (bytes) | Occupied (%) | Free/Headroom (bytes) | Free/Headroom (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Allocator (Dual-source) | `dump_device_memory_state()` | 73984 | 5.03% | 1396096 | 94.97% |
| Allocator (DRAM-only) | `dump_device_memory_state()` | 4352 | 0.30% | 1465728 | 99.70% |
| **Real runtime (bottleneck cores)** | Pass/fail probe + clash parse | **1249664** | **85.01%** | **220416** | **14.99%** |

**Explanation:** Allocator rows show what `dump_device_memory_state()` reports (allocator-managed blocks only). The real runtime row is inferred from the first failing `l1_kv_window_size` probe: when the L1 KV buffer clashes with static circular buffers, we parse the error to get the static CB end address. That address is the actual SRAM already occupied on the constraining decode cores (8×1 bottleneck range). The headroom is the remainder—the only space available for the L1 KV mirror.

## Real Window Probe

| Metric | Value |
| --- | ---: |
| Max passing `l1_kv_window_size` | `512` |
| Min failing `l1_kv_window_size` | `544` |
| First failure reason | `static_cb_l1_clash` |
| Static CB end address (bottleneck cores) | `1249664` bytes |
| Inferred SRAM occupied by static CBs | `1249664` bytes (85.01%) |
| L1 buffer start address (requested) | `1220352` |
| Real headroom on bottleneck cores | `220416` bytes (14.99%) |
| Requested L1 buffer size | `249728` bytes (16.99%) |
| Overlap (clash region) | `29312` bytes (1.99%) |

**Why the allocator view misleads:** The dump only tracks allocator-managed blocks. Static circular buffers (CBs) used by decode kernels are typically not allocator-managed, so they do not appear in the allocator's free-space numbers. DRAM-only mode shows ~99.7% free from the allocator's perspective, but the real bottleneck cores have ~85% of their L1 already occupied by static CBs, leaving only ~15% headroom for the L1 KV mirror.

**Interpretation:** At window size 544, the L1 KV mirror requires 249,728 bytes per bank on the constraining cores. The allocator places it starting at address 1,220,352 (top-down). Static CBs already occupy 0–1,249,664, so the requested buffer overlaps the static region by 29,312 bytes, causing the clash. The real upper bound is 512 tokens—the largest window that passes the actual workload. The exact per-bank footprint depends on interleaving and which cores hold the L1 KV cache; the probe confirms the limit.

## Compile-time CB size (debug log)

Static CB size per core from debug log (when available) or from probe clash.

| Metric | Value | Source |
| --- | ---: | --- |
| Total static CB size per core (bytes) | 1,249,664 | probe_clash |
| % of L1 per bank | 85.01% | — |

To refresh from debug log: run demo with `TT_LOGGER_LEVEL=Debug`, save log, then:
`python research_codes/parse_sdpa_cb_memory.py --log <log> --comparison <comparison.json> --output-json research_codes/l1_memory_compare_full_model_corrected/bottleneck_core_memory.json`

## Example commands

**Regenerate this summary from existing `comparison.json`** (no re-run of inference):

```bash
python research_codes/profile_l1_memory_usage.py --from-json research_codes/l1_memory_compare_full_model_corrected/comparison.json
```

To reflect **compile-time CB size from debug logs**: run the demo with `TT_LOGGER_LEVEL=Debug`, save log, then:
`python research_codes/parse_sdpa_cb_memory.py --log <log> --comparison research_codes/l1_memory_compare_full_model_corrected/comparison.json --output-json research_codes/l1_memory_compare_full_model_corrected/bottleneck_core_memory.json`, then regenerate the summary again.

**Full run** (creates comparison.json and summary from scratch; runs dual-source, DRAM-only, and real-window probe):

```bash
python research_codes/profile_l1_memory_usage.py \
  --dual-source-cmd 'pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" --max_generated_tokens 2 --stop_at_eos 0 --l1_kv_window_size 128' \
  --dram-only-cmd 'pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" --max_generated_tokens 2 --stop_at_eos 0 --l1_kv_window_size 0' \
  --working-directory . \
  --output-dir research_codes/l1_memory_compare_full_model_corrected \
  --snapshot-label after_inference \
  --real-window-cmd-template 'pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" --max_generated_tokens 2 --stop_at_eos 0 --l1_kv_window_size {window}' \
  --real-window-values 512,544 \
  --num-local-kv-heads 8 \
  --num-layers 32
```

Run from repo root.

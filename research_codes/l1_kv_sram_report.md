# L1 KV SRAM Analysis

- DRAM-only persistent KV mirror in L1: `0.000 MiB`
- Persistent L1 KV mirror: `0.250 MiB`
- Extra persistent SRAM over DRAM-only baseline: `0.250 MiB`
- Estimated transient SDPA CB pressure: `0.273 MiB`
- Observed L1 hit ratio from DPRINT: `100.000%`
- Effective hot-set bytes served from L1: `0.250 MiB`

This report uses the current DPRINT hit ratio as a baseline for future sharded/zero-copy work.
The hit ratio tells you how much of the decode working set is actually hot enough to justify keeping in SRAM.

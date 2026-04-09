# SDPA Decode L1 SRAM Optimization Implementation Plan

This plan outlines the analysis steps needed to recover the precise cumulative SRAM layout across the full Blackhole 8x8 core grid at the SDPA decode stage, so future L1 KV mirror placement can use the true remaining space of every core.

## Recent Findings Summary

*   **Footprint Discrepancy**: Static CBs for SDPA alone only use ~248 KiB (17.3%).
*   **The 85% Bottleneck**: The observed 85% occupancy (`1,249,664`) is the **cumulative** peak reached on row 0 (`[(0,0)-(7,0)]`) due to stacked CBs from previous operations (`nlp_concat_heads`, `nlp_create_qkv_heads`, etc.).
*   **Current Gap**: We still do **not** know the precise cumulative CB occupancy on the other rows (`y=1..7`) at the SDPA stage.
*   **Goal of This Phase**: Build a per-core cumulative CB map for **all 64 cores** at the SDPA stage, then derive the true remaining L1 headroom on every core before making any placement decision for the L1 KV mirror.

## Analysis Deliverables

This phase is analysis-only. We are **not** yet changing KV mirror placement or treating row 0 exclusion as an acceptable solution.

The required outputs are:

1.  A **64-core cumulative CB heatmap** at the SDPA stage.
2.  A **per-program contribution map** showing how much each cached program contributes to each core.
3.  A **per-core remaining-headroom map** that can later drive KV mirror placement.
4.  A verification of whether cached-but-idle programs continue to contribute SRAM pressure until the program cache is explicitly cleared.

For each core `(x, y)`, the quantity we ultimately care about is:

*   `remaining_l1_for_kv(x, y) = l1_top_down_limit(x, y) - cumulative_cb_end(x, y)`

In the simplest case, `l1_top_down_limit` is the per-core L1 size. If other top-down allocations are live at the SDPA stage, they must be included as well.

## Proposed Changes

### 1. Full-Grid Hardware Verification (Profiling)

We need to dump the exact cumulative CB peak address for *every* core in the 8x8 grid at the SDPA stage, not just row 0.

#### [MODIFY] [program.cpp](file:///home/masterjunmo/codes/tt-metal/tt_metal/impl/program/program.cpp)
*   Temporarily instrument `validate_circular_buffer_region` or `allocate_circular_buffers` to `log_info` the `cb_region_end` for every `CoreRange` in `cb_allocators_`.
*   Expand each `CoreRange` into its member cores so the output becomes a true **per-core** map rather than only a per-range summary.
*   Include enough program identity in the logs to distinguish contributions from different cached programs.
*   The instrumentation must produce:
    *   a **per-program core map**
    *   a **cumulative core map** by the time SDPA is launched
    *   a **derived headroom map** for all 64 cores

#### Required Questions This Instrumentation Must Answer
*   Are rows `1..7` lightly loaded, moderately loaded, or also heavily consumed once all cached decode programs are resident?
*   Is row 0 uniquely bad, or is there a broader front-loaded placement bias across the full grid?
*   Which specific programs dominate SRAM usage on each region of the grid?
*   Is the cumulative SDPA-stage footprint stable across iterations once the program cache is warm?

#### Logging Format Requirements
The temporary instrumentation should emit **structured, machine-parseable log lines** rather than free-form debug text.

Suggested format:

```text
L1_CB_MAP program_id=<id> phase=<allocate|validate> core_range=<[(x0,y0)-(x1,y1)]> core=<x,y> cb_region_end=<bytes> max_l1_size=<bytes> lowest_top_down_addr=<bytes_or_none>
```

Additional recommendations:

*   Emit one log line per **logical core**, even if the allocator entry was created from a larger `CoreRange`.
*   Include `program_id` from `ProgramImpl::id` so logs can be grouped by cached program.
*   Include `phase=allocate` when emitted from `allocate_circular_buffers()` and `phase=validate` when emitted from `validate_circular_buffer_region()`.
*   Include both `cb_region_end` and `max_l1_size` so the parser can directly compute occupied/free percentages.
*   Include `lowest_top_down_addr` when available so the same logs can later be used to estimate true top-down collision margins.

#### Parser / Artifact Requirements
The parser should consume the structured log lines and generate the following artifacts:

1.  `per_program_core_map.json`
    One entry per `program_id`, with `cb_region_end` for every core `(x, y)`.
2.  `cumulative_core_map.json`
    The cumulative `cb_region_end` on every core at the SDPA stage.
3.  `sdpa_stage_headroom_map.json`
    Per-core remaining headroom in bytes and percent.
4.  `sdpa_stage_heatmap.md`
    A human-readable 8x8 table for quick inspection.

#### Cumulative Reconstruction Strategy
Because the central question is cumulative residency across cached programs, the parser should support two complementary views:

*   **Observed program-local view**: each program's own per-core `cb_region_end`.
*   **Observed SDPA-stage cumulative view**: the per-core SRAM state measured when the SDPA program is validated/launched after cache warmup.

The SDPA-stage cumulative view is the authoritative one for future KV mirror placement. The per-program view is diagnostic and attribution-oriented.

### 2. Controlled Attribution Knobs (Analysis Only)

Expose internal SDPA parameters only as controlled analysis knobs to measure how much SRAM pressure comes from SDPA itself versus previously cached programs.

#### [MODIFY] [model_config.py](file:///home/masterjunmo/codes/tt-metal/models/tt_transformers/tt/model_config.py)
*   Implement environment variable overrides (e.g., `TT_SDPA_K_CHUNK_SIZE`) to override the hardcoded `128` (for Blackhole) in `SDPA_DECODE_PROGCFG`.
*   Use this **only** to isolate the incremental SDPA-local CB contribution after the cumulative baseline has already been measured.
*   This knob should answer:
    *   How much of the SDPA-stage pressure is already present before SDPA starts?
    *   How much additional per-core pressure is introduced by SDPA itself?
    *   Whether reducing SDPA-local CB usage changes only the SDPA delta or the full cumulative map.
*   This is an attribution tool during analysis, **not** yet a placement solution.

### 3. Cache Residency Verification

We need to verify whether previously cached programs that are no longer actively executing still contribute SRAM pressure at the SDPA stage.

#### [VERIFY] Program cache behavior during decode
*   Run the same decode workload twice:
    *   once with normal warm program cache behavior
    *   once with controlled program-cache clearing inserted for verification
*   Compare the full-grid cumulative CB maps between the two runs.
*   If clearing the program cache materially reduces SDPA-stage CB occupancy, that strongly supports the hypothesis that cached prior programs are retaining SRAM footprint.
*   If the maps are unchanged, then the SRAM pressure is dominated by currently active programs rather than cached inactive ones.

### 4. Layout Optimization Research (Deferred Until After Mapping)

Only after the full-grid cumulative map is known should we investigate how to place the L1 KV mirror across the entire 8x8 grid.

#### [RESEARCH] [attention.py](file:///home/masterjunmo/codes/tt-metal/models/tt_transformers/tt/attention.py)
*   Analyze `_create_l1_kv_sharded_memcfg` and related sharding logic to understand future placement options across the entire grid.
*   Do **not** assume that excluding row 0 is acceptable; the target is to use the remaining space of **every** core as much as possible.
*   Any future placement strategy should be driven by measured per-core headroom, not by a row-0-only heuristic.

## Open Questions

> [!IMPORTANT]
> 1. **Cached Program Residency**: Do previously compiled/cached decode programs continue to retain CB SRAM footprint at the SDPA stage until the program cache is explicitly cleared?
> 2. **Grid-Wide Distribution**: What is the exact cumulative CB occupancy on all 64 cores, not just row 0?
> 3. **Per-Program Attribution**: Which cached programs contribute the largest SRAM deltas on each region of the grid?
> 4. **Usable Headroom**: After accounting for cumulative CB occupancy, how much L1 remains available on each core for future KV mirror placement?
> 5. **Placement Feasibility**: Once the real headroom map is known, can the KV mirror be distributed to exploit residual space across the whole 8x8 grid?

## Verification Plan

### Automated Tests
1.  **Warm-Cache Grid Map Generation**: Run the demo once and parse the logs to create a cumulative 2D CB map of L1 occupancy at the SDPA stage:
    ```bash
    TT_LOGGER_LEVEL=Info pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" --num_layers 1
    ```
2.  **Cache-Cleared Comparison Run**: Repeat with controlled program-cache clearing to compare the cumulative maps and isolate cached-program residency effects.
3.  **Optional Attribution Sweep**: After the baseline map is stable, sweep `TT_SDPA_K_CHUNK_SIZE` to isolate the SDPA-local contribution to the cumulative footprint.
4.  **Artifact Validation**: Ensure the parser produces `per_program_core_map.json`, `cumulative_core_map.json`, `sdpa_stage_headroom_map.json`, and a readable 8x8 markdown heatmap.

### Manual Verification
*   Confirm that the generated artifacts include:
    *   a per-core cumulative CB map for all 64 cores
    *   a per-program contribution breakdown
    *   a derived per-core remaining-headroom map at the SDPA stage
*   Verify whether the cumulative map changes after clearing the program cache.
*   Confirm that the log format is stable enough to support repeated runs and diffing across configurations.

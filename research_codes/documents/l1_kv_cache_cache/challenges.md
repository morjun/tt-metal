# Adaptive L1 KV Cache — Key Technical Challenges

## 1. Why Non-Uniform Allocation Is Hard

### The fundamental constraint: HEIGHT_SHARDED requires uniform shard shapes

`TensorMemoryLayout::HEIGHT_SHARDED` in tt-metalium stores a tensor as equal-sized rectangular slabs, one per participating core. The `ShardSpec` carries a single `shard_shape` value that applies to **every core** identically. There is no API to assign different shard heights to different cores within one tensor.

This means:
- If core A has 1,119 KiB headroom and core B has 335 KiB headroom, a single `HEIGHT_SHARDED` tensor must be sized to fit the weakest core.
- All extra headroom on core A is wasted.

### Why we want non-uniformity

The headroom distribution across the 130 Tensix cores is strongly non-uniform:

| Headroom class | Cores | Tile-rows fitting (bfloat8_b, 32 layers) |
|---|---|---|
| 311 KiB | 8 | 1 |
| 335 KiB | 56 | 1 |
| 747 KiB | 8 | **2** |
| 1,119 KiB | 45 | **4** |
| 1,221 KiB | 13 | **4** |

With uniform cap (bottleneck = 311 KiB): **520 total tokens**.
With non-uniform assignment: **1,248 total tokens** — a **2.4× increase**.

### The cost formula that makes it hard

Each tile-row stored on one core costs (all 32 layers, K+V):
```
bytes_per_tile_row = tile_size × head_dim × elem_bytes × num_layers × 2
                   = 32 × 128 × 1 × 32 × 2 = 262,144 bytes = 256 KiB
```

This is expensive because all 32 layers share the same core's L1 budget simultaneously (KV cache tensors are persistent, not transient). The minimum addressable unit in TILE layout is one tile-row (32 tokens). Key implications:

- **You cannot go below 32 tokens as the minimum per-core assignment.** Padding a single token to 1 tile wastes 31/32 of the shard.
- **The n_kv_heads dimension is split across cores**, not replicated. Each core holds `1/N` of the total (head × sequence) space. A non-uniform split means different cores own different-sized windows into this joint space.
- **The ring-buffer write position must be mapped** to the correct tier and the correct local offset within that tier — this mapping changes every decode step.

### Why truly continuous non-uniform allocation is not feasible with existing APIs

A truly continuous non-uniform shard (variable shard height per individual core) requires:
1. Per-core L1 buffer allocation at arbitrary addresses, bypassing the unified `HEIGHT_SHARDED` abstraction.
2. A kernel that carries a per-core `(base_addr, shard_height)` table as runtime arguments.
3. The SDPA kernel to dynamically dispatch reads based on which shard owns a given tile index.

None of this is supported by the current ttnn op infrastructure. The **bucketed-tier approach** (multiple HEIGHT_SHARDED tensors) is the closest feasible approximation.

---

## 2. Feasibility of Multiple MemoryConfigs (Bucketed Tiers)

### The idea

Instead of one non-uniform HEIGHT_SHARDED tensor, create **N independent HEIGHT_SHARDED tensors** (one per headroom tier), each with a uniform shard size matching that tier's capacity:

```
Tier 0: 64 cores @ 1 tile-row each  →  HEIGHT_SHARDED, shard_shape=[32, 128]
Tier 2:  8 cores @ 2 tile-rows each →  HEIGHT_SHARDED, shard_shape=[64, 128]
Tier 3: 45 cores @ 4 tile-rows each →  HEIGHT_SHARDED, shard_shape=[128, 128]
Tier 4: 13 cores @ 4 tile-rows each →  HEIGHT_SHARDED, shard_shape=[128, 128]
```

Each tier covers a contiguous token range `[token_start_i, token_start_i + token_count_i)`.

### What this requires from ttnn

**Tensor allocation**: `ttnn.as_tensor(shape=(1, 8, tok_count_i, 128), memory_config=tier_memcfg_i)` for each tier — **supported today**. There is no restriction on creating multiple HEIGHT_SHARDED tensors with different CoreRangeSets.

**Constraint**: The same physical core cannot appear in two different tier CoreRangeSets (L1 conflict). Since each core belongs to exactly one headroom class, the tier CoreRangeSets are naturally disjoint. ✓

**paged_update_cache (write path)**: Each tier is a separate tensor. We call `paged_update_cache` once per tier per decode step. The C++ validation guard (patched in our earlier session) already accepts `HEIGHT_SHARDED` L1 tensors. ✓

**fill_cache (prefill write)**: Each tier receives a slice of the prefill sequence. `fill_cache` operates on one tensor at a time and is called per tier. ✓

**Known risk**: Allocating 5 tier tensors × 2 directions (K+V) × 32 layers = **320 separate L1 buffer allocations** across a single inference session. The tt-metalium allocator must track all of these as top-down buffers simultaneously. This has not been stress-tested at this scale on Blackhole and could hit allocator limits.

---

## 3. SDPA Kernel Scope

### Current kernel interface (single L1 tensor)

`reader_decode_all.cpp` takes exactly one L1 K tensor and one L1 V tensor:
- **Compile-time**: Two `TensorAccessorArgs` blocks (encodes interleaved vs. sharded geometry)
- **Runtime**: `l1_k_addr`, `l1_v_addr`, `l1_recent_window_start_tile`, `l1_recent_window_size_tiles`, `l1_sink_size_tiles`

Per-tile L1 vs. DRAM decision in `read_kv_mask_chunks_dual_source`:
```cpp
bool in_sink   = (global_seq_tile < l1_sink_size_tiles);
bool in_recent = (global_seq_tile >= l1_recent_window_start_tile) &&
                 (global_seq_tile <  l1_recent_window_start_tile + l1_recent_window_size_tiles);
// → read from l1_k_reader if in_sink || in_recent, else from DRAM k_reader
```

### What N-tier support requires

For N tiers, each tile must be mapped to its tier's reader:
```
for i in 0..N:
    if tier_start[i] <= global_seq_tile < tier_start[i] + tier_size[i]:
        read from l1_k_reader[i] at (global_seq_tile - tier_start[i])
        break
else:
    read from DRAM
```

This requires changes to **4 files** across 3 abstraction layers:

| File | Change | Invasiveness |
|---|---|---|
| `reader_decode_all.cpp` | Add N compile-time `TensorAccessorArgs` pairs + N runtime arg groups | Medium — additive |
| `dataflow_common.hpp` | Extend `read_kv_mask_chunks_dual_source` to N-tier dispatch | Medium — template `num_tiers` |
| `sdpa_decode_program_factory.cpp` | Route N tier addresses to per-core runtime args | **High** |
| `sdpa_decode_op.cpp` + pybind | Accept `std::vector<Tensor>` L1 tier inputs | Low — additive |

### The hardest part: per-core address routing in the program factory

In the current single-L1 case, the program factory passes `l1_k_addr = tensor.buffer()->address()` as one base address. The kernel uses the `TensorAccessorArgs` sharding geometry to compute the correct per-core local address from a tile index automatically.

For N tiers, the program factory must determine, for each SDPA worker core, **which addresses within each tier tensor correspond to its assigned KV head range**. This is complex because:
- Each tier has a different `CoreRangeSet` and `ShardSpec`
- The SDPA worker cores and the tier storage cores may not be the same physical cores
- If they differ, reads become **remote NOC reads** — negating the latency benefit of L1 caching

### Critical feasibility question: SDPA-tier core alignment

**The key property that makes L1 cache reads local** in the current single-tensor case: the HEIGHT_SHARDED cache tensor is assigned such that the core computing KV head `h` also locally stores the shard for head `h`. No NOC hop needed.

With multiple tiers and disjoint CoreRangeSets, this alignment must be preserved:
- The SDPA decode op assigns computation to cores based on `num_kv_heads` and `num_cores_per_head`
- The tier CoreRangeSets must be chosen such that each SDPA core's assignment corresponds to its locally stored tier shards
- **If the tier boundaries cut across KV head boundaries**, a single SDPA core may need data from a core it doesn't own — requiring a NOC read

> [!IMPORTANT]
> This alignment constraint must be verified **before Phase 3 implementation begins**. If SDPA core assignment is configurable via the program factory, tier CoreRangeSets can be designed to match. If it is fixed, the tier design must adapt.

### Summary: effort estimate

| Phase | Effort | Risk |
|---|---|---|
| Phase 1 (Python allocation) | Low — ✅ complete | Low |
| Phase 2 (C++ op interface) | Medium (~2 days) | Low |
| Alignment investigation | Medium (~1 day) | **Could block Phase 3** |
| Phase 3 (RISC-V kernel) | High (~1 week) | Medium |

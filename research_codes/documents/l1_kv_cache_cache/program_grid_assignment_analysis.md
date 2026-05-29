# Program-to-Grid Assignment Analysis: Llama 3.1 8B Decode on P150

This document captures the analysis of how TT-Metal programs are assigned to specific
core grids on the Blackhole P150 chip during Llama 3.1 8B Instruct decode (batch=1).
The goal is to explain *why* each program uses its particular core count rather than
maximizing core utilization.

> **Note on program IDs (PIDs)**: TT-Metal assigns program IDs sequentially in order of
> first creation. PIDs are **not stable** across runs — re-running the workload or clearing
> the compiled kernel cache will produce different PID assignments for the same operations.
> The PIDs and CB numbers in this document are sourced from
> `l1_cb_artifacts_noCacheFull_260414/` (run: `TT_METAL_LOG_L1_CB_MAP=1 TT_LOGGER_LEVEL=Info
> pytest models/tt_transformers/demo/simple_text_demo.py -k 'performance and batch-1'`).
> Use op name and core grid shape as the stable identifiers; treat PIDs as session-local labels.

## P150 Chip Layout

The Blackhole P150 has **130 usable Tensix cores** arranged in a 13×10 grid (x=columns 0..12,
y=rows 0..9). There are 8 DRAM banks. Each bank's optimal reader is a Tensix core placed one
column to the right of the DRAM NOC endpoint (at physical x+1) and at physical y ≥ 2 (physical
rows 0–1 are reserved for non-Tensix cores such as Ethernet and dispatch; Tensix compute workers
begin at physical y=2). On P150 this placement maps DRAM readers to logical y=8 (physical y=10).
Logical rows y=0 through y=9 are all full-capability Tensix cores — the y=0 hotspot discussed
later is unrelated to this physical-row numbering.

```
x →  0   1   2   3   4   5   6   7  │  8   9  10  11  12
     ──────────────────────────────────────────────────────
y=0 ┐                                │
y=1 │                                │
y=2 │  8×8 = 64-core compute zone    │  right wing
y=3 │  (L1-sharded storage cores)    │  (x=8..12)
y=4 │  (matmul compute, SDPA, etc.)  │
y=5 │                                │
y=6 │                                │
y=7 ┘                                │
     ──────────────────────────────────
y=8 [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓]  ← DRAM-reader row → bounding-box edge (45%)
     ──────────────────────────────────
y=9 [  full-chip bottom row (7%)     ]
```

All 130 cores are also overlaid with full-chip programs (rotary embedding, RMSNorm,
elementwise ops) that shard uniformly across the entire grid.

---

## Observed Grid Shapes and Their Programs

From the L1 CB map logs (`l1_cb_artifacts/per_program_core_map.json`):

| Grid shape | Core count | Coordinate range | Programs |
|---|---|---|---|
| Row y=0 only | 8 | x=0..7, y=0 | pid=5 (`rms_norm`, worst-case CB 79.5%; identity of tensor being normalized TBD) |
| Column x=0 only | 8 | x=0, y=0..7 | pids 1, 9, 27 (unknown; `nlp_create_qkv_heads_decode_interleaved` and `nlp_concat_heads_decode` use only globally-allocated CBs and do not appear here) |
| 8×8 inner | 64 | x=0..7, y=0..7 | FFN matmuls, SDPA decode, prefill mcast-2d |
| 8×9 bounding box | 72 | x=0..7, y=0..8 | `matmul_dram_sharded` (FFN w1/w2/w3, XQKV) |
| 13×9 bounding box | 117 | x=0..12, y=0..8 | `matmul_dram_sharded` — **WO (attention output)**, K=N=4096, M=1, 32 storage cores across all 13 columns |
| Full chip | 130 | x=0..12, y=0..9 | `rotary_embedding_llama`, `rms_norm`, `rms_norm_sharded`, `typecast`, `broadcast_height_and_width` |

---

## Big-Picture Core Zone Map

The following table maps the functional zones of the P150 die during Llama 3.1 8B Instruct decode.
Each zone is defined by the set of operations that allocate CBs there at runtime. These zones
**overlap**: the same physical core participates in multiple zones across different programs and
decode steps.

| Zone | Coordinates | Cores | Primary ops |
|---|---|---|---|
| 8-core row y=0 | x=0..7, y=0 | 8 | `rms_norm` (pid=5, dominant CB user, 79.5%); `paged_update_cache` (pid=83) also uses (0,0) but only on **1 core** with 9.7% CB |
| QKV / concat heads column | x=0, y=0..7 | 8 | `nlp_create_qkv_heads_decode_interleaved`, `nlp_concat_heads_decode` (use only globally-allocated CBs; appear in op log but not in per-core CB map) |
| 64-core compute block | x=0..7, y=0..7 | 64 | `matmul_mcast_2d_optimized` (prefill), `sdpa_decode` |
| 8×9 DRAM-sharded bbox | x=0..7, y=0..8 | 72 | `matmul_dram_sharded` (FFN w1/w2/w3, XQKV) |
| 13×9 DRAM-sharded bbox | x=0..12, y=0..8 | 117 | `matmul_dram_sharded` — **WO** (attention output projection, K=N=4096, M=1 decode token, 32 storage cores across all 13 columns; confirmed by wide-bbox log: M=1 K=128 N=128 tiles) |
| Full chip | x=0..12, y=0..9 | 130 | `rotary_embedding_llama`, `rms_norm` (non-sharded), `rms_norm_sharded`, `typecast`, `broadcast_height_and_width`; CCL (`all_gather_async`, `reduce_scatter_minimal_async`) in multi-device runs only |

For example, core (0,0) belongs to every zone in the table: it is in the KV head row, the QKV
column, the 64-core block, both bounding boxes, and the full-chip programs — all across different
ops in the same decode pass.

### Bounding-box mechanics

The `matmul_dram_sharded` factory does not allocate CBs separately on storage cores and
DRAM-reader cores. Instead it computes the **bounding-box rectangle** of their union and places
all CBs on every core within that rectangle:

```cpp
// matmul_op_multi_core_reuse_mcast_dram_sharded_program_factory.cpp, lines 288–293
CoreRange bounding_box = all_cores.bounding_box();   // all_cores = storage ∪ DRAM readers
std::set<CoreRange> bounding_box_set;
bounding_box_set.insert(bounding_box);
CoreRangeSet all_cores_in_rect_grid(bounding_box_set);
// All CBs created on the full rectangle:
CreateCircularBuffer(program, all_cores_in_rect_grid, src0_cb_config);  // line 469
CreateCircularBuffer(program, all_cores_in_rect_grid, src1_cb_config);  // line 483
// ... all subsequent CBs use all_cores_in_rect_grid
```

Cores that fall inside the rectangle but are neither storage nor DRAM reader ("gap cores") receive
CBs and run idle kernel code. These CBs consume real L1 on every gap core regardless of whether
that core contributes any compute.

**Implication for the L1 KV cache mirroring feature (future improvement)**: For the 13×9 bounding
box (WO matmul), gap cores include the entire right wing (x=8..12, y=0..7) and the full DRAM-reader
row (y=8, x=0..12). All 117 cores have L1 consumed by matmul CBs while this op is live.
The L1 KV cache mirror must coexist with these CB allocations, so effective headroom on right-wing
cores is reduced whenever WO runs. Allocating CBs only on the actual storage and DRAM-reader core
sets (rather than their bounding box) would eliminate this waste and is a concrete optimization
opportunity in the factory.

---

## Why Each Grid Size Is Chosen

### 1. DRAM-sharded matmuls — grid from tile-divisibility constraint

The core count for DRAM-sharded matmuls is determined by `find_grid_k_n(K_tiles, N_tiles)`:
find the **largest** number of cores ≤ 64 that evenly divides **both** K_tiles and N_tiles.
The divisibility requirement exists because each core must receive an integer number of tiles
for both the reduction (K) and output (N) dimensions.

```python
# model_config.py
def find_grid_k_n(K, N):
    max_cores = 8 * 8  # hard cap at 64 (8x8 sub-grid)
    possible_cores = [c for c in range(1, max_cores+1) if K % c == 0 and N % c == 0]
    possible_cores.sort(reverse=True)   # take the largest valid count
    ...
```

Results for Llama 3.1 8B (tile_size=32):

| Operation | k | n | K_tiles | N_tiles | Grid | #Cores |
|---|---|---|---|---|---|---|
| FFN gate (w1) | 4096 | 14336 | 128 | 448 | 8×8 | 64 |
| FFN up (w3) | 4096 | 14336 | 128 | 448 | 8×8 | 64 |
| FFN down (w2) | 14336 | 4096 | 448 | 128 | 8×8 | 64 |
| XQKV decode | 4096 | 6144 | 128 | 192 | 8×8 | 64 |
| Attn output (WO) | 4096 | 4096 | 128 | 128 | 4×8 | 32 |

**Why not 130 cores?** The cap of 64 (8×8) is a model-config constant inherited from
Wormhole B0, which has an 8×8 compute grid. The Blackhole P150's full 10×13 grid has not yet
been exploited by this model config.

### 2. DRAM-reader expansion — why bounding box is 8×9, not 8×8

After selecting 64 compute (L1-storage) cores — confirmed as **x=0..7, y=0..7** by the CB
map data — the factory calls `get_optimal_dram_bank_to_reader_assignment()`.

This function (`get_optimal_dram_to_physical_worker_assignment` in `core_assignment.cpp`) places
each DRAM reader at physical (dram_x+1, max(dram_y, 2)) and then converts to logical
coordinates. On P150, with NOC_1 as the preferred DRAM write NOC, this assignment maps all
8 DRAM readers to logical cores whose y-values include **y=8** (physical y=10 in the
functional-worker grid that starts at physical y=2).

The factory then creates semaphores and CBs on the **bounding box** of
`mcast_senders (L1 storage) ∪ mcast_receivers (DRAM readers)`. Because the L1 storage
occupies x=0..7, y=0..7 and the DRAM readers extend to y=8, the bounding box becomes
x=0..7, y=0..8 = **72 cores**.

```
Bounding box = (8×8 compute, y=0..7) + (DRAM reader row at y=8) = 8×9 = 72 cores logged in CB map
```

Empirical confirmation from `headroom_map.json`:
- y=0..7 cores (x=0..7): worst_pid = various compute programs — clearly in the compute zone
- y=8 cores (x=0..7): worst_pid = `matmul_dram_sharded` (pid=57, 81, etc.), 45.96% L1 — the bounding box extension, CBs allocated here by the factory even for cores that are passive bystanders within the rectangle
- y=9 cores: only touched by 130-core full-chip programs (6.8%)

**Why bounding box, not just storage+readers?** The factory does not allocate CBs on the
individual storage and DRAM-reader core sets separately. Instead it computes the bounding-box
rectangle of their union and allocates ALL CBs on that rectangle
(`matmul_op_multi_core_reuse_mcast_dram_sharded_program_factory.cpp` lines 288–293):

```cpp
CoreRange bounding_box = all_cores.bounding_box();  // all_cores = mcast_senders ∪ mcast_receivers
CoreRangeSet all_cores_in_rect_grid(bounding_box_set);
// Every CB is then created on the full rectangle:
CreateCircularBuffer(program, all_cores_in_rect_grid, src0_cb_config);  // line 469
CreateCircularBuffer(program, all_cores_in_rect_grid, src1_cb_config);  // line 483
// ... all subsequent CBs use all_cores_in_rect_grid
```

Cores that fall inside the rectangle but are neither L1 storage nor DRAM reader (the
"gap" cores) receive CBs and run idle kernel code. These CBs consume real L1 space on
every gap core.

For the **117-core variant**, the `input_all_storage_cores` spans all 13 device columns
(x=0..12). With DRAM readers at y=8 pushing max_y to 8, the bounding box is
x=0..12, y=0..8 = 13×9 = **117 cores** — and every one of those 117 cores receives CBs.

**Runtime identification (confirmed)**: The wide-bbox log (`TT_METAL_LOG_L1_CB_MAP`) shows:
```
>>> matmul_dram_sharded wide-bbox: M=1 K=128 N=128 per_core_M=1 per_core_N_storage=4
    bbox=[(0,0)..(12,8)] storage_cores=32
```
`K=128 tiles × 32 = 4096`, `N=128 tiles × 32 = 4096`, `M=1 tile` (one decode token).
This is the **WO (attention output projection)** matmul: the 32 attention head outputs
(each 128-dimensional) are concatenated into a 4096-vector and projected by WO back to
model dimension 4096. The 32 storage cores holding these activations are distributed
across all 13 chip columns (because SDPA decode scatters the head outputs across the full
chip width), producing the 13-column bounding box.

### 3. Full-chip (130 cores) — shard spec propagation

Elementwise ops (unary, binary, RMSNorm/LayerNorm) inherit the **shard spec of their input
tensor**. They do not independently choose a core count. If the upstream op placed its output
on all 130 cores, the downstream op runs on all 130 cores.

The rotary embedding (`rotary_embedding_llama_multi_core`) and broadcast ops follow this
pattern: they run wherever the Q/K head tensors are already resident.

Full-chip sharding is used for ops with small per-tile CB footprint (~7–13% L1 per core) where
there is no memory reason to restrict to a sub-grid.

### 4. 8-core programs — n_kv_heads constraint

Several programs run on exactly 8 cores, one per KV head. Two distinct layouts appear:

- **Column pattern (x=0, y=0..7)**: pids 1, 9, 27 (unknown — `nlp_create_qkv_heads_decode_interleaved`
  and `nlp_concat_heads_decode` are confirmed to run here but allocate only globally-allocated
  CBs and are invisible to the CB map; the actual unknown programs may be other column-using ops)
- **Row pattern (x=0..7, y=0)**: pid=5 (`rms_norm`, 8 cores WIDTH-sharded); pid=83
  (`paged_update_cache`, only 1 core at (0,0))

The two patterns reflect different shard orientations (HEIGHT vs WIDTH) for the 8-element KV
head dimension.

---

## L1 Pressure by Zone

Measured from `l1_cb_artifacts_noCacheFull_260414/headroom_map.json` (worst-case CB per core,
i.e. the single program with the highest `cb_region_end` for each core):

| Zone | Cores | Worst-case CB | % of L1 | Caused by (worst_case_pid) |
|---|---|---|---|---|
| y=0 row (x=0..7) | 8 | 1,249,664 B | **79.5%** | pid=5 `rms_norm` — tensor sharded on this 8-core row (identity TBD) |
| x=0 col y=1..7 | 7 | 1,225,088 B | **77.9%** | pid=41 `matmul_mcast_2d_optimized` |
| 64-core interior y=0 (x=1..7) | 7 | 1,249,664 B | **79.5%** | pid=5 `rms_norm` (same 8-core program as y=0 row) |
| 64-core interior y=1..7 (x=0..7) | 56 | 1,225,088 B | **77.9%** | pid=41 `matmul_mcast_2d_optimized` |
| y=8 row (x=0..7) — DRAM reader row | 8 | 722,944 B | **46.0%** | pid=49 `matmul_dram_sharded` (FFN bbox extension) |
| Right wing (x=8..12, y=0..8) | 45 | 360,832 B | **22.9%** | pid=91 `matmul_dram_sharded` WO (13-column bbox extension) |
| Bottom row (y=9) | 13 | 106,880 B | **6.8%** | pid=117 unknown (full-chip program) |

---

## The y=0 Hotspot: Root Cause

The 8 cores at y=0 (x=0..7) are the hottest on the chip. They are **not** DRAM reader cores
(those land at y=8). y=0 is the first row of the ordinary compute zone.

### What the data shows

From `l1_cb_artifacts_noCacheFull_260414/headroom_map.json`, all 8 cores at y=0 (x=0..7)
share the same worst-case CB:

| Core | worst_case_pid | op_name | cb_region_end | % of L1 |
|---|---|---|---|---|
| (0,0)..(7,0) | **5** | **`rms_norm`** | 1,249,664 B | **79.45%** |

From `per_program_core_map.json`, pid=5 is exactly the 8 cores {(0,0),(1,0),...,(7,0)} — the y=0
row. This rms_norm program allocates 1.25 MB of local CBs on each of these 8 cores.

### What paged_update_cache actually looks like

`paged_update_cache` (pid=83, `paged_update_cache_program_factory.cpp:45`) appears in the
per-core map with **1 core only** at `(0,0)` with cb_region_end = **152,704 B (9.7%)**. The
paged KV cache update for all 8 KV heads is handled by a single-core program, not an 8-core
one. It is far from the dominant user of y=0 L1.

### Why the previous analysis was wrong

Earlier analyses attributed the y=0 hotspot to the KV cache update operation, reasoning from
the model architecture (8 KV heads should map to 8 cores at y=0 in ROW_MAJOR sharding). That
was an incorrect inference: `paged_update_cache` in this model uses a **single-core** dispatch,
and the op that actually dominates the y=0 row is an `rms_norm` applied to a tensor sharded
WIDTH-wise across those 8 cores.

### What rms_norm is running at y=0?

pid=5 = `rms_norm` with 8 cores at x=0..7, y=0 means some tensor is WIDTH-sharded across
8 cores placed at y=0. The most likely candidates in the Llama attention path are:
- Per-head K-norm or Q-norm applied before rotary embedding (some Llama 3.x variants include this)
- RMSNorm applied to the KV projection output before paging, if that output is sharded at y=0

The exact identity requires tracing which tensor produces a WIDTH-sharded shard spec at y=0
with 8 shards in the attention path. The CB size of 1.25 MB at decode (batch=1, seq_len=1) is
unusually large for a simple normalization step and may indicate prefill-phase norms compiled
for a longer sequence or a larger intermediate buffer.

### The L1 KV cache tensor is invisible to this heatmap

`models/tt_transformers/tt/attention.py` stores the KV cache as:
```python
self.l1_kv_cache = [
    ttnn.as_tensor(k_or_v, ..., memory_config=ttnn.L1_MEMORY_CONFIG, ...)
    for k_or_v in [l1_cache_k, l1_cache_v]
]
```
`ttnn.L1_MEMORY_CONFIG` = `TensorMemoryLayout::INTERLEAVED, BufferType::L1`, allocating pages
top-down across all 130 cores. This is tracked in `lowest_top_down_addr`, not `cb_region_end`,
and is therefore invisible in the CB heatmap. The top-down allocation consumes only 4,352 B
per core in this run (`top_down_size_bytes` in headroom_map.json for y=0 cores).

---

## Program Identification Methodology

Each program ID was matched to a source factory by cross-referencing the core grid shape observed
in `per_program_core_map.json` with the core assignment logic in the factory source code. The
key principle is that every factory sets `all_cores` from either (a) the input/output tensor's
shard spec, or (b) a hardcoded formula (GCD, `compute_with_storage_grid_size`, etc.). We then
verify the CB index set against what the factory creates on those cores.

### `rms_norm` (pid=5) — actual y=0 row hotspot

**Observed**: 8 cores at x=0..7, y=0 (single row). Worst-case CB: **1,249,664 B (79.5%)**.

**Factory**: `ttnn/.../layernorm_op_multi_core.cpp:324`
(runtime confirmed: `>>> rms_norm program id=5`)

This is the dominant CB consumer on the y=0 row. The tensor it normalizes must be WIDTH-sharded
across exactly these 8 cores. The specific tensor and calling layer require further code
investigation (see "y=0 Hotspot" section above).

### `paged_update_cache` (pid=83) — single-core KV update

**Observed**: **1 core at (0,0) only**. CB: 152,704 B (9.7%).

**Factory**: `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/paged_update_cache_program_factory.cpp`
(runtime confirmed: `>>> paged_update_cache program id=83`)

Contrary to prior analysis, `paged_update_cache` in this model uses a **single-core** program,
not 8 cores. All 8 KV heads are managed through one compiled program running on core (0,0).
Its CB footprint is modest (9.7%) and does not contribute meaningfully to the y=0 hotspot.

### `nlp_create_qkv_heads_decode_interleaved` / `nlp_concat_heads_decode` — column pattern

Runtime confirmed: `>>> nlp_create_qkv_heads_decode_interleaved program id=77` and
`>>> nlp_concat_heads_decode program id=89`.

**Important**: pids 77 and 89 do **not appear** in `per_program_core_map.json`. These ops
allocate CBs backed only by globally-allocated output shard buffers
(`set_globally_allocated_address`). Globally-allocated CBs are skipped in
`allocate_circular_buffers()` and produce no `L1_CB_MAP` log entries. They have zero
`cb_region_end` contribution and are invisible to the CB heatmap entirely.

**Factory**: `ttnn/.../nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_program_factory.cpp`

Core grid assignment (lines 85–116): cores taken from the corresponding output tensor shard spec:
```cpp
auto q_cores = output[0].shard_spec().value().grid;  // ← inherited from Q output shard spec
auto k_cores = output[1].shard_spec().value().grid;  // ← inherited from K output shard spec
```

For 8 KV heads in COLUMN_MAJOR orientation:
`CoreRange({0,0},{0,7})` = x=0, y=0..7 (single column).

The unknown pids 1, 9, 27 (8 cores at x=0, y=0..7) that DO appear in the CB map are
separate programs that also run on the same column. Their identity is unresolved.

### `matmul_dram_sharded` (13-column / WO variant) — confirmed via wide-bbox log

**Observed**: 117 cores at x=0..12, y=0..8 (13×9 bounding box). `op_name = "matmul_dram_sharded"`.

**Why 117 and not 72?** The factory creates CBs on the bounding-box rectangle of
`input_all_storage_cores ∪ DRAM_readers` (see "DRAM-reader expansion" section above). For
the standard 72-core programs, storage spans x=0..7 → bounding box x=0..7, y=0..8 = 72.
For this program, storage spans **x=0..12** (13 columns) → bounding box x=0..12, y=0..8 = 117.

**What makes storage span 13 columns?** Confirmed by runtime logging (wide-bbox log):
`M=1, K=128, N=128 tiles`, `storage_cores=32`, `per_core_N_storage=4`.
This is the **WO (attention output projection)** matmul: K=N=4096 (tile_size=32 →
128 tiles each), M=1 decode token. The 32 storage cores hold the concatenated attention head
activations and are distributed across all 13 chip columns, because SDPA decode scatters
its per-head outputs across the full P150 width.

**Note**: The experimental `use_l1_weight_sharding` feature has been entirely removed from the codebase.

---

## Unknown PIDs: Instrumentation Coverage Gap

The initial artifacts (from an earlier run) had 47/66 programs (71%) with `op_name = "unknown"`.
After adding instrumentation to additional factories, a follow-up run identified many more
programs. The updated picture is below.

### Known programs in `l1_cb_artifacts_noCacheFull_260414/`

59 total programs. 24 identified (40.7%), 35 unknown (59.3%). The known programs:

| op_name | PID(s) | Cores | Notes |
|---|---|---|---|
| `rms_norm` | 5, 45 | 8, 1 | pid=5: 8 cores at y=0 (hotspot); pid=45: 1 core at (0,0) |
| `matmul_mcast_2d_optimized` | 7, 29, 33, 37, 41 | 64 each | 5 variants (different shapes/layers) |
| `rotary_embedding_llama` | 11, 13 | 130 each | Full-chip |
| `typecast` | 15, 23, 113 | 130 each | Full-chip; pid=129 is 2 cores |
| `typecast` | 129 | 2 | Only (0,0) and (0,1) |
| `broadcast_height_and_width` | 19 | 130 | Full-chip |
| `rms_norm_sharded` | 71, 99 | 32, 64 | Sharded norm variants |
| `matmul_dram_sharded` | 49, 73, 101, 107, 111 | 72 each | FFN/XQKV; 8×9 bbox |
| `rotary_embedding_llama_sharded` | 79, 81 | 1 each | Single core |
| `paged_update_cache` | 83 | **1** | Single core at (0,0); 9.7% CB |
| `sdpa_decode` | 85 | 64 | 8×8 |
| `matmul_dram_sharded` (WO) | 91 | 117 | 13×9 bbox; confirmed by wide-bbox log |

**Programs with no CB map entry** (globally-allocated CBs only, invisible to heatmap):
- `nlp_create_qkv_heads_decode_interleaved` (pid=77)
- `nlp_concat_heads_decode` (pid=89)

### CCL ops: single-device vs. multi-device

`all_gather_async` and `reduce_scatter_minimal_async` did **not appear** in the single-P150
run. These CCL operations are only dispatched in tensor-parallel (multi-device) mode.
Instrumentation was added (`all_gather_async_program_minimal_variants.cpp`,
`reduce_scatter_minimal_async_program.cpp`) and will produce op names in multi-device runs.
They are **not a source of unknowns** in single-device decode.

### `update_cache_multi_core` (non-paged) not observed

The non-paged `update_cache_op_multi_core.cpp` factory was instrumented but did not fire
in any run. Llama 3.1 8B exclusively uses `paged_update_cache`; the non-paged variant is
not invoked. Instrumentation was added as a precaution and is harmless.

### Remaining unknowns

After the new instrumentation, any remaining unknown programs in a fresh run are likely
small-core (1–2 core) initialization, synchronization, or single-tile dispatch ops that
execute once at model load time. They have negligible L1 footprint and are low-priority
identification targets.

---

## Implications for L1 KV Mirror Placement

**Important baseline**: The L1 KV cache itself (`ttnn.L1_MEMORY_CONFIG`, interleaved) is already
distributed across all 130 cores via top-down allocation (`lowest_top_down_addr`). This allocation
is separate from the CB stack measured here and reduces the available top-down headroom on every
core, not just y=0.

The `cb_region_end`-based headroom estimates below reflect remaining space for additional CBs only;
any mirror placement must also account for the existing interleaved KV cache pages.

**y=0 row (x=0..7)** — 79.5% worst-case CB (`rms_norm`, pid=5), gap_bytes = 318,848 B (~312 KB).
Insufficient for a meaningful additional KV window shard.

**64-core compute interior y=1..7 (x=0..7, 56 cores)** — 77.9% CB (`matmul_mcast_2d_optimized`,
pid=41), gap_bytes = 343,424 B (~335 KB). Very tight; these cores also carry interleaved KV
cache pages from top-down.

**Right wing (x=8..12, y=0..8)** — 22.9% CB (pid=91 `matmul_dram_sharded` WO bbox extension),
gap_bytes ≈ 764,928 B (~747 KB). Best candidate for additional L1 KV mirror CB allocation.
Also carries a small top-down allocation (~65 KB).

**DRAM reader row (y=8, x=0..7)** — 46% CB (`matmul_dram_sharded` FFN bbox extension),
gap_bytes ≈ 764,928 B. Reasonable headroom but these cores sit on the DRAM reader row and
overlap with active matmul data paths.

**Bottom row (y=9)** — 6.8% CB, gap_bytes ≈ 1,251,008 B (~1.22 MB). Best raw headroom on
the chip. Touched only by 130-core full-chip programs. Would require explicit shard placement
to use; carries more top-down allocation (~215 KB) than other zones.

---

## The 8×8 Core Cap: Current State

Three functions in `models/tt_transformers/tt/model_config.py` each hardcode a maximum of
8×8 = 64 compute cores for DRAM-sharded matmuls:

| Function | Lines | Cap |
|---|---|---|
| `find_grid(N)` | 2176–2196 | `max_rows = 8`, `max_cols = 8` |
| `find_prefill_grid(row, col)` | 2202–2222 | `max_rows = 8`, `max_cols = 8` |
| `find_grid_k_n(K, N)` | 2243–2261 | `max_rows = 8`, `max_cols = 8` |

These are plain Python constants — not device-queried, not guarded by `is_blackhole()` or
`is_wormhole_b0()`. A `# TODO Improve configuration for BH (higher core grid than WH)` comment
exists at `model_config.py:2204` acknowledging this gap.

`self.max_grid_size` (set to `compute_with_storage_grid_size()` at line 687, which returns
13×10 for P150) is available to the class but is not consulted by these three functions.

**GCD-based impact analysis for Llama 3.1 8B (tile_size=32):**

| Op | K_tiles | N_tiles | GCD(K,N) | GCD caps cores at | Raising cap to 130 gains |
|---|---|---|---|---|---|
| FFN gate/up (w1/w3) | 128 | 448 | 64 | 64 | nothing (already at cap) |
| FFN down (w2) | 448 | 128 | 64 | 64 | nothing |
| XQKV decode | 128 | 192 | 64 | 64 | nothing |
| Attn output (WO) | 128 | 128 | 128 | 128 | **+2× cores** (128 fits in 13×10) |
| LM head (per split) | 128 | 501 | 1 | 1 | nothing (N_tiles not divisible) |

`find_grid_k_n` requires each candidate core count to divide both K_tiles and N_tiles exactly.
The GCD of (K_tiles, N_tiles) is the hard ceiling on achievable cores for that op. Raising the
cap only matters when GCD > 64. For most Llama 3.1 8B ops the GCD is already 64, so the cap
has no effect. Only WO (K=N=4096, GCD=128) would benefit: 128 cores can be tiled as
e.g. 10×12+8 within the 13×10 grid.

`find_prefill_grid` searches each dimension independently and would also gain from the wider
13-column layout for large N splits.

---

## Data Sources

- `l1_cb_artifacts_noCacheFull_260414/per_program_core_map.json` — per-program core sets and op names (includes `wide_bbox_info` for DRAM-sharded programs with bbox wider than 8 columns)
- `l1_cb_artifacts_noCacheFull_260414/headroom_map.json` — per-core worst-case CB end, headroom, worst_pid
- `l1_cb_artifacts_noCacheFull_260414/cumulative_core_map.json` — max CB usage across all programs per core
- `models/tt_transformers/tt/model_config.py` — `find_grid`, `find_grid_k_n`, `find_prefill_grid`, core grid configs, `lm_head_core_grid`
- `models/tt_transformers/tt/attention.py` — L1 KV cache tensor layout (`ttnn.L1_MEMORY_CONFIG`, interleaved)
- `models/tt_transformers/tt/mlp.py` — Experimental `use_l1_weight_sharding` feature (entirely removed from the codebase)
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/paged_update_cache_program_factory.cpp` — y=0 hotspot (`paged_update_cache`)
- `ttnn/.../matmul_op_multi_core_reuse_mcast_dram_sharded_program_factory.cpp` — DRAM reader assignment, wide-bbox log (lines 295–306)

---

## Still Existing Questions

### Does the current SRAM usage analysis include space that kernel code and the program queue occupy?

**Short answer: No — the analysis already excludes both.**

#### Kernel RISCV binaries (IRAM)

Tensix cores contain dedicated instruction RAM (IRAM) that is architecturally separate from L1
data SRAM. When the TT-Metal runtime compiles a kernel, the RISCV binary is loaded into IRAM.
IRAM has no address overlap with L1 SRAM, so kernel code contributes **zero bytes** to
`cb_region_end` or `lowest_top_down_addr`. The `L1_CB_MAP` log and all derived metrics
(`gap_bytes`, `headroom_map.json`) measure only the L1 data SRAM window.

#### Program queue / dispatch communication buffers

The TT-Metal runtime reserves a fixed region at the very base of each core's L1 SRAM for
dispatch ring-buffers, command queues, and other runtime communication structures. This reserved
block is carved out once at device initialization, below `l1_unreserved_base`.

All values reported by the `L1_CB_MAP` log are measured **relative to `l1_unreserved_base`**:

```
cb_region_end   → bytes above l1_unreserved_base used by CBs (bottom-up)
max_l1_size     → 1,572,864 B (1.5 MB) = unreserved L1 capacity, NOT raw physical L1
lowest_top_down_addr → offset from l1_unreserved_base of the top-down tensor region
```

Because the origin is `l1_unreserved_base`, the dispatch reserved region is implicitly excluded
from every headroom measurement. The 1.5 MB figure is the usable window after the runtime carves
out its overhead; the raw physical L1 per core is larger.

#### Summary

| Memory kind | Location | Included in analysis? |
|---|---|---|
| RISCV kernel binaries | IRAM (separate hardware) | No — architecturally separate |
| Dispatch / program queue buffers | L1 below `l1_unreserved_base` | No — below measurement origin |
| Circular buffers (CBs) | L1 above `l1_unreserved_base`, bottom-up | Yes — `cb_region_end` |
| Tensor buffers (KV cache, weights) | L1 above `l1_unreserved_base`, top-down | Yes — `lowest_top_down_addr` |

The gap (`lowest_top_down_addr − cb_region_end`) reported throughout this document is the true
free headroom available for additional CB or tensor allocation within the unreserved L1 window.

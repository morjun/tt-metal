# L1 CB Footprint Analysis: Why Cores Outside the 8×8 SDPA Grid Have CB Pressure

**Source**: `TT_METAL_LOG_L1_CB_MAP=1` profiling of `simple_text_demo.py`
with `--num_layers 1 MESH_DEVICE=P150` on Blackhole P150.

---

## 0. What `cb_region_end` Actually Measures

Before reading the heatmap, it is critical to understand what the logged
`cb_region_end` value does and does **not** include.

### Only locally-allocated CBs are counted

`cb_region_end` is the end address of the **bottom-up, locally-allocated** circular
buffer region for a single program on a single core. Specifically, in
`allocate_circular_buffers()` (`program.cpp` line 885–887):

```cpp
if (circular_buffer->globally_allocated()) {
    continue;   // ← globally-allocated CBs are SKIPPED entirely
}
```

Globally-allocated CBs — those backed by a persistent tensor buffer via
`.set_globally_allocated_address(*buffer)` — are **excluded** from `cb_region_end`.

### The true L1 layout has two separate occupied regions

```
low addr  [l1_unreserved_base]
          ┌──────────────────────────────────────────┐
          │  LOCAL CBs (bottom-up, per-program)       │  ← cb_region_end tracks HERE
          │  Scratch CBs; same addresses reused by    │
          │  each program. NOT cumulative across       │
          │  programs (MAX, not SUM).                 │
          ├──────────────────────────────────────────┤ ← cb_region_end
          │                                           │
          │         FREE HEADROOM (gap)               │
          │  gap = lowest_top_down_addr - cb_region_end
          │                                           │
          ├──────────────────────────────────────────┤ ← lowest_top_down_addr
          │  GLOBALLY-ALLOCATED tensor buffers        │  ← NOT in cb_region_end
          │  (top-down; L1 Buffer defaults to         │
          │   bottom_up=false, i.e. top-down;         │
          │   buffer.cpp line 261)                    │
          │  top_down_size = max_l1 - lowest_top_down │
          └──────────────────────────────────────────┘
high addr [max_l1_size = 1,572,864 B]
```

### True simultaneous L1 occupancy

```
total_occupied  = cb_region_end  +  (max_l1 - lowest_top_down_addr)
free_headroom   = lowest_top_down_addr  -  cb_region_end
```

The `cb_region_end % of L1` numbers in the heatmap **understate** actual L1 pressure;
the true occupied percentage also includes the top-down tensor buffer region.

### What "cumulative" in the heatmap means

The heatmap shows the **MAX `cb_region_end` across all programs** per core — not a SUM.
Different programs' local CBs occupy the **same** bottom-up address range and overwrite
each other between executions. Program caching means the same program always lands at
the same bottom-up addresses across tokens, so `max` over programs is the right peak.

---

## 1. What Does `pid` (program_id) Mean and How to Locate Execution Logic?

### How `pid` is assigned

`program_id` (abbreviated `pid`) is a **process-global monotonic counter** incremented
when each `tt_metal::Program` object is constructed:

```cpp
// tt_metal/impl/program/program.cpp  line 255-261
std::atomic<uint64_t> detail::ProgramImpl::program_counter = 0;

detail::ProgramImpl::ProgramImpl() :
    ...
    id(program_counter++),   // ← this is the pid logged
```

`pid=0` is the first `Program` object created in the entire process. The counter is
never reset and has **nothing to do with OS process IDs**.

### How to map a pid to an operation

1. **Each ttnn operation creates exactly one `Program`** inside its `*_program_factory.cpp`.
   The pid is determined by the order in which those factories are called during the
   first forward pass (before program caching kicks in).

2. **Key source files to trace:**

   | Operation | Program factory |
   |---|---|
   | `nlp_concat_heads_decode` | `ttnn/.../nlp_concat_heads_decode/device/nlp_concat_heads_decode_program_factory.cpp` |
   | `nlp_create_qkv_heads_decode` | `ttnn/.../nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_program_factory.cpp` |
   | SDPA decode | `ttnn/.../scaled_dot_product_attention/device/...` |
   | Linear/matmul | `tt_metal/.../matmul/...` |

3. **Add debug prints** to each factory's top:
   ```cpp
   tt_metal::Program program = tt_metal::CreateProgram();
   log_info(tt::LogMetal, ">>> nlp_concat_heads_decode program id={}", program.get_id());
   ```

4. **Use `runtime_id`** to count re-dispatches: same pid with incrementing `runtime_id`
   means a cached program being re-run (token N, N+1, …).

### Known pid–operation mapping in this run (approximate, run-specific)

| pid | Inferred operation | Cores | `cb_region_end` (local CBs only) |
|---|---|---|---|
| `pid=3,11,13,15,…` | Full-grid ops (RMSNorm, all-reduce, barrier syncs) | 130 | 7–12% |
| `pid=5` | `nlp_concat_heads_decode` local CB overhead | 8 (y=0) | +~30% above SDPA |
| `pid=33` | SDPA decode kernel | 64 (x=0..7, y=0..7) | 48.7% |
| `pid=43` | Reducer sub-program for `nlp_concat_heads_decode` | 1 (at `(0,0)`) | same as pid=5 |
| `pid=47` | `nlp_create_qkv_heads_decode` input-shard CB | 72 (x=0..7, y=0..8) | 46.0% |
| `pid=71,89,99,105,…` | Other ops on the 9-row WQKV grid | 72–117 cores | 22–30% |

> [!NOTE]
> These pid values are **run-specific** — they shift if you add/remove operations,
> change layer count, or restart the process. Always re-derive from a fresh log.

---

## 2. Observed Heatmap (cumulative peak `cb_region_end` per core)

y \ x |  0     1     2     3     4     5     6     7  ||  8     9    10    11    12
------+------------------------------------------------++----------------------------
  0   | 79.5% 79.5% 79.5% 79.5% 79.5% 79.5% 79.5% 79.5% ||22.9% 22.9% 22.9% 22.9% 22.9%
  1   | 48.7% 48.7% 48.7% 48.7% 48.7% 48.7% 48.7% 48.7% ||22.9% 22.9% 22.9% 22.9% 22.9%
  2   | 48.7%  ...                                        ||22.9%  ...
  3   | 48.7%  ...                                        ||  ...
  4   | 48.7%  ...                                        ||  ...
  5   | 48.7%  ...                                        ||  ...
  6   | 48.7%  ...                                        ||  ...
  7   | 48.7% 48.7% 48.7% 48.7% 48.7% 48.7% 48.7% 48.7% ||22.9% 22.9% 22.9% 22.9% 22.9%
------+------------------------------------------------++----------------------------
  8   | 46.0% 46.0% 46.0% 46.0% 46.0% 46.0% 46.0% 46.0% ||22.9% 22.9% 22.9% 22.9% 22.9%
  9   | 12.9% 12.9% 12.9% 12.9% 12.9% 12.9% 12.9% 12.9% ||12.4% 12.4% 12.4% 12.4% 12.4%

`max_l1_size = 1,572,864 B`. Values are `cb_region_end` as a % of L1 (**local CBs only**).

> [!IMPORTANT]
> These percentages show **only the locally-allocated CB stack** (bottom-up).
> The top-down tensor buffer region (globally-allocated) is **not reflected** in these
> numbers. The true occupied % is higher. See §5 for full accounting.

There are **four distinct tiers**:
- **Tier A** (79.5%): x=0..7, y=0 — the "hot row" (local CB peak)
- **Tier B** (48.7%): x=0..7, y=1..7 — standard SDPA 8×8 workers
- **Tier C** (46.0%): x=0..7, y=8 — one row outside the 8×8 SDPA grid
- **Tier D** (22.9%): x=8..12 — all five right-side columns

---

## 3. Why Row 0 (y=0) Is Hotter Than Rows 1–7

### Short answer: `nlp_concat_heads_decode` leaves a larger local CB stack on y=0

The SDPA decode kernel (`pid=33`, 64 cores, all of 8×8) contributes **766,336 B (48.7%)**
to the local CB region on all 64 cores — this is Tier B.

Row 0 reaches **79.5% (1,249,664 B)** in the local CB region because `pid=5`
(`nlp_concat_heads_decode`) runs on 8 cores `(x=0..7, y=0)` and its local kernel config
and scratch CB descriptors stack on top of the SDPA local CBs, pushing the local CB
high-water mark higher on those cores.

The core range selection is determined by `nlp_concat_heads_decode_device_operation.cpp`
line 80:
```cpp
output_core_grid = tt::tt_metal::num_cores_to_corerangeset(
    num_heads, input_tensor.device()->compute_with_storage_grid_size(), true);
```
For 8 Q heads on a single P150, this resolves to `(0,0)` through `(7,0)` — row 0.

### Why the output tensor buffer is NOT part of the 79.5%

`cb_q_output` (`CBIndex::c_16`) is a **globally-allocated CB** backed by the output
tensor buffer (`set_globally_allocated_address(*output.buffer())`). Because globally-
allocated CBs are skipped in `allocate_circular_buffers()`, this tensor buffer resides
**in the top-down region** (at address ≥ `lowest_top_down_addr`) and is **not counted**
in the 79.5%. It shows up instead in `lowest_top_down_addr`.

### Why `(0,0)` (the reducer) has the same local CB `%` as other y=0 cores

`pid=43` (the reducer sub-program) runs only on `(0,0)`. Its local CB allocation does
not exceed `pid=5`'s local CB watermark on `(0,0)`. Since the heatmap takes
`MAX(cb_region_end across programs)`, core `(0,0)` shows the same 79.5% as `(1,0)...(7,0)`.

Both `pid=5` and `pid=43` reference the same `cb_q_output` output buffer — but that
buffer lives top-down and contributes identically (and only once) to `lowest_top_down_addr`
on `(0,0)`; it does **not** double-count in `cb_region_end`.

### Summary: Row-0 local CB decomposition

| Contributor | Program | Cores | Local `cb_region_end` |
|---|---|---|---|
| SDPA decode kernel | `pid=33` | all 64 (x=0..7, y=0..7) | 766,336 B (48.7%) |
| `nlp_concat_heads_decode` overhead | `pid=5` | 8 (x=0..7, y=0) | 1,249,664 B (79.5%) |

> [!IMPORTANT]
> The 79.5% is **local CB only**. The `nlp_concat_heads_decode` output tensor buffer
> additionally occupies the top-down region (quantified by `max_l1 - lowest_top_down_addr`),
> which is NOT included in these percentages but IS part of the true L1 footprint.

---

## 4. Why Row 8 (y=8, Outside the 8×8 Grid) Has 46.0% Usage

### Short answer: `nlp_create_qkv_heads_decode` operand grid spans 9 rows

Program `pid=47` allocates 72 cores: `x=0..7, y=0..8` (8 cols × 9 rows = 72 cores).
Its local `cb_region_end = 722,944 B (46.0%)`.

This is the **input tensor shard** for `nlp_create_qkv_heads_decode`, which reads the
fused WQKV matrix output. The WQKV matmul uses 9 rows for `dim=4096` on Blackhole,
so the input shard resides on `y=0..8`. The `nlp_create_qkv_heads_decode` program
registers a local CB on those 72 source cores — even though SDPA only uses y=0..7.

Similarly, `pid=71, pid=99, pid=105` follow the same pattern.

### Does row 8's CB serve itself, or is it read by others via NoC?

**Row 8 is a passive data source.** Its local CB holds a shard of the WQKV output.
The `nlp_create_qkv_heads_decode` reader kernel running on the 8×8 output cores issues
`noc_async_read()` to pull tiles from row 8's L1:

```cpp
noc_async_read(
    get_noc_addr(in0_mcast_noc_x[qkv_x], in0_mcast_noc_y[qkv_y], q_start_addr)
        + in_tile_offset_by_head,
    q_write_addr, SUBTILE_LINE_BYTES);
```
Row 8 cores do **not** run SDPA decode. They run the WQKV projection matmul that
produced the shard, and then their L1 is read remotely by the 8×8 SDPA/QKV cores.

> [!NOTE]
> Row 8 is a **passive participant**: its 46.0% local CB usage comes from the WQKV
> matmul output shard registered there. The 8×8 SDPA cores consume it over NoC.

---

## 5. Why the Right-Side Columns (x=8..12) Have 22.9% Usage

Multiple programs with `cores=130` span the full 13×10 grid (e.g. `pid=3,11,13,15,19`,
`pid=89`). These are RMSNorm, all-reduce syncs, FFN projections, etc. that use the full
`compute_with_storage_grid_size = (13, 10)`. Their local CBS peak at 22.9% on right-side
cores.

---

## 6. True L1 Occupancy and KV Mirror Headroom

### What the crash check actually does (per-program, not cumulative)

The clash detection in `validate_circular_buffer_region()`:
```cpp
if (lowest_address.value() < cb_region_end) {
    TT_THROW("Statically allocated circular buffers in program {} clash with L1 buffers...");
}
```
is evaluated **once per program dispatch**, using **that program's own `cb_region_end`** against
the allocator's **current `lowest_top_down_addr`** at the time that specific program runs.

The cumulative heatmap (which takes `MAX(cb_region_end)` per core and `MIN(lowest_top_down_addr)`
per core across ALL programs) produces a **pessimistic cross-program view** that can show
"total > 100%" without any actual crash, because the max-CB and min-top-down come from
**different programs running at different moments** with different top-down allocations alive.

### Empirical data: kv_window=512 profiling run (all 32 layers)

Two profiling runs were conducted and compared:

| Condition | `lowest_top_down_addr` during `nlp_concat` | KV mirror allocated footprint |
|---|---|---|
| KV mirror **disabled** (original run) | 1,568,512 B | — |
| KV mirror **window=512** (new run) | 1,289,984 B | **272 KiB** added |

#### Per-program crash safety (KV window=512 run)

| Program | Role | `cb_region_end` | `lowest_top_down_addr` | **gap** | Status |
|---|---|---|---|---|---|
| `pid=41` (SDPA decode) | 8×8 grid workers | 766,336 B | 1,289,984 B | **511 KiB** | ✅ safe |
| `pid=5/53` (`nlp_concat_heads_decode`) | y=0 row only | 1,249,664 B | 1,289,984 B | **39 KiB** | ✅ barely safe |

- **SDPA** has a large gap (511 KiB) — no collision risk even with window=512.
- **`nlp_concat_heads_decode`** is the tight one: adding window=512 shrinks its isolated gap from
  311 KiB → 39 KiB. It remains positive, so no crash, but this is the true headroom limit.

> [!IMPORTANT]
> The KV mirror payload limits calculations (253 KiB for 512 tokens) match the empirical allocation
> footprint (272 KiB with padding overhead). This 272 KiB chunk eats almost all of `nlp_concat`'s
> 311 KiB headroom, leaving only **39 KiB** on row 0. This is the real safety margin.

### Why does the cumulative heatmap show "110%" yet no crash?

| Measurement | Value | Source |
|---|---|---|
| Cumulative `MAX(cb_region_end)` on row-0 | 1,249,664 B (79.5%) | from `nlp_concat` (pid=5/53) |
| Cumulative `MIN(lowest_top_down_addr)` on row-0 | 1,079,360 B | from a different program that allocates more top-down tensors simultaneously |
| Cross-program "gap" | **−170,304 B** (−167 KiB) | **NOT a real per-program conflict** |

The `−170 KiB` comes from mixing: `nlp_concat`'s high `cb_region_end` (1,249,664) against
the lowest global `lowest_top_down_addr` (1,079,360) observed when some OTHER, lower-CB
program runs concurrently with more top-down allocations. At the moment `nlp_concat` actually
executes, the `lowest_top_down_addr` is 1,289,984 → gap = +39 KiB → no crash.

> [!WARNING]
> The cumulative heatmap's `total_occupied_pct > 100%` does **not** indicate an actual
> memory overflow. It is an artifact of mixing max and min values across different programs.
> Per-program safety must be assessed from `per_program_core_map.json`, not the heatmap.

### Actual headroom budget for additional KV mirror growth

The binding constraint is `nlp_concat_heads_decode` on row 0 with its **39 KiB remaining gap**.

| Region | `nlp_concat` gap (no KV mirror) | KV mirror footprint (window=512) | Remaining headroom |
|---|---|---|---|
| Row 0 (y=0, binding for `nlp_concat`) | 311 KiB | ~272 KiB | **39 KiB** |
| SDPA workers on row 0 (pid=41) | ~783 KiB | ~272 KiB | **511 KiB** |
| Right-side cores (x=8..12) | 973 KiB | ~272 KiB | **~701 KiB** |

39 KiB headroom means increasing the window size even slightly (beyond 512) will immediately exhaust row 0.

> [!CAUTION]
> With window=512 the system works, but the `nlp_concat` gap on row 0 is only **39 KiB**.
> The baseline KV footprint calculation (253 KiB payload) effectively leaves no room for scaling.
> Any further growth of window size on row-0 cores guarantees a collision on `nlp_concat`.
> Preferentially routing larger KV windows to idle right-side
> cores (x=8..12, ~701 KiB headroom remaining) remains the correct path for larger window support.

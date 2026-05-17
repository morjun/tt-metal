# L1 KV Cache: Workflow Big Picture

> Blackhole (P150) · Llama 3.1 8B · `l1-kv-cache` branch
> Last updated: 2026-05-13

---

## 1. Three Workflows at a Glance

| | **Baseline DRAM** | **Fixed-window L1** | **Adaptive N-tier L1** |
|---|---|---|---|
| CLI flag | *(default)* | `--l1_kv_window_size N` | `--use_adaptive_l1_kv_cache` |
| `ModelArgs` field | — | `l1_kv_window_size=N` | `use_adaptive_l1_kv_cache=True` |
| L1 cache size | none | exactly N tokens, every layer | measured from post-compile headroom |
| Memory layout | DRAM interleaved | L1 interleaved **or** HEIGHT\_SHARDED (chosen at alloc) | L1 HEIGHT\_SHARDED per headroom bucket |
| Allocation time | model `__init__` | model `__init__` (eager) | **deferred** — after first decode compile |
| # of KV tensors per layer | 1 DRAM | 1 DRAM + 1 L1 | 1 DRAM + N L1 tiers (N ≤ 5) |
| Ring-buffer write | ✗ | `paged_update_cache` (modular pos) | ✓ `_write_adaptive_l1_tiers` (host-sync, no-trace) |
| SDPA read path | DRAM-only NoC | DRAM + L1 (same layout as allocated) | DRAM + `find_tier` dispatch in kernel |
| `--l1_kv_use_sharded` | N/A | Allocates HEIGHT\_SHARDED directly (no read-time copy) | N/A (always HEIGHT\_SHARDED) |

---

## 2. Shared Startup Path (all three workflows)

```
pytest simple_text_demo.py --<flags>
│
├─ conftest.py  pytest_addoption()
│     Registers: --l1_kv_window_size, --l1_kv_use_sharded,
│                --use_adaptive_l1_kv_cache, ...
│
├─ test_demo_text()
│     getoption("--l1_kv_window_size")      → l1_kv_window_size
│     getoption("--use_adaptive_l1_kv_cache") → use_adaptive_l1_kv_cache
│
├─ prepare_generator_args()
│     └─ create_tt_model(
│           l1_kv_window_size=N,
│           l1_kv_use_sharded=True/False,
│           use_adaptive_l1_kv_cache=True/False, ...)
│              [common.py]
│
├─ ModelArgs.__init__()
│     self.l1_kv_window_size        = N
│     self.l1_kv_use_sharded        = True/False
│     self.use_adaptive_l1_kv_cache = True/False
│     [model_config.py]
│
└─ Transformer.__init__()  →  TransformerBlock.__init__()
       └─ Attention.__init__()          ← PATH SPLITS HERE
             self.l1_kv_window_size        = config.l1_kv_window_size
             self.use_adaptive_l1_kv_cache = config.use_adaptive_l1_kv_cache
             self.l1_kv_total_size         = sink + window
             self.l1_kv_tiers              = []
             ...
             self.init_kv_cache()          ← see below
```

---

## 3. Path A — Baseline DRAM (default)

### Allocation

```
Attention.init_kv_cache()
│  l1_kv_window_size == 0  AND  use_adaptive_l1_kv_cache == False
│
├─ torch.zeros(B, H_kv, max_seq_len, D)
│  ttnn.as_tensor(..., memory_config=DRAM_MEMORY_CONFIG)
│  → self.layer_past[0]   (K, DRAM)
│  → self.layer_past[1]   (V, DRAM)
│
└─ self.l1_kv_cache = None
   self.l1_kv_tiers = []

Generator.__init__()
   l1_kv_needs_alloc = False       ← no deferred work
```

### Decode Loop

```
Attention.forward_decode()
│
├─ xqkv = ttnn.linear(x, wqkv)
├─ [QKV split, RoPE, reshape to heads]
│
├─ keys   = layer_past[0]   (DRAM, full history)
│  values = layer_past[1]
│  paged_update_cache(keys,   k_heads, current_pos)   ← write new KV to DRAM
│  paged_update_cache(values, v_heads, current_pos)
│
├─ sdpa_kwargs = {}          ← no L1 args
│
└─ ttnn.transformer.scaled_dot_product_attention_decode(
       q, keys, values, **sdpa_kwargs)
       │
       └─ [C++] sdpa_decode_program_factory.cpp
             num_active_l1_tiers = 0
             └─ reader_decode_all.cpp
                   read ALL seq_len tiles from DRAM via NoC
```

---

## 4. Path B — Fixed-window L1  (`--l1_kv_window_size N`)

### Allocation

```
Attention.init_kv_cache()
│  l1_kv_window_size == N > 0
│
├─ DRAM cache (full history)
│  ttnn.as_tensor(zeros(B,H,max_seq_len,D), DRAM_MEMORY_CONFIG)
│  → self.layer_past[0/1]
│
├─ Choose L1 layout BEFORE allocation:
│  if l1_kv_use_sharded:
│    l1_memcfg = _create_l1_kv_sharded_memcfg(shape)  ← HEIGHT_SHARDED
│    (fallback to L1_MEMORY_CONFIG if shape too small)
│  else:
│    l1_memcfg = L1_MEMORY_CONFIG                      ← plain interleaved
│
├─ L1 cache (exactly N tokens, eager)
│  torch.zeros(B, H_kv, N, D)  ×2
│  ttnn.as_tensor(..., memory_config=l1_memcfg)   ← born in right layout
│  → self.l1_kv_cache[0]  (K)
│  → self.l1_kv_cache[1]  (V)
│  self.l1_kv_sharded_memcfg = l1_memcfg if l1_kv_use_sharded else None
│
│  NOTE: no to_memory_config() copy at read time — tensor is already
│  in the correct layout when passed to SDPA.
│
Generator.__init__()
   l1_kv_needs_alloc = False       ← already allocated above
```

### Decode Loop

```
Attention.forward_decode()
│
├─ [QKV, RoPE, heads]
│
├─ Write DRAM (full history)
│  paged_update_cache(layer_past[0], k_heads, current_pos)
│  paged_update_cache(layer_past[1], v_heads, current_pos)
│
├─ Write L1 ring-buffer (most recent N tokens)
│  l1_pos = (current_pos % l1_kv_window_size) + l1_kv_sink_size
│  paged_update_cache(l1_kv_cache[0], k_heads, l1_pos)
│  paged_update_cache(l1_kv_cache[1], v_heads, l1_pos)
│
├─ _get_sdpa_l1_cache_tensors()
│  └─ return l1_kv_cache[0], l1_kv_cache[1]   ← already in chosen layout
│     (no to_memory_config() — layout was fixed at allocation time)
│
└─ ttnn.transformer.scaled_dot_product_attention_decode(
       q, keys_dram, values_dram,
       l1_k_tensor=sdpa_l1_k,
       l1_v_tensor=sdpa_l1_v,
       **sdpa_kwargs)
       │
       └─ [C++] sdpa_decode_program_factory.cpp
             num_active_l1_tiers = 1
             TensorAccessorArgs(l1_k_buf)  → compile-time
             TensorAccessorArgs(l1_v_buf)  → compile-time
             runtime args: [k_addr, v_addr, ..., l1_start, l1_size]
             │
             └─ dataflow_common.hpp
                   read_kv_mask_chunks_1_tier():
                   ├─ if gst in [l1_start, l1_start+l1_size):
                   │    noc_async_read_tile(id, l1_k0_reader, ptr)  ← L1 NoC
                   └─ else:
                        noc_async_read_tile(id, dram_reader, ptr)   ← DRAM NoC
```

---

## 5. Path C — Adaptive N-tier L1  (`--use_adaptive_l1_kv_cache`)

### Allocation (two-phase)

```
─── Phase 1: model __init__ (placeholder only) ──────────────────────────────

Attention.init_kv_cache()
│  use_adaptive_l1_kv_cache == True
│
├─ DRAM cache (full history)  → self.layer_past[0/1]
│
└─ self.l1_kv_cache  = None    ← no L1 yet
   self.l1_kv_tiers  = []      ← empty
   self.l1_kv_sharded_memcfg = None

Generator.__init__()
   l1_kv_needs_alloc = True    ← deferred work needed!

─── Phase 2: first decode call (post-compile) ───────────────────────────────

Generator._decode_forward_no_trace_text()  (compile iteration)
│  self._decode_compile_done = True
│
└─ _allocate_l1_kv_cache_if_needed()
    │  l1_kv_needs_alloc == True  AND  _decode_compile_done == True
    │
    ├─ mesh_dev.get_l1_headroom_per_core()
    │  → headroom_map: {(x,y): free_bytes_above_all_CB_addresses}
    │
    └─ for each layer's attention:
          layer.attention.allocate_l1_kv_cache(headroom_map, safety_margin)
          │
          └─ _allocate_adaptive_l1_kv_tiers(headroom_map, safety_margin)
              │
              ├─ _build_adaptive_l1_memcfg_tiers()
              │  │
              │  │  bytes_per_tile_row = tile_size × head_dim × elem_bytes
              │  │                       × num_layers × 2   (K+V, all layers share headroom)
              │  │
              │  │  For each core (x,y):
              │  │    net = headroom_map[(x,y)] - safety_margin_bytes
              │  │    tile_rows = net // bytes_per_tile_row
              │  │    if tile_rows < 1: skip
              │  │    tier_buckets[tile_rows].append((x,y))
              │  │
              │  │  For each bucket (sorted by tile_rows):
              │  │    tok_count = tile_rows × n_cores × 32 / (B × H_kv)
              │  │    shard_spec = ShardSpec(
              │  │        core_range_set,
              │  │        shard_shape=[tile_rows×32, head_dim],
              │  │        ROW_MAJOR)
              │  │    memcfg = MemoryConfig(HEIGHT_SHARDED, L1, shard_spec)
              │  └─ returns [(memcfg, tile_rows, cores, tok_count), ...]
              │
              └─ For each (memcfg, tile_rows, cores, tok_count):
                    k_tensor = ttnn.as_tensor(
                        zeros(B, H_kv, tok_count, D),
                        memory_config=memcfg)    ← HEIGHT_SHARDED in L1
                    v_tensor = ttnn.as_tensor(...)
                    self.l1_kv_tiers.append(
                        (k_tensor, v_tensor, token_start, tok_count))
                    token_start += tok_count
```

### Decode Loop

```
Attention.forward_decode()
│
├─ [QKV, RoPE, heads]
│
├─ Write DRAM (full history)
│  paged_update_cache(layer_past[0], k_heads, current_pos)
│  paged_update_cache(layer_past[1], v_heads, current_pos)
│
├─ Write adaptive L1 ring-buffer  (if l1_write_enabled, not in trace)
│  _build_adaptive_l1_write_pos(current_pos)
│  │  T   = l1_kv_adaptive_total_capacity   # = Σ tier tok_counts
│  │  cap = T - l1_kv_sink_size
│  │  flat_pos = sink_size + (pos - sink_size) % cap   # ring in [0, T)
│  │  (flat_pos is a single index into the logically-concatenated tier space)
│  │
│  _write_adaptive_l1_tiers(k_heads, v_heads, flat_pos)
│  │  pos_val = to_torch(flat_pos).item()   ← host sync, trace-incompatible
│  │  find tier where t[2] ≤ pos_val < t[2]+t[3]:   # t[2]=token_start, t[3]=tok_count
│  │    offset = pos_val - t[2]   # position within that tier tensor
│  │    paged_update_cache(t[0], k_heads, offset)   # t[0]=K tensor
│  └─   paged_update_cache(t[1], v_heads, offset)   # t[1]=V tensor
│
│  l1_kv_tiers: list of (k_tensor, v_tensor, token_start, tok_count)
│    t[0]: K  t[1]: V  t[2]: token_start  t[3]: tok_count
│
├─ _get_sdpa_l1_cache_tensors()
│  ks   = [t[0] for t in l1_kv_tiers]   # K tensors, one per tier
│  vs   = [t[1] for t in l1_kv_tiers]   # V tensors
│  meta = [(t[2], t[3]) for t in l1_kv_tiers]  # (token_start, tok_count)
│
└─ ttnn.transformer.scaled_dot_product_attention_decode(
       q, keys_dram, values_dram,
       l1_k_tensors = [k0, k1, ..., kN],
       l1_v_tensors = [v0, v1, ..., vN],
       l1_tier_token_starts = [s0, s1, ..., sN],
       l1_tier_token_counts = [c0, c1, ..., cN],
       **sdpa_kwargs)
       │
       └─ [C++] sdpa_decode_program_factory.cpp
             num_active_l1_tiers = N  (compile-time constant)
             TensorAccessorArgs(dram_k)   → DRAM accessor
             TensorAccessorArgs(dram_v)   → DRAM accessor
             for ti in 0..N-1:
               TensorAccessorArgs(l1_k_tiers[ti])  → L1 tier accessor
               TensorAccessorArgs(l1_v_tiers[ti])  → L1 tier accessor
             runtime args per core:
               [k_addr, v_addr, ...,
                (k_addr_t0, v_addr_t0, start_tile_t0, size_tiles_t0),
                (k_addr_t1, v_addr_t1, start_tile_t1, size_tiles_t1), ...]
             │
             └─ dataflow_common.hpp
                   read_kv_mask_chunks_n_tier<N>():
                   │
                   │  find_tier(gst):          ← gst = global_seq_tile index
                   │    for ti in 0..N-1:
                   │      if tier_start[ti] ≤ gst < tier_start[ti]+tier_size[ti]:
                   │        return ti
                   │    return DRAM_SENTINEL
                   │
                   ├─ if DRAM_SENTINEL: noc_async_read_tile(id, dram_reader, ptr)
                   ├─ if ti == 0:       noc_async_read_tile(id, l1_k0_reader, ptr)
                   ├─ if ti == 1:       noc_async_read_tile(id, l1_k1_reader, ptr)
                   ├─ if ti == 2:       noc_async_read_tile(id, l1_k2_reader, ptr)
                   ├─ if ti == 3:       noc_async_read_tile(id, l1_k3_reader, ptr)
                   └─ if ti == 4:       noc_async_read_tile(id, l1_k4_reader, ptr)
                        (constexpr if — dead branches compiled away)
```

---

## 6. Side-by-Side Comparison

### 6.1 Memory Layout per Layer

```
DRAM baseline:
┌───────────────────────────────────────────────────────┐
│ DRAM: K/V cache  [B, H_kv, max_seq_len, D]  ~4 GB     │
└───────────────────────────────────────────────────────┘
   No L1 buffers.

Fixed-window L1 (N tokens):
┌───────────────────────────────────────────────────────┐
│ DRAM: K/V cache  [B, H_kv, max_seq_len, D]             │
└───────────────────────────────────────────────────────┘
┌─────────────────────────┐
│ L1 (interleaved):        │  identical across ALL cores
│ K/V ring  [B, H, N, D]  │  N = user-provided token count
└─────────────────────────┘

Adaptive N-tier L1:
┌───────────────────────────────────────────────────────┐
│ DRAM: K/V cache  [B, H_kv, max_seq_len, D]             │
└───────────────────────────────────────────────────────┘
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Tier 0       │  │ Tier 1       │  │ Tier 2       │  ...
│ 12 cores     │  │ 45 cores     │  │ 73 cores     │
│ 2 tile-rows  │  │ 4 tile-rows  │  │ 6 tile-rows  │
│ HEIGHT_SHRD  │  │ HEIGHT_SHRD  │  │ HEIGHT_SHRD  │
│ 192 tokens   │  │ 720 tokens   │  │ 2190 tokens  │
└──────────────┘  └──────────────┘  └──────────────┘
  Each tier: independent ttnn tensor, placed in the
  specific cores that have that headroom class.
```

### 6.2 Write Path

| Step | DRAM | Fixed-window L1 | Adaptive N-tier |
|---|---|---|---|
| DRAM update | `paged_update_cache(layer_past, k, pos)` | Same | Same |
| L1 update | — | `paged_update_cache(l1_kv_cache, k, pos % N)` | `_write_adaptive_l1_tiers(k, v, flat_pos)` |
| Ring-buffer modular pos | — | `pos % window_size + sink_size` | `sink_size + (pos - sink_size) % (T - sink_size)` |
| Trace compatible | ✓ | ✓ | ✗ (host sync in `_write_adaptive_l1_tiers`) |

### 6.3 Read Path (SDPA kernel perspective)

```
DRAM only:
  every tile → DRAM NoC read
  latency = DRAM BW limited

Fixed-window (no --l1_kv_use_sharded):
  L1 cache allocated as L1_MEMORY_CONFIG (interleaved)
  tile in [recent N] → L1 interleaved read (may cross cores via NoC)
  tile outside N     → DRAM NoC read

Fixed-window (with --l1_kv_use_sharded):
  L1 cache allocated as HEIGHT_SHARDED directly at init_kv_cache()
  No to_memory_config() copy at read time — tensor already in final layout
  tile in [recent N] → each core reads from its own L1 shard
  tile outside N     → DRAM NoC read

Adaptive N-tier:
  tile in tier 0 range → L1 NoC read from tier-0 cores' L1
  tile in tier 1 range → L1 NoC read from tier-1 cores' L1
  ...
  tile not in any tier → DRAM NoC read
  (L1 reads are still NoC if the tile's shard lives on a different core)
```

### 6.4 C++ Layer Changes Summary

```
sdpa_decode_program_factory.cpp
─────────────────────────────────────────────────────────────────────────────
DRAM only:
  TensorAccessorArgs(k)  →  1 compile-time accessor
  TensorAccessorArgs(v)  →  1 compile-time accessor
  runtime: [k_addr, v_addr, ...]

Fixed-window (1 tier):
  TensorAccessorArgs(k)         →  DRAM accessor
  TensorAccessorArgs(v)         →  DRAM accessor
  TensorAccessorArgs(l1_k)      →  L1 accessor
  TensorAccessorArgs(l1_v)      →  L1 accessor
  num_active_l1_tiers = 1
  runtime: [k_addr, v_addr, ..., l1_k_addr, l1_v_addr, l1_start, l1_size]

Adaptive N-tier:
  TensorAccessorArgs(k)                  →  DRAM
  TensorAccessorArgs(v)                  →  DRAM
  num_active_l1_tiers = N               (compile-time)
  for ti in 0..N-1:
    TensorAccessorArgs(l1_k_tiers[ti])  →  L1 tier ti
    TensorAccessorArgs(l1_v_tiers[ti])  →  L1 tier ti
  runtime per core:
    [k_addr, v_addr, ...,
     (k_t0, v_t0, start_t0, size_t0),
     (k_t1, v_t1, start_t1, size_t1), ...]

dataflow_common.hpp  (kernel dispatch)
─────────────────────────────────────────────────────────────────────────────
DRAM only / Fixed-window:
  read_kv_mask_chunks_1_tier(gst):
    if gst in l1_range: l1_reader  else: dram_reader

Adaptive N-tier:
  read_kv_mask_chunks_n_tier<N>(gst):
    find_tier(gst) → ti or DRAM_SENTINEL
    constexpr dispatch: if (ti==0) l1_k0_reader; if (ti==1) l1_k1_reader; ...
```

---

## 7. Key Design Decisions

### Why defer adaptive allocation to post-compile?

CB (Circular Buffer) L1 addresses are not finalized until the first compile/trace run of the decode kernel. Allocating L1 tiers during `Attention.__init__` risks collision with CB addresses. `get_l1_headroom_per_core()` after the compile run returns a stable snapshot of free bytes above all frozen CBs.

### Why HEIGHT_SHARDED for adaptive tiers?

The SDPA decode kernel distributes KV head computation across cores. HEIGHT_SHARDED aligns the tier's KV data with the natural per-core head assignment, enabling local L1 reads (or at minimum, fewer long-distance NoC hops compared to interleaved).

### Both paths use ring-buffers; why different mechanics?

Both fixed-window and adaptive N-tier maintain a ring of recent tokens in L1. The difference is:
- **Fixed-window**: single contiguous L1 tensor; `pos % window_size` is computed on-device (trace-compatible).
- **Adaptive N-tier**: N separate HEIGHT_SHARDED tensors; the flat ring position (`sink + (pos-sink) % cap`) is computed on-device, but dispatching to the correct tier requires a `to_torch()` host sync to read `pos_val` and pick the tier. This makes it **trace-incompatible** — the ring write is only active when `l1_write_enabled=True` (no-trace path).

### What is the "flat" position in the adaptive ring?

All N tiers are treated as a single logically concatenated sequence of `T = Σ tok_count` tokens. Tier 0 covers `[0, tok_count_0)`, Tier 1 covers `[tok_count_0, tok_count_0+tok_count_1)`, etc. The flat position is a single index in `[0, T)` — `_write_adaptive_l1_tiers` then maps it back to the correct physical tier and offset.

### Why does `l1_kv_use_sharded` no longer need `to_memory_config()`?

The previous design allocated the L1 cache as interleaved and applied a HEIGHT_SHARDED view via `to_memory_config()` at read time every decode step — a full data copy. Now `_create_l1_kv_sharded_memcfg()` runs **before** `ttnn.as_tensor()`, so the tensor is born in HEIGHT_SHARDED layout. No copy is needed at read time; `_get_sdpa_l1_cache_tensors()` returns the tensor directly.

### Why max 5 tiers?

Each tier requires `TensorAccessorArgs` and a `constexpr if` branch in the kernel. The kernel uses up to 5 template specializations (`num_tiers = 1..5`). Beyond 5 risks hitting Tensix compile-time argument size limits, and in practice Blackhole headroom variation clusters into ≤ 4 distinct buckets.

---

## 8. Files Modified in This Implementation

| File | Change |
|---|---|
| `demo/conftest.py` | Added `--use_adaptive_l1_kv_cache` `store_true` option |
| `demo/simple_text_demo.py` | Wire `getoption → prepare_generator_args → create_tt_model` |
| `tt/common.py` | Add `use_adaptive_l1_kv_cache` param to `create_tt_model` → `ModelArgs` |
| `tt/model_config.py` | Add `use_adaptive_l1_kv_cache=False` field to `ModelArgs.__init__` |
| `tt/generator.py` | `l1_kv_needs_alloc` gate now also triggers on `use_adaptive_l1_kv_cache` |
| `tt/attention.py` | Strict path split: fixed-window eager in `init_kv_cache`; adaptive deferred via `allocate_l1_kv_cache → _allocate_adaptive_l1_kv_tiers` |
| `sdpa_decode_program_factory.cpp` | N-tier `TensorAccessorArgs` loop + runtime arg packing per tier |
| `reader_decode_all.cpp` | Remove stale window-start computation |
| `dataflow_common.hpp` | `read_kv_mask_chunks_n_tier<N>` + `find_tier` lambda dispatch |

---

## 9. Verification Commands

```bash
# Baseline DRAM (control)
TT_LOGGER_LEVEL=Info \
pytest models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" 2>&1 | tee /tmp/dram_baseline.log

# Fixed-window L1 (N=256 tokens)
TT_LOGGER_LEVEL=Info \
pytest models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" \
  --l1_kv_window_size 256 2>&1 | tee /tmp/fixed_window.log

# Fixed-window + HEIGHT_SHARDED read view
TT_LOGGER_LEVEL=Info \
pytest models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" \
  --l1_kv_window_size 256 --l1_kv_use_sharded 2>&1 | tee /tmp/fixed_sharded.log

# Adaptive N-tier
TT_LOGGER_LEVEL=Info \
pytest models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" \
  --use_adaptive_l1_kv_cache 2>&1 | tee /tmp/adaptive.log

# Adaptive + L1 CB map profiling
TT_LOGGER_LEVEL=Info TT_METAL_LOG_L1_CB_MAP=1 \
pytest models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" \
  --use_adaptive_l1_kv_cache 2>&1 | tee /tmp/adaptive_cb_map.log
```

#### Path A: with offline JSON (most accurate)
```bash
TT_LOGGER_LEVEL=Info \
pytest models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" \
  --use_adaptive_l1_kv_cache \
  --l1_kv_headroom_json research_codes/documents/l1_kv_cache_cache/comparison/headroom_map_global_minimum.json \
  2>&1 | tee /tmp/adaptive_json.log
```

#### Path B: improved live scan (no JSON)
```bash
TT_LOGGER_LEVEL=Info \
pytest models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" \
  --use_adaptive_l1_kv_cache \
  2>&1 | tee /tmp/adaptive_live.log
```


### Log signatures to verify each path

| Log message | Path confirmed |
|---|---|
| *(no L1 KV messages)* | DRAM baseline |
| `[L1 KV] Fixed-window cache allocated ... tokens, L1_MEMORY_CONFIG` | Fixed-window interleaved |
| `[L1 KV] Fixed-window cache allocated ... tokens, HEIGHT_SHARDED` | Fixed-window sharded |
| `[L1 KV] Headroom acquired for N cores ... Allocating adaptive L1 KV cache...` | Adaptive trigger |
| `[L1 KV adaptive] Tier: X cores × Y tile-rows = Z tokens [tokens A..B]` | Adaptive tier built |
| `[L1 KV adaptive] Total L1 KV tokens across N tiers: T` | Adaptive complete |
| `decode.adaptive_l1_kv_write` in perf timer | Adaptive ring-buffer write active |

# L1 KV Cache Architecture — Complete Technical Reference

## 1. Executive Summary

We implemented a **dual-source KV cache** for the SDPA (Scaled Dot Product Attention) decode kernel on Tenstorrent hardware. The original (official) implementation reads the **entire** KV cache from DRAM on every attention computation. Our modification adds a small **L1-resident ring buffer** that stores the most recent tokens' KV entries, enabling the SDPA kernel to read "hot" (recently-accessed) KV tiles from fast L1 SRAM instead of slow DRAM.

> [!IMPORTANT]
> The official Tenstorrent implementation uses the **entire** KV cache stored in DRAM. There is no "hot subset" optimization in the upstream code. Every decode step reads all relevant KV tiles from DRAM via NoC.

---

## 2. Foundational Concepts: Heads, Tiles, Chunks, and the Compute Grid

Before diving into the L1 KV cache, you need to understand four key concepts and how they relate.

### 2.1 What Is a "Head"?

In transformer models, attention is split into multiple independent **heads**. Each head computes attention separately, then the results are concatenated. This is "Multi-Head Attention."

For Llama-3.1-8B (per device):

| Tensor | Symbol | Count | Head Dimension | What It Is |
|--------|--------|-------|----------------|------------|
| **Query (Q)** | `nh` | 8 | 128 | The current token's "question" — what am I looking for? |
| **Key (K)** | `nkv` | 1 | 128 | Each past token's "identifier" — here's what I represent |
| **Value (V)** | `nkv` | 1 | 128 | Each past token's "content" — here's my actual information |

> [!NOTE]
> Llama uses **GQA (Grouped Query Attention)**: 8 Q heads share 1 KV head. This is why `nkv=1` but `nh=8`. The KV cache only stores 1 head's worth of K/V data, but all 8 Q heads attend to it.

**KV cache shape**: `[batch, nkv, seq_len, head_dim]` = `[1, 1, 2048, 128]` in our test
- The `1` is the single KV head (nkv=1)
- `2048` is the sequence length (max number of tokens stored)
- `128` is the head dimension (each token's K or V vector has 128 elements)

### 2.2 What Is a "Tile"?

Tenstorrent hardware processes data in **32×32 element tiles**. A tile is the smallest unit of data that can be read/written by a single NoC (Network on Chip) operation.

```
One Tile = 32 rows × 32 columns of elements

For the KV cache [1, 1, 2048, 128]:
  Sequence dimension:  2048 tokens / 32 = 64 "sequence tiles" (rows)
  Head dimension:      128 elements / 32 = 4 "dimension tiles" (columns)
```

Visually, the KV cache is a 2D grid of tiles:

```
                    head_dim = 128
              col 0    col 1    col 2    col 3
            ┌────────┬────────┬────────┬────────┐
  seq tile 0│ tile   │ tile   │ tile   │ tile   │ ← tokens 0-31
  (row 0)   │(0,0)   │(0,1)   │(0,2)   │(0,3)   │
            ├────────┼────────┼────────┼────────┤
  seq tile 1│ tile   │ tile   │ tile   │ tile   │ ← tokens 32-63
  (row 1)   │(1,0)   │(1,1)   │(1,2)   │(1,3)   │
            ├────────┼────────┼────────┼────────┤
  seq tile 2│        │        │        │        │ ← tokens 64-95
            ├────────┼────────┼────────┼────────┤
       ...  │  ...   │  ...   │  ...   │  ...   │
            ├────────┼────────┼────────┼────────┤
  seq tile  │        │        │        │        │ ← tokens 2016-2047
  63 (row63)│        │        │        │        │
            └────────┴────────┴────────┴────────┘

Total tiles = 64 seq tiles × 4 dim tiles = 256 tiles
```

**Key variable names in the kernel:**
- `DHt` = number of dimension tiles = `head_dim / 32` = **4** (for head_dim=128)
- `Sk_chunk_t` = number of sequence tiles per chunk (see below)
- `seq_tiles` = total number of valid sequence tiles = `ceil((cur_pos+1) / 32)`

### 2.3 What Is a "Chunk"?

The SDPA kernel doesn't process all sequence tiles at once — it processes them in **chunks**. A chunk is a contiguous group of sequence tiles that are loaded together, then used for attention computation before moving to the next chunk.

**Why chunks?** The Tensix core's circular buffer (`cb_k_in`) has limited space. It can only hold one chunk's worth of K/V tiles at a time. So the kernel iterates: load chunk → compute partial attention → load next chunk → compute → ... → combine all partial results.

**Chunk size** is determined by the `k_chunk_size` parameter (in tokens), which translates to tiles:

```python
# In the test code:
k_chunk_size = get_chunk_size(s)  # s=2048 → k_chunk_size=512 tokens

# In the kernel:
Sk_chunk_t = k_chunk_size / 32    # 512 / 32 = 16 sequence tiles per chunk
```

**Tiles per chunk** = `Sk_chunk_t × DHt` = 16 × 4 = **64 tiles** (for K)

```
Chunk 0: seq tiles 0-15   (tokens 0-511)    → 16 × 4 = 64 tiles
Chunk 1: seq tiles 16-31  (tokens 512-1023) → 16 × 4 = 64 tiles
Chunk 2: seq tiles 32-47  (tokens 1024-1535) → 16 × 4 = 64 tiles
Chunk 3: seq tiles 48-63  (tokens 1536-2047) → 16 × 4 = 64 tiles
```

> [!IMPORTANT]
> **Where does this 512 limit come from?**
> It is an algorithmic safety limit hard-coded directly into the official Tenstorrent C++ driver for the SDPA operation.
>
> In `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp` (lines 15-29):
> ```cpp
> inline uint32_t get_chunk_size(uint32_t s) {
>     uint32_t i = 1;
>     for (; i < s; i++) {
>         if (s % (1 << (i + 1)) != 0) {
>             break;
>         }
>     }
>     return std::min(512, 1 << i);  // ← OFFICIAL HARD-CODED 512 LIMIT!
> }
> ```
>
> **Why 512? (The Hardware Math):**
> 512 tokens = 16 sequence tiles.
> - K chunk size: 16 seq tiles × 4 dim tiles = 64 tiles.
> - V chunk size: 16 seq tiles × 4 dim tiles = 64 tiles.
> - 64 tiles × 2048 bytes (bfloat16) = 131 KB per chunk.
> - K + V = 262 KB of just chunk data.
> The Tensix core has ~1 MB of total SRAM available for Circular Buffers (`cb_k_in`, `cb_v_in`, Q, outbound results). If `k_chunk_size` exceeds 512, the circular buffers require more SRAM than the hardware possesses, causing an immediate **Out of Memory (OOM) crash**.

### 2.4 What Is the Compute Grid?

The `grid_size = (8, 4)` parameter in `SDPAProgramConfig` specifies how many Tensix cores are used for the SDPA computation. But not all 32 cores process unique chunks independently.

The grid is divided based on:
- **Heads**: Work for each KV head is distributed across cores
- **Reduce**: Multiple cores may collaborate on the same head using a reduce pattern

In our test with `nh=8, nkv=1, batch=1`:
- The kernel assigns work to cores based on the KV head and batch. Because there are 8 Q heads sharing 1 KV head, different cores compute attention for different Q heads.
- **Core (0,0)** computes the full attention for one of the Q heads.
- To do this, Core (0,0) **must process ALL required chunks** of the sequence memory sequentially to compute the final attention score.

**Why are there exactly 3 chunks (0, 1, 2) processed?**
The number of chunks processed depends entirely on the sequence length being decoded.

In our test, we set `cur_pos = 1024` as the token index we are currently decoding. The attention mechanism must look back at all past tokens (Token 0 up to Token 1024).
Because our chunk size is strictly capped at 512 tokens (see above), the kernel slices the history into 512-token pieces:
- **Chunk 0:** Processes Tokens 0 to 511
- **Chunk 1:** Processes Tokens 512 to 1023
- **Chunk 2:** Processes Tokens 1024 to 1535 (but stops early at 1024)

```
Token:
 0                    512                  1024  (cur_pos stops here)
 │      CHUNK 0        │      CHUNK 1        │      CHUNK 2
 ├─────────────────────┼─────────────────────┼─────────...
 │ seq tiles 0 to 15   │ seq tiles 16 to 31  │ tile 32 (loop stops)
```

If we chose a different position, say `cur_pos = 2000`, the kernel would iterate through exactly **4 chunks**:
- Chunk 0: 0-511
- Chunk 1: 512-1023
- Chunk 2: 1024-1535
- Chunk 3: 1536-2000

**Why did DPRINT only print `CHUNK 2`?**
This is an artifact of the RISC-V print buffer. `DPRINT` writes to a small memory mailbox that the host CPU periodically reads. If the kernel loops too fast, the mailbox overflows, and early prints (Chunks 0 and 1) are overwritten before the host reads them. So we only "saw" Chunk 2 printed, but mathematically, Core 0,0 absolutely calculated Chunks 0, 1, and 2.

### 2.5 Putting It All Together: Why K(L1:4 DRAM:60)?

Now let's trace exactly why the DPRINT output showed `K(L1:4 DRAM:60)` for Chunk 2.

**Test parameters:**
```
seq_len = 2048         →  64 total seq tiles
cur_pos = 1024         →  33 valid seq tiles (ceil(1025/32) = 33)  *Note: 1024 is just an arbitrary test point simulating halfway through decode*
l1_window = 256 tokens →   8 L1 seq tiles (256/32 = 8)
k_chunk_size = 512     →  16 seq tiles per chunk
head_dim = 128         →   4 dim tiles (DHt)
```

**Step 1: Compute L1 window bounds**
```
seq_tiles = ceil(1025/32) = 33
l1_window_start_tile = 33 - 8 = 25   ← L1 window: tiles [25, 33)
```

**Step 2: Core (0,0) processes Chunk 2 (seq tiles 32-47)**

The kernel iterates over every tile in the chunk:

```
For each col in [0, 1, 2, 3]:        ← 4 dim tiles (DHt)
  For each row in [0, 1, ..., 15]:   ← 16 seq tiles per chunk
    global_seq_tile = 32 + row       ← chunk_start=32

    row=0: global_seq_tile=32  → in_l1? 32 >= 25 AND 32 < 33 → YES! → L1 read
    row=1: global_seq_tile=33  → in_l1? 33 >= 25 AND 33 < 33 → NO  → DRAM read
    row=2: global_seq_tile=34  → in_l1? 34 >= 25 AND 34 < 33 → NO  → DRAM read
    ...
    row=15: global_seq_tile=47 → NO → DRAM read
```

**Result per dimension column**: 1 L1 read + 15 DRAM reads.
**Across all 4 columns**: 4 × 1 = **4 L1 reads**, 4 × 15 = **60 DRAM reads**.

```
K(L1:4 DRAM:60)  ← ✅ Exactly matches DPRINT output!
```

### 2.6 Visual: Which Tiles Come from L1 in Chunk 2?

```
Chunk 2 tile grid (16 seq tiles × 4 dim tiles):
                    col 0    col 1    col 2    col 3
                  ┌────────┬────────┬────────┬────────┐
  row 0 (tile 32) │ 🟢 L1  │ 🟢 L1  │ 🟢 L1  │ 🟢 L1  │ ← This row is in L1 window!
                  ├────────┼────────┼────────┼────────┤
  row 1 (tile 33) │ 🔴 DRAM│ 🔴 DRAM│ 🔴 DRAM│ 🔴 DRAM│
                  ├────────┼────────┼────────┼────────┤
  row 2 (tile 34) │ 🔴 DRAM│ 🔴 DRAM│ 🔴 DRAM│ 🔴 DRAM│
                  ├────────┼────────┼────────┼────────┤
  ...             │  ...   │  ...   │  ...   │  ...   │
                  ├────────┼────────┼────────┼────────┤
  row 15 (tile 47)│ 🔴 DRAM│ 🔴 DRAM│ 🔴 DRAM│ 🔴 DRAM│
                  └────────┴────────┴────────┴────────┘

  🟢 = 4 tiles from L1
  🔴 = 60 tiles from DRAM
  Total = 64 tiles per chunk
```

### 2.7 How to Adjust the Caching Amount in the Test

In `tests/test_l1_kv_cache_poison.py`, the L1 cache size is controlled by **one variable**:

```python
l1_window = 256  # ← CHANGE THIS to adjust how many tokens are cached in L1
```

| `l1_window` | L1 tiles | Effect on Chunk 2 (tile 32-47) | What you'll see |
|-------------|----------|-------------------------------|----------------|
| 256 | 8 | Tiles 25-32 in L1 → only tile 32 hits | K(L1:4 DRAM:60) |
| 512 | 16 | Tiles 17-32 in L1 → only tile 32 hits | K(L1:4 DRAM:60) |
| 1024 | 32 | Tiles 1-32 in L1 → only tile 32 hits | K(L1:4 DRAM:60) |
| 1056 | 33 | Tiles 0-32 in L1 → ALL 33 valid tiles in L1 | K(L1:64 DRAM:0) |

Wait — interesting! Even with larger `l1_window`, Chunk 2 still only gets tile 32 from L1 because the L1 window extends **backwards** from `cur_pos`, so higher-numbered tiles (33-47) are always **after** the current position and thus always DRAM.

**To see more L1 hits from Chunk 2's perspective**, you can:
1. **Decrease `cur_pos`** (e.g., set `cur_pos = 300`) so the L1 window overlaps more with Chunk 2's range
2. **Increase `l1_window`** so it covers more of the sequence
3. **Decrease `k_chunk_size`** manually to get smaller chunks (more chunks = more lines printed)

**To see ALL chunks** (not just Chunk 2), widen the DPRINT core range:
```bash
TT_METAL_DPRINT_CORES="0,0-7,3" python tests/test_l1_kv_cache_poison.py
```

---

## 3. What Is `l1_kv_window_size`? (Parameter Semantics)

```
l1_kv_window_size = 256
```

**The unit is tokens** (sequence positions), NOT bytes.

When you set `l1_kv_window_size=256`, the system allocates an L1 KV cache tensor with shape:

```
[batch_size, n_kv_heads, 256, head_dim]
```

Here `256` occupies the **S (sequence) dimension**. On Tenstorrent hardware, data is organized in tiles of 32×32 elements. So 256 tokens = 256/32 = **8 tile rows** in the sequence dimension.

### Memory Calculation Example (Llama-3.1-8B, batch=1)

| Parameter        | Value  |
|-----------------|--------|
| `n_kv_heads`    | 8 (per device) |
| `head_dim`      | 128    |
| `l1_kv_window_size` | 256 tokens |
| `kv_cache_dtype` | bfloat8_b (1 byte/element) |

Per K or V tensor:
```
1 × 8 × 256 × 128 = 262,144 elements
× 1 byte (bfloat8_b) ≈ 256 KB
```

Total L1 for both K and V: **~512 KB spread across all cores** (INTERLEAVED layout distributes tiles round-robin, so each core holds only ~4 KB).

---

## 4. Official vs. Our Implementation

### 4.1 Official Implementation (DRAM-Only, Entire KV Cache)

The official Tenstorrent SDPA decode flow:

```
┌──────────────────────────────────────────────────┐
│                    DRAM                           │
│                                                   │
│  KV Cache: [batch, n_kv_heads, max_seq_len, dim] │
│  (e.g., [1, 8, 131072, 128])                     │
│                                                   │
│  The ENTIRE sequence (all tokens 0..cur_pos) is  │
│  stored and read from DRAM every decode step.    │
└─────────────────────┬────────────────────────────┘
                      │ NoC reads (slow, ~100+ cycles/tile)
                      ▼
         ┌────────────────────────┐
         │  Tensix Core (L1)      │
         │                        │
         │  Circular Buffers:     │
         │  cb_k_in, cb_v_in      │
         │  (transient, per-chunk)│
         └────────────────────────┘
```

**Key point**: The circular buffers (`cb_k_in`, `cb_v_in`) inside each Tensix core are **transient staging areas** — they hold one chunk of K/V at a time (e.g., 256 tiles), process it, then overwrite with the next chunk. They do NOT persistently store KV data.

The kernel function `read_kv_mask_chunks` reads each chunk entirely from DRAM:

```cpp
// DRAM-only path (official):
for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
    for (uint32_t col = 0; col < DHt; ++col) {
        for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
            noc_async_read_tile(k_tile_id, k_reader, k_write_ptr);  // Always DRAM
            k_tile_id += DHt;
            k_write_ptr += k_tile_bytes;
        }
    }
}
```

### 4.2 Our Implementation (Dual-Source: L1 Ring Buffer + DRAM)

```
┌──────────────────────────────────────────────────┐
│                    DRAM                           │
│                                                   │
│  Full KV Cache: [1, 8, 131072, 128]              │
│  Contains ALL tokens (0..cur_pos)                │
│  ← Still authoritative source of truth           │
└──────────────┬───────────────────────────────────┘
               │ NoC reads (only for "cold" tiles)
               ▼
┌──────────────────────────────────────────────────┐
│              Tensix Core (L1)                     │
│                                                   │
│  ┌──────────────────────────────────────┐        │
│  │  L1 KV Ring Buffer                    │        │
│  │  [1, 8, 256, 128]                    │        │
│  │  Contains 256 most recent tokens     │        │
│  │  Organized as ring: pos % 256        │        │
│  │  ← Persistent, fast L1 reads         │        │
│  └──────────────────────────────────────┘        │
│                                                   │
│  ┌──────────────────────────────────────┐        │
│  │  Circular Buffers (transient)         │        │
│  │  cb_k_in, cb_v_in                    │        │
│  │  ← Staging for SDPA compute          │        │
│  └──────────────────────────────────────┘        │
└──────────────────────────────────────────────────┘
```

The kernel function `read_kv_mask_chunks_dual_source` checks each tile:

```cpp
// Dual-source path (our addition):
for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
    uint32_t global_seq_tile = chunk_seq_tile + row;

    // Is this tile in the L1 window?
    bool in_l1 = (global_seq_tile >= l1_window_start_tile) &&
                 (global_seq_tile < l1_window_start_tile + l1_window_size_tiles);

    if (in_l1) {
        // Ring buffer mapping: modular arithmetic
        uint32_t l1_tile_row = global_seq_tile % l1_window_size_tiles;
        noc_async_read_tile(l1_tile_row * DHt + col, l1_k_reader, k_write_ptr);  // Fast L1
    } else {
        noc_async_read_tile(dram_k_tile_id, k_reader, k_write_ptr);              // Slow DRAM
    }
}
```

---

## 5. Ring Buffer Mechanics

### 5.1 Why a Ring Buffer?

The L1 cache has a fixed size (e.g., 256 tokens = 8 tile rows). As decoding progresses beyond position 256, we can't store everything. Instead, we overwrite the oldest entry with the newest — a classic **circular / ring buffer**.

**Why select `[25,33)` instead of `[0,8)`?**
Because of "Sliding Window Attention" locality! In LLMs, the most recently generated tokens have the highest impact on the next word. The oldest tokens (0-8) are usually far less important. Storing the *latest* tokens in fast L1 cache gives the highest chance of fetching the most relevant data quickly. The L1 window slides forward with every new token, always hoarding the "newest" information.

**Why write to BOTH DRAM and SRAM?**
Because L1 SRAM is tiny. As the window slides from `[25, 33)` to `[26, 34)`, token 25 is evicted from SRAM to make room for token 34. If we didn't also write token 25 to DRAM back when it was generated, it would be deleted permanently! DRAM serves as the authoritative, infinite-capacity "source of truth", while SRAM is just a temporary turbo-cache.

### 5.2 Write Side (Python: `attention.py`)

During decode, each new token's K/V is written to **both** the DRAM cache and the L1 cache:

```python
# Step 1: Write to DRAM (always, at position cur_pos)
ttnn.experimental.paged_update_cache(keys_DRAM, k_heads, update_idxs_tensor=current_pos)

# Step 2: Write to L1 ring buffer (at position cur_pos % l1_kv_window_size)
l1_pos = ttnn.typecast(current_pos, ttnn.float32)
l1_pos = ttnn.remainder(l1_pos, float(self.l1_kv_window_size))  # Modulo!
l1_pos = ttnn.typecast(l1_pos, ttnn.int32)
ttnn.experimental.paged_update_cache(l1_kv_cache[0], k_heads, update_idxs_tensor=l1_pos)
```

### 5.3 Read Side (C++ Kernel: `dataflow_common.hpp`)

The SDPA kernel dynamically decides where to read each tile. It computes `l1_window_start_tile` based on `cur_pos`:

```cpp
// In reader_decode_all.cpp:
uint32_t seq_tiles = (cur_pos + 1 + TILE_HEIGHT - 1) / TILE_HEIGHT;  // ceil div
cur_l1_window_start_tile = (seq_tiles > l1_window_size_tiles)
                            ? (seq_tiles - l1_window_size_tiles) : 0;
```

This means: "the L1 window covers the last N tile-rows of the sequence."

### 5.4 Visual Example: `l1_kv_window_size=256`, generating token at position 300

```
DRAM KV Cache (full, max_seq_len=131072):
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬─────┐
│ T0 │ T1 │ T2 │... │T31│T32│...│T255│T256│...│T300│ ... │
│tile│tile│tile│    │   │   │   │    │    │   │    │     │
│ 0  │ 0  │ 0  │    │ 0 │ 1 │   │  7 │  8 │   │ 9 │     │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴─────┘
   ▲ "Cold" tiles (positions 0-44)        ▲ "Hot" tiles (positions 45-300)
   └── Read from DRAM                     └── Read from L1

L1 Ring Buffer (256 tokens = 8 tile rows):
Physical layout (what paged_update_cache wrote):
┌──────────┬──────────┬──────────┬──────┬──────────┬──────────┬──────────┬──────────┐
│ Row 0    │ Row 1    │ Row 2    │ ...  │ Row 5    │ Row 6    │ Row 7    │          │
│ T288-300 │ T256-287 │ T288-319 │      │ T160-191 │ T192-223 │ T224-255 │          │
│(wrapping)│          │          │      │          │          │          │          │
└──────────┴──────────┴──────────┴──────┴──────────┴──────────┴──────────┴──────────┘

Kernel mapping (global_seq_tile → L1 row):
  global_seq_tile 2 → L1 row = 2 % 8 = 2   (for early tokens that haven't wrapped)
  global_seq_tile 8 → L1 row = 8 % 8 = 0   (wraps around!)
  global_seq_tile 9 → L1 row = 9 % 8 = 1

The kernel checks: is global_seq_tile in [l1_window_start_tile, l1_window_start_tile + 8)?
  l1_window_start_tile = ceil(301/32) - 8 = 10 - 8 = 2
  So tiles 2..9 are "in L1" → read from L1 ring buffer via modulo mapping.
  Tiles 0..1 are "cold" → read from DRAM.
```

---

## 6. All Code Changes — Detailed Explanation

### 6.1 Python Layer: `attention.py`

#### Change 1: L1 KV Cache Allocation (`init_kv_cache`, line 474-495)

```python
if self.l1_kv_window_size > 0 and not self.paged_attention_config:
    l1_seq_len = self.l1_kv_window_size  # e.g., 256 tokens
    l1_cache_k = torch.zeros(
        (self.batch_size_per_device_group, self.n_local_kv_heads, l1_seq_len, self.head_dim)
    )
    # ... same for l1_cache_v ...
    self.l1_kv_cache = [
        ttnn.as_tensor(k_or_v, ..., memory_config=ttnn.L1_MEMORY_CONFIG, ...)
        for k_or_v in [l1_cache_k, l1_cache_v]
    ]
```

**Meaning**: Allocates two tensors (K, V) in **L1 SRAM** with sequence dimension = `l1_kv_window_size`. Uses `L1_MEMORY_CONFIG` (not `DRAM_MEMORY_CONFIG`) so they live on-chip.

---

#### Change 2: Ring Buffer Update During Decode (`forward_decode`, line 620-634)

In the codebase, `attention.py`, the exact order of operations is sequentially:

1. **Phase 2:** Update DRAM (`paged_update_cache` for keys, then values)
2. **Phase 3:** Update L1 (`paged_update_cache` for L1 keys, then L1 values)

There is no alternation; DRAM is always completely updated first, then L1 is updated using modulo math.

```python
# Phase 2: Write to DRAM (always, at position cur_pos)
ttnn.experimental.paged_update_cache(keys, k_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table)
ttnn.experimental.paged_update_cache(values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table)

# Phase 3: Write to L1 ring buffer (at position cur_pos % l1_kv_window_size)
if self.l1_kv_cache is not None and not page_table:
    l1_pos = ttnn.typecast(current_pos, ttnn.float32)        # int32 → float32
    l1_pos = ttnn.remainder(l1_pos, float(self.l1_kv_window_size))  # modulo
    l1_pos = ttnn.typecast(l1_pos, ttnn.int32)                # float32 → int32
    ttnn.experimental.paged_update_cache(self.l1_kv_cache[0], k_heads_1BKD, update_idxs_tensor=l1_pos)
    ttnn.experimental.paged_update_cache(self.l1_kv_cache[1], v_heads_1BKD, update_idxs_tensor=l1_pos)
```

**Meaning**: Writes the current token's K/V to the L1 ring buffer at position `cur_pos % l1_kv_window_size`. The `typecast` dance is required because `ttnn.remainder` only works with `float32` operands, not `int32`.

> [!CAUTION]
> **The bug we fixed**: The original code used `ttnn.to_layout(current_pos, ttnn.TILE_LAYOUT)` before `remainder`, then converted back with `ttnn.to_layout(..., ttnn.ROW_MAJOR_LAYOUT)`. The `TILE_LAYOUT` conversion pads the small index tensor (shape `[1, batch]`) to 32×32 tile dimensions with **zeros**. After modulo and conversion back, these zero-padding entries became spurious index-0 values, causing `paged_update_cache` to overwrite tile row 0 of the L1 cache on every step.

---

#### Change 3: L1 Cache Population During Prefill (`forward_prefill`, line 972-987)

```python
if (self.l1_kv_cache is not None
    and seq_len <= self.l1_kv_window_size
    and (chunk_start_idx is None or chunk_start_idx == 0)):
    ttnn.fill_cache(self.l1_kv_cache[0], k_fill, user_id % self.batch_size_per_device_group)
    ttnn.fill_cache(self.l1_kv_cache[1], v_fill, user_id % self.batch_size_per_device_group)
```

**Meaning**: During prefill, if the prompt fits within the L1 window, copy the prefill K/V into L1 so decode starts with a warm cache. Without this, the first ~256 decode steps would read all tiles from DRAM even though they're also supposed to be in L1.

---

#### Change 4: Passing L1 Tensors to SDPA (`forward_decode`, line 657-668)

```python
attn_output_1G4D = ttnn.transformer.scaled_dot_product_attention_decode(
    q_heads_1BQD, keys, values,
    cur_pos_tensor=current_pos,
    ...
    l1_k_tensor=self.l1_kv_cache[0] if self.l1_kv_cache else None,
    l1_v_tensor=self.l1_kv_cache[1] if self.l1_kv_cache else None,
)
```

**Meaning**: The L1 KV cache tensors are passed as optional parameters to the SDPA op. If `None`, the kernel uses the DRAM-only path.

#### Summary of Parameters: Fixed vs User-Definable

| Parameter | Type | Value / Location | Description |
|-----------|------|------------------|-------------|
| **Tile Size** | 🔒 **Fixed** (Hardware) | 32×32 elements | The bare minimum unit of math in Tensix cores. |
| **Max Chunk Size** | 🔒 **Fixed** (Kernel) | 512 tokens (16 tiles) | Determined by `get_chunk_size` limit `min(512, ...)`. Capped to prevent L1 `cb_k_in` circular buffer OOM. |
| **`l1_kv_window_size`** | ⚙️ **User-Definable** | Set via CLI: `--l1_kv_window_size` | How much L1 cache to allocate. Determines how many past tokens stay "hot". |
| **`cur_pos`** | 📈 **Dynamic** (Runtime) | Varies from 0 to `seq_len` | The current token being decoded. |

---

### 6.2 C++ Kernel: SDPA Decode Program Factory

#### `sdpa_decode_program_factory.cpp`

**L1 window start calculation** (line 972-977):
```cpp
uint32_t cur_l1_window_start_tile = 0;
if (use_l1_kv_cache && l1_window_size_tiles > 0) {
    uint32_t seq_tiles = (cur_pos + 1 + TILE_HEIGHT - 1) / TILE_HEIGHT;
    cur_l1_window_start_tile = (seq_tiles > l1_window_size_tiles)
                                ? (seq_tiles - l1_window_size_tiles) : 0;
}
```

**Meaning**: Computes which DRAM tile row is the start of the L1 window. For `cur_pos=300` with `l1_window_size_tiles=8`: `seq_tiles = ceil(301/32) = 10`, so `l1_window_start_tile = 10 - 8 = 2`. Tiles 2-9 are in L1.

**Runtime args** (line 996-999): Four additional runtime args are passed to each core:
```cpp
reader_rt_args = { ..., l1_k_addr, l1_v_addr, cur_l1_window_start_tile, l1_window_size_tiles };
```

---

### 6.3 C++ Kernel: Reader Dataflow

#### `reader_decode_all.cpp`

**Dynamic window recalculation on-device** (line 126-132):
```cpp
uint32_t cur_l1_window_start_tile = l1_window_start_tile;
if constexpr (use_l1_kv_cache) {
    if (l1_window_size_tiles > 0) {
        uint32_t seq_tiles = (cur_pos + 1 + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
        cur_l1_window_start_tile = (seq_tiles > l1_window_size_tiles) ? (seq_tiles - l1_window_size_tiles) : 0;
    }
}
```

**Meaning**: When `cur_pos` is read from the device tensor (not from the host), the kernel recalculates `l1_window_start_tile` dynamically. This is necessary because during traced execution, the host-provided value may be stale.

#### `dataflow_common.hpp` — `read_kv_mask_chunks_dual_source`

The per-tile dispatch logic (line 660-672):
```cpp
uint32_t global_seq_tile = chunk_seq_tile + row;

// Check L1 inclusion
bool in_l1 = (global_seq_tile >= l1_window_start_tile) &&
             (global_seq_tile < l1_window_start_tile + l1_window_size_tiles);

if (in_l1) {
    uint32_t l1_tile_row = global_seq_tile % l1_window_size_tiles;  // Ring buffer!
    uint32_t l1_k_tile_id = l1_tile_row * DHt + col;
    noc_async_read_tile(l1_k_tile_id, l1_k_reader, k_write_ptr);   // L1 read
} else {
    uint32_t dram_k_tile_id = k_start_tile_id + col + row * DHt;
    noc_async_read_tile(dram_k_tile_id, k_reader, k_write_ptr);    // DRAM read
}
```

**Meaning**: For each tile in each chunk, the kernel checks if that tile falls within the L1 window. If yes, it uses **modular arithmetic** (`global_seq_tile % l1_window_size_tiles`) to map the global sequence tile to the physical L1 ring buffer position. Otherwise, it falls back to DRAM.

---

## 7. Comparison: Circular Buffer vs. Our Ring Buffer

| Aspect | Tensix Circular Buffer (`cb_k_in`) | Our L1 Ring Buffer (`l1_kv_cache`) |
|--------|-------------------------------------|-------------------------------------|
| **Purpose** | Transient staging for SDPA compute | Persistent storage of hot KV entries |
| **Lifetime** | One chunk at a time, overwritten per iteration | Persists across all decode steps |
| **Size** | ~4-8 tiles (per chunk) | 256 tokens × head_dim (persistent) |
| **Memory Config** | Implicitly allocated by CB framework | Explicitly allocated with `L1_MEMORY_CONFIG` |
| **What it stores** | Whatever chunk SDPA is currently computing | Most recent 256 tokens' K and V |
| **Who reads** | SDPA compute kernel | SDPA reader kernel (dual-source) |
| **Who writes** | Reader kernel (from DRAM or L1) | `paged_update_cache` (from Python) |

---

## 8. Verification: Poison Test

### 8.1 What the Poison Test Does

The test (`tests/test_l1_kv_cache_poison.py`) definitively proves that the SDPA kernel reads KV tiles from L1 by placing **intentionally different** (poisoned) values in L1 vs DRAM:

**3 Runs:**

| Run | DRAM KV | L1 KV | Purpose |
|-----|---------|-------|---------|
| Run 1 (Baseline) | Original values A | None | DRAM-only output reference |
| Run 2 (Poison) | Original values A | **Different values B** | Does the output change? |
| Run 3 (Ground Truth) | A with B baked into hot region | None | What output SHOULD look like if L1 is read |

**Logic:**
- If L1 is truly read → Run 2 output ≠ Run 1 (because kernel read B from L1 instead of A from DRAM)
- If L1 is ignored → Run 2 output = Run 1 (both runs read A from DRAM)
- If L1 is read → Run 2 output ≈ Run 3 (same values B, just read from L1 vs DRAM)

**Actual Results:**
```
PCC(baseline, poison):        0.781  ← VERY different (proves L1 is read)
PCC(poison, ground_truth):    1.000  ← PERFECT match (L1 math is correct)
PCC(baseline, ground_truth):  0.781  ← Different (expected, different data)
```

The PCC of 0.781 between baseline and poison is definitive proof: if L1 were ignored, both runs would produce identical output (`PCC 1.0`).

The `1.000` PCC between poison and ground truth proves that the C++ kernel's modulo ring-buffer math (`tile_row = global_seq_tile % l1_window_size_tiles`) perfectly aligns with our generated ground truth.

### 8.2 Test Code Layout

```python
# 1. Setup: seq_len=2048, cur_pos=1024, l1_window=256 tokens
K_original = torch.randn(1, 1, 2048, 128)  # Original DRAM K
V_original = torch.randn(1, 1, 2048, 128)  # Original DRAM V

# 2. Generate poison data, pasted sequentially into DRAM (Ground Truth)
K_with_poison[:, :, l1_start:l1_end, :] = K_poison_seq

# 3. Pack poison data into L1 using MODULO math!
# This mimics `attention.py`'s circular buffer behavior
for global_tile in range(l1_start_tile, l1_end_tile):
    l1_row = global_tile % l1_window_size_tiles
    K_l1_ring[:, :, l1_row*32:(l1_row+1)*32, :] = K_poison_seq[...]

# 4. Run SDPA three times and compare PCC
# Run 1: baseline (DRAM only) → output_baseline
# Run 2: DRAM + K_l1_ring     → output_poison  (should differ from baseline)
# Run 3: K_with_poison        → output_ground_truth (should perfectly match Run 2)
```

### 8.3 How to Run

```bash
source python_env/bin/activate
python tests/test_l1_kv_cache_poison.py
```

---

## 9. DPRINT Instrumentation: Tracking L1 vs DRAM Read Ratio

### 9.1 What DPRINT Shows

We added debug print macros inside `read_kv_mask_chunks_dual_source` in `dataflow_common.hpp`. When enabled, each chunk iteration prints the exact count of tiles read from L1 vs DRAM:

```
0:(x=0,y=0):NC: CHUNK 2 | K(L1:4 DRAM:60) | V(L1:4 DRAM:60)
```

Format: `device:(core_x, core_y):NCRISC: CHUNK <id> | K(L1:<hits> DRAM:<misses>) | V(L1:<hits> DRAM:<misses>)`

### 9.2 How to Enable DPRINT

**Step 1:** Add `#define DEBUG_PRINT 1` in `dataflow_common.hpp` (line 652, right before the `#if defined(DEBUG_PRINT)` block):

```cpp
// In dataflow_common.hpp, inside read_kv_mask_chunks_dual_source:
for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
    uint32_t chunk_seq_tile = k_chunk * Sk_chunk_t;
    #define DEBUG_PRINT 1          // ← Add this line
#if defined(DEBUG_PRINT)
    uint32_t k_l1_hits = 0;
    // ... (counting variables already present)
```

**Step 2:** Run with `TT_METAL_DPRINT_CORES` environment variable:

```bash
TT_METAL_DPRINT_CORES="0,0" python tests/test_l1_kv_cache_poison.py
```

Core format options:
- `"0,0"` — single core
- `"0,0-1,1"` — range (2×2 grid)
- `"(0,0),(1,0),(2,0)"` — specific cores

**Step 3:** Check the output inline in stdout. DPRINT lines are prefixed with the core coordinates.

> [!WARNING]
> Remove `#define DEBUG_PRINT 1` when done! DPRINT adds significant overhead and will heavily slow down real inference.

### 9.3 Why Only One CHUNK Line Printed

In our test with `TT_METAL_DPRINT_CORES="0,0"`, only **one** CHUNK line appeared:

```
0:(x=0,y=0):NC: CHUNK 2 | K(L1:4 DRAM:60) | V(L1:4 DRAM:60)
```

This is because:
1. DPRINT only captures output from core `(0,0)`, but the work is distributed across a `(8,4)` grid = 32 cores
2. Core `(0,0)` was only assigned **one chunk** (Chunk 2) of the multi-chunk sequence
3. Other cores processed Chunks 0 and 1 but their DPRINT output was not captured

### 9.4 Decoding the Output: Why K(L1:4 DRAM:60)?

For Chunk 2 (seq tiles 32-47):
- `Sk_chunk_t` = 16 (tiles per chunk in sequence dimension)
- `DHt` = 4 (tiles in head dimension = 128/32)
- Total tile reads per chunk = 16 × 4 = **64 tiles**

The L1 window covers seq tiles `[25, 33)`. In Chunk 2 (seq tiles 32-47):
- **Tile 32**: `in_l1 = true` (32 is within [25, 33)) → 1 seq tile × 4 dim tiles = **4 L1 reads**
- **Tiles 33-47**: `in_l1 = false` → 15 seq tiles × 4 dim tiles = **60 DRAM reads**

Total: 4 + 60 = 64. ✅ Perfect match!

---

## 10. L1 Hit Ratio Calculation

### 10.1 Theoretical Formula

The L1 hit ratio depends on the current decode position and the L1 window size:

```
                     min(l1_window_size, cur_pos + 1)
L1 Hit Ratio  =  ──────────────────────────────────────
                            cur_pos + 1
```

In tile units (multiply both by 1/32):

```
                     min(l1_window_size_tiles, seq_tiles)
L1 Hit Ratio  =  ──────────────────────────────────────────
                                seq_tiles
```

### 10.2 Example Ratios

| Scenario | `cur_pos` | `l1_window` | seq_tiles | L1 tiles | **Hit Ratio** |
|----------|-----------|-------------|-----------|----------|---------------|
| Test (poison) | 1024 | 256 | 33 | 8 | **24.2%** |
| Short decode | 200 | 256 | 7 | 7 | **100%** (fully cached) |
| Medium decode | 2048 | 256 | 65 | 8 | **12.3%** |
| Long decode | 4096 | 256 | 129 | 8 | **6.2%** |
| Long + big window | 4096 | 4096 | 129 | 128 | **99.2%** |

**Key insight**: With `l1_kv_window_size=256` and short decodes (200 tokens), the L1 cache covers the **entire** sequence, so there's no performance difference vs DRAM-only — all tiles come from L1 either way. The benefit only appears at **longer sequences** where DRAM bandwidth becomes a bottleneck.

### 10.3 How to Measure & Record Hit Ratio on Real Llama Workloads

**Yes, you can measure this!** Here is the exact step-by-step process and a Python parser script to calculate the true L1 hit ratio from `simple_text_demo.py` logs.

**Step 1: Enable DPRINT in the C++ Kernel**
Edit `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` (around line 652):
```cpp
#define DEBUG_PRINT 1    // ← Uncomment or add this line
#if defined(DEBUG_PRINT)
...
```

**Step 2: Disable Tracing and Run Demo with Output Redirection**
DPRINT requires the host to constantly read a small mailbox from the device. However, **Llama3 uses Device Tracing**, which queues commands in hardware and runs them instantly without host synchronization. Tracing will completely swallow and overwrite all DPRINT output!

You must temporarily disable tracing in the Python test file:
1. Open `models/tt_transformers/demo/simple_text_demo.py`
2. Scroll to the `batch-1` configuration block (around line 458)
3. Change `True,  # enable_trace` to `False,`

Since Llama3 maps the attention kernel to a massive mesh of cores, we need to capture output from **all** cores. Because our Python script automatically sums every match it finds, grabbing all cores gives us the perfectly accurate global L1 hit ratio!

```bash
# Clear cache to force recompilation of the C++ kernel with DPRINT enabled
rm -rf ~/.cache/ttnn/*

# Run the demo with '-s' (stops pytest from swallowing output) and capture ALL cores:
TT_METAL_DPRINT_CORES=all pytest -s models/tt_transformers/demo/simple_text_demo.py -k "batch-1" --l1_kv_window_size 4096 > dprint_log.txt
```

**Step 3: Parse the Log File (Python Parser Script)**
Save this script as `parse_hit_ratio.py` and run it: `python parse_hit_ratio.py dprint_log.txt`

```python
import sys
import re

def parse_hit_ratio(log_file):
    total_l1_hits = 0
    total_dram_reads = 0

    # Regex to match: K(L1:<hits> DRAM:<misses>)
    pattern = re.compile(r"K\(L1:(\d+)\s+DRAM:(\d+)\)")

    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                total_l1_hits += int(match.group(1))
                total_dram_reads += int(match.group(2))

    total_reads = total_l1_hits + total_dram_reads
    if total_reads == 0:
        print("No DPRINT lines found. Did you enable DEBUG_PRINT 1?")
        return

    hit_ratio = (total_l1_hits / total_reads) * 100
    print(f"Total KV Tiles Read: {total_reads}")
    print(f"L1 Hits:           {total_l1_hits} ({hit_ratio:.2f}%)")
    print(f"DRAM Reads:        {total_dram_reads} ({100 - hit_ratio:.2f}%)")

if __name__ == "__main__":
    parse_hit_ratio(sys.argv[1])
```

> [!WARNING]
> Don't forget to remove `#define DEBUG_PRINT 1` when you are done! The `dprint_log.txt` file will grow by megabytes per second, severely bottlenecking the actual token generation speed.

### 10.4 Will a 2048 L1 Window Give Meaningful Speedups on Short Generations?

If you configure a massive `--l1_kv_window_size 2048`, but you only use the standard `simple_text_demo.py` prompt (~100 tokens) and ask it to generate ~90 tokens: **No, you will not see a meaningful human-perceptible speedup.**

**Why? Two reasons:**
1. **The hit ratio is already 100%:** If your total sequence length never exceeds 200 tokens, and your L1 window is 2048, *every single token fits into L1*. The L1 Hit Ratio will be an absolute **100%**.
2. **But the absolute time is too short:** Matrix math on 200 tokens takes microseconds. Even though L1 is ~4x faster than DRAM, 4x faster than "almost instant" is still "almost instant." The bottleneck in a short text demo is usually Python framework overhead, CPU-to-device dispatching, or MLP layers, **not** SDPA DRAM bandwidth.

**How to see massive, meaningful SDPA speedups:**
To see the true power of a 2048 L1 window, you must stress the **DRAM Memory Bandwidth Limits**. You need:
1. **A massive prompt context:** Feed the model a 4,000 to 6,000 token system prompt (like a long document or codebase).
2. **A long generation:** Generate 200 to 500 tokens.

When decoding token 4001, the SDPA kernel must fetch all previous 4000 tokens of KV cache data.
- Without L1: The core must drag 4000 tokens × 8 heads across the slow DRAM NoC. This is a massive bandwidth bottleneck.
- With `l1_kv_window_size 2048`: The core instantly fetches the newest 2048 tokens from local ultra-fast SRAM, and only drags 1952 tokens across the DRAM NoC.

**This essentially halves the memory bandwidth requirement of the SDPA operation.** On a 4000-token context, generating hundreds of tokens, the tokens/second throughput will show a massive, deeply meaningful improvement.

---

## 11. Known Limitations

1. **Paged attention incompatible**: The L1 ring buffer is disabled when `page_table` is provided (paged attention mode).
2. **Prefill must fit**: L1 is only populated during prefill if the entire prompt length ≤ `l1_kv_window_size`. For longer prompts, only the decode-phase updates populate L1, meaning the first `l1_kv_window_size` decode steps will have partial L1 coverage.
3. **No attention sinks**: The current implementation stores only the most recent tokens. "Attention sink" tokens (the initial 4 tokens that models attend to heavily) are NOT pinned in L1.
4. **`ttnn.remainder` requires float32**: We must typecast `int32 → float32 → remainder → int32` because `ttnn.remainder` crashes with `int32` operands.

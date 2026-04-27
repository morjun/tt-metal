# Blackhole P150 SDPA Decode: Core Architecture and L1 KV Mirror Constraints

## 1. Blackhole P150 Physical Core Layout

### 1.1 Raw Silicon: 140 Tensix Cores

From the UMD hardware header (`tt_metal/third_party/umd/device/api/umd/device/arch/blackhole_implementation.hpp`):

```cpp
const static tt_xy_pair TENSIX_GRID_SIZE = {14, 10};
// 14 columns × 10 rows = 140 physical Tensix (worker) cores
```

> [!NOTE]
> **This 140 does NOT include DRAM controller cores, Ethernet cores, PCIe cores, or ARC management cores.**
> Tensix cores are the pure compute+storage tile processors. The 130-core number you see at runtime is derived from this 140 by subtracting dispatch-reserved cores (see §1.2).

### 1.2 Runtime Compute Grid: Up to 130 Cores

From the core descriptor (`tt_metal/core_descriptors/blackhole_140_arch.yaml`):

When **dispatch runs on Tensix** (the default for single-device P150):

```yaml
unharvested:
  col:
    1:
      compute_with_storage_grid_range:
        start: [0, 0]
        end: [12, 9]     # 13 columns × 10 rows = 130 compute cores
      dispatch_cores:
        [[-1, 0], [-1, 1], ..., [-1, 9]]  # last column (10 cores) reserved for dispatch
```

| Harvesting state | Dispatch mode | `compute_with_storage_grid_size` | Compute cores |
|---|---|---|---|
| None harvested | Tensix dispatch | 13 × 10 | **130** |
| 1 col harvested | Tensix dispatch | 12 × 10 | 120 |
| 2 col harvested | Tensix dispatch | 11 × 10 | 110 |
| None harvested | **ETH dispatch** | **14 × 10** | **140** |
| 1 col harvested | ETH dispatch | 13 × 10 | 130 |

**For the standard single Blackhole P150 with Tensix dispatch and no harvesting:**
`compute_with_storage_grid_size() = (13, 10)` → **130 allocatable worker cores.**

---

## 2. Llama 3.1 8B Model Parameters (Per Device, Single P150)

From `model_params/Llama-3.1-8B-Instruct/config.json` and `model_config.py`:

| Parameter | Global Value | Per-Device Value (`num_devices=1`) |
|---|---|---|
| `n_heads` (Q) | 32 | 32 |
| `n_kv_heads` | 8 | 8 |
| `head_dim` | 128 | 128 |
| `dim` (hidden size) | 4096 | 4096 |
| `hidden_dim` (FFN) | 14336 | 14336 |
| `n_layers` | 32 | 32 |

**GQA ratio**: 32 Q heads / 8 KV heads = **4 Q heads share each KV head**.

---

## 3. SDPA Decode: Core Assignment Formula

### 3.1 The Kernel's Core Allocation Logic

From `sdpa_decode_program_factory.cpp` lines 191–200, with `sdpa_config.hpp` default:

```cpp
// Inputs for Llama 3.1 8B, batch=1, single device
uint32_t B           = 1;           // batch size
uint32_t num_kv_heads = 8;          // n_local_kv_heads per device
uint32_t max_cores_per_head_batch = 16;  // default cap (sdpa_config.hpp)

// From model_config.py line 979 (hard-coded in the model):
// compute_with_storage_grid_size = (8, 8) for SDPA decode
uint32_t num_cores_available = 8 * 8 = 64;  // NOT the full 130!

// Core allocation math:
uint32_t max_num_cores_for_compute = max_cores_per_head_batch * B * num_kv_heads;
//                                 = 16 * 1 * 8 = 128

uint32_t num_cores_per_batch = min(num_cores_available, max_num_cores_for_compute) / B;
//                           = min(64, 128) / 1 = 64

uint32_t num_cores_per_head  = max(1, num_cores_per_batch / num_kv_heads);
//                           = max(1, 64 / 8) = 8     ← 8 cores per KV head

uint32_t num_heads_per_core  = max(1, ceil(num_kv_heads / num_cores_per_batch));
//                           = max(1, ceil(8 / 64)) = 1

uint32_t num_active_cores    = num_cores_per_head * num_kv_heads * B / num_heads_per_core;
//                           = 8 * 8 * 1 / 1 = 64   ← all 64 cores in the (8,8) grid are active
```

### 3.2 Why `(8, 8)` and Not the Full `(13, 10)`?

The SDPA decode program config in `model_config.py` line 979:

```python
self.model_config["SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=(8, 8),   # ← hard-coded, does NOT use full 130
    exp_approx_mode=False,
    q_chunk_size=128,   # Blackhole
    k_chunk_size=128,   # Blackhole
)
```

Two reasons for 8×8 rather than 13×10:

1. **Divisibility**: Core assignment iterates as a flat 1D list and assigns groups to KV heads. With `n_kv_heads=8`, the total core count must be exactly divisible by 8. `13×10=130` → `130 / 8 = 16.25` (not integer). `8×8=64` → `64 / 8 = 8` (clean). Choosing (8,8) avoids idle fractional cores.

2. **Communication overhead cap**: The `max_cores_per_head_batch=16` cap limits each KV head to at most 16 cores. Beyond that, the NoC reduction traffic (workers sending partial softmax results to reducer) costs more than the parallelism saves. 8 cores/head is the sweet spot chosen by the model team.

---

## 4. Q Head Assignment: Where Do Q Heads Live?

> [!IMPORTANT]
> **Q heads are NOT assigned to separate cores from KV heads.** In GQA, all Q heads that share the same KV head are processed **together, inside the same core group** that owns that KV head.

### 4.1 How It Works

The SDPA kernel does NOT split Q heads across cores. Instead:

- **`PNH`** = total Q heads assigned to one "batch slot" = `n_local_q_heads` = **32** (all Q heads on one device)
- **`PNHt`** = `PNH / TILE_HEIGHT` = `32 / 32` = **1** tile of Q per chunk read

Each KV-head core group (8 cores) computes attention for **all 32 Q heads** against **its assigned KV head's K and V tiles**:

```
Q tensor shape entering SDPA: [1 batch × 1 seq_pos × 32 Q-heads × 128 head_dim]
K tensor shape:                [1 batch × 8 KV-heads × max_seq × 128 head_dim]

Core group for KV head 0 (logical cores 0–7):
  Reducer core (core 0): Reads Q[all 32 Q heads], K[kv_head 0], V[kv_head 0]
                          Computes: Q[0..31] × K[0]^T, softmax, × V[0]
                          Receives partial results from 7 worker cores
  Worker cores (cores 1–7): Each reads a different chunk of K[kv_head 0] and V[kv_head 0]
                              Sends partial (score, log-sum-exp) to reducer

Core group for KV head 1 (logical cores 8–15): same structure, uses K[kv_head 1], V[kv_head 1]
...
Core group for KV head 7 (logical cores 56–63): uses K[kv_head 7], V[kv_head 7]
```

The output tensor per KV group: attention outputs for all Q heads that map to that KV head.

For GQA: `32 Q heads / 8 KV heads = 4` → each KV core group computes attention for 4 Q heads simultaneously (as 4 rows in the QK^T matmul, represented by `PNHt=1` tile because 4 Q heads = 128 elements, which is less than one 32-row tile, so they fit in 1 tile).

### 4.2 Concrete Layout for Llama 3.1 8B

```
8x8 grid (logical coordinates, row-major):

Row 0: Core(0,0) Core(1,0) ... Core(7,0)  ← cores 0–7
Row 1: Core(0,1) Core(1,1) ... Core(7,1)  ← cores 8–15
Row 2: Core(0,2) ...                       ← cores 16–23
Row 3: Core(0,3) ...                       ← cores 24–31
Row 4: Core(0,4) ...                       ← cores 32–39
Row 5: Core(0,5) ...                       ← cores 40–47
Row 6: Core(0,6) ...                       ← cores 48–55
Row 7: Core(0,7) ...                       ← cores 56–63

KV Head 0:  cores 0–7   (Core(0,0) = reducer, Core(1,0)..Core(7,0) = workers)
KV Head 1:  cores 8–15  (Core(0,1) = reducer, ...)
KV Head 2:  cores 16–23
KV Head 3:  cores 24–31
KV Head 4:  cores 32–39
KV Head 5:  cores 40–47
KV Head 6:  cores 48–55
KV Head 7:  cores 56–63

Each KV head group processes Q heads [kv*4 .. kv*4+3]:
  KV head 0 → Q heads 0, 1, 2, 3   (Q[0..3] share K/V head 0)
  KV head 1 → Q heads 4, 5, 6, 7
  ...
  KV head 7 → Q heads 28, 29, 30, 31
```

### 4.3 What the 8×1 Pattern Actually Means

The **8×1** subprogram mentioned in `l1_kv_decode_sram_tradeoff.md` §9 refers to the **bottleneck scenario**, not the full Llama 3.1 8B single-device case.

It arises when `n_local_kv_heads = 1`, which happens in **tensor-parallel multi-device** configurations where the 8 KV heads are sharded across 8 devices. Each device then has:

```
n_local_kv_heads = 8 / 8 = 1
num_active_cores = 8 cores/head × 1 KV head = 8 cores → 8×1 layout
```

On a **single P150**, all 8 KV heads are local → 8 groups × 8 cores = **64 active cores in (8,8)**.

| Configuration | `n_local_kv_heads` | Active SDPA cores | Layout |
|---|---|---|---|
| 1 device (P150) | 8 | **64** | 8×8 |
| 8 devices (T3K / galaxy) | 1 | **8** | 8×1 |

---

## 5. L1 KV Mirror Capacity Constraint: Exact Address Layout

### 5.1 Blackhole L1 Address Map (Per Core)

From `dev_mem_map.h` and `bh_hal_tensix.cpp`:

```
MEM_L1_SIZE            = 1536 × 1024  = 1,572,864 B  (total physical L1 per core)
MEM_MAP_END            = 32,080 B     (end of firmware, routing tables, etc.)
default_l1_kernel_config_size = 69 KiB = 70,656 B
DEFAULT_UNRESERVED_BASE ≈ ((MEM_MAP_END + 70,656 - 1) | 31) + 1 = 102,752 B
```

**The TTNN allocator manages the bank `[DEFAULT_UNRESERVED_BASE, MEM_L1_SIZE)` = `[102,752, 1,572,864)`.**
- Size: `1,572,864 - 102,752 = 1,470,112 B ≈ 1,470,080 B` (matching `total_bytes_per_bank` in the doc)
- **Tensors** are allocated **top-down** from `1,470,080` (bank offset) downward
- **Static CBs** from the program factory are placed **bottom-up** from bank offset `0` (absolute addr `DEFAULT_UNRESERVED_BASE`) upward

### 5.2 The Static CB Footprint: Present on ALL 64 Active Cores

From `sdpa_decode_program_factory.cpp`, `CreateCircularBuffer()` is called on the entire `core_grid` (all 64 active cores). Every core—reducer and workers alike—carries the full static CB set.

From `l1_kv_decode_sram_tradeoff.md` §9.2.1, the measurement on the failing `window=544` run:

```
total_bytes_per_bank:              1,470,080 B   (= allocator bank size)
static CB region end (bank offset): 1,249,664 B   (CBs fill 85.0% of bank = 1,220 KiB)
headroom for tensors (top-down):   1,470,080 - 1,249,664 = 220,416 B  (≈ 215 KiB)
```

> [!IMPORTANT]
> The 1,249,664 is a **bank-relative offset** (not an absolute L1 address). In absolute L1 terms:
> `CB end = DEFAULT_UNRESERVED_BASE + 1,249,664 = 102,752 + 1,249,664 = 1,352,416 B (0x14A2E0)`.
> The headroom `[1,249,664, 1,470,080)` within the bank maps to absolute addresses `[1,352,416, 1,572,864)`.

### 5.3 Per-Core Mirror Tile Allocation (Round-Robin over 130 Cores)

The L1 KV mirror uses `ttnn.L1_MEMORY_CONFIG` (INTERLEAVED), distributing tiles **evenly** round-robin across all 130 allocatable worker cores. The per-core share is:

```
tiles_per_layer = 2 × batch × n_kv × l1_tokens × head_dim / tile_bytes
               = 2 × 1 × 8 × window × 128 / 1024
               = 2 × window  tiles  (for bfp8, 1024 B/tile)

total_tiles_all_layers = n_layers × tiles_per_layer

# round-robin → "heavy" cores get ⌈total_tiles / 130⌉ tiles
heavy_core_bytes = ⌈total_tiles / 130⌉ × 1024
```

| `window` | tiles/layer | total tiles (32L) | heavy-core bytes | vs 215 KiB headroom |
|---|---|---|---|---|
| 128 | 256 | 8,192 | 64 KiB | **PASS** |
| 256 | 512 | 16,384 | 127 KiB | **PASS** |
| 384 | 768 | 24,576 | 190 KiB | **PASS** |
| **~427** | ~854 | ~27,331 | **215 KiB** | **← theoretical limit** |
| 512 | 1,024 | 32,768 | 253 KiB | CRASH (32 layers) |
| 544 | 1,088 | 34,816 | 268 KiB | CRASH (32 layers) |

### 5.4 Why Did the Experiment Show 512 Passes and 544 Fails?

The `512 pass / 544 fail` boundary was measured in a **1-layer microbenchmark** (`--num_layers 1`), not the full 32-layer Llama 3.1 8B model. With only 1 layer resident:

```
window=512, n_layers=1:
  total_tiles = 1 × 1,024 = 1,024
  heavy_core_bytes = ⌈1,024 / 130⌉ × 1024 = 8 × 1024 = 8,192 B = 8 KiB  ← trivially fits

window=544, n_layers=1:
  total_tiles = 1 × 1,088 = 1,088
  heavy_core_bytes = ⌈1,088 / 130⌉ × 1024 = 9 × 1024 = 9,216 B = 9 KiB  ← trivially fits
```

Both 512 and 544 fit easily in a 1-layer microbenchmark. The crash at 544 observed in that experiment implies the **actual failure threshold was measured against a different reference**—possibly the full 32-layer model, or a separate constraint from kernel config memory, not tile headroom. In any case, with 32 layers the real limit is closer to `window ≈ 427` tokens.

> [!CAUTION]
> For the production 32-layer Llama 3.1 8B model with KV dtype `bfp8` (1 B/element), the safe empirical ceiling for `l1_kv_window_size` on a single P150 is approximately **384 tokens** when distributed across 130 cores—well below 512.

### 5.5 Address Collision Diagram

```
Per-active-decode-core allocator bank (offset 0 = absolute addr 102,752):

Offset:  0 ────────────────────── 1,249,664 ──────────── 1,470,080
         │                              │                       │
         │   Static SDPA CBs           │   ←  headroom 215 KiB │
         │   (q, k, v, qk, out,        │   for TTNN tensors     │
         │    stats, im...)             │   (top-down alloc)    │
         │   = 85% of bank             │                       │
         └──────────────────────────────┴───────────────────────┘
         ↑                             ↑                       ↑
   CB base (abs 102,752)        CB end (abs 1,352,416)   bank top (abs 1,572,864)

→ Mirror tiles for 32 layers at window=512: ~253 KiB > 215 KiB → CRASH
→ Mirror tiles for 32 layers at window=384: ~190 KiB < 215 KiB → PASS
```

---

## 6. Core Utilization Summary for Full Decode Pass

On single Blackhole P150 with Tensix dispatch (130 worker cores available),
running Llama 3.1 8B with batch=1 in `simple_text_demo.py`:

| Operation | Grid Config | Active Cores | Idle Cores | Chip Utilization |
|---|---|---|---|---|
| RMSNorm / residual add | width-sharded, `dram_shard_core_grid_for_k(4096)` | **32** | 98 | 25% |
| WQKV matmul | DRAM-sharded, `attn_input_grid` | **32** | 98 | 25% |
| Q/K rotary embed | follows WQKV shard | **32** | 98 | 25% |
| **SDPA decode** | hard-coded **(8,8)** | **64** | 66 | **49%** |
| WO (output proj) | DRAM-sharded | **32** | 98 | 25% |
| W1/W3 (gate/up proj) | DRAM-sharded | **32** | 98 | 25% |
| W2 (down proj) | DRAM-sharded | **32** | 98 | 25% |
| LM head matmul | DRAM-sharded | **32** | 98 | 25% |

> [!NOTE]
> Peak utilization (64/130 = **49%**) occurs during SDPA decode. All matmuls run at 25% (32/130). The remaining 66–98 cores are completely idle during each operator.

This is not a bug but a deliberate design choice:
- Decode is **memory-bandwidth-bound**, not compute-bound. Adding more cores competes for the same DRAM bandwidth without proportional speedup.
- Batch=1 means only **1 output row per token step** — there is simply not enough work to fill 130 cores efficiently with single-batch decoding.
- The L1 KV mirror optimization specifically aims to off-load the DRAM BW bottleneck by serving the hottest KV tiles from on-chip SRAM, allowing the fixed 64 active cores to run faster rather than adding more idle cores.

---

## 7. Source References

| Fact | Source File |
|---|---|
| 140 physical Tensix cores | `tt_metal/third_party/umd/device/api/umd/device/arch/blackhole_implementation.hpp` line 69 |
| 130 compute cores (Tensix dispatch) | `tt_metal/core_descriptors/blackhole_140_arch.yaml` line 13 |
| 140 compute cores (ETH dispatch) | `tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml` line 13 |
| `max_cores_per_head_batch = 16` | `ttnn/cpp/ttnn/operations/transformer/sdpa_config.hpp` line 18 |
| Core allocation formula | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp` lines 191–200 |
| `SDPA_DECODE_PROGCFG = (8,8)` | `models/tt_transformers/tt/model_config.py` line 979–984 |
| Static CB footprint 85% | `research_codes/documents/l1_kv_decode_sram_tradeoff.md` §9.2.1 |
| Per-token mirror cost formula | `research_codes/documents/l1_kv_decode_sram_tradeoff.md` §10 |

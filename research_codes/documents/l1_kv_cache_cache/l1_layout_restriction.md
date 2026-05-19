# When tt-metal Allows L1 Sharded vs Interleaved Allocation

> Reference doc — what rules gate sharded-vs-interleaved L1 in tt-metal, and how
> to predict whether a given op accepts one, the other, or both.
> Last updated: 2026-05-18

There is **no single master rule**. Layout support is enforced at two independent
layers — the **allocator** (storage) and the **operation** (compute kernel). A
buffer with a perfectly legal sharded L1 layout can still be rejected by the next
ttnn op it flows into. Conversely, an op can fail validate even with a buffer the
allocator was happy to create.

This document inventories both layers, with file:line citations, so future debug
sessions don't have to re-derive the rules.

---

## 1. Allocator layer (storage gates)

Tt-metal's allocator owns L1 partitioning into three buffer types — `L1`,
`L1_SMALL`, and (separately, not part of this discussion) `DRAM`. Each has a
different sharded/interleaved policy:

| Buffer type | Interleaved | Sharded | Notes |
|---|:---:|:---:|---|
| `L1` (main) | ✓ (default) | ✓ | Both legal. Interleaved has an extra `interleaved_address_limit` clamp. |
| `L1_SMALL` | ✗ | ✓ (mandatory) | Sharded-only. Interleaved fails at allocator level. |
| `DRAM` | ✓ | ✗ (irrelevant for this doc) | DRAM has its own banking; L1 sharding doesn't apply. |

### 1.1 The five gates that fire

| # | File:line | Condition | When it fires |
|---|---|---|---|
| 1 | `tt_metal/impl/allocator/bank_manager.cpp:121` | `validate_num_banks`: rejects interleaved on `L1_SMALL` or when `disable_interleaved` is set | Construction-time validation of bank count |
| 2 | `tt_metal/impl/allocator/allocator.cpp:144` | `TT_FATAL(num_cores.has_value(), "L1_SMALL only supports sharded allocations")` | Attempting to allocate L1_SMALL without a shard spec |
| 3 | `tt_metal/impl/allocator/allocator.cpp:116-118` | `if (config_->disable_interleaved) TT_FATAL(num_cores.has_value(), …)` | Global flag forces all allocations sharded |
| 4 | `tt_metal/impl/allocator/bank_manager.cpp:410-413` | `if (!is_sharded && buffer_type_ == L1) address_limit = interleaved_address_limit_` | Non-sharded L1 must respect the storage-region boundary; sharded L1 ignores it |
| 5 | `tt_metal/impl/buffers/buffer.cpp:60-68` | `validate_buffer_parameters`: sharded layout requires shard spec; interleaved must have none | Layout↔spec consistency check at Buffer construction |
| 6 | `tt_metal/impl/buffers/buffer.cpp:157-158` | Sub-device-bound buffers must be sharded L1 | `Buffer::create` with `sub_device_id` set |

### 1.2 Available `TensorMemoryLayout` enum values

```cpp
INTERLEAVED       // 0 — DRAM-style addressing across all banks
                  //     For L1: every L1 bank holds a slice; default for L1
SINGLE_BANK       // 1 — one specific bank; rarely used
HEIGHT_SHARDED    // 2 — split tensor along height dim across cores
WIDTH_SHARDED     // 3 — split along width dim
BLOCK_SHARDED     // 4 — split into 2D blocks across a core grid
```

For L1, all of `INTERLEAVED`, `HEIGHT_SHARDED`, `WIDTH_SHARDED`, `BLOCK_SHARDED`
are legal at the allocator level. For `L1_SMALL`, only the three sharded variants.

### 1.3 Allocator is mostly permissive

The allocator itself rarely refuses a sharded L1 buffer that has a valid
`ShardSpec` / `BufferDistributionSpec`. The constraints that bite in practice
live one layer up.

---

## 2. Operation layer (compute gates)

This is where most "your buffer must be INTERLEAVED" errors come from. Every
ttnn op's `validate()` function declares its own input layout requirements. There
is **no global convention** — each op file enforces its own rules. The pattern is
"opt-in": an op supports sharded inputs only if its program factory has been
written to handle them.

### 2.1 Common ops requiring INTERLEAVED L1

These ops will `TT_FATAL` if you pass them sharded L1 inputs:

| Op | File:line | Assertion text |
|---|---|---|
| `matmul` (multicast / 1d / 2d) | `ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:856-859` | `Input tensor B must have INTERLEAVED memory layout, got: {}` |
| `matmul` (generic) | `…matmul_op.cpp:2353-2363` | Inputs A, B, and Output must all be INTERLEAVED |
| `embedding` | `ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp:26-27` | `Embedding does not currently support sharded inputs` / `…sharded weights` |
| `binary` (broadcast) | `ttnn/cpp/ttnn/operations/eltwise/binary/device/broadcast_height_and_width_multi_core_program_factory.cpp:167` | `src1_buffer must be interleaved` |
| `interleaved_to_sharded` | `ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp:30` | `Input tensor memory layout must be INTERLEAVED but got {}` |
| `rotary_embedding_llama` (prefill mode) | `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_device_operation.cpp:98` | Input must be INTERLEAVED in prefill |

### 2.2 Common ops requiring SHARDED L1

These ops will `TT_FATAL` if you pass them interleaved inputs:

| Op | File:line | Assertion text |
|---|---|---|
| `nlp_concat_heads_decode` | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp:33` | `Input tensor must be sharded` |
| `nlp_create_qkv_heads_decode` | (same family) | sharded inputs required |
| `rotary_embedding_llama` (decode mode) | `…rotary_embedding_llama_device_operation.cpp:60` | Input must be HEIGHT_SHARDED |
| `sharded_to_interleaved` | (counterpart of above) | Input must be sharded |

### 2.3 Ops accepting both

`reshape`, `clone`, `to_layout`, `to_memory_config`, several `eltwise_unary`
variants — these branch on the input's memory layout at validate time and pick
the appropriate kernel. Pass either layout; they'll do the right thing or
internally invoke a layout conversion.

---

## 3. The empirical conventions

These aren't enforced by the framework — they're patterns the repo's ops follow.

### 3.1 Pattern A — decode-path attention ops

Designed **sharded-first**. The fast path requires sharded inputs because the
op parallelizes over batch via core-grid sharding.

Examples: `nlp_create_qkv_heads_decode`, `nlp_concat_heads_decode`,
`paged_cache::update`, `sdpa_decode`, `rotary_embedding_llama` in decode mode.

### 3.2 Pattern B — weight-bearing matmul / embedding / multicast ops

Designed **interleaved-first**. Weights typically live in DRAM and are read
through interleaved address generators that broadcast to every core. Teaching
these ops to consume sharded weights requires writing a new program factory.

Examples: `matmul` (multicast variants), `embedding`, multi-core `binary`
broadcast.

### 3.3 Pattern C — layout conversion ops

Always require exactly one layout on input. They exist *to* convert layouts, so
there's no "accept both" option.

Examples: `interleaved_to_sharded`, `sharded_to_interleaved`, `reshard`.

### 3.4 Pattern D — element-wise / norm

Usually accept both but with internal branches that have their own assertion
paths. The `broadcast` and `multi-core` *variants* of binary ops often add
fresh INTERLEAVED-only requirements that the base op doesn't impose.

### 3.5 Pattern E — mode-dispatched ops

The same op may flip its requirements based on a `mode` flag. RoPE is the
canonical example: prefill wants interleaved, decode wants sharded. The op's
`validate()` reads the mode and dispatches.

---

## 4. How to predict whether an op accepts your buffer

For each op a buffer flows through, in order:

1. Open the op's device-operation file (`…device/…_device_operation.cpp`).
2. Find its `validate()` function (or `validate_with_output_tensors`).
3. Read every `TT_FATAL` / `TT_ASSERT` that mentions `memory_config`, `layout`,
   `is_sharded`, `INTERLEAVED`, or `HEIGHT/WIDTH/BLOCK_SHARDED`. Each one is a
   constraint your buffer must satisfy.

There is no shortcut. The graph of "which ops accept which layouts" is not
published anywhere in tt-metal; the source of truth is the assertion text.

---

## 5. Bypassing an assertion

Removing or relaxing a `TT_FATAL` is *not* the same as adding support. The
kernel that runs after the validate likely makes layout-dependent assumptions:

- **Address generators** — interleaved address gen calls
  (`InterleavedAddrGenFast`, `InterleavedAddrGen`) compute bank+offset
  differently from the sharded path (which uses a precomputed core mapping). If
  the kernel was written for interleaved and you feed it sharded, the addresses
  it reads/writes will be wrong even though the validate "passed".
- **CB sizing** — circular buffers are sized in `*_program_factory.cpp` based on
  assumed shard shapes vs interleaved tile counts. A mismatch shows up as
  partial outputs or corruption, not a crash.
- **Multicast / reader-kernel patterns** — multicast matmul broadcasts B from
  one bank to many cores. If B is sharded across cores, the broadcast is
  meaningless and you'll read undefined memory.

If you bypassed an `Input tensor B must have INTERLEAVED memory layout`
assertion in `matmul_op.cpp` and saw "gibberish output" or "weird results", the
kernel is computing addresses for the wrong layout. The correct fix is to
either:

1. Insert an `interleaved_to_sharded` / `sharded_to_interleaved` conversion op
   before the matmul (cheap; ttnn provides this).
2. Add a sharded variant to the matmul's program factory and dispatch on layout
   in `validate()`. This is what "Add sharded support to op X" PRs in the
   tt-metal repo do — see e.g. `Add Conv3d sharding support by removing
   interleaved-only restrictions (fixes #34943)` for the shape of such a
   change.

---

## 6. What this means for the adaptive L1 KV cache work

The adaptive KV scheme allocates `HEIGHT_SHARDED L1` buffers (legal at the
allocator level — Layer 1 gate 1 says yes). These tensors are read by SDPA
decode (Pattern A — sharded-first, legal at Layer 2) and written by
`paged_cache::update_cache`.

### 6.1 The `paged_update_cache` validate had to be relaxed

`paged_update_cache` was an op that **looked** like Pattern B (interleaved-only)
because its `validate()` originally said:

```cpp
TT_FATAL(
    cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
    "Only interleaved cache is supported");
```

This is the assertion that was relaxed in commit `0d8d04c5a2` to:

```cpp
TT_FATAL(
    cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
        (cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
         cache_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::L1),
    "Cache must be INTERLEAVED (DRAM/L1) or HEIGHT_SHARDED in L1; …");
```

Located at `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/paged_cache_operation.cpp:59-64`.

### 6.2 Why this bypass is actually safe

Section 5 of this doc warns that bypassing layout assertions is dangerous
because kernels often hard-code address-gen for one layout. **For
`paged_update_cache` specifically, that warning does not apply.** Inspecting
the program factory and kernel:

`paged_update_cache_program_factory.cpp:193`:
```cpp
TensorAccessorArgs(dst_buffer).append_to(reader_compile_time_args);
```

`kernels/dataflow/reader_update_cache_interleaved_start_id.cpp:38, 55`:
```cpp
constexpr auto s0_args = TensorAccessorArgs<18>();
const auto s0 = TensorAccessor(s0_args, cache_addr, cache_tile_bytes);
```

The kernel uses the unified `TensorAccessor` API. `TensorAccessorArgs(buffer)`
reads the buffer's actual memory layout at program-construction time and
encodes it into compile-time args, which the kernel's `TensorAccessor` then
uses to dispatch the correct address-gen path internally (interleaved or
sharded). The reader-writer kernels are **layout-agnostic** despite the
`interleaved_start_id` in their filename — the name dates from before the
unified accessor migration.

So the original `"Only interleaved cache is supported"` assertion was leftover
conservatism. The kernel was already capable of handling sharded; only the
validate refused to admit it.

### 6.3 Lesson — assertion text is not always accurate

This is a worked example of the broader pattern: a `TT_FATAL` saying *"only X
is supported"* might mean any of:

1. The kernel actually requires X (bypassing breaks correctness silently).
2. The kernel handles both, but only X was tested when the op landed
   (bypassing is safe, modulo testing).
3. The kernel migrated to a unified accessor since the op landed but no one
   relaxed the assertion (bypassing is safe and the assertion is a bug —
   should be a PR).

Distinguishing them requires reading the program factory and the kernel(s),
not just the assertion. `paged_update_cache`'s case is **#3** — the kernel was
modernized to `TensorAccessor` but the validate wasn't.

### 6.4 So is the layout layer a constraint for adaptive KV?

It *was*, before the bypass. After bypass: no. The KV tensor flows through:

| Hop | Op | Layout requirement | Status |
|---|---|---|---|
| Allocate | `MeshBuffer::create` with sharded memcfg | sharded ok | ✓ (Layer 1) |
| Write | `paged_update_cache` | originally INTERLEAVED-only; **bypassed** | ✓ (kernel supports both) |
| Read | `sdpa_decode` | sharded ok (Pattern A) | ✓ |

The bypass is what makes the whole pipeline work. After it, the remaining
constraint is the **shared L1 algorithm-space** (covered in
[per_core_validate_walkthrough.md §7-bis](./per_core_validate_walkthrough.md)),
which is an allocator-internals concern that survives regardless of layout
permissions.

---

## 7. Pointers

| Concern | File |
|---|---|
| Layer-1 storage gates | `tt_metal/impl/allocator/{allocator,bank_manager}.cpp`, `tt_metal/impl/buffers/buffer.cpp` |
| Layer-2 op gates | each op's `…/device/…_device_operation.cpp::validate()` |
| `TensorMemoryLayout` enum | `tt_metal/api/tt-metalium/buffer_types.hpp` |
| Conversion op reference | `ttnn/cpp/ttnn/operations/data_movement/sharded/` |
| Decode-mode shard example | `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/` (mode-dispatched) |
| Companion docs | [`per_core_validate_walkthrough.md`](./per_core_validate_walkthrough.md), [`l1_kv_cache_architecture.md`](./l1_kv_cache_architecture.md) |

# Multi-allocator L1 changes (single AllocatorID → per-tier AllocatorIDs)

Commits: `e49b79d` ("improve allocation", C++ infrastructure + Python tier→id) and
`e80a4a5` ("Improve allocation 2", interleaved-adaptive layout + flag plumbing), on
branch `l1-kv-cache`.

## 1. Problem being solved

The adaptive L1 KV cache builds N tiers on physically disjoint core sets (one tier per
headroom class). Originally every tier shared a single L1 allocator (`AllocatorID{1}`)
backed by one `FreeListOpt`. That single shared address line stacked the *cumulative*
depth of all tiers onto one axis, so a tier living on cores A would push a tier living on
disjoint cores B below B's CB-region floor, tripping the `program.cpp` "circular buffers
clash with L1 buffers" assert at runtime. This forced a large `runtime_pad_bytes` to keep
KV above all CB regions, which is "Problem 1: low SRAM utilization" from
`perf_walkthrough_l1_vs_dram.md`.

The fix is to give each tier its own L1 allocator (independent free list / address line)
so each tier stacks only within its own cores' physical gap, plus make the cross-allocator
conflict accounting core-aware so disjoint tiers do not reserve address space on cores
they do not occupy.

Net effect (measured): viable `runtime_pad_bytes` floor dropped ~576 KB → ~32 KB (~18x),
max KV capacity 192 → 256 tokens at the committed default. See
[l1-kv-per-tier-allocator memory] and `perf_walkthrough_l1_vs_dram.md` §3.

## 2. The single → multiple AllocatorID change, by layer

### A. Provision N L1 allocators with a clique dependency graph
`tt_metal/impl/allocator/l1_banking_allocator.cpp` (`init_compute_and_storage_l1_bank_manager`)

- `AllocatorID{0}` = default (model intermediates, interleaved buffers).
- `AllocatorID{1..K}` = one dedicated allocator per adaptive KV tier, `K =
  kNumL1KvTierAllocators = 8`.
- All K+1 allocators share the same physical L1 address space but each tracks an
  independent free list.
- The dependency graph is a **clique**: every allocator depends on every other. Combined
  with core-aware subtraction (layer C), tier↔tier edges are no-ops for disjoint tiers
  (so tiers pack independently) but stay correct if the disjointness invariant is ever
  violated. The model↔tier edges are load-bearing: they stop KV on disjoint cores from
  inflating `AllocatorID{0}`'s counter and pushing model intermediates above CB regions.
- `K=8` covers the ≤5 headroom gap classes plus margin. MUST stay in sync with
  `MAX_KV_TIER_ALLOCATORS` in `models/tt_transformers/tt/attention.py`.

### B. `allocator_id` plumbing: MemoryConfig → Buffer → Allocator
The tier's chosen allocator id rides on the tensor's `MemoryConfig` and must survive every
rebuild on the way to the allocator:

- `MemoryConfig` carries an `allocator_id` field (default 0); pybind exposes it
  (`ttnn-pybind/tensor.cpp`).
- `ttnn/core/tensor/tensor_spec.cpp` propagates `allocator_id()` through all 4 MemoryConfig
  rebuild sites (sharded-spec rebuild, etc.). This was the silent bug: rebuilds dropped the
  id back to 0, so KV never reached `AllocatorID{1}` until fixed.
- `Buffer` gains an `allocator_id_` field + `allocator_id()` accessor and `create(...,
  uint32_t allocator_id = 0)` parameter (`tt-metalium/buffer.hpp`, `buffers/buffer.cpp`).
- `Allocator::allocate_buffer` (`allocator.cpp`) now passes `AllocID{buffer->allocator_id()}`
  to the L1 `BankManager` instead of the old hardcoded `kDefaultAllocId{0}`;
  `deallocate_buffer` likewise routes to the buffer's allocator id.

### C. Core-aware dependency subtraction
`tt_metal/impl/allocator/bank_manager.{hpp,cpp}`

- `compute_merged_allocated_ranges(allocator_id, buffer_cores)` and
  `compute_available_addresses(..., buffer_cores)` now take the requesting buffer's cores.
- New member `l1_buffer_cores_` records, per allocator, `{address → CoreRangeSet}` for
  every live L1 allocation (recorded on allocate, erased on free).
- For an L1 **sharded** request (non-empty `buffer_cores`), a dependent allocation is
  subtracted only if its recorded cores intersect `buffer_cores` (or it was recorded with
  the empty-set "interleaved / all-cores" sentinel, which always conflicts). Computed
  fresh each call (result depends on `buffer_cores`).
- For DRAM/TRACE/L1_SMALL, and for L1 **interleaved** requests (empty `buffer_cores`
  sentinel = touches every bank), it falls back to the cached "include every dependent
  allocation" path.

This is what lets two tiers on disjoint cores ignore each other's address ranges.

### D. Alignment fix (the semaphore FATAL)
`allocator_algorithm.hpp` + `bank_manager.cpp::allocate_buffer`

- Added public `alignment()` and `min_allocation_size()` getters on the allocator
  algorithm.
- The multi-allocator address-selection path now sizes/aligns with `alloc->alignment()`
  and `alloc->min_allocation_size()` instead of `BankManager.alignment_bytes_`. On
  Blackhole the BankManager L1 alignment is 16 B but `FreeListOpt` is initialized with the
  DRAM alignment (64 B); without this a small (e.g. 4 B semaphore) buffer placed near the
  top of a free block is chosen at a 16 B-aligned address that `allocate_at_address` then
  rejects because its internally-rounded size overruns the block (the
  `bank_manager.cpp:485 address.has_value()` FATAL).

### E. Aggregate-across-allocators queries
`tt_metal/impl/allocator/allocator.cpp`

- `get_lowest_occupied_l1_address` and `lowest_occupied_l1_address_for_cores` now loop over
  `l1_manager_->allocator_ids()` and take the min across every allocator, instead of
  querying a hardcoded `{0}`/`{1}` subset. The per-cores variant feeds the `program.cpp`
  CB-clash validate, so missing a tier here could let a tier buffer overlap a CB region
  without the assert firing.
- `BankManager::allocator_ids()` accessor added (`bank_manager.hpp`).
- `extract_state` uses `extract_merged_state()` for L1 (merge all allocators) instead of
  only `AllocatorID{0}`.

### F. Relaxed stale single-allocator guards
`bank_manager.cpp`

- `shrink_size` / `reset_size` dropped their `TT_FATAL(num_allocators() == 1, ...)` guards
  and now operate on the passed `allocator_id` (default `AllocID{0}`), matching their
  existing `get_allocator_from_id` bodies. These were a guaranteed crash-on-contact once
  N>1.

### G. Python side — tier → allocator_id
`models/tt_transformers/tt/attention.py`

- e49b79d (sharded path, `_build_adaptive_l1_memcfg_tiers`): each accepted tier gets
  `allocator_id = tier_index + 1`, clamped to `MAX_KV_TIER_ALLOCATORS = 8` (with a warning;
  `get_allocator_from_id` FATALs on id ≥ num_allocators). The id is set on the tier's
  `ttnn.MemoryConfig`, so the K/V tensors and the re-seeded sink inherit it.
- e80a4a5 (interleaved-adaptive, option 2): the adaptive cache can instead use ONE
  INTERLEAVED L1 tier spanning all banks (`ttnn.L1_MEMORY_CONFIG`, `allocator_id=0`), sized
  to `l1_kv_total_size` (sink + window), with no per-core sharding gate. This reuses the
  same machinery (sink/ring/l1_only, n-tier SDPA reader whose `TensorAccessor` handles an
  interleaved tier transparently) and coexists with the model's interleaved buffers,
  reaching higher capacity than sharding without sharded source-core read serialization.
  e80a4a5 also plumbs the enabling flag through `conftest.py`, `simple_text_demo.py`,
  `common.py`, and `model_config.py`. (This flag was later folded into the single
  `--l1_kv_mode` selector; see `l1_kv_modes.md`.)

## 3. Invariants and scoped-out items

- **Disjointness invariant:** each compute core belongs to exactly one headroom class, so
  each L1 KV tier occupies a unique core set. The clique's tier↔tier edges therefore no-op
  under core-aware subtraction; they exist only as a correctness backstop.
- **`override_state` (allocator.cpp):** still restores merged L1 state onto `AllocatorID{0}`
  only. This is conservative (over-reserves, never under-reserves, cannot miss a CB clash)
  and is not exercised by single-device trace decode. Its only caller is unit-mesh
  aggregation (`unit_mesh_utils.cpp::synchronize_parent_allocator_with_submeshes`). Must be
  generalized to per-allocator restore before unit-mesh aggregation runs with N>1 L1
  allocators.
- **K=8 cap:** if a future model yields >8 tiers, the overflow tier is clamped onto id 8
  (shared line, clash risk) with a warning. Bump `kNumL1KvTierAllocators` (C++) and
  `MAX_KV_TIER_ALLOCATORS` (Python) together.
- **Sub-device allocators** reuse `init_compute_and_storage_l1_bank_manager`, so they
  inherit the N-allocator clique automatically.

## 4. File map

| file | change |
|---|---|
| `impl/allocator/l1_banking_allocator.cpp` | provision K+1 L1 allocators, clique deps (A) |
| `impl/allocator/bank_manager.{hpp,cpp}` | core-aware subtraction, `l1_buffer_cores_`, alignment fix, `allocator_ids()`, relaxed FATALs (C,D,F) |
| `impl/allocator/algorithms/allocator_algorithm.hpp` | `alignment()` / `min_allocation_size()` getters (D) |
| `impl/allocator/allocator.cpp` | per-buffer `allocator_id` routing, aggregate lowest-occupied queries, merged extract_state, override_state note (B,E) |
| `api/tt-metalium/buffer.hpp`, `impl/buffers/buffer.cpp` | `allocator_id_` field + create param (B) |
| `ttnn/.../memory_config/memory_config.{hpp,cpp}`, `ttnn-pybind/tensor.cpp` | MemoryConfig `allocator_id` field + pybind (B) |
| `ttnn/core/tensor/tensor_spec.cpp`, `tensor_impl.cpp` | propagate `allocator_id` through MemoryConfig rebuilds (B) |
| `models/tt_transformers/tt/attention.py` | tier→`allocator_id` (e49b79d); interleaved-adaptive single tier (e80a4a5) (G) |
| `models/tt_transformers/tt/{model_config.py,common.py}`, `demo/{conftest.py,simple_text_demo.py}` | flag plumbing (e80a4a5) (G) |

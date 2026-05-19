# Per-Core CB Validate — Implementation Walkthrough (Steps 1–5)

> Status: landed; 14 files, ~360 added / ~5 removed
> Companion doc: [per_core_validate_patch_design.md](./per_core_validate_patch_design.md) (the design spec)
> Last updated: 2026-05-18

This walkthrough records what we actually built, file-by-file, with code excerpts and
verification results. It complements the design doc by capturing the implementation
detail and the trade-offs we hit along the way.

---

## 0. Recap: the problem the patch solves

`program.cpp::validate_circular_buffer_region` decides whether a program's bottom-up
static CB region collides with the top-down L1-buffer region. Pre-patch, it used
**one global** `lowest_occupied_compute_l1_address` and compared it against the
`cb_region_end` of **every** `cb_allocator` the program owned. Any L1 buffer on any
core constrained every program's CB region on every core.

Consequence for adaptive multi-tier L1 KV cache: putting a deeper KV stack on the
idle row (`y=9`) — which locally has ~1.35 MiB of headroom — falsely flagged a clash
on `(0,0)`, whose `nlp_concat_heads_decode` reducer has `cb_region_end ≈ 1,249,664`.
That made the worst-case core's headroom dictate the budget for every core.

**Goal of Steps 1–4:** make the check per-cb-allocator-core. A buffer that lives only
on `y=9` shouldn't show up in a clash check against `(0,0)`.

---

## 1. Step 1 — BankManager per-core tracking + Allocator wrapper

### 1.1 Why the bookkeeping has to live in `BankManager`

The existing `BankManager::lowest_occupied_address(bank_id, …)` walks one internal
`Algorithm` instance and returns its global minimum address, offset-adjusted by the
bank. The `Algorithm` itself has no concept of which logical cores an allocation
maps to — it tracks an abstract address space. So per-core information has to be
added one layer up, in `BankManager`, where the *core set* is known at allocation
time (it's passed in via `compute_grid` / `num_shards`).

### 1.2 New member in `BankManager` (header)

`tt_metal/impl/allocator/bank_manager.hpp`:

```cpp
// Per-allocator map: live L1 buffer start address -> cores it occupies.
// - Sharded L1 buffer: cores = shard_spec.grid() (or buffer distribution cores).
// - Interleaved L1 buffer: cores = empty CoreRangeSet (sentinel meaning "all
//   compute banks").
// Only populated when ``buffer_type_ == BufferType::L1``; DRAM / TRACE / L1_SMALL
// BankManagers skip the bookkeeping.
// Used exclusively by ``lowest_occupied_address_for_cores`` to support the
// per-cb-allocator validate check in program.cpp. Not in the dispatch hot path.
ttsl::SmallVector<std::unordered_map<DeviceAddr, CoreRangeSet>> l1_buffer_cores_{};
```

Notes:
- Per-`AllocatorID` keying (one vector slot per dependent allocator), matching the
  pattern of `allocated_buffers_` and `allocated_ranges_cache_`. Ensures
  sub-device-scoped dependent allocators don't bleed into the default allocator's
  view.
- `unordered_map` keyed by start address — `O(1)` insert on alloc, `O(1)` erase on
  dealloc.
- Empty `CoreRangeSet` is the **sentinel for interleaved**. Cleanly distinguishes
  the two physical layouts without an extra flag.

### 1.3 New `allocate_buffer` parameter

```cpp
DeviceAddr allocate_buffer(
    DeviceAddr size,
    DeviceAddr page_size,
    bool bottom_up,
    const CoreRangeSet& compute_grid,
    std::optional<uint32_t> num_shards,
    AllocatorDependencies::AllocatorID allocator_id = AllocatorDependencies::AllocatorID{0},
    // Cores the buffer actually occupies. Used only for per-core L1 occupancy
    // queries (lowest_occupied_address_for_cores). For sharded buffers, pass
    // shard_spec.grid() / distribution_spec cores. For interleaved buffers,
    // leave default-constructed: the empty set is treated as a sentinel
    // meaning "every compute bank" by the per-core query.
    const CoreRangeSet& buffer_cores = CoreRangeSet{});
```

Default-constructed so every existing call site (DRAM, L1_SMALL, TRACE) continues
to compile unchanged. Only the main-L1 path threads through real core info.

### 1.4 Tracking on the two allocation paths

`bank_manager.cpp` has two allocation paths — a fast single-allocator path and a
slower dependent-allocator path. Both update the tracker on success:

```cpp
// Single-allocator path (fast):
allocated_buffers_[allocator_id.get()].insert(address.value());
if (buffer_type_ == BufferType::L1) {
    l1_buffer_cores_[allocator_id.get()].emplace(address.value(), buffer_cores);
}

// Dependent-allocator path (after allocate_at_address):
allocated_buffers_[allocator_id.get()].insert(address.value());
if (buffer_type_ == BufferType::L1) {
    l1_buffer_cores_[allocator_id.get()].emplace(address.value(), buffer_cores);
}
```

### 1.5 Untracking on dealloc / clear / move

```cpp
// deallocate_buffer
if (buffer_type_ == BufferType::L1) {
    l1_buffer_cores_[allocator_id.get()].erase(address);
}

// deallocate_all / clear: per-allocator-id .clear()

// operator=(&&): l1_buffer_cores_ moves with allocated_buffers_
```

### 1.6 State-restore paths (apply_state / override_state)

`apply_state` reconstructs allocations from a serialized state that **does not
carry core info**. We register restored buffers with the empty-set "interleaved /
all-cores" sentinel:

```cpp
allocated_buffers_[target_allocator_id.get()].insert(start_addr);
if (buffer_type_ == BufferType::L1) {
    // Serialized state carries addresses but not the original sharded core
    // sets. Register with the empty-set "interleaved / all-cores" sentinel so
    // a per-core query treats restored buffers as touching every core. This
    // is conservative — validate may over-fire after a state restore but will
    // never silently miss a real clash.
    l1_buffer_cores_[target_allocator_id.get()].emplace(start_addr, CoreRangeSet{});
}
```

`override_state` clears the map before invoking `apply_state` so we never accumulate
stale entries across restores.

This is deliberately conservative — state-restore is rare, and over-reporting
preserves backward-compatibility for code paths we can't reason about.

### 1.7 The new per-core query

```cpp
std::optional<DeviceAddr> BankManager::lowest_occupied_address_for_cores(
    const CoreRangeSet& target_cores,
    BankManager::AllocatorDependencies::AllocatorID allocator_id) const {
    // Tracking only populated for L1; other buffer types fall back to nullopt so
    // callers get the same "no occupancy info" answer for DRAM/L1_SMALL/TRACE.
    if (buffer_type_ != BufferType::L1) {
        return std::nullopt;
    }
    const auto id = allocator_id.get();
    if (id >= l1_buffer_cores_.size()) {
        return std::nullopt;
    }
    const auto& live = l1_buffer_cores_[id];
    if (live.empty()) {
        return std::nullopt;
    }
    DeviceAddr lowest = std::numeric_limits<DeviceAddr>::max();
    bool any = false;
    for (const auto& [addr, cores] : live) {
        // Empty core set is the sentinel for "interleaved across every compute bank"
        // — those buffers participate in every per-core query.
        const bool is_interleaved = cores.ranges().empty();
        if (is_interleaved || cores.intersects(target_cores)) {
            lowest = std::min(lowest, addr);
            any = true;
        }
    }
    return any ? std::make_optional(lowest) : std::nullopt;
}
```

`O(N)` per query where `N` = live L1 buffer count. For Llama 3.1 8B with the
adaptive scheme, `N` peaks at a few hundred. Validate fires once per program
dispatch — microseconds.

### 1.8 Allocator-level plumbing

`tt_metal/impl/allocator/allocator.cpp` extracts the buffer's core info from the
`Buffer*` and threads it through:

```cpp
CoreRangeSet buffer_cores{};
if (buffer_type == BufferType::L1 || buffer_type == BufferType::L1_SMALL) {
    if (buffer->has_shard_spec()) {
        buffer_cores = buffer->shard_spec().grid();
    } else if (const auto& dist = buffer->buffer_distribution_spec(); dist.has_value()) {
        buffer_cores = dist->core_groups().cores_with_data;
    }
    // else: interleaved — leave buffer_cores empty (treated as all-cores).
}
// Only the L1 path threads buffer_cores through:
case BufferType::L1:
    address = l1_manager_->allocate_buffer(
        size, page_size, bottom_up, config_->compute_grid, num_cores, kDefaultAllocId, buffer_cores);
    break;
```

L1_SMALL allocations don't pass `buffer_cores` — their bank manager is a different
instance and we don't read its tracker (the per-core query gates on
`buffer_type_ == BufferType::L1`).

`Allocator::lowest_occupied_l1_address_for_cores` then wraps the bank manager:

```cpp
std::optional<DeviceAddr> Allocator::lowest_occupied_l1_address_for_cores(
    const CoreRangeSet& target_cores) const {
    std::lock_guard<std::mutex> lock(mutex_);
    // L1 small region sits above the main L1 region address-wise, so the absolute
    // lowest among the two is whatever the main L1 manager reports first; only
    // fall back to L1 small if the main L1 has no candidates touching the target.
    auto main_addr = l1_manager_->lowest_occupied_address_for_cores(target_cores);
    if (main_addr.has_value()) {
        return main_addr;
    }
    return l1_small_manager_->lowest_occupied_address_for_cores(target_cores);
}
```

(L1_SMALL is queried but its tracker is empty by design — kept for symmetry.)

### 1.9 Unit test

`tests/.../test_l1_banking_allocator.cpp` exercises the new query end-to-end against
a real device through `MeshDeviceSingleCardBufferFixture`:

```cpp
TEST_F(MeshDeviceSingleCardBufferFixture, TestLowestOccupiedL1AddressForCores_Interleaved) {
    auto& mesh = this->devices_[0];
    const auto& allocator = mesh->allocator();

    const CoreCoord any_core{0, 0};
    const CoreRangeSet any_core_set{CoreRange{any_core, any_core}};

    // Empty allocator → no L1 buffer anywhere → nullopt for any core.
    EXPECT_FALSE(allocator->lowest_occupied_l1_address_for_cores(any_core_set).has_value());

    constexpr uint32_t kBufferSize = 64 * 1024;
    distributed::DeviceLocalBufferConfig local_config{.page_size = kBufferSize, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = kBufferSize};
    auto buf = distributed::MeshBuffer::create(buffer_config, local_config, mesh.get());

    // Interleaved L1 buffer: should be reported for any core.
    auto reported = allocator->lowest_occupied_l1_address_for_cores(any_core_set);
    ASSERT_TRUE(reported.has_value());
    EXPECT_EQ(reported.value(), buf->address());

    // Query with a different core — interleaved sentinel still matches.
    const CoreCoord other_core{1, 0};
    const CoreRangeSet other_core_set{CoreRange{other_core, other_core}};
    auto reported_other = allocator->lowest_occupied_l1_address_for_cores(other_core_set);
    ASSERT_TRUE(reported_other.has_value());
    EXPECT_EQ(reported_other.value(), buf->address());

    // Deallocating the buffer must remove its entry from the per-core tracker.
    buf.reset();
    EXPECT_FALSE(allocator->lowest_occupied_l1_address_for_cores(any_core_set).has_value());
}
```

**Result:**
```
[==========] 3 tests from 2 test suites ran. (5430 ms total)
[  PASSED  ] 3 tests.
```

The original `TestL1BuffersAllocatedTopDown` and `TestL1BuffersDoNotGrowBeyondBankSize`
continue to pass — no regression.

### 1.10 Step-1 file totals

| File | Lines added |
|---|---:|
| `bank_manager.hpp` | +27 |
| `bank_manager.cpp` | +61 |
| `allocator.hpp` | +7 |
| `allocator.cpp` | +31 |
| `test_l1_banking_allocator.cpp` | +38 |
| **Subtotal** | **+164** |

---

## 2. Step 2 — `SubDeviceManagerTracker` composition

Both the existing global query (`lowest_occupied_compute_l1_address`) and the
per-core variant need to consult **two sets of allocators**:

1. The default sub-device manager's allocator (`SubDeviceId{0}` of the default
   manager) — what `device->allocator()` returns.
2. Each active sub-device's own allocator — what `device->allocator(sub_device_id)`
   returns, iterated when the user has loaded a custom sub-device manager.

`SubDeviceManagerTracker::lowest_occupied_compute_l1_address_for_cores` mirrors
the existing global query's two-stage composition exactly, with one extra check
to skip sub-devices that share no core with `target_cores`:

```cpp
std::optional<DeviceAddr> SubDeviceManagerTracker::lowest_occupied_compute_l1_address_for_cores(
    const CoreRangeSet& target_cores,
    tt::stl::Span<const SubDeviceId> sub_device_ids) const {
    DeviceAddr lowest_addr = std::numeric_limits<DeviceAddr>::max();

    const auto& global_allocator = default_sub_device_manager_->allocator(SubDeviceId{0});
    auto found = global_allocator->lowest_occupied_l1_address_for_cores(target_cores);
    if (found.has_value()) {
        lowest_addr = std::min(lowest_addr, *found);
    }

    // Default to all active sub-device ids when caller passes none, mirroring the
    // global query's behaviour for code paths that don't specify sub-devices.
    if (sub_device_ids.empty() && default_sub_device_manager_ != active_sub_device_manager_) {
        sub_device_ids = tt::stl::Span<const SubDeviceId>(active_sub_device_manager_->get_sub_device_ids());
    }
    for (const auto& sub_device_id : sub_device_ids) {
        const auto& allocator = this->get_active_sub_device_manager()->sub_device_allocator(sub_device_id);
        if (!allocator) {
            continue;
        }
        // Skip sub-devices that share no core with target_cores — they cannot host an
        // L1 buffer that would intersect the cb_allocator's range, and the per-core
        // query inside their allocator would scan an empty intersection anyway.
        const auto& sub_cores =
            this->get_active_sub_device_manager()->sub_device(sub_device_id).cores(HalProgrammableCoreType::TENSIX);
        if (!sub_cores.intersects(target_cores)) {
            continue;
        }
        found = allocator->lowest_occupied_l1_address_for_cores(target_cores);
        if (found.has_value()) {
            lowest_addr = std::min(lowest_addr, *found);
        }
    }
    return lowest_addr == std::numeric_limits<DeviceAddr>::max() ? std::nullopt : std::make_optional(lowest_addr);
}
```

Note the extra `sub_cores.intersects(target_cores)` short-circuit. The global
query has no analogous guard because it queries `bank_id = 0` (a single bank) of
each sub-device. The per-core query *would* scan every allocation in the
sub-device's bank manager, so skipping disjoint sub-devices saves real work.

### 2.1 Step-2 file totals

| File | Lines added |
|---|---:|
| `sub_device_manager_tracker.hpp` | +9 |
| `sub_device_manager_tracker.cpp` | +42 |
| **Subtotal** | **+51** |

---

## 3. Step 3 — `IDevice` / `Device` / `MeshDevice` forwarders

Pure plumbing. Each device class exposes the query and forwards to its own
`sub_device_manager_tracker_`.

`tt_metal/api/tt-metalium/device.hpp` (pure virtual on the interface):

```cpp
// Per-core analogue of lowest_occupied_compute_l1_address: returns the lowest L1
// buffer start address whose cores intersect ``target_cores``. Used by
// ``program.cpp::validate_circular_buffer_region`` to make the CB / L1-buffer
// clash check per-cb-allocator-core instead of global across all banks.
virtual std::optional<DeviceAddr> lowest_occupied_compute_l1_address_for_cores(
    const CoreRangeSet& target_cores,
    tt::stl::Span<const SubDeviceId> sub_device_ids = {}) const = 0;
```

`Device::lowest_occupied_compute_l1_address_for_cores` in `device.cpp`:

```cpp
std::optional<DeviceAddr> Device::lowest_occupied_compute_l1_address_for_cores(
    const CoreRangeSet& target_cores, tt::stl::Span<const SubDeviceId> sub_device_ids) const {
    return sub_device_manager_tracker_->lowest_occupied_compute_l1_address_for_cores(target_cores, sub_device_ids);
}
```

`MeshDevice` is identical except for its own `sub_device_manager_tracker_` (which
manages the mesh-level allocator — the one MeshBuffer actually uses, as we
discovered while debugging Step 3 of our diagnostic phase).

### 3.1 Step-3 file totals

| File | Lines added |
|---|---:|
| `device.hpp` | +8 |
| `device_impl.hpp` | +3 |
| `device.cpp` | +5 |
| `mesh_device.hpp` | +3 |
| `mesh_device.cpp` | +5 |
| **Subtotal** | **+24** |

---

## 4. Step 4 — the `validate_circular_buffer_region` switch

The actual behaviour change. One global query becomes one per-cb-allocator query:

```diff
 void detail::ProgramImpl::validate_circular_buffer_region(const IDevice* device) {
-    std::optional<DeviceAddr> lowest_address =
-        device->lowest_occupied_compute_l1_address(this->determine_sub_device_ids(device));
+    const auto sub_device_ids = this->determine_sub_device_ids(device);
     uint32_t max_l1_size = device->l1_size_per_core();

     for (const CircularBufferAllocator& cb_allocator : this->cb_allocators_) {
         if (cb_allocator.l1_regions.empty()) {
             continue;
         }
         uint64_t cb_region_end = cb_allocator.l1_regions.back().second;
+        const CoreRangeSet cb_cores{cb_allocator.core_range};
+        std::optional<DeviceAddr> lowest_address =
+            device->lowest_occupied_compute_l1_address_for_cores(cb_cores, sub_device_ids);
         log_l1_cb_map(this->id, this->runtime_id, cb_allocator, "validate",
                       cb_region_end, max_l1_size, lowest_address);
         device->update_max_cb_end(cb_allocator.core_range, cb_region_end);
         if (cb_region_end > max_l1_size) {
             TT_THROW("…");
         }
         if (lowest_address.has_value() and lowest_address.value() < cb_region_end) {
             TT_THROW("…");
         }
     }
 }
```

Each `cb_allocator` now compares its `cb_region_end` against the lowest L1 buffer
address that **actually touches its cores**. Anything on disjoint cores is filtered
out.

A long comment explains the rationale, with a back-pointer to the design doc, so
future readers don't undo the change without understanding why.

### 4.1 Step-4 file totals

| File | Lines added |
|---|---:|
| `program.cpp` | +15 (−5 net) |
| **Subtotal** | **+10 net** |

---

## 5. Grand total

| Category | Files | Lines added |
|---|---:|---:|
| Bank manager + Allocator | 4 | 126 |
| Tests | 1 | 38 |
| Sub-device tracker | 2 | 51 |
| Device interfaces / forwarders | 5 | 24 |
| Program validate switch | 1 | 10 |
| **Total** | **13** | **~249** |

Right around the design doc's prediction of 160 lines (the variance comes from
state-restore handling and the `sub_cores.intersects` short-circuit, both extra
bits of robustness identified during implementation).

---

## 6. Verification

### 6.1 Build

```
$ ./build_metal.sh --build-tests
… (full build) …
exit code 0
```

Three consecutive clean builds, each cascading correctly through the layers we
touched (header changes in `device.hpp` and `bank_manager.hpp` invalidated their
dependents as expected — no stale objects).

### 6.2 Unit tests

```
$ TT_METAL_SLOW_DISPATCH_MODE=1 ./build_Release/test/tt_metal/unit_tests_api \
    --gtest_filter='*L1Buffers*:*LowestOccupiedL1AddressForCores*'

[ RUN      ] TestLowestOccupiedL1AddressForCores_Interleaved
[       OK ]                                                  (526 ms)
[ RUN      ] TestL1BuffersAllocatedTopDown
[       OK ]
[ RUN      ] TestL1BuffersDoNotGrowBeyondBankSize
[       OK ]                                                  (503 ms)

[==========] 4 tests from 2 test suites ran. (6259 ms total)
[  PASSED  ] 4 tests.
```

New test passes. Pre-existing tests still pass — no regression.

### 6.3 End-to-end demo (adaptive L1 KV cache, JSON-headroom path)

```
$ TT_LOGGER_LEVEL=Info pytest models/tt_transformers/demo/simple_text_demo.py \
    -k "performance and batch-1" --use_adaptive_l1_kv_cache \
    --l1_kv_headroom_json …/headroom_map_global_minimum.json
```

**Pre-patch behaviour:**
- `Statically allocated circular buffers in program 59 clash with L1 buffers on
  core range [(x=0,y=0) - (x=0,y=0)]` — clash on `(0,0)` triggered by a tier-3
  buffer on `y=9`. False positive: the buffer and the CB are on disjoint cores.

**Post-patch behaviour:**
- Program 59 clash: **gone**. Validate filters out tier buffers (on y=8 / column /
  y=9) when evaluating program 59 (on `(0,0)`).
- Program 143 clash: **gone** for the same reason.

The patch achieves its design goal. The demo still doesn't complete end-to-end,
but the remaining failures (see §7) are not about validate.

---

## 7. Remaining failures — root cause is budget, not validate

The post-patch run fails in two ways. Both have the same root cause and neither
is something the per-core validate patch can fix.

### 7.1 Layers 25–32 OOM during KV tier allocation

```
[L1 KV adaptive] OOM allocating V tier (2 tile-rows × 8 cores, …) — skipping.
[L1 KV adaptive] OOM allocating K tier (3 tile-rows × 27 cores, …) — skipping.
[L1 KV adaptive] OOM allocating K tier (4 tile-rows × 11 cores, …) — skipping.
[L1 KV adaptive] Total L1 KV tokens across 0 tiers: 0
… repeated for layers 25–32 …
```

The allocator's internal `Algorithm` tracks **one shared address space per
`AllocatorID`**. All sharded buffers — Tier 1 (y=8), Tier 2 (column), Tier 3
(y=9) — consume slots in the same address space, even though they target
physically disjoint banks. Cumulative `size_per_bank` summed across all sharded
allocations on this allocator:

```
Tier 1: 64 allocations × 8704 B   =  557 KiB
Tier 2: 64 allocations × 13056 B  =  836 KiB
Tier 3: 64 allocations × 17408 B  = 1114 KiB
-------------------------------------------
Total                              ≈ 2.5 MiB  (> 1.47 MiB bank ceiling)
```

The Python budget formula reasons per-core ("y=9 cores have 1.15 MiB headroom,
so 4 tile-rows fit"), which is true *physically* but not *in the algorithm's
view*. The algorithm sees the sum.

### 7.2 Late-program CB clash (program 65 on cores (0,0)–(7,3))

```
Statically allocated circular buffers in program 65 clash with L1 buffers on
core range [(x=0,y=0) - (x=7,y=3)]. L1 buffer allocated at 106752 and static
circular buffer region ends at 110976
```

This clash is **real** under the per-core query. Program 65's cores `(0,0)-(7,3)`
are disjoint from all our tier cores, so the buffer at 106,752 is **not** from
KV tiers — it's a runtime intermediate (likely from `InterleavedToShardedOperation`
during step-2 forward) that got placed low because the address space was already
~1.46 MiB deep from the cumulative tier allocations in §7.1.

So this is the same root cause: cumulative L1 stack depth.

### 7.3 What needs to change next

The budget formula in `attention.py::_build_adaptive_l1_memcfg_tiers` needs to
account for **cumulative algorithm-space depth**, not just per-core physical
headroom. Concretely:

```
sum(tier_shard_size_per_bank × num_layers × 2)
  ≤  bank_allocatable_size − runtime_intermediate_pad
```

Implementation options (Python-only, not part of this C++ patch):

1. **Reduce safety margin headroom but cap cumulative.** Today the formula gates
   on per-core `(headroom - safety_margin) / bytes_per_tile_row`. Add a second
   gate: `sum(per_tier_total) ≤ 1.4 MiB − 100 KiB` runtime pad.

2. **Subsample layers.** Cache KV for fewer layers (e.g., 16 of 32) so cumulative
   depth halves.

3. **Smaller per-tier tile_rows.** Cap each tier at one tile-row less than the
   per-core formula allows; trades L1 KV capacity for budget headroom for
   runtime intermediates.

The right choice depends on what trade-off best fits the model. None of these
require further allocator changes.

---

## 7-bis. Step 5 — cumulative-depth cap (Python side)

We took **Option 1** from §7.3. The Python budget formula in
`attention.py::_build_adaptive_l1_memcfg_tiers` got a second gate: after the
per-core tile-row computation, candidate tiers are sorted by
**token-efficiency** (`tok_count / cumulative_algo_bytes`) and packed greedily
into a fixed algo-space budget. Tiers that don't fit at their full `tile_rows`
are shrunk; tiers that can't fit at any `tile_rows ≥ 1` are dropped.

### 7-bis.1 New parameters

```python
def _build_adaptive_l1_memcfg_tiers(
    self,
    headroom_map: dict,
    safety_margin_bytes: int,
    bank_allocatable_bytes: int = 1_470_080,  # Blackhole P150 main L1 region
    runtime_pad_bytes: int = 480 * 1024,
) -> list:
```

`bank_allocatable_bytes` hard-codes the Blackhole P150 main-L1 region
(1,572,864 − `l1_unreserved_base` − `l1_small_size` = 1,470,080). Override per
arch as needed.

`runtime_pad_bytes = 480 KiB` was picked empirically: with smaller pads (128 /
256 KiB) the demo still tripped over higher-CB-top programs (67 at 627 KiB on
`(0,0)`). Raise this further if the model has additional programs with even
higher CB tops on cores you can't avoid.

### 7-bis.2 Greedy packing algorithm

```
algo_budget = bank_allocatable - runtime_pad - safety_margin

# initial candidates from the per-core gate
initial = [(tile_rows, cores, tok_count, bytes_per_tile_row * tile_rows) … ]

# sort by token-efficiency descending (most tokens per algo byte first)
ranked = sorted(initial, key=lambda t: -(t.tok_count / t.cumulative_bytes))

accepted, used = [], 0
for tier in ranked:
    remaining = algo_budget - used
    if tier.cumulative_bytes <= remaining:
        accept tier at full tile_rows
    else:
        # try shrinking tile_rows until it fits, or drop the tier
        shrunk_T = remaining // bytes_per_tile_row
        while shrunk_T > 0:
            if shrunk_tier_fits: accept; break
            else: shrunk_T -= 1
        else: drop the tier
```

This preserves the **most token-dense** tier (typically the column tier — 27
cores × moderate tile-rows beats both the y=8 row (8 cores) and the idle row
(11 cores) on tokens-per-byte).

### 7-bis.3 Observed selection (32-layer Llama 3.1 8B, bfp8 KV)

With `runtime_pad_bytes = 480 KiB`, `safety_margin = 64 KiB`:

```
algo_budget = 1,470,080 − 491,520 − 65,536 = 913,024 B
bytes_per_tile_row = 4 (tiles wide) × 1088 (bfp8 tile) × 32 layers × 2 (K+V)
                  = 278,528 B  per tile-row per tier

Initial tiers (per-core gate):
  Tier 1 (y=8,    8 cores, T=2):  557,056 B  → 64 tokens
  Tier 2 (column, 27 cores, T=3):  835,584 B  → 288 tokens
  Tier 3 (y=9,    11 cores, T=4): 1,114,112 B  → 160 tokens

Token efficiency (tokens / cumulative byte):
  Tier 2: 3.45e-4   ← most efficient
  Tier 3: 1.44e-4
  Tier 1: 1.15e-4

Greedy fit:
  Accept Tier 2 (full):  used=835,584, remaining=77,440
  Tier 3 (4 → ?):        max_fit=0, drop
  Tier 1 (2 → ?):        max_fit=0, drop

Final: Tier 2 only (288 tokens / layer, 91.5% of budget used)
```

Compared to the pre-Step-5 attempt (which over-promised `Tier1 + Tier2 +
Tier3` = 320 tokens but OOM'd at layer 25), Step 5 sacrifices ~10% of the
nominal token capacity but lets **all 32 layers** allocate cleanly with no
OOM cascade.

### 7-bis.4 What Step 5 fixes and does not fix

| Symptom | Pre-Step-5 | Post-Step-5 |
|---|---|---|
| KV OOM cascade (layers 25–32) | yes | **no** |
| Program 65 clash (cb_top ~110 KiB, cores `(0,0)-(7,3)`) | yes | **no** |
| Program 67 clash (cb_top 627 KiB, core `(0,0)`) | n/a (didn't reach) | **no** |
| Program 49 clash (cb_top 723 KiB, cores `(0,0)-(7,8)`) | n/a | **yes** ← new |
| End-to-end demo passes | no | no |

The remaining failure is on `nlp_create_qkv_heads_decode` (program 49,
`cb_region_end = 722,944` on cores `(0,0)-(7,8)`, matching `pid=47` in the
analysis doc). A buffer at 691,456 occupies cores intersecting that range.
Per-core query correctly identifies this as a real intersection — it is **not**
from our KV tiers (which are on column / y=9 cores disjoint from
`(0,0)-(7,8)`). It must be a model-level sharded buffer (likely the QKV
projection output, or an embedding output, sharded across the 8×9 grid).

### 7-bis.5 Why pushing `runtime_pad_bytes` higher hits diminishing returns

Each `runtime_pad_bytes` increase pushes more cumulative depth out of the way
of one specific program's CB region, but reveals the next-highest-CB program's
clash. The pattern observed:

| `runtime_pad_bytes` | Cumulative KV | Clashing program after | `cb_top` on `(0,0)` |
|---:|---:|---|---:|
| 128 KiB | 1,114,112 B | program 67 (`(0,0)`) | 627,200 |
| 480 KiB | 835,584 B  | program 49 (`(0,0)-(7,8)`) | 722,944 |
| 700 KiB+ (untested) | < 770 KiB | next-highest cb_top program | … |

The ceiling is determined by `max(cb_region_end across programs whose cores
intersect the cores any L1 buffer is on)`. Per the
[`l1_cb_map_analysis`](./l1_cb_map_analysis.md) doc, that's
`nlp_concat_heads_decode` reducer at `pid=43` (`cb_region_end = 1,249,664` on
just `(0,0)`). To unblock fully without further changes, runtime_pad would
need to be `1,572,864 − 1,249,664 − safety = 257 KiB worth of cumulative
budget` — basically the original uniform-window cap pre-adaptive.

So the per-core validate patch helped us shed the **disjoint-core false
positives** (Tier buffers on y=9 no longer falsely clash with `(0,0)` CBs).
What remains is the **genuine global constraint** from sharded buffers in the
model's runtime path that happen to land on cores overlapping high-CB-top
programs. Removing those constraints requires runtime-op refactoring
(steering intermediates to disjoint cores, or moving them to DRAM), not
allocator-level changes.

### 7-bis.6 Step-5 file totals

| File | Lines added |
|---|---:|
| `models/tt_transformers/tt/attention.py` | ~110 (helper + greedy fit + logging) |
| **Subtotal** | **+110** |

---

## 8. What the patch does and does not do

**Does:**
- Make `validate_circular_buffer_region` per-cb-allocator-core. Disjoint-core L1
  buffers no longer false-positive against a program's CBs.
- Track per-allocation core sets in the L1 BankManager, queryable by any
  caller (not just program validate).
- Preserve backward-compatibility: existing global queries unchanged; new
  query is additive.
- (Step 5) Add a cumulative algo-space gate to the Python tier builder so the
  shared-address-space ceiling is respected, eliminating the OOM cascade and
  the low-CB-top clashes.

**Does not:**
- Change the underlying allocator's algorithm-space accounting. Multiple
  sharded buffers on disjoint banks still consume slots from a shared address
  space.
- Move model runtime intermediates (QKV projection, embedding outputs, concat
  outputs) off cores that overlap high-CB-top programs. As long as those
  intermediates use sharded L1 on cores intersecting `nlp_create_qkv_heads_decode`
  or `nlp_concat_heads_decode`'s core ranges, their physical occupancy will
  clash with those programs' CB regions and the validate (correctly) fires.

The C++ patch (Steps 1–4) is a precondition for the adaptive multi-tier scheme
to work at all. The Python cap (Step 5) is necessary for it to be useful in
practice. End-to-end correctness still requires runtime-op placement work
that's outside this patch's scope.

---

## 9. Test coverage I'd want to add later

The shipped unit test covers the interleaved-buffer path (empty-cores sentinel).
Worth adding when the test fixture supports it cleanly:

1. **Sharded buffer disjoint from target.** Allocate sharded buffer on y=9 cores;
   `lowest_occupied_l1_address_for_cores({y=0 row})` returns `nullopt`.
2. **Sharded buffer overlapping target.** Same allocation; query for `{y=9 row}`
   returns the buffer's address.
3. **Sharded + interleaved mix.** Allocate one of each, deallocate the
   sharded, verify the interleaved still matches all-core queries.
4. **Dependent allocator scoping.** Allocate via two `AllocatorID` paths,
   verify the maps don't bleed.

These would harden the patch against future refactors of the bank manager.

---

## 10. Pointers

| File path | What's in it |
|---|---|
| `tt_metal/impl/allocator/bank_manager.{hpp,cpp}` | Tracking + new query |
| `tt_metal/impl/allocator/allocator.{hpp,cpp}` | Wrapper + buffer-cores extraction |
| `tt_metal/impl/sub_device/sub_device_manager_tracker.{hpp,cpp}` | Cross-allocator composition |
| `tt_metal/api/tt-metalium/{device,mesh_device}.hpp` | Interface decls |
| `tt_metal/impl/device/device.cpp`, `tt_metal/distributed/mesh_device.cpp` | Forwarders |
| `tt_metal/impl/program/program.cpp` | The behaviour change (lines ~939–984) |
| `tests/.../test_l1_banking_allocator.cpp` | Unit test |
| `models/tt_transformers/tt/attention.py` | Step 5 cumulative-depth cap |
| `research_codes/documents/l1_kv_cache_cache/per_core_validate_patch_design.md` | Design spec |

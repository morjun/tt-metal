# Per-Core CB Validate Check — Patch Design (Option 2)

> Status: design proposal, not yet implemented
> Branch: `l1-kv-cache`
> Last updated: 2026-05-17

---

## 1. Why this patch exists

`tt_metal/impl/program/program.cpp::validate_circular_buffer_region` decides whether
a program's bottom-up static CB region collides with the top-down L1 buffer region.
Today it uses **one global** `lowest_occupied_compute_l1_address` and compares it
against the `cb_region_end` of **every** `cb_allocator` the program owns:

```cpp
std::optional<DeviceAddr> lowest_address =
    device->lowest_occupied_compute_l1_address(this->determine_sub_device_ids(device));

for (const auto& cb_allocator : this->cb_allocators_) {
    uint64_t cb_region_end = cb_allocator.l1_regions.back().second;
    if (lowest_address.has_value() && lowest_address.value() < cb_region_end) {
        TT_THROW("clash on core range {} …", cb_allocator.core_range.str(), …);
    }
}
```

`lowest_address` is the **minimum** L1 buffer start across **every** bank tracked by
the sub-device allocator. Any L1 buffer on **any** core constrains **every** program's
CB region on **every** core. Concretely on Blackhole P150 Llama 3.1 8B:

| Core | `max(cb_region_end across programs)` |
|---|---:|
| `(x=0..7, y=0)` — `nlp_concat_heads_decode` row | **1,249,664 B** |
| `(x=0..7, y=1..7)` — SDPA workers | 1,225,088 B |
| `(x=0..7, y=8)` — WQKV producer | 722,944 B |
| `(x=8..10, y=0..8)` — column / right-side | ≤ 360,832 B |
| `(x=0..10, y=9)` — idle row | ≤ 218,816 B |

Because `(0,0)`'s worst-case CB top is 1,249,664 B, **any** L1 buffer below
1,249,664 B globally trips the check. That caps per-bank L1 KV cache at
`1,572,864 − 1,249,664 = 323,200 B` — even on the idle row, which locally has
~1.35 MiB of free space. Adaptive multi-tier sizing is squashed to the
worst-case core's headroom.

**The fix in one line:** for each `cb_allocator`, only consider L1 buffers
that *actually occupy that allocator's cores*.

---

## 2. New API surface

Two new query methods (one per-buffer-aware, one per-core composed on top):

### 2.1 `Allocator::lowest_occupied_l1_address_for_cores`

```cpp
// In tt_metal/impl/allocator/allocator.hpp
std::optional<DeviceAddr> lowest_occupied_l1_address_for_cores(
    const CoreRangeSet& target_cores) const;
```

Returns the lowest start address across all currently-live L1 buffers whose
core set intersects `target_cores`. Implemented via the bank manager (§3).

### 2.2 `IDevice::lowest_occupied_compute_l1_address_for_cores`

```cpp
// In tt_metal/api/tt-metalium/device.hpp
virtual std::optional<DeviceAddr> lowest_occupied_compute_l1_address_for_cores(
    const CoreRangeSet& target_cores,
    tt::stl::Span<const SubDeviceId> sub_device_ids = {}) const = 0;
```

Composed across the default sub-device allocator **and** the active sub-device
allocators (mirroring the existing
`lowest_occupied_compute_l1_address(sub_device_ids)`). Implemented on both
`Device` and `MeshDevice`.

The existing global `lowest_occupied_compute_l1_address(...)` remains — it has
other callers (e.g. shrink-allocator paths) and is still useful for diagnostics.

---

## 3. Per-buffer core tracking in `BankManager`

`BankManager` currently stores allocated *addresses* but not *which cores each
address belongs to*. We need that mapping.

### 3.1 New member

```cpp
// In tt_metal/impl/allocator/bank_manager.hpp
struct AllocatedL1Buffer {
    DeviceAddr address;        // start address (top-down)
    DeviceAddr size_per_bank;  // shard size (sharded) or per-bank slice (interleaved)
    CoreRangeSet cores;        // empty => all compute banks (interleaved)
};

// Keyed by AllocatorID so dependent-allocator paths stay isolated.
std::unordered_map<AllocatorDependencies::AllocatorID,
                   std::vector<AllocatedL1Buffer>> l1_buffer_cores_;
```

### 3.2 Update sites

* `BankManager::allocate_buffer` (both the no-dependencies path and the
  dependencies path) already receives a `const CoreRangeSet& compute_grid` arg
  and knows `is_sharded`. On success, push back an entry:
  - sharded → `cores = shard_spec.grid()` (passed in via `compute_grid`)
  - interleaved → `cores = CoreRangeSet{}` (sentinel meaning "all banks")
* `BankManager::deallocate_buffer` already takes `address`; locate the entry
  by `address` and erase. (O(N) erase is fine; allocations are sparse.)
* `BankManager::clear` / `deallocate_all` → clear the vector.

### 3.3 New query

```cpp
// In tt_metal/impl/allocator/bank_manager.cpp
std::optional<DeviceAddr> BankManager::lowest_occupied_address_for_cores(
    const CoreRangeSet& target_cores,
    AllocatorDependencies::AllocatorID allocator_id) const {
    if (buffer_type_ != BufferType::L1) {
        return std::nullopt;  // only meaningful for L1
    }
    DeviceAddr min_addr = std::numeric_limits<DeviceAddr>::max();
    const auto it = l1_buffer_cores_.find(allocator_id);
    if (it == l1_buffer_cores_.end()) return std::nullopt;
    for (const auto& buf : it->second) {
        const bool is_interleaved = buf.cores.ranges().empty();
        if (is_interleaved || buf.cores.intersects(target_cores)) {
            min_addr = std::min(min_addr, buf.address);
        }
    }
    return min_addr == std::numeric_limits<DeviceAddr>::max()
        ? std::nullopt : std::make_optional(min_addr);
}
```

`CoreRangeSet::intersects(other)` already exists (it powers the existing
`cb_allocator.core_range.intersects(...)` calls).

---

## 4. Allocator wrapper

```cpp
// In tt_metal/impl/allocator/allocator.cpp
std::optional<DeviceAddr> Allocator::lowest_occupied_l1_address_for_cores(
    const CoreRangeSet& target_cores) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return l1_manager_->lowest_occupied_address_for_cores(target_cores);
}
```

Header declaration to match. No behaviour change for existing
`get_lowest_occupied_l1_address(bank_id)` callers.

---

## 5. SubDeviceManagerTracker composition

Mirror `SubDeviceManagerTracker::lowest_occupied_compute_l1_address`. The
sub-device's cores are intersected with `target_cores` so that sub-devices
which don't overlap the program's CB cores don't contribute.

```cpp
// In tt_metal/impl/sub_device/sub_device_manager_tracker.cpp
std::optional<DeviceAddr> SubDeviceManagerTracker::
lowest_occupied_compute_l1_address_for_cores(
    const CoreRangeSet& target_cores,
    tt::stl::Span<const SubDeviceId> sub_device_ids) const {

    DeviceAddr lowest = std::numeric_limits<DeviceAddr>::max();
    auto fold = [&](std::optional<DeviceAddr> v) {
        if (v.has_value()) lowest = std::min(lowest, *v);
    };

    fold(default_sub_device_manager_->allocator(SubDeviceId{0})
             ->lowest_occupied_l1_address_for_cores(target_cores));

    if (sub_device_ids.empty() && default_sub_device_manager_ != active_sub_device_manager_) {
        sub_device_ids = active_sub_device_manager_->get_sub_device_ids();
    }
    for (const auto& sub_device_id : sub_device_ids) {
        const auto& alloc = active_sub_device_manager_->sub_device_allocator(sub_device_id);
        if (!alloc) continue;
        const auto& sub_cores =
            active_sub_device_manager_->sub_device(sub_device_id)
                .cores(HalProgrammableCoreType::TENSIX);
        // If this sub-device shares any core with the target, query it.
        if (sub_cores.intersects(target_cores)) {
            fold(alloc->lowest_occupied_l1_address_for_cores(target_cores));
        }
    }
    return lowest == std::numeric_limits<DeviceAddr>::max()
        ? std::nullopt : std::make_optional(lowest);
}
```

---

## 6. Device & MeshDevice overrides

Pure plumbing — both forward to their respective trackers.

```cpp
// Device::lowest_occupied_compute_l1_address_for_cores
return sub_device_manager_tracker_->lowest_occupied_compute_l1_address_for_cores(
    target_cores, sub_device_ids);
```

```cpp
// MeshDevice::lowest_occupied_compute_l1_address_for_cores
return sub_device_manager_tracker_->lowest_occupied_compute_l1_address_for_cores(
    target_cores, sub_device_ids);
```

`IDevice` gets the pure-virtual declaration; existing concrete devices add the
override.

---

## 7. The actual validate change

The minimum diff in `program.cpp`:

```diff
 void detail::ProgramImpl::validate_circular_buffer_region(const IDevice* device) {
-    std::optional<DeviceAddr> lowest_address =
-        device->lowest_occupied_compute_l1_address(this->determine_sub_device_ids(device));
+    const auto sub_device_ids = this->determine_sub_device_ids(device);
     uint32_t max_l1_size = device->l1_size_per_core();

     for (const auto& cb_allocator : this->cb_allocators_) {
         if (cb_allocator.l1_regions.empty()) {
             continue;
         }
         uint64_t cb_region_end = cb_allocator.l1_regions.back().second;
         log_l1_cb_map(this->id, this->runtime_id, cb_allocator, "validate",
                       cb_region_end, max_l1_size, lowest_address);
         device->update_max_cb_end(cb_allocator.core_range, cb_region_end);
         if (cb_region_end > max_l1_size) {
             TT_THROW("Statically allocated circular buffers on core range {} grow to {} B …",
                      cb_allocator.core_range.str(), cb_region_end, max_l1_size);
         }
+        CoreRangeSet cb_cores{cb_allocator.core_range};
+        std::optional<DeviceAddr> lowest_address =
+            device->lowest_occupied_compute_l1_address_for_cores(cb_cores, sub_device_ids);
         if (lowest_address.has_value() && lowest_address.value() < cb_region_end) {
             TT_THROW(
                 "Statically allocated circular buffers in program {} clash with L1 buffers "
                 "on core range {}. L1 buffer allocated at {} and static circular buffer "
                 "region ends at {}",
                 this->id, cb_allocator.core_range.str(),
                 lowest_address.value(), cb_region_end);
         }
     }
 }
```

`log_l1_cb_map`'s `lowest_address` argument changes meaning (now per-cb-allocator
rather than global) — either rename the parameter, or accept the slight semantic
shift; the value is still "the address the assertion would compare against".

---

## 8. File-by-file summary

| File | Change | Lines (est.) |
|---|---|---:|
| `tt_metal/api/tt-metalium/device.hpp` | Add `lowest_occupied_compute_l1_address_for_cores` pure virtual. | 4 |
| `tt_metal/impl/device/device_impl.hpp` | Override decl. | 2 |
| `tt_metal/impl/device/device.cpp` | Trivial forwarder to tracker. | 6 |
| `tt_metal/api/tt-metalium/mesh_device.hpp` | Override decl. | 2 |
| `tt_metal/distributed/mesh_device.cpp` | Trivial forwarder. | 6 |
| `tt_metal/impl/sub_device/sub_device_manager_tracker.hpp` | New method decl. | 4 |
| `tt_metal/impl/sub_device/sub_device_manager_tracker.cpp` | Composed query across default + active sub-devices. | ~35 |
| `tt_metal/impl/allocator/allocator.hpp` | New method decl. | 2 |
| `tt_metal/impl/allocator/allocator.cpp` | Locked wrapper around `BankManager`. | 6 |
| `tt_metal/impl/allocator/bank_manager.hpp` | Add `AllocatedL1Buffer` struct, `l1_buffer_cores_` map, query decl. | 12 |
| `tt_metal/impl/allocator/bank_manager.cpp` | Track on allocate / untrack on deallocate / query impl / clear handling. | ~70 |
| `tt_metal/impl/program/program.cpp` | Switch validate to per-cb-allocator query. | ~10 |

Total: ~160 lines of changes across 12 files. No public Python API surface
change. `update_max_cb_end` / `get_l1_headroom_per_core` instrumentation is
unaffected.

---

## 9. Correctness, performance, risk

**Correctness.**
* Sharded buffers: `core_ranges()` from `shard_spec.grid()` is the authoritative
  set the bank manager already uses to pick banks; reusing it for tracking is
  consistent.
* Interleaved buffers: every L1 compute bank holds a slice; the sentinel
  "empty core range = all cores" makes them participate in every per-core
  query, which is the safe answer.
* Backward compat: the existing global query is untouched. The validate change
  is strictly more permissive (each program is checked against a smaller
  candidate set of buffers), so any case that previously *passed* continues
  to pass. Cases that previously *failed* unnecessarily (the adaptive KV
  scenario) now pass when the offending buffer doesn't live on the program's
  cores.

**Performance.**
* `lowest_occupied_address_for_cores` is O(N · R) where N = live L1 buffers
  and R = ranges in the cb_allocator's core_range. In Llama 3.1 8B with the
  adaptive scheme, N peaks around `3 tiers × 32 layers × 2 (K+V) + small
  count of model intermediates` ≈ a few hundred. R is typically 1–3.
  Validate runs once per program dispatch; cost is microseconds.
* If profiling later shows hot-path pressure, the structure trivially upgrades
  to an interval tree keyed by core, or a per-core sorted list.

**Risk areas to test.**
1. **Deallocation paths.** Every site that calls `BankManager::deallocate_buffer`
   (single-allocator and dependent-allocator paths) must remove the
   tracking entry. Easy to miss; cover with a unit test that allocates,
   deallocates, allocates again, and asserts the second query returns the
   expected value.
2. **Dependent allocators / sub-device managers.** The existing dependent-
   allocator logic in `BankManager::allocate_buffer` uses cross-allocator
   ranges; ensure the new tracking is scoped per `AllocatorID` (it is, via
   the map key) so sub-device buffers don't accidentally leak into the
   default sub-device's view.
3. **Globally-allocated CBs (`set_globally_allocated_address`).** These are
   already excluded from `cb_region_end` (program.cpp:885–887) but the
   buffer they reference is a regular L1 buffer that *does* show up in the
   bank manager. Sanity-check that the validate logic still makes sense
   when the buffer's address sits below its own program's CB region (this
   is currently allowed; nothing in this patch changes it).
4. **Watcher dump / debug printout.** `log_l1_cb_map` consumes
   `lowest_address`; if its semantics changes to per-cb-allocator, update
   the log message or split into a separate log line so post-hoc analysis
   (e.g. `l1_cb_map_analysis.md`) still parses.
5. **CoreRangeSet semantics.** Verify that `CoreRangeSet::intersects(other)`
   handles the "empty (all-cores) sentinel" the way we want — likely we want
   a helper `is_all_cores()` rather than overloading `empty()`. Cheap to
   change later.

**Test plan.**
* New unit test in `tt_metal/tests/api/test_allocator.cpp`:
  - Allocate a sharded L1 buffer on `(x=0..7, y=9)`.
  - Assert `lowest_occupied_l1_address_for_cores({y=0 row})` returns `nullopt`.
  - Assert `lowest_occupied_l1_address_for_cores({y=9 row})` returns the
    sharded buffer's address.
  - Allocate an interleaved L1 buffer; assert both queries now return its
    address (it spans every bank).
  - Deallocate the sharded buffer; assert the y=0 query is still
    answered by the interleaved buffer, y=9 query likewise.
* Integration test: run `simple_text_demo.py` with
  `--use_adaptive_l1_kv_cache --l1_kv_headroom_json …` and confirm the
  program-59 CB clash no longer fires.
* Regression: run the existing program-cache and SDPA decode unit tests
  to confirm no validate regressions on the legacy global path.

---

## 10. Rollout

1. Land the BankManager tracking + new query API behind the existing global
   API (parallel implementation, no behaviour change). Land tests.
2. Switch `validate_circular_buffer_region` to call the per-core query.
3. Confirm adaptive KV scenario passes end-to-end.
4. Keep the legacy global query for diagnostics / shrink paths.

No model-side or Python-side change is required. The patch is internal to
tt-metal's allocator and program-validation layer; existing
`get_l1_headroom_per_core` and the model's adaptive tier builder are
unaffected.

# L1 KV Decode SRAM Tradeoff And Overhead Analysis

## Executive Summary

- Yes: the "L1 KV mirror" means our extra L1-resident KV ring buffer in `models/tt_transformers/tt/attention.py`, not the native SDPA circular buffers.
- In decode, the native SDPA circular-buffer footprint is essentially the same in both DRAM-only and dual-source modes.
- The current dual-source design adds the L1 KV mirror on top of the native circular buffers. It does not replace them.
- For the current Blackhole decode configuration, the native SDPA circular buffers are about `248 KiB` per active core, while the current `l1_kv_window_size=128` mirror is about `256 KiB` per layer.
- The first profiling report in `research_codes/l1_kv_perf/summary.md` is a workload-average over all decode calls, not a steady-state per-token number. Its large `46-50 ms` L1-path averages are dominated by one expensive first decode iteration.
- After removing that first-call spike, the steady-state overhead in the 1-layer microbenchmark is only about `0.27 ms / layer / token`. Multiplying that by roughly `32` layers gives about `8.7 ms / token`, which is consistent with the roughly `10 ms / token` slowdown observed in full `simple_text_demo`.

## 1. What Takes SRAM In Decode?

There are two different SRAM consumers relevant to this discussion:

1. Native SDPA circular buffers in `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
2. Our extra L1 KV mirror in `models/tt_transformers/tt/attention.py`

These are different things.

### 1.1 Native SDPA Circular Buffers

The decode kernel always allocates a fixed set of circular buffers for:

- Q staging
- K staging
- V staging
- mask / scalar / statistics tiles
- intermediate QK tiles
- intermediate output accumulation tiles
- final output staging

The key tile-count sizing logic is:

- `q_tiles = PNHt * DHt`
- `k_tiles = Sk_chunk_t_cb_size * DHt * 2`
- `v_tiles = Sk_chunk_t_cb_size * vDHt * 2`
- `qk_tiles = PNHt * Sk_chunk_t_cb_size`
- `out_im_tiles = PNHt * vDHt`
- `statistics_tiles = PNHt`

This comes directly from `sdpa_decode_program_factory.cpp`.

Important point:

- These circular-buffer capacities are driven by head count, head dimension, chunk size, and kernel parallelization.
- They are not materially larger in DRAM-only mode.
- They are not materially smaller in dual-source mode.
- Dual-source changes where K/V tiles are fetched from before entering the same circular buffers.

So the native circular-buffer SRAM is basically a constant decode cost for both designs.

### 1.2 Our L1 KV Mirror

The current dual-source implementation allocates an extra persistent L1 cache:

- `self.l1_kv_cache[0]`: K mirror
- `self.l1_kv_cache[1]`: V mirror

This is the "ring buffer as KV mirror" in `attention.py`.

Its persistent size is:

```text
2 * batch * num_kv_heads * total_l1_tokens * head_dim * bytes_per_element
```

where:

- `2` is K + V
- `total_l1_tokens = l1_kv_sink_size + l1_kv_window_size`

For the current example:

- `batch = 1`
- `num_kv_heads = 8`
- `head_dim = 128`
- `l1_kv_sink_size = 0`
- `l1_kv_window_size = 128`
- `kv dtype = bfloat8_b = 1 byte / element`

the mirror size is:

```text
2 * 1 * 8 * 128 * 128 * 1 = 262,144 B = 256 KiB = 0.25 MiB
```

This matches `research_codes/l1_kv_sram_report.md`.

## 2. DRAM-Only Versus Dual-Source SRAM Comparison

### 2.1 DRAM-Only Decode

DRAM-only decode still uses the native SDPA circular buffers.

It does **not** allocate the extra persistent L1 KV mirror.

So its decode SRAM picture is:

- transient native SDPA circular buffers
- no persistent L1 KV mirror

### 2.2 Current Dual-Source Decode

Current dual-source decode uses:

- the same native SDPA circular buffers
- plus the persistent L1 KV mirror
- plus transient clone/update traffic to keep that mirror fresh

So the real tradeoff today is:

```text
same native transient CB SRAM
+ extra persistent L1 KV mirror
+ extra transient L1 update work
```

That is why Phase 5 talks about "redundancy" and "zero-copy direction". The problem is not that DRAM-only has a bigger circular buffer. The problem is that the current dual-source implementation keeps both:

- the native SDPA staging path
- and our extra mirrored L1 KV structure

at the same time.

## 3. Refined Native Circular-Buffer Estimate

The earlier SRAM report used a simplified estimate. For the current Blackhole decode configuration, a more accurate native SDPA circular-buffer estimate is:

- `q_chunk_size = 128`
- `k_chunk_size = 128`
- `head_dim = 128`, so `DHt = 4`
- `32` Q heads, so `PNHt = 1`
- `8` KV heads
- `max_cores_per_head_batch = 16`, which gives `8` cores per head for batch-1 decode on an `8x8` grid

Under these conditions:

- `q_tiles = 4`
- `k_tiles = 32`
- `v_tiles = 32`
- `qk_tiles = 4`
- `out_im_tiles = 4`
- `statistics_tiles = 1`
- `intermed_output_tiles = 42`

Using the current decode path assumptions:

- Q / stats / mask / intermediate tiles are effectively `Float16_b` sized
- K / V staging uses KV-cache dtype (`bfloat8_b`)

the native SDPA circular-buffer footprint is about:

```text
253,952 B = 248 KiB per active core
```

The biggest contributors are:

- `c_1` K staging: `32 KiB`
- `c_2` V staging: `32 KiB`
- `c_19` intermed output: `84 KiB`
- several smaller `2-8 KiB` scalar / stats / output buffers

This is an important comparison point:

- native CB footprint: about `248 KiB` per active core
- current L1 KV mirror: about `256 KiB` per layer

They are comparable in magnitude, but they are not the same type of allocation:

- CB footprint is transient and kernel-local
- L1 mirror is persistent and model-level

## 4. Is The Native Circular Buffer Bigger In DRAM-Only?

No, not in the way that matters here.

For a fixed decode configuration, the SDPA decode program factory allocates the same circular buffers regardless of whether K/V tiles come from:

- DRAM only
- L1 sink + ring + DRAM fallback

The reader kernel changes source selection. The CB capacities do not shrink just because we added an L1 mirror.

So if the goal is to reason about SRAM tradeoff, the correct comparison is:

### DRAM-only

- native decode circular buffers only

### Current dual-source

- native decode circular buffers
- plus our persistent L1 mirror
- plus extra update path overhead

That is exactly why the Phase 5 "sharded/zero-copy" direction matters:

- sharded L1 tries to align the persistent layout better with the reader
- zero-copy means reducing or eliminating redundant staging / mirroring so the L1-resident hot KV data becomes more directly usable

## 5. Why The First Timing Report Looked Much Worse Than The User's 10 ms / Token Observation

The first profile report in `research_codes/l1_kv_perf/summary.md` is a **whole-workload average** over all `16` decode calls in the benchmark command:

```text
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" --num_layers 1 --max_generated_tokens 16 --stop_at_eos 0 ...
```

That means each reported "avg ms" is:

```text
sum over all decode calls / number of decode calls
```

and therefore includes the first decode iteration, which is much more expensive than the steady-state tokens.

### 5.1 Example: The L1 Clone Path

From `research_codes/l1_kv_perf/dual_source_report.json`:

- `decode.l1_clone_path.sum_ms = 736.191`
- `decode.l1_clone_path.max_ms = 734.356`
- `decode.l1_clone_path.count = 16`

So almost the entire reported average:

```text
736.191 / 16 = 46.012 ms
```

comes from one large first-call spike.

The steady-state estimate excluding that max is:

```text
(736.191 - 734.356) / 15 = 0.122 ms / token / layer
```

### 5.2 Example: The L1 KV Write Path

From the same report:

- `decode.l1_kv_write.sum_ms = 810.318`
- `decode.l1_kv_write.max_ms = 809.521`
- `decode.l1_kv_write.count = 16`

So the workload average:

```text
810.318 / 16 = 50.645 ms
```

is again dominated by the first call.

The steady-state estimate excluding that max is:

```text
(810.318 - 809.521) / 15 = 0.053 ms / token / layer
```

### 5.3 Model Forward Delta

The same issue exists at the higher-level `decode.model_forward` line:

#### Workload-average view

- dual-source avg: `1071.715 ms`
- DRAM-only avg: `1006.918 ms`
- delta: `+64.797 ms`

#### Steady-state estimate excluding the largest first iteration

- dual-source steady: about `3.746 ms / layer / token`
- DRAM-only steady: about `3.474 ms / layer / token`
- delta: about `+0.271 ms / layer / token`

That `0.271 ms / layer / token` number is the one that should be compared to your "about 10 ms longer per token" observation.

## 6. Connecting The Profiling To The ~10 ms / Token Slowdown In Full Demo

The profiling runs were deliberately done with:

- `num_layers = 1`

to isolate the per-layer dual-source cost.

Your real `simple_text_demo` observation is from the full model, which is roughly `32` layers for Llama 3.1 8B.

If the isolated 1-layer steady-state overhead is about:

```text
+0.271 ms / layer / token
```

then scaling that to the full model gives:

```text
0.271 * 32 = 8.67 ms / token
```

That is very close to the roughly `10 ms / token` slowdown you observed.

So the two observations are consistent:

- the first report overstates per-token cost because it smears a large first-call spike across the workload
- the 1-layer steady-state delta, when scaled by full model depth, matches the observed full-model token slowdown reasonably well

## 7. What The Thresholded Rerun Shows

The thresholded rerun in `research_codes/l1_kv_perf_thresholded/summary.md` used:

- `--l1_kv_min_expected_hit_ratio 0.95`

On that short-context workload, the expected hit ratio was only about `0.792`, so the new gate disabled the extra L1 update path.

That changed the workload-average `decode.model_forward` delta from:

- `+64.797 ms` in the original report

to:

- `+21.691 ms` in the thresholded report

and the explicit `decode.l1_clone_path` / `decode.l1_kv_write` lines disappeared from the thresholded summary.

This confirms that:

- the extra L1 maintenance path is a real overhead source
- when L1 will not be used, skipping the writes removes most of that overhead

But it is not a complete solution:

- if you want L1 reads to stay enabled, you still need the mirror to be kept up to date
- therefore the long-term fix is not only gating; it is reducing the cost of keeping the L1 hot set alive, and eventually reducing the duplication between the mirror and native staging

## 8. Practical Interpretation For The Remaining Roadmap

### Phase 2

The memory picture is now clearer:

- DRAM-only does not save SRAM by shrinking native CBs
- dual-source pays an extra persistent mirror cost
- the current mirror is approximately the same size as a single-core native CB footprint, but persistent and multiplied across layers

### Phase 3

The immediate decode overhead is mostly:

- maintaining the mirror
- not the SDPA reader itself

### Phase 4

Sink pinning is still useful because it improves what portion of the mirror is actually valuable hot data.

### Phase 5

This is where the deeper win must come from:

- reduce persistent duplication
- reduce transient staging duplication
- make the persistent L1 layout more directly useful to the reader

That is the "sharded/zero-copy direction".

## 9. Can We Measure Actual SRAM Usage As A Percentage?

Yes, but there is an important caveat:

- the often-quoted `~210 MB` number is chip-total worker SRAM
- decode does **not** get to use that as one flat global pool
- allocations are still constrained by per-bank / per-core local L1 limits and fragmentation

So the most meaningful measurements are:

1. per-bank or per-core usage
2. largest free contiguous block per bank
3. summed chip-wide usage as a secondary number

### 9.1 Was There Unused L1 In The Original DRAM-Only Scenario?

Yes.

More precisely:

- there was enough **per-core unreserved L1** left over after the existing weights, activations, and circular buffers
- that spare capacity allowed us to allocate the extra KV mirror

This does **not** mean there was one big globally idle `210 MB` pool waiting to be used.

It means:

- some worker-local L1 banks still had enough free allocatable space
- and our current mirror fit into that remaining budget for this workload

That matches what `attention.py` is already doing with:

- `ttnn.get_max_worker_l1_unreserved_size()`

which is explicitly a **per-worker** quantity, not a full-chip quantity.

### 9.2 What Can We Measure Today?

TTNN already exposes two useful mechanisms.

#### A. In-process snapshot API

Use:

- `ttnn.get_memory_view(device, ttnn.BufferType.L1)`

This returns a `MemoryView` with:

- `num_banks`
- `total_bytes_per_bank`
- `total_bytes_allocated_per_bank`
- `total_bytes_free_per_bank`
- `largest_contiguous_bytes_free_per_bank`
- `block_table`

This is the quickest way to ask:

- how much L1 is allocatable per worker bank?
- how much of that is currently allocated?
- how much contiguous headroom remains?

#### B. Full memory dump reports

Use:

- `ttnn.device.EnableMemoryReports()`
- `ttnn.device.dump_device_memory_state(device, prefix="...")`

This produces:

- `l1_usage_summary.csv`
- `memory_usage_summary.csv`
- `detailed_memory_usage.csv`

These are better when you want:

- bank-by-bank usage
- fragmentation analysis
- a saved artifact for comparing DRAM-only vs dual-source runs

### 9.3 What Percentages Make Sense?

There are three useful percentages.

#### Per-bank average allocation percentage

```text
allocated_pct_per_bank =
    total_bytes_allocated_per_bank / total_bytes_per_bank
```

This is the best first number to quote.

#### Chip-wide summed allocation percentage

```text
allocated_pct_chip =
    (num_banks * total_bytes_allocated_per_bank) /
    (num_banks * total_bytes_per_bank)
```

Numerically, this is the same ratio as the per-bank average if all banks are symmetric, but it is still useful for reporting "out of total allocatable L1 on chip".

#### Per-bank headroom / fragmentation percentage

```text
largest_free_pct_per_bank =
    largest_contiguous_bytes_free_per_bank / total_bytes_per_bank
```

This number is often more actionable than total free space, because an allocation can fail even when total free bytes look comfortable if the remaining space is too fragmented.

### 9.4 Why Per-Core Matters More Than Whole-Chip Percentage

Suppose the chip has a lot of total free SRAM left overall, but the specific worker cores used by decode have one or two banks close to full.

Then:

- the global percentage may still look fine
- but the decode program can still become constrained or fail

So for our L1-KV work, the best hierarchy is:

1. participating worker-bank peak usage
2. largest free block on participating banks
3. chip-wide summed percentage

### 9.5 Minimal Example

The following code is enough to snapshot current L1 allocation after model load or after a decode step:

```python
import ttnn

view = ttnn.get_memory_view(device, ttnn.BufferType.L1)

per_bank_alloc_pct = 100.0 * view.total_bytes_allocated_per_bank / view.total_bytes_per_bank
per_bank_free_pct = 100.0 * view.total_bytes_free_per_bank / view.total_bytes_per_bank
per_bank_largest_free_pct = (
    100.0 * view.largest_contiguous_bytes_free_per_bank / view.total_bytes_per_bank
)

chip_total_bytes = view.num_banks * view.total_bytes_per_bank
chip_allocated_bytes = view.num_banks * view.total_bytes_allocated_per_bank
chip_alloc_pct = 100.0 * chip_allocated_bytes / chip_total_bytes

print(f"L1 banks: {view.num_banks}")
print(f"Per-bank allocatable bytes: {view.total_bytes_per_bank}")
print(f"Per-bank allocated bytes: {view.total_bytes_allocated_per_bank}")
print(f"Per-bank allocated %: {per_bank_alloc_pct:.2f}")
print(f"Per-bank free %: {per_bank_free_pct:.2f}")
print(f"Per-bank largest contiguous free %: {per_bank_largest_free_pct:.2f}")
print(f"Chip-total allocatable bytes: {chip_total_bytes}")
print(f"Chip-total allocated bytes: {chip_allocated_bytes}")
print(f"Chip-total allocated %: {chip_alloc_pct:.2f}")
```

### 9.6 Best Practical Plan For Our Comparison

To compare DRAM-only and dual-source accurately, the most useful measurement plan is:

1. Run DRAM-only model load + one decode step.
2. Capture `get_memory_view(..., BufferType.L1)` immediately after decode.
3. Dump `dump_device_memory_state(..., prefix="dram_only_...")`.
4. Repeat for dual-source.
5. Compare:
   - allocated bytes per bank
   - largest contiguous free bytes per bank
   - chip-total allocated percentage
   - detailed block tables on the participating decode banks

That will tell us not only "how much SRAM is used", but also whether the current KV mirror is consuming the exact scarce per-core headroom that the native decode path wants for its own staging and future optimizations.

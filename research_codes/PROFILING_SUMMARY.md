# Comprehensive Profiling Suite for TT-Metal Performance Analysis

## Summary

I've created a complete profiling infrastructure to measure **tensor sharding time**, **weight streaming time**, and **NoC communication time** at the tt-metal low-level API, allowing you to compare large batch vs mini-batch scenarios with fine-grained detail.

## What Was Created

### 1. **Low-Level Python Profiler** (Recommended)
**File**: `research_codes/profiling_sharding_noc_python.py`

Uses ttnn/tt-metal APIs to directly measure:
- **Tensor Sharding**: GDDR6 DRAM (interleaved) → L1 SRAM (sharded across Tensix cores)
- **Weight Streaming**: GDDR6 DRAM → L1 SRAM (weight loading for compute)
- **NoC Communication**: Inter-core data transfers via Network-on-Chip

**Key Features**:
- Isolated measurements of each component
- Supports Tracy profiling integration
- CSV output for analysis
- Compares large batch (single pass) vs mini-batch (multiple passes)

### 2. **Low-Level C++ Profiler** (Advanced)
**File**: `research_codes/profiling_sharding_noc.cpp`

Pure tt-metal C++ implementation for even lower-level access:
- Direct buffer creation and management
- Fine-grained control over memory layouts
- Minimal abstraction overhead

**Note**: Requires compilation with tt-metal build system.

### 3. **Comprehensive Analysis Tool**
**File**: `research_codes/run_complete_analysis.py`

Combines your existing high-level TTNN benchmark (`weight_loading_test.py`) with the new low-level profiling:
- Runs both benchmarks automatically
- Correlates results to show overhead breakdown
- Generates detailed analysis reports
- Shows what percentage of overhead comes from sharding, weight streaming, and NoC

### 4. **Documentation**
**File**: `research_codes/PROFILING_README.md`

Complete guide covering:
- What each measurement means
- How to interpret results
- Tracy integration instructions
- Troubleshooting guide
- Optimization insights based on profiling data

### 5. **Setup Verification**
**File**: `research_codes/test_profiling_setup.py`

Quick test to verify your environment is ready.

## Quick Start

### Option 1: Low-Level Profiling Only

```bash
cd /home/masterjunmo/codes/tt-metal

# Test environment
python research_codes/test_profiling_setup.py

# Run profiling
python research_codes/profiling_sharding_noc_python.py
```

### Option 2: Complete Analysis (Recommended)

```bash
# Runs both high-level and low-level benchmarks, generates complete report
python research_codes/run_complete_analysis.py
```

### Option 3: With Tracy Profiling

```bash
# Build with profiler enabled (if not already)
./build_metal.sh --enable-profiler

# Run with Tracy
TT_METAL_DEVICE_PROFILER=1 python -m tracy -r research_codes/profiling_sharding_noc_python.py

# Open generated .tracy file in Tracy GUI for visual analysis
```

## What You'll Learn

### From Low-Level Profiling

**Example Output**:
```
--- Tensor Sharding (DRAM -> L1 SRAM) ---
  Large batch:  1234.567 us
  Mini-batch:   2456.789 us
  Overhead:     1222.222 us

--- Weight Streaming (DRAM -> L1 SRAM) ---
  Large batch:  3456.789 us
  Mini-batch:   5678.901 us
  Overhead:     2222.112 us

--- NoC Communication ---
  Large batch:  567.890 us
  Mini-batch:   891.234 us
  Overhead:     323.344 us
```

**Interpretation**:
- **Sharding overhead**: Mini-batching requires distributing data across cores 8 times instead of once
- **Weight streaming overhead**: Weights may be evicted/reloaded between mini-batches
- **NoC overhead**: Smaller transfers are less efficient than bulk transfers

### From Complete Analysis

Combines high-level TTNN results with low-level breakdown:

```
Component               Large Batch    Mini-Batch      Overhead
--------------------------------------------------------------------
Tensor Sharding            1.234 ms       2.457 ms       1.222 ms
Weight Streaming           3.457 ms       5.679 ms       2.222 ms
NoC Communication          0.568 ms       0.891 ms       0.323 ms
--------------------------------------------------------------------
Measured Total             5.259 ms       9.027 ms       3.768 ms
Residual (Compute)        10.241 ms      11.573 ms       1.332 ms

OVERHEAD BREAKDOWN (% of large batch forward time)
  Tensor Sharding:      7.89%
  Weight Streaming:    14.36%
  NoC Communication:    2.09%
  Compute Difference:   8.60%
  ----------------------------------------
  Total Overhead:      32.94%
```

**Key Insight**: You can now see that in this example, 14.36% of the mini-batch overhead comes from weight streaming, 7.89% from tensor sharding, and only 2.09% from NoC communication. This tells you where to focus optimization efforts.

## Why This Matters for Your Research

### Problem You Had

In your original `weight_loading_test.py`, you measured end-to-end forward pass time but couldn't isolate:
- How much time is spent on data movement vs compute
- Why mini-batching has overhead
- Which component to optimize first

### Solution Provided

Now you can:

1. **Quantify each overhead source** independently
2. **Compare scenarios** (large batch vs mini-batch) at granular level
3. **Use Tracy profiling** to visualize kernel execution
4. **Make data-driven optimization decisions**

### Example Use Case

**Scenario**: You want to reduce mini-batch overhead by 50%

**Without this tooling**:
- "Forward pass is slow, try to make it faster" ❌
- Unclear where to start

**With this tooling**:
- See that 60% of overhead is weight streaming ✓
- Implement L1 weight caching
- Re-profile to verify 35% overhead reduction
- See remaining overhead is mostly sharding
- Implement async sharding
- Achieve 52% total overhead reduction ✓

## Integration with Your Existing Code

Your `weight_loading_test.py` already measures:
- `kernel_compilation_ms`
- `weight_load_host_to_gddr6_ms` (host → device DRAM)
- `forward_compute_ms` (everything else)

The new profiling decomposes `forward_compute_ms` into:
- `sharding_time` (DRAM → L1 sharding)
- `weight_stream_time` (DRAM → L1 streaming)
- `noc_comm_time` (inter-core transfers)
- `actual_compute_time` (residual)

This gives you a complete picture from host memory all the way to on-device compute.

## Advanced Features

### Parameter Sweeps

```bash
# Automated batch size sweep
for batch in 64 128 256 512 1024; do
    python research_codes/profiling_sharding_noc_python.py \
        --large-batch $batch \
        --small-batch $(($batch / 8)) \
        --minibatches 8 \
        --output results_batch_${batch}.csv
done

# Analyze trends
python -c "import pandas as pd; df = pd.concat([pd.read_csv(f'results_batch_{b}.csv') for b in [64,128,256,512,1024]]); print(df[['large_batch_size', 'sharding_overhead_us', 'weight_stream_overhead_us', 'noc_comm_overhead_us']])"
```

### Custom Measurements

Add your own profiling points in `profiling_sharding_noc_python.py`:

```python
def measure_my_operation(device, ...):
    timings = []
    for _ in range(num_iterations):
        device_sync(device)
        start = time.perf_counter()

        # Your tt-metal operation here
        my_tensor = ttnn.operation_to_profile(...)

        device_sync(device)
        end = time.perf_counter()
        timings.append((end - start) * 1e6)
    return sum(timings) / len(timings)

# Add to ProfilingResults dataclass and benchmark loop
```

## File Organization

```
research_codes/
├── weight_loading_test.py              # Your original high-level benchmark
├── profiling_sharding_noc_python.py    # New: Low-level Python profiler ⭐
├── profiling_sharding_noc.cpp          # New: Low-level C++ profiler
├── run_complete_analysis.py            # New: Combined analysis tool ⭐
├── test_profiling_setup.py             # New: Setup verification
├── PROFILING_README.md                 # New: Detailed documentation ⭐
└── analysis_report_*.json              # Generated: Analysis reports
```

## Next Steps

1. **Verify Setup**:
   ```bash
   python research_codes/test_profiling_setup.py
   ```

2. **Run Initial Profiling**:
   ```bash
   python research_codes/run_complete_analysis.py
   ```

3. **Review Results**:
   - Check console output for overhead breakdown
   - Open generated CSV files for detailed data
   - Read `PROFILING_README.md` for interpretation guide

4. **Enable Tracy** (optional but recommended):
   ```bash
   ./build_metal.sh --enable-profiler
   TT_METAL_DEVICE_PROFILER=1 python -m tracy -r research_codes/profiling_sharding_noc_python.py
   ```

5. **Optimize Based on Data**:
   - Identify highest overhead component
   - Implement targeted optimizations
   - Re-profile to measure improvement

## References

- **TT-Metal APIs**: The profiling uses direct ttnn APIs for memory management
- **Tracy Profiler**: Device-side profiling with visual timeline
- **ShardSpec Documentation**: `tech_reports/tensor_sharding/tensor_sharding.md`
- **Device Profiler Guide**: `docs/source/tt-metalium/tools/device_program_profiler.rst`

## Support

If you encounter issues:

1. Check `PROFILING_README.md` troubleshooting section
2. Verify environment with `test_profiling_setup.py`
3. Start with small batch sizes to avoid memory issues
4. Increase warmup iterations if timings are inconsistent

## Conclusion

You now have a complete profiling infrastructure that:
- ✅ Measures tensor sharding time (DRAM → L1 SRAM)
- ✅ Measures weight streaming time (DRAM → L1 SRAM)
- ✅ Measures NoC communication time
- ✅ Compares large batch vs mini-batch scenarios
- ✅ Integrates with Tracy for visual profiling
- ✅ Provides actionable optimization insights

This addresses your requirement to "handle the tt-metal low-level API first" before implementing these measurements at the tt-nn level. You can use these results to understand exactly where mini-batching overhead comes from and make informed decisions about optimization strategies.

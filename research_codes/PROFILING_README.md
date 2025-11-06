# Low-Level Profiling for Tensor Sharding, Weight Streaming, and NoC Communication

This directory contains tools for measuring fine-grained performance metrics at the tt-metal level, including:

1. **Tensor Sharding Time**: Time to convert tensors from GDDR6 DRAM (interleaved layout) to L1 SRAM (sharded across Tensix cores)
2. **Weight Streaming Time**: Time to load weights from GDDR6 DRAM to L1 SRAM for compute operations
3. **NoC Communication Time**: Time for inter-core data transfers via the Network-on-Chip

## Files

- `profiling_sharding_noc_python.py`: Python implementation using ttnn APIs with detailed measurements
- `profiling_sharding_noc.cpp`: C++ implementation using tt-metal low-level APIs (requires compilation)
- `weight_loading_test.py`: Original high-level ttnn benchmark (for comparison)

## Why These Measurements Matter

At the tt-nn level (as in `weight_loading_test.py`), it's difficult to isolate individual operations because:
- Forward passes are atomic and include multiple phases (sharding, compute, communication)
- Weight loading and tensor movement are abstracted away
- TTNN operations bundle multiple low-level operations together

This low-level profiling decomposes the operations to measure:
- **Sharding overhead**: How much extra time is spent distributing data across cores in mini-batch scenarios
- **Weight streaming overhead**: Cost of repeatedly loading weights for multiple mini-batches vs. loading once for a large batch
- **NoC communication overhead**: Inter-core communication costs that scale with mini-batching

## Quick Start

### Python Version (Recommended)

```bash
# Basic run
cd /home/masterjunmo/codes/tt-metal
python research_codes/profiling_sharding_noc_python.py

# With custom configuration
python research_codes/profiling_sharding_noc_python.py \
    --large-batch 512 \
    --small-batch 64 \
    --minibatches 8 \
    --in-features 4096 \
    --out-features 4096 \
    --iterations 10

# With Tracy profiling (for detailed kernel-level analysis)
TT_METAL_DEVICE_PROFILER=1 python -m tracy -r research_codes/profiling_sharding_noc_python.py
```

### C++ Version (Advanced)

First, build tt-metal with profiling enabled:

```bash
# Build with profiler support
./build_metal.sh --enable-profiler

# Compile the profiling tool
cd build_Release
cmake --build . --target profiling_sharding_noc

# Run
./research_codes/profiling_sharding_noc \
    --large-batch 256 \
    --small-batch 32 \
    --minibatches 8 \
    --iterations 5
```

## Understanding the Measurements

### 1. Tensor Sharding Time

**What it measures**: Time to convert a tensor from interleaved DRAM layout to sharded L1 layout.

**Large batch scenario**:
- Single tensor of shape [1, 1, 256, 4096] loaded from DRAM
- Distributed across Tensix cores in L1 (e.g., 8x8 grid = 64 cores)
- Each core gets shard: [4, 4096] (batch/num_cores rows)

**Mini-batch scenario** (8 x 32):
- 8 separate tensors of shape [1, 1, 32, 4096] loaded from DRAM
- Each distributed independently across cores
- Each core processes multiple small shards sequentially

**Expected result**: Mini-batch has overhead due to repeated sharding operations.

### 2. Weight Streaming Time

**What it measures**: Time to transfer weight matrices from DRAM to L1 for matmul operations.

**Large batch scenario**:
- Weights [4096, 4096] loaded once from DRAM to L1
- Kept in L1 for entire batch computation

**Mini-batch scenario**:
- Same weights [4096, 4096] potentially evicted/reloaded multiple times
- L1 cache pressure higher with multiple passes
- Weights may need to stream from DRAM for each mini-batch

**Expected result**: Mini-batch may have overhead if weights are evicted between passes.

### 3. NoC Communication Time

**What it measures**: Inter-core data transfer time via Network-on-Chip.

**Large batch scenario**:
- Larger data chunks transferred between cores
- More efficient NoC utilization with bulk transfers
- Better amortization of NoC overhead

**Mini-batch scenario**:
- Smaller data chunks transferred multiple times
- Higher relative overhead per transfer
- More frequent NoC arbitration and setup costs

**Expected result**: Mini-batch has overhead due to less efficient NoC utilization.

## Interpreting Results

### Sample Output

```
======== Profiling Results ========

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

--- Total Operation Time ---
Large batch:  5259.246 us
Mini-batch:   9026.924 us
Total Overhead: 3767.678 us (71.63%)
```

### What the Numbers Mean

- **Positive overhead**: Mini-batching is slower (expected for most cases)
- **Sharding overhead %**: Percentage of extra time spent on tensor layout conversions
- **Weight streaming overhead %**: Cost of repeated weight loading
- **NoC overhead %**: Inter-core communication inefficiency
- **Total overhead**: Combined effect of all three factors

### CSV Output

Results are automatically saved to CSV for analysis:

```csv
timestamp,large_batch_size,small_batch_size,num_minibatches,in_features,out_features,...
2025-11-06 14:30:00,256,32,8,4096,4096,1234.567,2456.789,1222.222,...
```

You can aggregate multiple runs to analyze:
- Scaling with batch size
- Impact of feature dimensions
- Consistency across runs

## Using Tracy for Detailed Profiling

Tracy provides kernel-level profiling with visual timeline:

```bash
# 1. Build with profiler enabled (if not already done)
./build_metal.sh --enable-profiler

# 2. Run with Tracy
TT_METAL_DEVICE_PROFILER=1 python -m tracy -r research_codes/profiling_sharding_noc_python.py

# 3. Tracy outputs a .tracy file - open with Tracy profiler GUI
# Download Tracy: https://github.com/wolfpld/tracy/releases
```

In Tracy you can see:
- Individual kernel execution times
- Data movement operations
- NoC transactions
- Memory allocation/deallocation
- Core utilization

## Integration with Your Existing Workflow

### Comparing with TTNN-Level Results

Your `weight_loading_test.py` measures end-to-end forward pass time. Use these low-level measurements to understand the breakdown:

```python
# In weight_loading_test.py, you measure:
# - forward_compute_ms: Total forward pass time

# With low-level profiling, you now know:
# - How much is tensor sharding
# - How much is weight streaming
# - How much is NoC communication
# - How much is actual compute (residual)

actual_compute_time = forward_compute_ms - (sharding_time + weight_stream_time + noc_comm_time)
```

### Optimization Insights

Based on profiling results:

1. **High sharding overhead** → Consider:
   - Keeping data pre-sharded in L1
   - Using async sharding operations
   - Batching sharding operations

2. **High weight streaming overhead** → Consider:
   - Weight caching in L1
   - Weight pre-fetching
   - Better L1 memory management

3. **High NoC overhead** → Consider:
   - Different core grid layouts
   - Larger data chunks per transfer
   - Pipeline parallelism to hide NoC latency

## Advanced Usage

### Parameter Sweeps

```bash
# Sweep batch sizes
for batch in 64 128 256 512; do
    python research_codes/profiling_sharding_noc_python.py \
        --large-batch $batch \
        --small-batch $(($batch / 8)) \
        --minibatches 8 \
        --output results_batch_${batch}.csv
done

# Analyze results
python analyze_profiling_results.py results_batch_*.csv
```

### Combining with Device Profiler

```bash
# Enable device-side profiling markers
export TT_METAL_DEVICE_PROFILER=1

# Run with device profiling
python research_codes/profiling_sharding_noc_python.py

# Results include both host and device timings
```

### Custom Measurements

Modify the Python script to add your own measurements:

```python
def measure_custom_operation(device, ...):
    """Add your custom measurement here."""
    timings = []
    for _ in range(num_iterations):
        device_sync(device)
        start = time.perf_counter()

        # Your operation here

        device_sync(device)
        end = time.perf_counter()
        timings.append((end - start) * 1e6)

    return sum(timings) / len(timings)
```

## Troubleshooting

### "Device not found" error

Ensure tt-metal environment is properly set up:
```bash
export ARCH_NAME=wormhole_b0  # or your architecture
export TT_METAL_HOME=/home/masterjunmo/codes/tt-metal
source python_env/bin/activate
```

### Out of memory errors

Reduce tensor sizes or batch sizes:
```bash
python profiling_sharding_noc_python.py \
    --large-batch 128 \
    --in-features 2048 \
    --out-features 2048
```

### Inconsistent timings

Increase warmup and measurement iterations:
```bash
python profiling_sharding_noc_python.py \
    --warmup 5 \
    --iterations 20
```

## References

- [TT-Metal Documentation](https://docs.tenstorrent.com/tt-metal/)
- [Tracy Profiler Guide](../../docs/source/tt-metalium/tools/tracy_profiler.rst)
- [Device Program Profiler](../../docs/source/tt-metalium/tools/device_program_profiler.rst)
- [Tensor Sharding Tech Report](../../tech_reports/tensor_sharding/tensor_sharding.md)

## Contributing

To add new measurements or improve profiling:

1. Add measurement function in Python version
2. Update ProfilingResults dataclass
3. Update CSV output format
4. Update this README with interpretation guidance

## License

SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0

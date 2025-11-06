# TT-Metal Profiling Tools - Complete Index

## 📦 Complete File List

### ⭐ Main Profiling Tools

| File | Purpose | Size | When to Use |
|------|---------|------|-------------|
| `profiling_sharding_noc_python.py` | Low-level Python profiler | 20KB | **Primary tool** - Detailed breakdown of sharding, weight streaming, NoC |
| `run_complete_analysis.py` | Combined analysis script | 12KB | **Best for insights** - Runs both high & low level, generates reports |
| `run_profiling.sh` | Convenience wrapper script | 5KB | **Easy access** - Simple commands for common tasks |

### 📚 Documentation

| File | Purpose | Size | What You'll Find |
|------|---------|------|------------------|
| `PROFILING_SUMMARY.md` | Complete overview | 10KB | What was created, why it matters, how to use it |
| `PROFILING_README.md` | Detailed guide | 10KB | Technical details, interpretation, troubleshooting |
| `QUICK_REFERENCE.txt` | Cheat sheet | 9KB | Commands, options, common patterns |
| `INDEX.md` | This file | - | Navigation and quick links |

### 🔧 Supporting Tools

| File | Purpose | Size | When to Use |
|------|---------|------|-------------|
| `test_profiling_setup.py` | Setup verification | 1.5KB | First-time setup, troubleshooting |
| `profiling_sharding_noc.cpp` | C++ profiler | 19KB | Advanced use, minimal overhead |

### 📊 Original Tool

| File | Purpose | Size | Notes |
|------|---------|------|-------|
| `weight_loading_test.py` | High-level TTNN benchmark | 32KB | Your existing tool - now complemented with low-level profiling |

---

## 🚀 Quick Start Guide

### First Time Setup

```bash
# 1. Verify your environment
python research_codes/test_profiling_setup.py

# 2. Read the quick reference
cat research_codes/QUICK_REFERENCE.txt

# 3. Run a quick test
./research_codes/run_profiling.sh quick
```

### Common Tasks

```bash
# Standard profiling run
./research_codes/run_profiling.sh standard

# Full analysis (high-level + low-level)
./research_codes/run_profiling.sh full

# Batch size sweep
./research_codes/run_profiling.sh sweep

# With Tracy profiling
./research_codes/run_profiling.sh tracy
```

---

## 📖 Documentation Reading Order

For **first-time users**:
1. `QUICK_REFERENCE.txt` - Get familiar with commands (5 min)
2. `PROFILING_SUMMARY.md` - Understand what you can achieve (10 min)
3. Run `./research_codes/run_profiling.sh quick` - See it in action (2 min)
4. `PROFILING_README.md` - Deep dive when needed (reference)

For **experienced users**:
1. `QUICK_REFERENCE.txt` - Command syntax
2. Source code - `profiling_sharding_noc_python.py` for customization

---

## 🎯 What Each Tool Measures

### `profiling_sharding_noc_python.py`

Measures at **tt-metal level**:

✓ **Tensor Sharding**: GDDR6 DRAM (interleaved) → L1 SRAM (sharded)
  - Distributing input tensors across Tensix cores
  - Comparing single large batch vs multiple small batches

✓ **Weight Streaming**: GDDR6 DRAM → L1 SRAM
  - Loading weights for compute operations
  - Overhead of repeated loading in mini-batch scenarios

✓ **NoC Communication**: Inter-core data transfers
  - Network-on-Chip efficiency
  - Impact of transfer sizes

### `weight_loading_test.py` (Your Original)

Measures at **ttnn level**:

✓ **Kernel Compilation**: One-time JIT compilation cost
✓ **Host→Device Transfer**: CPU memory → GDDR6 DRAM
✓ **Forward Pass**: End-to-end execution time
  - Includes all sharding, weight loading, compute, communication
  - Cannot isolate individual components

### `run_complete_analysis.py` (New Combined Tool)

Combines both:

✓ **End-to-End Time** from high-level benchmark
✓ **Component Breakdown** from low-level profiling
✓ **Overhead Analysis** showing what percentage each component contributes
✓ **Actionable Insights** for optimization

---

## 🔍 Use Case Examples

### Use Case 1: "Why is mini-batching slow?"

```bash
# Run complete analysis
./research_codes/run_profiling.sh full

# Check the output - you'll see:
# - Sharding overhead: X%
# - Weight streaming overhead: Y%
# - NoC communication overhead: Z%
# → Now you know where to optimize!
```

### Use Case 2: "Does batch size affect overhead?"

```bash
# Run batch size sweep
./research_codes/run_profiling.sh sweep

# Analyze results across different batch sizes
python -c "
import pandas as pd
df = pd.concat([pd.read_csv(f'results_batch_{b}.csv')
                for b in [64,128,256,512]])
print(df[['large_batch_size', 'sharding_overhead_us',
          'weight_stream_overhead_us', 'overhead_percent']])
"
```

### Use Case 3: "Where are my kernels spending time?"

```bash
# Build with profiler
./build_metal.sh --enable-profiler

# Run with Tracy
./research_codes/run_profiling.sh tracy

# Open generated .tracy file in Tracy GUI
# → Visual timeline of kernel execution
```

### Use Case 4: "I want to add custom measurements"

```python
# Edit profiling_sharding_noc_python.py

# Add your measurement function (around line 150):
def measure_my_custom_operation(device, ...):
    timings = []
    for _ in range(num_iterations):
        device_sync(device)
        start = time.perf_counter()

        # Your tt-metal operation here
        result = ttnn.my_operation(...)

        device_sync(device)
        end = time.perf_counter()
        timings.append((end - start) * 1e6)
    return sum(timings) / len(timings)

# Add to ProfilingResults dataclass (around line 50)
# Add to benchmark loop (around line 450)
# Add to CSV output (around line 120)
```

---

## 📊 Expected Output Examples

### From `profiling_sharding_noc_python.py`:

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

### From `run_complete_analysis.py`:

```
OVERHEAD BREAKDOWN (% of large batch forward time)
  Tensor Sharding:      7.89%
  Weight Streaming:    14.36%
  NoC Communication:    2.09%
  Compute Difference:   8.60%
  ----------------------------------------
  Total Overhead:      32.94%
```

---

## 🛠️ Troubleshooting Guide

### Problem: "Device not found"

```bash
# Check environment
echo $ARCH_NAME  # Should be: wormhole_b0 or similar
echo $TT_METAL_HOME  # Should point to tt-metal directory

# Activate environment
source python_env/bin/activate

# Test setup
python research_codes/test_profiling_setup.py
```

### Problem: "Out of memory"

```bash
# Use smaller configuration
./research_codes/run_profiling.sh quick

# Or manually reduce sizes
python research_codes/profiling_sharding_noc_python.py \
    --large-batch 128 \
    --in-features 2048 \
    --out-features 2048
```

### Problem: "Inconsistent timings"

```bash
# Increase warmup and iterations
python research_codes/profiling_sharding_noc_python.py \
    --warmup 5 \
    --iterations 20
```

---

## 🎓 Learning Path

### Beginner

1. ✅ Run `test_profiling_setup.py` - Verify environment
2. ✅ Read `QUICK_REFERENCE.txt` - Learn commands
3. ✅ Run `./run_profiling.sh quick` - See it work
4. ✅ Review output - Understand the numbers

### Intermediate

1. ✅ Run `./run_profiling.sh full` - Complete analysis
2. ✅ Read `PROFILING_SUMMARY.md` - Understand methodology
3. ✅ Run `./run_profiling.sh sweep` - Compare scenarios
4. ✅ Analyze CSV outputs - Find patterns

### Advanced

1. ✅ Read `PROFILING_README.md` - Deep dive
2. ✅ Build with profiler - `./build_metal.sh --enable-profiler`
3. ✅ Run Tracy profiling - Visual analysis
4. ✅ Customize `profiling_sharding_noc_python.py` - Add measurements
5. ✅ Integrate findings - Optimize your code

---

## 📁 File Dependency Graph

```
run_profiling.sh
  ├── test_profiling_setup.py
  ├── profiling_sharding_noc_python.py ⭐
  ├── run_complete_analysis.py
  │    ├── weight_loading_test.py
  │    └── profiling_sharding_noc_python.py ⭐
  └── Documentation
       ├── PROFILING_README.md
       ├── PROFILING_SUMMARY.md
       └── QUICK_REFERENCE.txt
```

---

## 🎯 Key Takeaways

1. **Low-level profiling** isolates individual operations that ttnn bundles together
2. **Combined analysis** shows both end-to-end time AND breakdown
3. **Tracy integration** provides visual kernel-level profiling
4. **Automated tools** make it easy to run and analyze
5. **Extensible design** allows adding custom measurements

---

## 📞 Need Help?

| Question | Answer |
|----------|--------|
| How do I start? | `python research_codes/test_profiling_setup.py` |
| What command should I use? | Check `QUICK_REFERENCE.txt` |
| How do I interpret results? | Read `PROFILING_SUMMARY.md` section "What You'll Learn" |
| Where's the detailed guide? | `PROFILING_README.md` |
| Can I customize measurements? | Yes - edit `profiling_sharding_noc_python.py` |
| How do I use Tracy? | `./run_profiling.sh tracy` |

---

## ✅ Checklist for Your Research

- [ ] Setup verified: `python research_codes/test_profiling_setup.py`
- [ ] Quick test run: `./research_codes/run_profiling.sh quick`
- [ ] Full analysis done: `./research_codes/run_profiling.sh full`
- [ ] Results understood: Read output + `PROFILING_SUMMARY.md`
- [ ] Batch sweep completed: `./research_codes/run_profiling.sh sweep`
- [ ] Tracy profiling explored: `./research_codes/run_profiling.sh tracy`
- [ ] Custom measurements added: Edit `profiling_sharding_noc_python.py`
- [ ] Optimization implemented: Based on profiling insights
- [ ] Re-profiled after optimization: Measure improvement

---

**Created**: 2025-11-06
**Purpose**: Low-level profiling for tensor sharding, weight streaming, and NoC communication
**Author**: AI Assistant for masterjunmo
**Status**: Ready for use ✓

---

## 🚀 Ready to Start?

```bash
# Verify everything is ready
python research_codes/test_profiling_setup.py

# Run your first benchmark
./research_codes/run_profiling.sh quick

# Get complete analysis
./research_codes/run_profiling.sh full
```

**Happy profiling! 🎉**

# Research Codes Documentation

This directory contains scripts for analyzing minibatching overhead compared to single large batch execution, using TT_METAL_PROFILER to measure time of each stage in a forward pass using RISC-V cycles.

## Scripts Overview

### 1. `weight_loading_test.py`

**Purpose**: Benchmark script that measures weight loading and forward pass performance for both large batch and mini-batch scenarios.

**What it does**:
- Creates a linear layer (4096 → 4096) with configurable batch sizes
- Measures timing for:
  - Kernel compilation (one-time cost)
  - Weight loading from host DRAM to device GDDR6
  - Forward pass computation (includes weight streaming, NoC communication, and compute)
- Supports two scenarios:
  - **Large batch**: Single forward pass with batch size 256
  - **Mini-batch**: 8 forward passes with batch size 32 each (total 256 samples)
- Uses Python `time.perf_counter()` for high-resolution timing
- Includes `device_synchronize()` calls to ensure accurate measurements
- Saves results to CSV file (`benchmark_results.csv`)

**Usage**:
```bash
# Run both scenarios
python3 research_codes/weight_loading_test.py

# Run only large batch
python3 research_codes/weight_loading_test.py --only-large

# Run only mini-batch
python3 research_codes/weight_loading_test.py --only-mini

# Custom configuration
python3 research_codes/weight_loading_test.py \
    --large-batch-size 256 \
    --small-batch-size 32 \
    --minibatches 8 \
    --measure-iters 5
```

**Output**:
- Console output with timing breakdown
- CSV file with detailed metrics (if `--output-csv` specified or default `benchmark_results.csv`)

**Key Metrics**:
- `large_forward_compute_ms`: Large batch forward pass time
- `mini_forward_compute_ms`: Mini-batch total forward pass time (sum of 8 forwards)
- `overhead_ms`: Difference between mini-batch and large batch times
- `overhead_percentage`: Overhead as percentage of large batch time

---

### 2. `device_profile_analysis.py`

**Purpose**: Analyzes device profiler CSV output to extract cycle-accurate timing measurements for each stage of execution.

**What it does**:
- Parses `generated/profiler/.logs/profile_log_device.csv` (generated when `TT_METAL_DEVICE_PROFILER=1` is set)
- Extracts device frequency from CSV header (supports Blackhole, Wormhole, Grayskull)
- Groups zones by RISC type (BRISC, NCRISC, TRISC) and operation (`run_host_id`)
- Calculates:
  - **Wall clock time**: Timeline span from earliest start to latest end
  - **Parallel work**: Timeline span × number of cores (accounts for parallelism)
  - **Component breakdown**: Weight streaming (BRISC), NoC communication (NCRISC), Computation (TRISC)
  - **Effective parallelism**: Total work / wall clock time
  - **Efficiency**: Parallelism / theoretical maximum (650x for 130 cores × 5 RISCs)
- Handles unmatched zones (zones with START but no END marker)
- Automatically detects large batch vs mini-batch scenarios
- Extracts Python timing from benchmark CSV for comparison
- Intelligently groups operations to capture complete forward passes

**Usage**:
```bash
# Analyze current device profile
python3 research_codes/device_profile_analysis.py
```

**Prerequisites**:
- Must run with `TT_METAL_DEVICE_PROFILER=1` first to generate profile CSV
- Requires `benchmark_results.csv` for Python time comparison (optional, falls back to defaults)

**Output**:
- Console output with detailed analysis:
  - Device wall clock time
  - Python measurement time (if available)
  - Parallel work breakdown by component
  - Parallelism and efficiency metrics
  - Per-forward averages (for mini-batch)

**Key Features**:
- **Accurate cycle-to-time conversion**: Reads actual device frequency from CSV header
- **Parallel work calculation**: Uses timeline span × cores (not sum of durations)
- **Operation detection**: Automatically finds forward pass operations
- **Unmatched zone handling**: Reports and accounts for incomplete zones

---

### 3. `run_full_device_profile.py`

**Purpose**: Automated workflow script that orchestrates the complete profiling and analysis pipeline.

**What it does**:
1. Clears old device profile data
2. Runs large batch profiling with device profiler enabled
3. Analyzes large batch profile and saves results to JSON
4. Clears profile data for mini-batch
5. Runs mini-batch profiling with device profiler enabled
6. Analyzes mini-batch profile and saves results to JSON
7. **Appends comparison data to CSV file** for cumulative data collection
8. Optionally generates markdown report (if requested)

**Usage**:
```bash
# Run complete workflow
python3 research_codes/run_full_device_profile.py

# With custom CSV output path
python3 research_codes/run_full_device_profile.py --output-csv custom_results.csv
```

**Output Files**:
- `research_codes/large_batch_profile.json`: Large batch analysis results
- `research_codes/mini_batch_profile.json`: Mini-batch analysis results
- `research_codes/device_profile_comparison.csv`: **Cumulative comparison data** (appended each run)
- `research_codes/DEVICE_PROFILE_RESULTS.md`: Markdown report (if generated)

**CSV Output Format** (`device_profile_comparison.csv`):
The CSV file contains one row per run with the following columns:
- `timestamp`: When the run was executed
- `large_wall_clock_ms`: Device wall clock time for large batch
- `large_python_ms`: Python measurement time for large batch
- `large_total_work_ms`: Total parallel work for large batch
- `large_parallelism`: Effective parallelism for large batch
- `large_efficiency_pct`: Parallelism efficiency percentage
- `large_weight_streaming_ms`: Weight streaming work (BRISC)
- `large_noc_communication_ms`: NoC communication work (NCRISC)
- `large_computation_ms`: Computation work (TRISC)
- `mini_per_fwd_wall_clock_ms`: Per-forward device wall clock for mini-batch
- `mini_per_fwd_python_ms`: Per-forward Python time for mini-batch
- `mini_per_fwd_total_work_ms`: Per-forward total parallel work
- `mini_per_fwd_parallelism`: Per-forward effective parallelism
- `mini_per_fwd_efficiency_pct`: Per-forward efficiency
- `mini_per_fwd_weight_streaming_ms`: Per-forward weight streaming
- `mini_per_fwd_noc_communication_ms`: Per-forward NoC communication
- `mini_per_fwd_computation_ms`: Per-forward computation
- `work_overhead_pct`: Percentage overhead of mini-batch work vs large batch
- `wall_clock_overhead_pct`: Percentage overhead of mini-batch wall clock vs large batch
- `cores_used_large`: Number of cores used in large batch
- `cores_used_mini`: Number of cores used in mini-batch

**Key Features**:
- **Cumulative data collection**: Each run appends to CSV for trend analysis
- **Automatic operation detection**: Intelligently finds forward pass operations
- **Error handling**: Stops on failures with clear error messages
- **Progress reporting**: Shows detailed progress for each step

---

### 4. `compare_core_usage.py`

**Purpose**: Simple utility script to compare core usage patterns between large batch and mini-batch scenarios.

**What it does**:
- Loads device profile CSV
- Identifies large batch operation (last `run_host_id`)
- Identifies mini-batch operation (previous `run_host_id`)
- Compares:
  - Number of unique cores used
  - Number of TRISC zones
  - Core usage ratio vs expected ratio (based on batch size)
- Prints analysis in Korean

**Usage**:
```bash
# Compare core usage (requires device profile CSV)
python3 research_codes/compare_core_usage.py
```

**Output**:
- Console output showing:
  - Core counts for both scenarios
  - TRISC zone counts
  - Ratio analysis (actual vs expected)
  - Conclusion about over-parallelization

**Note**: This script is a simple analysis tool and may need adjustment if the device profile contains multiple operations.

---

### 5. `run_full_comparison.sh`

**Purpose**: Bash script wrapper for running the complete profiling workflow.

**What it does**:
- Runs large batch profiling with minimal iterations (for quick testing)
- Analyzes large batch results
- Runs mini-batch profiling with minimal iterations
- Analyzes mini-batch results
- Saves large batch analysis to `/tmp/large_batch_analysis.txt`

**Usage**:
```bash
bash research_codes/run_full_comparison.sh
```

**Note**: This script uses minimal iterations (0 warmup, 1 measurement) for quick testing. For production analysis, use `run_full_device_profile.py` instead.

---

## Workflow

### Typical Usage Flow

1. **Run benchmark with device profiler**:
   ```bash
   TT_METAL_DEVICE_PROFILER=1 python3 research_codes/weight_loading_test.py --only-large
   ```

2. **Analyze device profile**:
   ```bash
   python3 research_codes/device_profile_analysis.py
   ```

3. **Or run complete automated workflow**:
   ```bash
   python3 research_codes/run_full_device_profile.py
   ```

### For Cumulative Data Collection

Run `run_full_device_profile.py` multiple times to build up a dataset in `device_profile_comparison.csv`:

```bash
# Run multiple times to collect data
for i in {1..10}; do
    echo "Run $i/10"
    python3 research_codes/run_full_device_profile.py
    sleep 5  # Allow device to cool down
done
```

Then analyze the CSV file to see trends over time.

---

## Key Concepts

### Device Profiling

- **TT_METAL_DEVICE_PROFILER=1**: Enables cycle-accurate device-side profiling
- **Profile CSV**: Contains zone start/end timestamps in RISC-V cycles
- **Zones**: Instrumented code regions (weight streaming, NoC communication, computation)
- **RISC Types**:
  - BRISC: Boot RISC (weight streaming from GDDR6 to L1)
  - NCRISC: Network-on-Chip RISC (inter-core communication)
  - TRISC: Tensor RISC (computation)

### Time Measurements

- **Python time**: Measured using `time.perf_counter()`, includes `device_synchronize()` overhead
- **Device wall clock**: Calculated from cycle timestamps, represents actual device execution time
- **Parallel work**: Timeline span × number of cores (accounts for parallel execution)
- **Effective parallelism**: Total work / wall clock time

### Discrepancies

- **Python > Device**: Python time includes synchronization overhead
- **Device > Python**: Device profiler captures overlapping work or zones Python timer misses

---

## File Structure

```
research_codes/
├── README.md                          # This file
├── weight_loading_test.py             # Benchmark script
├── device_profile_analysis.py         # Device profile analyzer
├── run_full_device_profile.py         # Automated workflow
├── compare_core_usage.py              # Core usage comparison utility
├── run_full_comparison.sh             # Bash wrapper script
├── benchmark_results.csv              # Python benchmark results
├── device_profile_comparison.csv      # Cumulative comparison data (generated)
├── large_batch_profile.json           # Large batch analysis (generated)
├── mini_batch_profile.json          # Mini-batch analysis (generated)
└── DEVICE_PROFILE_RESULTS.md         # Markdown report (if generated)
```

---

## Troubleshooting

### Device profile not found
- Ensure `TT_METAL_DEVICE_PROFILER=1` is set before running benchmarks
- Check that `generated/profiler/.logs/profile_log_device.csv` exists

### Python time not found
- Ensure `benchmark_results.csv` exists (generated by `weight_loading_test.py`)
- Script will fall back to default values if CSV not found

### Unmatched zones warning
- Some zones may not have END markers (normal in some cases)
- Script accounts for unmatched zones in timeline calculations

### Frequency extraction failed
- Check CSV header format matches expected pattern
- Script will fall back to default 1350 MHz if extraction fails

---

## Notes

- All timing measurements use RISC-V cycles converted to milliseconds using device frequency
- Parallel work calculation uses timeline span × cores (not sum of durations) to account for parallelism
- Device profiler captures cycle-accurate timestamps, while Python timer includes host-side overhead
- Mini-batch analysis automatically estimates number of forward passes based on operation patterns

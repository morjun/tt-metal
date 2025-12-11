# How to Trace from Python Code to Kernel Source Files

This guide explains how to find which kernel source files are actually executed when running a Python test.

## Step-by-Step Process

### Step 1: Start with Your Python Test File

```python
# research_codes/weight_loading_test.py
from models.common.helper_funcs import Linear as TtLinear
linear = TtLinear(in_features, out_features, w_tt, b_tt, ...)
output = linear(x_tt)  # This calls ttnn.linear internally
```

### Step 2: Trace to the Helper Function

```python
# models/common/helper_funcs.py
def Linear(...):
    def linear_(activation):
        return ttnn.linear(activation, weight_T, bias=bias, ...)
    return linear_
```

**Key**: The helper calls `ttnn.linear()` - this is the entry point.

### Step 3: Find the C++ Operation Implementation

```bash
# Search for where ttnn.linear is implemented
grep -r "def linear\|ttnn.linear" ttnn/ttnn/operations/
```

You'll find:
- `ttnn/ttnn/operations/matmul.py` - Python wrapper
- `ttnn/cpp/ttnn/operations/matmul/matmul.cpp` - C++ implementation
- `ttnn/cpp/ttnn/operations/matmul/matmul.hpp` - Header

The C++ code calls matmul program factories.

### Step 4: Identify Which Program Factory is Used

The matmul operation selects a program factory based on:
- Program config (if provided)
- Tensor shapes and memory layouts
- Device capabilities

```bash
# Check which program factories exist
ls ttnn/cpp/ttnn/operations/matmul/device/matmul_op_*.cpp
```

Common factories:
- `matmul_op_multi_core_reuse_program_factory.cpp`
- `matmul_op_multi_core_reuse_optimized_program_factory.cpp`
- `matmul_op_multi_core_reuse_mcast_1d_program_factory.cpp`
- `matmul_op_multi_core_reuse_mcast_2d_program_factory.cpp`

### Step 5: Check the Cache to See Which Kernels Were Compiled

**This is the most reliable method!**

```bash
# Find recently compiled kernels
find ~/.cache/tt-metal-cache -type d -name "*reader*" | grep bmm

# Or list all kernels in a specific cache entry
ls -la ~/.cache/tt-metal-cache/*/kernels/ | grep reader
```

Example output:
```
reader_bmm_tile_layout_in0_receiver
reader_bmm_tile_layout_in0_sender_padding
reader_bmm_tile_layout_in1_sender_writer_padding
```

These are the kernels that were **actually compiled and used**.

### Step 6: Find the Kernel Source Files

Once you know the kernel names, find the source files:

```bash
# Search for kernel files
find ttnn/cpp/ttnn/operations/matmul/device/kernels -name "*reader_bmm_tile_layout_in0_sender_padding*"
find ttnn/cpp/ttnn/operations/matmul/device/kernels -name "*reader_bmm_tile_layout_in1_sender_writer_padding*"
```

Or use glob:
```bash
find . -path "*/kernels/dataflow/reader_bmm_tile_layout*.cpp"
```

### Step 7: Verify Which Program Factory Uses These Kernels

```bash
# Search for kernel file references in program factories
grep -r "reader_bmm_tile_layout_in0_sender_padding" \
  ttnn/cpp/ttnn/operations/matmul/device/
```

This shows which factory creates these kernels.

## Complete Example Workflow

```bash
# 1. Run your test to generate cache
export TT_METAL_DEVICE_PROFILER=1
python3 research_codes/weight_loading_test.py --only-large

# 2. Check which kernels were compiled
find ~/.cache/tt-metal-cache -type d -name "*reader*" | \
  grep bmm | head -5

# 3. Find the source files
KERNEL_NAME="reader_bmm_tile_layout_in1_sender_writer_padding"
find . -path "*/kernels/dataflow/${KERNEL_NAME}.cpp"

# 4. Check which program factory uses it
grep -r "${KERNEL_NAME}" \
  ttnn/cpp/ttnn/operations/matmul/device/matmul_op_*.cpp

# 5. Verify by checking build logs
find ~/.cache/tt-metal-cache -name "build.log" \
  -path "*${KERNEL_NAME}*" | head -1 | xargs tail -20
```

## Alternative: Check Build Logs Directly

```bash
# Find build logs for specific kernels
find ~/.cache/tt-metal-cache -name "build.log" \
  -path "*reader_bmm*" | head -3

# Check what was compiled
find ~/.cache/tt-metal-cache -name "build.log" \
  -path "*reader_bmm_tile_layout_in1_sender*" | \
  head -1 | xargs grep "g++" | head -1
```

## Understanding Kernel Naming Conventions

- `reader_bmm_tile_layout_in0_*` - Reads input tensor (in0/activation)
- `reader_bmm_tile_layout_in1_*` - Reads weight tensor (in1/weight)
- `*_sender_*` - Multicast sender (reads from DRAM, multicasts to receivers)
- `*_receiver_*` - Multicast receiver (receives multicast data)
- `*_writer_*` - Also writes output
- `*_padding` - Handles padding

## Quick Reference Commands

```bash
# Find all matmul reader kernels
find ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow \
  -name "reader_bmm*.cpp"

# Check which kernels are in cache
ls ~/.cache/tt-metal-cache/*/kernels/ 2>/dev/null | grep reader

# Find program factory that uses a kernel
grep -l "reader_bmm_tile_layout_in1_sender_writer_padding" \
  ttnn/cpp/ttnn/operations/matmul/device/matmul_op_*.cpp

# Check if kernel has profiling zones
grep -n "DeviceZoneScoped" \
  ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp
```

## Why This Matters

Different matmul configurations use different kernels:
- **Simple matmul**: `reader_bmm_tile_layout.cpp` (single reader)
- **Optimized matmul**: `reader_bmm_tile_layout_in0.cpp` + `reader_writer_bmm_tile_layout_in1.cpp`
- **Multicast matmul**: `reader_bmm_tile_layout_in0_sender_padding.cpp` + `reader_bmm_tile_layout_in0_receiver.cpp` + `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` + `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`

**Always check the cache** to see which kernels are actually used for your specific test case!

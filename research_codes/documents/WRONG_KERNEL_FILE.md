# ⚠️ 중요: 잘못된 커널 파일을 수정했습니다!

## 🔍 문제 발견

사용자가 `bmm_large_block_zm.cpp` 파일에 custom profiling zone을 추가했지만, **실제로 실행되는 커널 파일은 `bmm_large_block_zm_fused_bias_activation.cpp`입니다!**

## 📊 증거

### 1. 캐시 분석
JIT 빌드 캐시를 확인하면:
- ✅ `bmm_large_block_zm_fused_bias_activation` → 캐시에 있음
- ❌ `bmm_large_block_zm` → 캐시에 없음

### 2. 코드 분석
matmul operation factory들을 확인하면:
- `matmul_op_multi_core_reuse_optimized_program_factory.cpp` → `bmm_large_block_zm_fused_bias_activation.cpp` 사용
- `matmul_op_multi_core_reuse_mcast_1d_program_factory.cpp` → `bmm_large_block_zm_fused_bias_activation.cpp` 사용
- `matmul_op_multi_core_reuse_mcast_2d_program_factory.cpp` → `bmm_large_block_zm_fused_bias_activation.cpp` 사용
- `matmul_op_multi_core_reuse_mcast_dram_sharded_program_factory.cpp` → `bmm_large_block_zm_fused_bias_activation.cpp` 사용

**단 하나만**:
- `matmul_op_multi_core_reuse_program_factory.cpp` → `bmm_large_block_zm.cpp` 사용

대부분의 경우 `bmm_large_block_zm_fused_bias_activation.cpp`가 사용됩니다!

## ✅ 해결 방법

### 방법 1: 올바른 파일 수정 (권장)

`bmm_large_block_zm_fused_bias_activation.cpp` 파일에도 동일한 custom profiling zone을 추가해야 합니다:

```bash
# 파일 확인
cat ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp | head -30
```

이 파일에도 다음을 추가해야 합니다:
1. `#include "tools/profiler/kernel_profiler.hpp"`
2. `DeviceZoneScopedMainChildN("TRISC-MATMUL-COMPUTE")`
3. `DeviceZoneScopedN("MM-INIT")` 등 custom zones

### 방법 2: 어떤 파일이 사용되는지 확인

실제로 어떤 matmul operation factory가 사용되는지 확인하려면:

```python
# Python 코드에서 확인
import ttnn
# ... your matmul operation ...
# 코드 실행 후 로그 확인
```

또는 실행 시 로그에서 확인:
```bash
# 빌드 로그 확인
find ~/.cache/tt-metal-cache/ -name "build.log" | xargs grep -l "bmm_large" | head -5
```

## 📝 권장 작업

1. **`bmm_large_block_zm_fused_bias_activation.cpp` 파일 확인**:
   ```bash
   grep -n "DeviceZoneScoped\|kernel_profiler" \
     ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp
   ```

2. **없다면 추가**:
   - `bmm_large_block_zm.cpp`에 추가한 것과 동일한 방식으로
   - `bmm_large_block_zm_fused_bias_activation.cpp`에도 추가

3. **캐시 삭제 및 재실행**:
   ```bash
   rm -rf ~/.cache/tt-metal-cache/*
   export TT_METAL_DEVICE_PROFILER=1
   python3 your_program.py
   ```

## 🔍 확인 방법

실행 후 다음을 확인:
```bash
# 1. 빌드 로그에서 custom zone 확인
find ~/.cache/tt-metal-cache/ -path "*bmm_large_block_zm_fused*/trisc*/build.log" | \
  xargs grep -i "MM-INIT\|BATCH-ITERATION\|TRISC-MATMUL"

# 2. new_zone_src_locations.log 확인
grep -i "bmm_large_block_zm_fused" generated/profiler/.logs/new_zone_src_locations.log

# 3. profile_log_device.csv 확인
grep -E "MM-INIT|BATCH-ITERATION|TRISC-MATMUL-COMPUTE" \
  generated/profiler/.logs/profile_log_device.csv | head -10
```

## 📚 관련 파일

- 수정한 파일: `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm.cpp`
- **실제 사용되는 파일**: `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`
- 예시 파일: `research_codes/EXAMPLE_bmm_large_block_zm_PROFILED.cpp`

# Custom Profiling Zone 추가 완료

## 📝 수정된 파일

`ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`

이 파일은 `weight_loading_test.py`의 `TtLinear` 함수가 `ttnn.linear`을 호출할 때 사용되는 커널 파일입니다.

## ✅ 추가된 Custom Zones

다음과 같은 profiling zones를 추가했습니다:

### 1. Main Zone
- `TRISC-MATMUL-FUSED-COMPUTE`: 전체 커널 실행 범위 (함수 시작 부분)

### 2. Initialization
- `MM-BLOCK-INIT`: Matmul block 초기화

### 3. Batch & Block Iterations
- `BATCH-ITERATION`: 각 batch 반복
- `BLOCK-DIM-ITERATION`: Block dimension 반복 (h_dim × w_dim)
- `INNER-BLOCK-ITERATION`: Inner dimension block 반복

### 4. Data Movement
- `CB-WAIT-FRONT`: Circular buffer 대기 (idle time 측정)
- `CB-POP-FRONT`: Circular buffer pop

### 5. Computation
- `SUBBLOCK-COMPUTE`: 각 subblock 계산 범위
- `MATMUL-BLOCK`: 실제 matmul block 계산 (핵심 연산)
- `RELOAD-PARTIAL`: Partial result 재로드

### 6. Output Processing
- `PACK-OUTPUT`: 최종 결과 packing
- `PACK-PARTIAL`: Partial result packing

### 7. Bias & Activation (if enabled)
- `FUSE-BIAS`: Bias fusion 전체 범위
- `ADD-BIAS`: Bias 추가 연산
- `ACTIVATION`: Activation 함수 적용
- `PACK-BIAS-OUTPUT`: Bias가 포함된 출력 packing

### 8. Optional Operations
- `UNTILIZE-OUTPUT`: Untilize 연산 (if enabled)

## 🔧 다음 단계

1. **JIT 캐시 삭제**:
   ```bash
   rm -rf ~/.cache/tt-metal-cache/*
   ```

2. **환경변수 설정**:
   ```bash
   export TT_METAL_DEVICE_PROFILER=1
   ```

3. **프로그램 실행**:
   ```bash
   python3 research_codes/weight_loading_test.py --only-large
   ```

4. **결과 확인**:
   ```bash
   grep -E "TRISC-MATMUL-FUSED-COMPUTE|MM-BLOCK-INIT|BATCH-ITERATION|MATMUL-BLOCK|FUSE-BIAS" \
     generated/profiler/.logs/profile_log_device.csv | head -20
   ```

## 📊 예상 결과

`profile_log_device.csv`에서 다음과 같은 zone들이 나타나야 합니다:

```
TRISC-MATMUL-FUSED-COMPUTE
  MM-BLOCK-INIT
  BATCH-ITERATION
    BLOCK-DIM-ITERATION
      INNER-BLOCK-ITERATION
        CB-WAIT-FRONT
        SUBBLOCK-COMPUTE
          MATMUL-BLOCK        ← 핵심 연산!
          PACK-OUTPUT
        CB-POP-FRONT
      FUSE-BIAS
        ADD-BIAS
        ACTIVATION
        PACK-BIAS-OUTPUT
```

## 🔍 확인 방법

빌드 로그에서 custom zone 확인:
```bash
find ~/.cache/tt-metal-cache/ -path "*bmm_large_block_zm_fused*/trisc*/build.log" | \
  xargs grep -i "TRISC-MATMUL-FUSED\|MM-BLOCK-INIT\|MATMUL-BLOCK\|FUSE-BIAS" | head -10
```

`new_zone_src_locations.log` 확인:
```bash
grep -i "TRISC-MATMUL-FUSED\|MM-BLOCK-INIT\|MATMUL-BLOCK\|FUSE-BIAS" \
  generated/profiler/.logs/new_zone_src_locations.log
```

## 📌 참고사항

- 이 커널은 **bias가 있을 때** 사용됩니다
- `weight_loading_test.py`는 bias를 사용하므로 이 커널이 실행됩니다
- Custom zone은 `PROFILE_KERNEL`이 정의되어 있을 때만 활성화됩니다

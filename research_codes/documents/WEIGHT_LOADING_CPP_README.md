# C++ Weight Loading Benchmark

## 개요

Python 코드로는 device profiler의 `run_host_id`와 forward pass를 정확히 매핑할 수 없습니다. 각 `linear()` 호출이 새로운 program을 생성하므로, 하나의 forward pass가 여러 `run_host_id`를 가질 수 있고, 이를 정확히 식별하기 어렵습니다.

## 해결 방안: C++ 코드 작성

C++ tt-metal 코드를 작성하여 각 forward pass에 명시적인 profiler 마커를 추가하면, device profiler에서 정확한 forward pass 경계를 식별할 수 있습니다.

## 현재 상태

`weight_loading_benchmark.cpp`는 템플릿 코드입니다. 완전한 구현을 위해서는:

1. **Tensor 생성 및 초기화**
   - `create_random_tensor()` 함수 완전 구현
   - Host에서 random data 생성
   - ttnn tensor format으로 변환
   - Device로 전송

2. **Device 초기화**
   - Device 객체 생성 및 초기화
   - Command queue 설정

3. **Tensor Slicing**
   - Mini-batch 시나리오에서 large input tensor를 slice하는 기능
   - `ttnn::slice()` 또는 유사한 함수 사용

4. **Device Profiler 마커 통합**
   - 각 forward pass에 `ZoneScopedN("ForwardPass_N")` 마커 추가
   - Device profiler가 이 마커를 사용하여 정확한 경계 식별

## 주요 개선 사항

### Python 코드의 문제점
- `run_host_id`와 forward pass의 1:1 매핑 불가능
- 시간 기반 그룹화는 근사치일 뿐
- `op_hash` 기반 그룹화도 정확하지 않음

### C++ 코드의 장점
- 각 forward pass에 명시적 마커 설정 가능
- Device profiler가 마커를 사용하여 정확한 경계 추적
- `ZoneScopedN()` 마커가 device profiler CSV에 기록됨
- 정확한 forward pass별 시간 측정 가능

## 다음 단계

1. `weight_loading_benchmark.cpp` 완전 구현
2. Build system에 추가 (CMakeLists.txt 등)
3. Device profiler와 통합 테스트
4. Python 분석 코드 업데이트 (C++ 마커 기반 분석)

## 참고

- Device profiler 마커 사용법: `docs/source/tt-metalium/tools/device_program_profiler.rst`
- C++ tt-metal API: `ttnn/cpp/ttnn/operations/matmul/matmul.hpp`
- 예제 코드: `tests/ttnn/unit_tests/gtests/test_matmul_benchmark.cpp`

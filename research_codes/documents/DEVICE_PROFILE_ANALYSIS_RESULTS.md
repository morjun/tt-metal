# Device Profile Analysis Results

## 실행 환경
- **Device**: Blackhole @ 1350 MHz (1.35 GHz)
- **Configuration**: warmup_iters=1, measure_iters=1
- **Scenario**: Mini-batch (b=32, 8 forward passes per sequence)

## 📊 핵심 결과

### Single Forward Pass 분석 (마지막 forward pass)

**Device Wall Clock**: 0.114215 ms (실제 device 실행 시간)
**Python Measurement**: 0.180175 ms (device + sync overhead)
**Sync Overhead**: 0.065960 ms (36.6%)

#### 세부 Operation 시간 분석:

| Component | Span (ms) | Cores | Work (ms) | % of Total |
|-----------|-----------|-------|-----------|------------|
| **Weight Streaming (BRISC)** | 0.114215 | 128 | 14.62 | 33.7% |
| **NoC Communication (NCRISC)** | 0.112079 | 128 | 14.35 | 33.1% |
| **Computation (TRISC)** | 0.112150 | 128 | 14.36 | 33.1% |
| **TOTAL** | - | - | 43.32 | 100.0% |

**Effective Parallelism**: 379.3x
- Total work: 43.32 ms
- Wall clock: 0.114 ms
- Efficiency: 58.4% (vs theoretical max 650x = 130 cores × 5 RISCs)

### 전체 Sequence 분석 (8 forward passes)

**Total Device Wall Clock**: 0.466101 ms (8개 forward pass의 합)
**Total Python Measurement**: 1.441401 ms
**Total Sync Overhead**: 0.975300 ms (67.7%)

#### 전체 Sequence Parallel Work:

| Component | Total Work (ms) | % of Total |
|-----------|-----------------|------------|
| **Weight Streaming (BRISC)** | 59.68 | 34.0% |
| **NoC Communication (NCRISC)** | 58.25 | 33.2% |
| **Computation (TRISC)** | 57.42 | 32.7% |
| **TOTAL** | 175.35 | 100.0% |

**Total Effective Parallelism**: 376.2x
**Total Efficiency**: 57.9%

### Per-Forward 평균 (8개 평균)

**Per-forward Device Wall Clock**: 0.058263 ms
**Per-forward Python Measurement**: 0.180175 ms

| Component | Per-Forward Work (ms) | % of Total |
|-----------|----------------------|------------|
| **Weight Streaming** | 7.46 | 34.0% |
| **NoC Communication** | 7.28 | 33.2% |
| **Computation** | 7.18 | 32.7% |
| **TOTAL** | 21.92 | 100.0% |

## 🔍 개별 Forward Pass 분석

### 패턴 분석:
8개의 forward pass가 **교대로 2가지 패턴**을 보임:

#### 패턴 1: "Fast" Forward Pass (홀수 번째: 1, 3, 5, 7)
- Wall Clock: ~0.002 ms (매우 빠름)
- Zones: 520개
- **특징**: TRISC (Computation) 작업이 0.00 ms
- Weight Streaming: ~58-59%
- NoC Communication: ~41-42%
- 추정: Weight/Data 전송만 하고 실제 computation은 하지 않음 (준비 단계?)

#### 패턴 2: "Full" Forward Pass (짝수 번째: 2, 4, 6, 8)
- Wall Clock: ~0.114 ms
- Zones: 1280개
- **특징**: 모든 component가 균등하게 작동
- Weight Streaming: ~33.7%
- NoC Communication: ~33.1%
- Computation: ~33.1%
- 추정: 실제 matmul 연산 수행

### 개별 Forward Pass 상세:

| Forward | Type | Wall Clock (ms) | BRISC (ms) | NCRISC (ms) | TRISC (ms) | Total Work (ms) | Parallelism |
|---------|------|-----------------|------------|-------------|------------|-----------------|-------------|
| 1 | Fast | 0.002362 | 0.31 (58.6%) | 0.22 (41.4%) | 0.00 (0.0%) | 0.52 | 221.7x |
| 2 | **Full** | 0.114136 | 14.61 (33.7%) | 14.35 (33.1%) | 14.36 (33.1%) | 43.31 | 379.5x |
| 3 | Fast | 0.002456 | 0.32 (58.9%) | 0.22 (41.1%) | 0.00 (0.0%) | 0.54 | 220.8x |
| 4 | **Full** | 0.114117 | 14.61 (33.7%) | 14.34 (33.1%) | 14.35 (33.1%) | 43.30 | 379.5x |
| 5 | Fast | 0.002267 | 0.29 (58.0%) | 0.21 (42.0%) | 0.00 (0.0%) | 0.51 | 224.1x |
| 6 | **Full** | 0.114241 | 14.62 (33.8%) | 14.35 (33.1%) | 14.36 (33.1%) | 43.32 | 379.2x |
| 7 | Fast | 0.002307 | 0.30 (58.2%) | 0.21 (41.8%) | 0.00 (0.0%) | 0.51 | 223.2x |
| 8 | **Full** | 0.114215 | 14.62 (33.7%) | 14.35 (33.1%) | 14.36 (33.1%) | 43.32 | 379.3x |

## 💡 주요 인사이트

### 1. 3-Way 균등 분산
**Full forward pass**에서 세 가지 operation이 거의 균등하게 시간을 소비:
- Weight Streaming (SRAM weight streaming): ~33.7%
- NoC Communication: ~33.1%
- Tensor Computation: ~33.1%

이는 매우 **균형잡힌 워크로드**를 나타냄.

### 2. Fast vs Full 패턴
- **Fast pass**: Weight/Data 준비만 (computation 없음)
- **Full pass**: 실제 matmul 포함
- 이 패턴은 mini-batch 처리 방식의 특징으로 보임

### 3. 효율성
- **Parallelism**: 379.3x (single forward pass)
- **Efficiency**: 58.4% (theoretical max 대비)
- 여전히 개선 여지가 있음 (41.6% unused capacity)

### 4. Sync Overhead
- Single forward: 36.6% (0.066 ms / 0.180 ms)
- Full sequence: 67.7% (0.975 ms / 1.441 ms)
- Sequence에서 sync overhead가 더 큼 → forward pass 사이의 gap (1.45 ms)

### 5. Gap Between Forward Passes
- Timeline span: 1.914 ms (first start to last end)
- Actual execution: 0.466 ms
- **Gap**: 1.448 ms (75.6% of timeline)
- 이 gap이 mini-batching overhead의 주요 원인!

## 🎯 최적화 제안

### 1. Gap 최소화
- Forward pass 사이의 gap (1.45 ms)을 줄이기
- Pipelining: 다음 forward pass를 미리 준비
- Batching: 여러 forward pass를 하나의 큰 batch로

### 2. Fast Pass 최적화
- Fast pass가 왜 필요한지 조사
- 가능하면 Fast pass를 Full pass와 병합

### 3. Parallelism 향상
- 현재 58.4% → 목표 70-80%
- Unused capacity (41.6%)를 활용

### 4. Weight Streaming 최적화
- Weight가 매번 streaming되는 것으로 보임
- Weight caching/reuse 고려

## 📁 생성된 파일

```
research_codes/tracy_output/
├── .logs/
│   ├── profile_log_device.csv (12MB) - Device profiling data
│   ├── tracy_ops_times.csv - Host-side zone timings
│   ├── tracy_ops_data.csv - Operation metadata
│   └── tracy_profile_log_host.tracy (1.5MB) - Tracy capture file
├── reports/
│   └── weight_loading_test/2025_11_09_22_48_27/
│       ├── ops_perf_results_*.csv - Performance analysis
│       └── profile_log_device.csv (copy)
└── benchmark_results_tracy.csv - Python benchmark results
```

## 🔧 사용 방법

### 1. 새 데이터로 분석 실행
```bash
# 1. Clean previous output
rm -rf research_codes/tracy_output

# 2. Run Tracy profiling (warmup=1, measure=1)
bash research_codes/run_tracy_weight_test.sh

# 3. Run device profile analysis
python3 research_codes/device_profile_analysis.py
```

### 2. Tracy GUI로 시각화
```bash
# Tracy profiler GUI 실행 후:
# File → Open → research_codes/tracy_output/.logs/tracy_profile_log_host.tracy
```

## ✅ 결론

TRACY profiling과 device profiler를 통해 각 forward pass의 세부 operation 시간을 성공적으로 분석했습니다:

1. ✅ **Weight Streaming**: ~33.7% (14.62 ms per full forward pass)
2. ✅ **NoC Communication**: ~33.1% (14.35 ms per full forward pass)
3. ✅ **Tensor Computation**: ~33.1% (14.36 ms per full forward pass)

**Mini-batching overhead**의 주요 원인:
- Forward pass 사이의 gap: 1.45 ms (75.6% of timeline)
- Sync overhead: 0.98 ms (67.7% of total Python time)

이 데이터를 바탕으로 최적화 방향을 명확히 할 수 있습니다!

---
생성일: 2025년 11월 9일
분석 도구: TRACY Profiler + tt-metal Device Profiler

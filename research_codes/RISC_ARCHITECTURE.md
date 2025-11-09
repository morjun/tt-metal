# Tensix Core Architecture - RISC Processor 설명

## 📐 Tenstorrent Blackhole P150A 아키텍처

### Tensix Core 구조

각 Tensix 코어는 **5개의 RISC-V 프로세서**를 포함합니다:

```
┌─────────────────────────────────────────────────────────┐
│                    Tensix Core                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │  BRISC   │  │ NCRISC   │  │  TRISC (x3)          │  │
│  │ (Data    │  │ (NoC)    │  │  • Unpack TRISC      │  │
│  │Movement 0)│  │          │  │  • Math TRISC        │  │
│  │          │  │          │  │  • Pack TRISC        │  │
│  └──────────┘  └──────────┘  └──────────────────────┘  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │         L1 SRAM (1.5MB)                         │   │
│  │  - Circular buffers                             │   │
│  │  - Intermediate data                            │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────┐                       ┌──────────┐       │
│  │  NoC 0   │◄──────────────────────┤  NoC 1   │       │
│  └──────────┘                       └──────────┘       │
└─────────────────────────────────────────────────────────┘
```

### 각 프로세서의 역할

#### 1. BRISC (Binary RISC) - Data Movement 0
- **주요 기능**: GDDR6 DRAM ↔ L1 SRAM 데이터 이동
- **담당 작업**:
  - Input 텐서를 DRAM에서 L1으로 로드 (input sharding)
  - Weight 행렬을 DRAM에서 L1으로 스트리밍 (weight streaming)
  - DRAM 읽기 작업 관리
- **커널**: Reader kernel 실행
- **측정 가능**: BRISC kernel time = Input sharding + Weight streaming 시간

#### 2. NCRISC (Network-on-Chip RISC) - Data Movement 1
- **주요 기능**: NoC를 통한 inter-core 통신
- **담당 작업**:
  - 계산 결과를 L1에서 DRAM으로 전송 (output gathering)
  - Core 간 데이터 라우팅
  - NoC 트랜잭션 관리
- **커널**: Writer kernel 실행
- **측정 가능**: NCRISC kernel time = Output gathering + NoC 통신 시간

#### 3. TRISC (Tensor RISC) - Compute (x3)
각 Tensix 코어는 **3개의 TRISC 프로세서**를 가짐:

**a) Unpack TRISC**
- FPU/SFPU에 입력할 데이터 언팩
- Tile format 변환

**b) Math TRISC**
- FPU (Matrix unit) 제어 - 행렬 곱셈
- SFPU (Vector unit) 제어 - element-wise 연산
- 실제 계산 수행

**c) Pack TRISC**
- 계산 결과를 메모리 저장 포맷으로 팩
- Circular buffer로 출력

- **커널**: Compute kernel 실행
- **측정 가능**: TRISC kernel time = 실제 계산 시간 (matmul 등)

## 🔢 실제 측정 결과 (Blackhole P150A)

### 사용된 하드웨어 리소스
```
Tensix 코어 수:        130 cores
RISC-V 프로세서 총 개수: 650 processors
  = 130 cores × 5 processors/core
  = 130 BRISC + 130 NCRISC + 390 TRISC (130×3)
```

### Device Operation 분해 (개별 operation, 예: matmul kernel)
```
Component              | Accumulated Time | Processors | Per-Processor
-----------------------|------------------|------------|---------------
BRISC (data movement)  | 15.196 ms        | 130 BRISC  | 0.117 ms
NCRISC (NoC)           | 14.946 ms        | 130 NCRISC | 0.115 ms
TRISC (compute)        | 44.906 ms        | 390 TRISC  | 0.115 ms
-----------------------|------------------|------------|---------------
Total work             | 75.048 ms        | 650 total  | 0.115 ms avg
Device operation time  | 0.124-0.164 ms   | (parallel) |
```

### Python Forward Pass (ttnn.matmul 호출)
```
Python forward pass time: 0.237 ms

포함 내역:
  - Device operation(s):        ~0.164 ms (실제 device 작업)
  - Host-Device sync:           ~0.050 ms (동기화 오버헤드)
  - Python overhead:            ~0.023 ms (Python 실행 오버헤드)
```

### Parallelism Factor 계산
```
Parallelism = Total work / Wall clock time
            = 75.048 ms / 0.124 ms
            = 605x

이는 650개의 RISC-V 프로세서가 병렬로 작동함을 의미
(실제로는 일부 overhead와 동기화로 인해 100% 효율은 아님)
```

## 📊 데이터 흐름

### Typical Forward Pass Flow

```
Host DRAM (CPU)
      ↓ (PCIe transfer, ~17ms for 4096×4096 weights)
GDDR6 DRAM (Device)
      ↓ (BRISC loads, measured by BRISC kernel time)
L1 SRAM (각 Tensix core에 1.5MB)
      ↓ (TRISC processes)
Compute Units (FPU/SFPU)
      ↓ (TRISC packs)
L1 SRAM (results)
      ↓ (NCRISC writes, measured by NCRISC kernel time)
GDDR6 DRAM (Device)
```

### Matmul Example (B=256, 4096×4096)

**Device-side (하나의 device operation)**:

1. **BRISC 작업 (15.2ms accumulated across 130 cores)**:
   - Input sharding: 256 batch를 130 cores에 분배
   - Weight streaming: 4096×4096 weight를 L1으로 로드
   - 각 core는 할당된 portion만 로드
   - Per-core time: ~0.117 ms

2. **TRISC 작업 (44.9ms accumulated across 390 TRISCs)**:
   - 390개 TRISC (130 cores × 3)가 동시에 계산
   - 각 core는 자신의 input/weight tile에 대해 matmul 수행
   - FPU (matrix unit)에서 실제 곱셈 실행
   - Per-TRISC time: ~0.115 ms

3. **NCRISC 작업 (14.9ms accumulated across 130 cores)**:
   - 130개 NCRISC가 각 core의 결과를 DRAM으로 전송
   - NoC를 통해 데이터 라우팅
   - Sharded output을 DRAM으로 gather
   - Per-core time: ~0.115 ms

**Timing Summary**:
- Device operation wall clock: **0.124-0.164 ms** (병렬 실행)
- Python forward pass: **0.237 ms** (device op + sync + overhead)
- 차이 (~0.07 ms) = Host-Device 동기화 + Python overhead

## 🎯 Weight Streaming Overhead 측정

### Large Batch vs Mini-batch 비교

**Large Batch (B=256, 1 forward pass)**:
- BRISC: Weight streaming **1번**
- Wall clock: 0.239 ms

**Mini-batch (b=32, 8 forward passes)**:
- BRISC: Weight streaming **8번**
- Wall clock: 1.443 ms

**Overhead 계산**:
```
Total overhead = 1.443 - 0.239 = 1.204 ms
Extra streaming = 8 - 1 = 7 times
Per-streaming cost = 1.204 / 7 = 0.172 ms
```

이는 **weight streaming이 각 forward pass마다 ~0.172ms 소요**됨을 의미

### BRISC Time 해석

```
BRISC accumulated time = 15.2 ms (130 cores에 분산)
Wall clock per core = 15.2 / 130 = 0.117 ms
```

이는 다음을 포함:
- Input sharding time (배치 크기에 비례)
- Weight streaming time (~0.172ms wall clock)
- DRAM ↔ L1 전송 overhead

## 💡 왜 600배 Parallelism?

### 오해하기 쉬운 점
❌ "600개의 코어가 있다" → 틀림
✓ "650개의 RISC-V 프로세서가 병렬로 작동한다" → 맞음

### 실제 계산
```
130 Tensix cores × 5 processors/core = 650 RISC-V processors

각 프로세서가 평균 0.115ms씩 작업 수행
→ 총 작업량 = 650 × 0.115 = 74.75ms

하지만 모두 병렬로 실행되므로
→ 실제 경과 시간 = 0.124ms

Parallelism = 74.75 / 0.124 ≈ 600x
```

### 왜 100% 효율이 아닌가?
- Synchronization overhead (circular buffer wait)
- Data dependency (BRISC → TRISC → NCRISC 순서 필요)
- Memory bandwidth 제약
- 일부 cores는 idle time 존재 가능

## 📝 핵심 요약

1. **130 Tensix cores** 사용 (Blackhole P150A)
2. **각 core = 5 RISC-V processors**:
   - 1 BRISC (data in)
   - 1 NCRISC (data out)
   - 3 TRISC (compute)
3. **총 650 processors** 병렬 실행
4. **BRISC time** = GDDR6 → L1 데이터 이동 시간 측정 가능
5. **Parallelism ~600x** = 650 processors의 병렬 실행 효율

이 아키텍처 덕분에 `profile_log_device.csv` 파싱으로 **GDDR6 → L1 SRAM의 정확한 데이터 이동 시간**을 측정할 수 있습니다!

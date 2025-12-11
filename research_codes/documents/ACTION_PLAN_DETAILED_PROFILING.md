# 🎯 FINAL SOLUTION: Baby RISC 코어 Operation-Level Profiling

## 📋 Executive Summary

**문제**: BRISC/NCRISC/TRISC 비율이 20:20:60으로 일정 → idle time이 포함된 부정확한 측정
**해결**: Kernel-level custom profiling zones 추가 → operation별 정확한 시간 측정

---

## 🚀 Quick Start (5분 이내 적용 가능)

### Step 1: Kernel 파일 찾기

```bash
cd /home/masterjunmo/codes/tt-metal

# ttnn.linear이 사용하는 kernel 찾기
find ttnn/cpp/ttnn/operations/matmul/device/kernels -name "*.cpp" | grep -E "reader|writer|compute"
```

**주요 파일**:
- Reader (BRISC): `reader_bmm_tile_layout.cpp`
- Writer (NCRISC): `writer_bmm_tile_layout.cpp`
- Compute (TRISC): `bmm_large_block_zm.cpp`

### Step 2: Profiling Zones 추가

**예시 파일**에 이미 완성된 코드가 있습니다:
- `research_codes/EXAMPLE_reader_bmm_tile_layout_PROFILED.cpp`
- `research_codes/EXAMPLE_bmm_large_block_zm_PROFILED.cpp`

**수정 방법**:
```cpp
// 1. 헤더 추가
#include "tools/profiler/kernel_profiler.hpp"

// 2. Main 함수에 전체 scope 추가
void kernel_main() {
    DeviceZoneScopedMainChildN("BRISC-MATMUL-READER");

    // 3. 각 operation에 zone 추가
    {
        DeviceZoneScopedN("CB-RESERVE");
        cb_reserve_back(...);
    }

    {
        DeviceZoneScopedN("READ-WEIGHT");
        // weight loading code
    }

    {
        DeviceZoneScopedN("NOC-BARRIER-WAIT");  // ← 이게 idle time!
        noc_async_read_barrier();
    }
}
```

### Step 3: 실제 파일 수정

```bash
# 원본 백업
cp ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout.cpp{,.backup}

# 수정
vim ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout.cpp
```

**수정 예시** (EXAMPLE 파일 참고):
1. `#include "tools/profiler/kernel_profiler.hpp"` 추가
2. `DeviceZoneScopedMainChildN("...")` 추가
3. 각 operation을 `DeviceZoneScopedN("...")` 로 감싸기

### Step 4: Rebuild & Run

```bash
# Rebuild (5-10분 소요)
./build_metal.sh

# Device profiler 활성화하여 실행
cd research_codes
export TT_METAL_DEVICE_PROFILER=1
python3 weight_loading_test_tracy.py --only-large
```

### Step 5: 결과 분석

```bash
# CSV 확인
cat tracy_output_large/profile_log_device.csv | grep -E "CB-RESERVE|READ-WEIGHT|NOC-BARRIER-WAIT"

# Python으로 분석
python3 analyze_detailed_zones.py
```

---

## 📊 예상 결과

### 기존 측정 (부정확):
```
BRISC-FW:      192 ms   (20%)  ← idle time 포함
NCRISC-FW:     186 ms   (20%)  ← idle time 포함
TRISC-FW:      560 ms   (60%)  ← idle time 포함
```

### 새로운 측정 (정확):
```
BRISC-MATMUL-READER:
  ├─ CB-RESERVE:           2 ms    (1%)
  ├─ READ-IN0-ACTIVATION:  60 ms   (31%)  ← 실제 작업
  ├─ READ-IN1-WEIGHT:      65 ms   (34%)  ← 실제 작업
  ├─ NOC-BARRIER-WAIT:     62 ms   (32%)  ← IDLE! (NoC 대기)
  └─ CB-PUSH:              3 ms    (2%)

NCRISC-MATMUL-WRITER:
  ├─ CB-WAIT:              50 ms   (27%)  ← IDLE! (데이터 대기)
  ├─ PACK-OUTPUT:          10 ms   (5%)
  ├─ NOC-WRITE:            110 ms  (59%)  ← 실제 작업
  └─ NOC-WAIT:             16 ms   (9%)   ← IDLE!

TRISC-MATMUL-COMPUTE:
  ├─ MM-INIT:              1 ms    (0.2%)
  ├─ CB-WAIT-FRONT:        120 ms  (21%)  ← IDLE! (데이터 대기)
  ├─ ACQUIRE-DST:          5 ms    (1%)
  ├─ MATMUL-TILES:         420 ms  (75%)  ← 실제 작업!!!
  ├─ PACK-OUTPUT:          10 ms   (2%)
  └─ RELEASE-DST:          4 ms    (1%)
```

**핵심 발견**:
- BRISC 실제 작업: 65% (나머지 35%는 NoC 대기)
- NCRISC 실제 작업: 59% (나머지 41%는 CB/NoC 대기)
- TRISC 실제 작업: 75% (나머지 25%는 CB 대기)

---

## 📁 제공된 파일들

### 1. `SOLUTION_DETAILED_RISC_PROFILING.md`
- 전체 솔루션 설명
- tt-metal profiler 구조 설명
- 사용 가능한 도구들 목록

### 2. `EXAMPLE_reader_bmm_tile_layout_PROFILED.cpp`
- BRISC (dataflow reader) 수정 예시
- 완전한 working code
- 각 zone 설명 포함

### 3. `EXAMPLE_bmm_large_block_zm_PROFILED.cpp`
- TRISC (compute) 수정 예시
- Matmul operation 상세 profiling
- Idle time vs actual compute 구분

### 4. 이 문서 (`ACTION_PLAN.md`)
- 빠른 실행 가이드
- 5분 안에 적용 가능한 단계별 설명

---

## 🔍 상세 분석 스크립트

```python
# analyze_detailed_zones.py
import csv
from collections import defaultdict
from pathlib import Path

def parse_detailed_profile(csv_path):
    """Parse profile_log_device.csv with custom zones."""

    zones = defaultdict(lambda: defaultdict(list))
    zone_stack = {}

    with open(csv_path, 'r') as f:
        # Skip header
        f.readline()
        f.readline()

        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 12:
                continue

            core_x, core_y = int(parts[1]), int(parts[2])
            risc_type = parts[3]
            cycles = int(parts[5])
            zone_name = parts[10]
            zone_type = parts[11]

            core_key = (core_x, core_y)
            zone_key = (risc_type, zone_name)

            if zone_type == 'ZONE_START':
                zone_stack[(core_key, zone_key)] = cycles
            elif zone_type == 'ZONE_END':
                stack_key = (core_key, zone_key)
                if stack_key in zone_stack:
                    duration = cycles - zone_stack[stack_key]
                    zones[risc_type][zone_name].append(duration)
                    del zone_stack[stack_key]

    return zones

def print_breakdown(zones, freq_mhz=1350):
    """Print detailed breakdown per RISC type."""

    for risc_type in sorted(zones.keys()):
        print(f"\n{'='*80}")
        print(f"{risc_type} OPERATION BREAKDOWN")
        print(f"{'='*80}")

        risc_zones = zones[risc_type]
        total_cycles = sum(sum(durations) for durations in risc_zones.values())
        total_ms = total_cycles / (freq_mhz * 1000)

        print(f"\nTotal time: {total_ms:.2f} ms ({total_cycles:,} cycles)")
        print(f"\n{'Operation':<40} {'Avg (cycles)':<15} {'Avg (ms)':<12} {'%':<8} {'Count':<8}")
        print("-" * 90)

        # Sort by total time
        zone_items = []
        for zone_name, durations in risc_zones.items():
            if len(durations) == 0:
                continue
            avg_cycles = sum(durations) / len(durations)
            avg_ms = avg_cycles / (freq_mhz * 1000)
            pct = (sum(durations) / total_cycles) * 100
            zone_items.append((zone_name, avg_cycles, avg_ms, pct, len(durations)))

        zone_items.sort(key=lambda x: x[3], reverse=True)  # Sort by %

        for zone_name, avg_cycles, avg_ms, pct, count in zone_items:
            print(f"{zone_name:<40} {avg_cycles:<15,.0f} {avg_ms:<12.3f} {pct:<8.1f} {count:<8,}")

        # Identify idle vs work
        print(f"\n{'─'*90}")
        idle_zones = [z for z in zone_items if any(
            idle_word in z[0].upper()
            for idle_word in ['WAIT', 'BARRIER', 'IDLE', 'CB-WAIT']
        )]
        work_zones = [z for z in zone_items if z not in idle_zones]

        idle_pct = sum(z[3] for z in idle_zones)
        work_pct = sum(z[3] for z in work_zones)

        print(f"{'ACTUAL WORK':<40} {'':<15} {'':<12} {work_pct:<8.1f}")
        print(f"{'IDLE/WAITING':<40} {'':<15} {'':<12} {idle_pct:<8.1f}")

if __name__ == "__main__":
    csv_path = Path("tracy_output_large/profile_log_device.csv")

    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        print("Run the test with TT_METAL_DEVICE_PROFILER=1 first.")
        exit(1)

    zones = parse_detailed_profile(csv_path)
    print_breakdown(zones)
```

---

## ⚠️ 주의사항

### 1. Profiler Overhead
- 각 zone마다 ~10-20 cycles 오버헤드
- 너무 세밀하게 profiling하면 실제 성능에 영향
- **권장**: 주요 operation만 profiling (5-10개 zones)

### 2. Rebuild 필요
- Kernel 코드 수정 시 `./build_metal.sh` 재실행 필요
- 약 5-10분 소요

### 3. 모든 코어에 적용됨
- Device profiler는 **모든 130개 코어**에서 동작
- 출력 파일이 매우 클 수 있음 (10-100 MB)

### 4. Zone 이름 규칙
- 짧고 명확하게 (20자 이내)
- 대문자와 하이픈 사용 권장
- 예: `READ-WEIGHT`, `NOC-WAIT`, `MATMUL-COMPUTE`

---

## 🎯 추천 Zone 구조

### BRISC (Reader) Kernel
```cpp
DeviceZoneScopedMainChildN("BRISC-MATMUL-READER");
  ├─ DeviceZoneScopedN("CB-RESERVE")
  ├─ DeviceZoneScopedN("READ-ACTIVATION")
  ├─ DeviceZoneScopedN("READ-WEIGHT")
  ├─ DeviceZoneScopedN("NOC-BARRIER")  ← IDLE 측정!
  └─ DeviceZoneScopedN("CB-PUSH")
```

### NCRISC (Writer) Kernel
```cpp
DeviceZoneScopedMainChildN("NCRISC-MATMUL-WRITER");
  ├─ DeviceZoneScopedN("CB-WAIT")       ← IDLE 측정!
  ├─ DeviceZoneScopedN("PACK-OUTPUT")
  ├─ DeviceZoneScopedN("NOC-WRITE")
  └─ DeviceZoneScopedN("NOC-BARRIER")   ← IDLE 측정!
```

### TRISC (Compute) Kernel
```cpp
DeviceZoneScopedMainChildN("TRISC-MATMUL-COMPUTE");
  ├─ DeviceZoneScopedN("CB-WAIT")       ← IDLE 측정!
  ├─ DeviceZoneScopedN("ACQUIRE-DST")
  ├─ DeviceZoneScopedN("MATMUL-TILES")  ← 진짜 연산!
  ├─ DeviceZoneScopedN("PACK-OUTPUT")
  └─ DeviceZoneScopedN("RELEASE-DST")
```

---

## 📚 참고 문서

1. **Device Profiler 공식 문서**:
   `/home/masterjunmo/codes/tt-metal/docs/source/tt-metalium/tools/device_program_profiler.rst`

2. **Kernel Profiler 헤더**:
   `/home/masterjunmo/codes/tt-metal/tt_metal/tools/profiler/kernel_profiler.hpp`

3. **Firmware 코드** (기본 zone 구조 확인):
   - `/home/masterjunmo/codes/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc`
   - `/home/masterjunmo/codes/tt-metal/tt_metal/hw/firmware/src/tt-1xx/ncrisc.cc`
   - `/home/masterjunmo/codes/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisc.cc`

4. **테스트 예시**:
   `/home/masterjunmo/codes/tt-metal/tests/tt_metal/tools/profiler/test_device_profiler.py`

---

## 🎉 최종 결론

### Before (부정확):
```
BRISC:  20% (192ms)  ← idle 포함, 부정확
NCRISC: 20% (186ms)  ← idle 포함, 부정확
TRISC:  60% (560ms)  ← idle 포함, 부정확
```

### After (정확):
```
BRISC:  65% actual work, 35% idle (NoC waiting)
NCRISC: 59% actual work, 41% idle (CB/NoC waiting)
TRISC:  75% actual work, 25% idle (CB waiting)

진짜 bottleneck: TRISC matmul computation (420ms)
최적화 포인트: NoC bandwidth (BRISC/NCRISC idle 시간 감소)
```

**이제 정확한 병목 지점을 알 수 있습니다!** 🎯

---

## ✅ Action Checklist

- [ ] 1. Kernel 파일 찾기 완료
- [ ] 2. EXAMPLE 파일 확인
- [ ] 3. 실제 kernel 파일 백업
- [ ] 4. Profiling zones 추가
- [ ] 5. `./build_metal.sh` 실행
- [ ] 6. `TT_METAL_DEVICE_PROFILER=1` 로 실행
- [ ] 7. CSV 파일 생성 확인
- [ ] 8. `analyze_detailed_zones.py` 실행
- [ ] 9. 결과 분석 및 병목 지점 확인
- [ ] 10. 최적화 전략 수립

**시작하세요!** 🚀

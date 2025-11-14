# 🎯 SOLUTION: Baby RISC 코어별 상세 Operation 추적 방법

## 문제 인식
- **현상**: BRISC/NCRISC/TRISC 비율이 20:20:60으로 일정 → 비정상
- **원인 가설**: 로그에 idle time까지 포함되어 있어서 실제 작업량을 반영하지 못함
- **목표**: 각 RISC 코어가 **실제로 어떤 operation을 언제 수행**하는지 시간별로 추적

---

## ✅ 해결 방법: Kernel-Level Custom Profiling

tt-metal에는 **이미 완벽한 도구가 내장**되어 있습니다!

### 1. DeviceZoneScopedN - Kernel Level Profiling

**위치**: `/home/masterjunmo/codes/tt-metal/tt_metal/tools/profiler/kernel_profiler.hpp`

**사용법**:
```cpp
#include <tools/profiler/kernel_profiler.hpp>

void MAIN {
    DeviceZoneScopedN("Weight-Loading");
    // weight loading code here

    DeviceZoneScopedN("NoC-Transfer");
    // NoC transfer code here

    DeviceZoneScopedN("Computation");
    // actual computation here
}
```

### 2. 현재 프로파일러가 측정하는 Zone들

**Firmware에 기본 탑재된 zones** (자동으로 측정됨):

#### BRISC (brisc.cc:412)
```cpp
DeviceZoneScopedMainN("BRISC-FW");      // Firmware overhead
    // 내부에서:
    DeviceZoneScopedMainChildN("BRISC-KERNEL");  // 실제 kernel 실행
```

#### NCRISC (ncrisc.cc:XXX)
```cpp
DeviceZoneScopedMainN("NCRISC-FW");     // Firmware overhead
    DeviceZoneScopedMainChildN("NCRISC-KERNEL"); // 실제 kernel 실행
```

#### TRISC (trisc.cc:XXX)
```cpp
DeviceZoneScopedMainN("TRISC-FW");      // Firmware overhead
    DeviceZoneScopedMainChildN("TRISC-KERNEL");  // 실제 kernel 실행
```

### 3. 현재 측정되는 데이터 구조

`profile_log_device.csv` 출력 예시:
```
PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, run host ID, trace id, trace id counter, zone name, type, source line, source file, meta data
0,1,2,BRISC,843,1113515784757,0,1024,,,BRISC-FW,ZONE_START,412,brisc.cc,
0,1,2,BRISC,24300,1113515785466,0,1024,,,BRISC-KERNEL,ZONE_START,64,brisck.cc,
0,1,2,BRISC,24300,1113515919138,0,1024,,,BRISC-KERNEL,ZONE_END,64,brisck.cc,
0,1,2,BRISC,843,1113515920000,0,1024,,,BRISC-FW,ZONE_END,412,brisc.cc,
```

**문제점**: `BRISC-FW`는 전체 firmware loop를 측정 → idle time 포함!

---

## 🔥 실제 해결책: Custom Zone 추가

### Step 1: ttnn.linear kernel 코드 수정

ttnn.linear 내부에서 사용하는 dataflow kernel에 custom zone 추가:

```cpp
// reader_bmm_tile_layout.cpp 같은 dataflow kernel 파일

#include <tools/profiler/kernel_profiler.hpp>

void kernel_main() {
    DeviceZoneScopedMainN("BRISC-KERNEL");

    {
        DeviceZoneScopedN("CB-RESERVE");
        // circular buffer reservation
        cb_reserve_back(cb_id_in0, num_tiles);
    }

    {
        DeviceZoneScopedN("DRAM-READ");
        // actual DRAM read
        noc_async_read(...);
    }

    {
        DeviceZoneScopedN("NOC-WAIT");
        // wait for NoC
        noc_async_read_barrier();
    }

    {
        DeviceZoneScopedN("CB-PUSH");
        // push to CB
        cb_push_back(cb_id_in0, num_tiles);
    }
}
```

### Step 2: Compute kernel도 동일하게

```cpp
// matmul kernel

#include <tools/profiler/kernel_profiler.hpp>

void kernel_main() {
    DeviceZoneScopedMainN("TRISC-KERNEL");

    {
        DeviceZoneScopedN("CB-WAIT-FRONT");
        cb_wait_front(cb_id_in0, num_tiles);
    }

    {
        DeviceZoneScopedN("MATMUL-COMPUTE");
        // actual matmul
        matmul_tiles(...);
    }

    {
        DeviceZoneScopedN("CB-POP");
        cb_pop_front(cb_id_in0, num_tiles);
    }
}
```

### Step 3: 실행 및 분석

```bash
# Device profiler 활성화
export TT_METAL_DEVICE_PROFILER=1

# 실행
python3 your_test.py

# 출력 확인
cat generated/profiler/profile_log_device.csv
```

**결과 예시**:
```
0,1,2,BRISC,xxx,1113515785466,0,1024,,,CB-RESERVE,ZONE_START,10,reader.cpp,
0,1,2,BRISC,xxx,1113515785500,0,1024,,,CB-RESERVE,ZONE_END,10,reader.cpp,
0,1,2,BRISC,xxx,1113515785501,0,1024,,,DRAM-READ,ZONE_START,15,reader.cpp,
0,1,2,BRISC,xxx,1113515900000,0,1024,,,DRAM-READ,ZONE_END,15,reader.cpp,
0,1,2,BRISC,xxx,1113515900001,0,1024,,,NOC-WAIT,ZONE_START,20,reader.cpp,
0,1,2,BRISC,xxx,1113515918000,0,1024,,,NOC-WAIT,ZONE_END,20,reader.cpp,
```

이제 **정확히 어느 operation이 얼마나 걸리는지** 볼 수 있습니다!

---

## 📊 분석 도구

### 1. Built-in Analysis Tool

```bash
cd tools/tracy
python3 process_device_log.py
```

자동으로 각 zone의 통계 생성:
- Average duration
- Min/Max duration
- Count
- Total time

### 2. Custom Python Analysis

```python
import csv
from collections import defaultdict

# Parse profile_log_device.csv
zones = defaultdict(list)

with open('profile_log_device.csv', 'r') as f:
    # Skip header lines
    f.readline()
    f.readline()

    reader = csv.reader(f)
    zone_stack = {}

    for row in reader:
        risc_type = row[3]
        cycles = int(row[5])
        zone_name = row[10]
        zone_type = row[11]  # ZONE_START or ZONE_END

        key = (risc_type, zone_name)

        if zone_type == 'ZONE_START':
            zone_stack[key] = cycles
        elif zone_type == 'ZONE_END' and key in zone_stack:
            duration = cycles - zone_stack[key]
            zones[key].append(duration)
            del zone_stack[key]

# Print statistics
for (risc, zone), durations in zones.items():
    if len(durations) == 0:
        continue
    avg = sum(durations) / len(durations)
    print(f"{risc:10} {zone:20} {avg:15.2f} cycles (count: {len(durations)})")
```

---

## 🎯 적용 예시: ttnn.linear 분석

### 찾아야 할 kernel 파일들:

```bash
# BRISC/NCRISC (dataflow) kernels
find ttnn -name "*reader*.cpp" -o -name "*writer*.cpp"

# TRISC (compute) kernels
find ttnn -name "*matmul*.cpp" -o -name "*compute*.cpp"
```

### ttnn.linear에서 사용하는 주요 kernel:

1. **Reader (BRISC)**:
   - `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_*.cpp`

2. **Writer (NCRISC)**:
   - `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/writer_bmm_*.cpp`

3. **Compute (TRISC)**:
   - `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_*.cpp`

### 수정 예시:

```bash
# 1. Reader kernel 찾기
grep -r "reader_bmm" ttnn/cpp/ttnn/operations/matmul/device/kernels/

# 2. 해당 파일 열기
vim ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout.cpp

# 3. Custom zone 추가
# (위의 예시대로)

# 4. Rebuild
./build_metal.sh

# 5. 실행 및 분석
export TT_METAL_DEVICE_PROFILER=1
python3 your_test.py
```

---

## 💡 추가 기능

### 1. Event Recording

특정 시점의 이벤트 기록:

```cpp
DeviceRecordEvent(event_id);  // 단순 타임스탬프 기록
```

### 2. Timestamped Data

데이터와 함께 타임스탬프 기록:

```cpp
DeviceTimestampedData("buffer-size", buffer_size_bytes);
```

### 3. Accumulated Time

반복되는 코드 섹션의 총 시간 누적:

```cpp
DeviceZoneScopedSumN1("inner-loop");
// 여러 번 호출되면 자동으로 합산
```

---

## 📈 예상 결과

Custom zone 추가 후 예상되는 분석:

```
BRISC Operation Breakdown:
  CB-RESERVE      : 100 cycles   (1%)
  DRAM-READ       : 5000 cycles  (50%)   ← 실제 작업
  NOC-WAIT        : 4500 cycles  (45%)   ← idle (NoC waiting)
  CB-PUSH         : 400 cycles   (4%)

NCRISC Operation Breakdown:
  CB-POP          : 200 cycles   (2%)
  NOC-WRITE       : 6000 cycles  (60%)   ← 실제 작업
  NOC-WAIT        : 3500 cycles  (35%)   ← idle
  CB-RELEASE      : 300 cycles   (3%)

TRISC Operation Breakdown:
  CB-WAIT         : 2000 cycles  (10%)   ← idle (waiting for data)
  MATMUL-COMPUTE  : 18000 cycles (90%)   ← 실제 작업
```

이제 **진짜 작업 vs idle time**을 명확히 구분할 수 있습니다!

---

## 🚀 Action Items

1. ✅ **Kernel 파일 찾기**
   ```bash
   find ttnn/cpp/ttnn/operations/matmul -name "*.cpp" | head -20
   ```

2. ✅ **DeviceZoneScopedN 추가**
   - Reader kernel: weight loading operation별로 zone 추가
   - Writer kernel: output writing operation별로 zone 추가
   - Compute kernel: computation operation별로 zone 추가

3. ✅ **Rebuild & Run**
   ```bash
   ./build_metal.sh
   export TT_METAL_DEVICE_PROFILER=1
   python3 research_codes/weight_loading_test_tracy.py
   ```

4. ✅ **분석**
   ```bash
   python3 tools/tracy/process_device_log.py
   # 또는 custom analysis script
   ```

---

## 📚 Reference

- **Kernel Profiler 문서**: `docs/source/tt-metalium/tools/device_program_profiler.rst`
- **Kernel Profiler 헤더**: `tt_metal/tools/profiler/kernel_profiler.hpp`
- **Firmware 코드**: `tt_metal/hw/firmware/src/tt-1xx/{brisc,ncrisc,trisc}.cc`
- **Example 테스트**: `tests/tt_metal/tools/profiler/test_device_profiler.py`

---

## 🎉 Summary

**현재 문제**: FW-level zone만 측정 → idle time 포함
**해결책**: Kernel-level custom zone 추가 → operation별 정확한 시간 측정

**장점**:
- ✅ Operation-level granularity
- ✅ Idle time vs actual work 구분
- ✅ NoC waiting time 정확히 측정
- ✅ 130개 코어 모두 독립적으로 추적
- ✅ Cycle-accurate measurement

**단점**:
- ⚠️ Kernel 코드 수정 필요 (rebuild)
- ⚠️ Profiler overhead (~10-20 cycles per zone)

하지만 이게 **유일하고 가장 정확한 방법**입니다!

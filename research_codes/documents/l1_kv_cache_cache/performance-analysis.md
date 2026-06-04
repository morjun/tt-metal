## Analysis

- L1 SRAM 사용 시 성능이 DRAM Baseline보다 안 나오는 이유

  - SDPA Attention 연산은 헤드당 8개 코어가 관여하며, Llama 3.1 8B 모델의 헤드 개수가 8개이므로 총 8x8개 Tensix 코어가 attention 연산을 수행하게 된다.

  - 현재의 Sharded layout 구현 방식: 각 KV캐시 티어(그룹, 서로 다른 memory config)는 서로 disjoint한 토큰 범위에 해당하는 KV캐시를 저장하고 있다. 또한 각 그룹 내에서도 KV캐시 부분을 다시 height-shard하여, 코어마다 서로 다른 토큰 범위를 할당받게 된다(KV 캐시의 height 축 = 시퀀스, width 축 = 헤드).

  - Height-sharded layout의 경우, 1개 코어가 1개 Tile-row 분량의 KV캐시 블록을 가지고 있어서 그리드 내 모든 코어의 request가 계속해서 집중되는 현상이 생겨 NoC 병목 + Head-of-Line (HoL) Blocking이 발생한다.

    - 또한, 코어당 할당 가능한 최소 용량이 1개 tile-row 분량이므로 interleaved 대비 최소 요구공간이 커서 이미 SRAM utilization이 높은 8x8 grid에 속하는 코어의 SRAM에는 아예 할당이 불가능해지는 상황이 발생한다.

  - Interleaved layout의 경우, 각 코어가 1개 타일만큼의 KV캐시를 가지고 있다. 마찬가지로 그리드 내 모든 코어의 요청이 순간적으로 1개 코어에 집중되지만 해당 hotspot 이 ~~130~~ -> 110개 코어에 걸쳐 계속해서 전환되므로 트래픽 분산 효과가 발생해 throughput이 현재의 height-sharded 레이아웃 대비 더 좋게 나온다.

- Width-sharded 나 Block-sharded layout의 경우 head-dim을 고려하지 않고 자르면 K, V벡터의 헤드차원이 쪼개져서 부분합 계산과 Collective Communication 이 필요해져 추가적인 sync overhead가 발생하므로 적합하지 않다.

  - 대안: **Width-sharded 하되, head-dim에 맞춰서 자르기** (구현 예정)
  - 효과: 한 코어에 집중되는 순간적으로 집중되는 요청 개수를 64개 -> 8개로 줄일 수 있다.
  - 예상 문제점: Llama-3.1 8B 모델의 head-dim이 128임을 고려할 때, 한 코어당 할당량이 지나치게 커질 수 있음

### L1-only Mode

- l1-only mode 사용 시, DRAM 접근이 아예 발생하지 않으므로 DRAM에서 가져오는 circular buffer 영역(`cb_k_in_dram`)이 아예 할당되지 않는다. 그만큼 SRAM 여유 공간이 늘어난다. (Interleaved config에서 l1-only mode 사용 시 약 950개 토큰 분량의 KV 캐시 저장 가능)



## Hybrid Approach (Sharded + Interleaved)

  - 현재 Interleaved config에 l1-only mode를 적용한다면, (문서 주장에 따르면) DRAM이 원래 사용했어야 될 CB가 할당되지 않아 그만큼 더 많은 공간이 남아 Bottleneck core의 여유 공간이 늘어나서, 결과적으로 전체 코어의 Utilization이 올라간다. 하지만 본질적으로 모든 코어가 동일한 용량만큼 조각을 나눠가지기 때문에 bottleneck 코어에 의해 할당가능 용량이 cap된다는 것은 변하지 않는다. 여유 공간이 더 많이 남는 코어까지 Utilization을 최대화하기 위해서 Hybrid approach를 사용하는 것은 합리적인가? 즉 interleaved하고 공간이 남는 코어들한테는 추가적으로 KV캐시를 shard 하는 것이다.

    - 다만 dynamic branching 복잡성이 더해져 실제 성능은 더욱 떨어질 수 있다.

  - **결론(아래 재검증 반영): hybrid의 유일한 가치는 "용량"이고 "latency"가 아니다.** SDPA-decode가
    compute-bound이고 L1==DRAM이므로(아래 §2), 코어 여유 공간을 더 짜내 KV를 더 많이 L1에 올려도
    decode latency는 안 줄어든다. capacity가 latency로 환산되는 영역(read>compute)은 32-layer가
    L1에 담을 수 있는 ~960 tok보다 훨씬 위에 있다. 따라서 hybrid 구현의 우선순위는 낮다(§To-do 참고).

## Profiling 결과: SDPA-decode는 compute-bound (L1==DRAM) — 재검증본 (2026-06-02)

> 이 절은 이전 버전의 측정표(특히 "L1(l1_only) @ 1,765 / 3,864 tok")를 **무효 처리**하고
> device에서 재측정한 결과로 교체한 것이다. 무효 사유와 올바른 방법론을 아래에 명시한다.

### 0. 이전 표가 틀렸던 이유 (l1_only `cur_pos` clamp)

l1_only 모드에서 reader 커널은 `cur_pos`를 `total_l1_tokens - 1`로 **클램프**한다
(`reader_decode_all.cpp:174-184`). `cur_pos`가 SDPA의 chunk 반복 범위(`k_chunk_end`)를 결정하므로,
32-layer에서 L1 용량이 ~960 tok인 상태로 "1,765 / 3,864 tok context"를 돌리면 op는 실제로 ~960 tok만
연산한다. 따라서 이전 표의 "L1(l1_only) @ 1,765 = 23.7µs"는 물리적으로 나올 수 없는 값이다
(클램프되면 ~960-tok latency ≈ 11µs여야 함). 그 수치를 뒷받침하는 profiler artifact도 디스크에
남아있지 않다(유일하게 남은 report는 4행짜리이고 SDPA 행이 없음) → **무효**.

추가로 중요한 점: **non-l1_only(DRAM fallback)로 바꿔도 "L1 vs DRAM" 비교가 되지 않는다.** prefill된
context의 KV는 DRAM에 있고 L1 ring은 decode-write로만 채워지므로, prefill body는 항상 DRAM에서 읽힌다.
어떤 context의 KV를 통째로 L1에 올리는 유일한 방법은 **l1_only + full-prefill seed + (L1 용량 ≥ context)**다.
~960 cap은 32-layer **배포 제약**일 뿐 **op 제약이 아니다.** `--num_layers 2`로 KV를 작게 만들면
1,792 tok까지 full-resident가 가능하므로, 이 조건에서 측정한 것이 "진짜 L1 vs DRAM" op latency다.

### 1. 방법론 (재현 가능)

- 빌드: 현재 `build_Release` (`ENABLE_TRACY=ON`). 재빌드 불필요 — 순수 config/Python 변경.
- profiler: `python -m tracy -r -p -v -m pytest ...`. 지표는 op CSV의
  `DEVICE KERNEL DURATION PER CORE AVG [ns]` (집계 `DEVICE KERNEL DURATION` 컬럼은 timestamp
  overflow로 손상되어 per-core 값을 사용).
- 모델: **2-layer**. op latency는 layer 수와 무관하며, KV를 L1에 full-resident시키기 위한 선택.
- trace는 코드에서 강제 off(`simple_text_demo.py:931`, `enable_trace = False`)되어 per-op profiling 가능.
- `--paged_attention 0`: L1 KV 경로는 `not page_table` 게이트라 paging을 꺼야 동작.
- `--instruct 0`: instruct 클리핑의 off-by-one assert를 피하려고 non-instruct 사용(토큰 그대로 슬라이스).
- **context 제어**: 토큰 수가 고정된 prompt 파일(512/1024/1792 tok; `prompt_ctxN.json`)을 주입하고
  `--max_seq_len 2048` 고정. prefill seq len = 실제 토큰 수, decode `cur_pos`는 거기서 시작.
  (`--max_seq_len`은 op latency에 영향 없음; SDPA는 `cur_pos`에만 의존.)
- **l1_only full-resident(클램프 없음 보장)**: `--l1_kv_mode interleaved --l1_kv_only_mode
  --l1_kv_window_size (ctx+128)` → tier 용량 > context. 로그의 "Total L1 KV tokens"가 context보다
  큰지, 그리고 L1/DRAM 생성 토큰이 byte-identical인지로 확인.
- 샘플 수: SDPA-decode 행 = 2 layers × 8 decode step = 16개/config, 평균.

드라이버/추출 스크립트, prompt, 원시 CSV, run.log는 모두
`research_codes/documents/l1_kv_cache_cache/reprofile/`에 보존(label별 디렉터리).

명령 (한 config):
```bash
cd /home/masterjunmo/codes/tt-metal
# run_one.sh LABEL MODE CTX WINDOW ONLY(0|1)   (max_seq_len 2048 고정, num_layers 2)
bash research_codes/documents/l1_kv_cache_cache/reprofile/run_one.sh dram_1024     dram        1024 0    0
bash research_codes/documents/l1_kv_cache_cache/reprofile/run_one.sh l1_1024       interleaved 1024 1152 1   # full-resident
bash research_codes/documents/l1_kv_cache_cache/reprofile/run_one.sh hybrid_1024   interleaved 1024 256  0   # DRAM fallback
```
run_one.sh가 내부에서 실행하는 명령(직접 재현용):
```bash
export PATH="$PWD/python_env/bin:$PATH"
TT_METAL_HOME="$PWD" PYTHONPATH="$PWD/tools" python -m tracy -r -p -v \
  -m pytest "models/tt_transformers/demo/simple_text_demo.py::test_demo_text[blackhole-mesh_device0-device_params0-performance-batch-1]" \
  --input_prompts research_codes/documents/l1_kv_cache_cache/reprofile/prompt_ctx1024.json \
  --max_seq_len 2048 --max_generated_tokens 8 --num_layers 2 --instruct 0 --paged_attention 0 \
  --l1_kv_mode interleaved --l1_kv_window_size 1152 --l1_kv_only_mode
# 결과 CSV: generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
# 추출:    python research_codes/documents/l1_kv_cache_cache/reprofile/extract.py <csv>
```
prompt 파일 생성(토큰 수 고정):
```bash
# reprofile/long_local_prompt.json(로컬, 네트워크 불필요)을 모델 토크나이저로 인코딩 후 N 토큰으로 슬라이스
# -> prompt_ctx512.json / prompt_ctx1024.json / prompt_ctx1792.json (재인코딩 시 정확히 512/1024/1792)
```

### 2. 결과 (SDPA-decode op, per-core avg, n=16/config)

| context | DRAM | L1 full-resident (l1_only) | Hybrid (non-l1_only, partial L1) |
|---|---|---|---|
| 512   | 9.86 µs  | 9.88 µs  | — |
| 1,024 | 18.59 µs | 18.90 µs | 18.78 µs (window 288, ~770 tok는 DRAM에서) |
| 1,792 | 24.72 µs | 24.61 µs | 25.01 µs (window 544, ~1,250 tok는 DRAM에서) |

- L1 full-resident와 DRAM의 **생성 토큰이 byte-identical**(같은 prompt, temperature 0 argmax)
  → L1 경로 수치 정확성 확인, 클램프 없음(전체 context 연산).
  - **정정(human-readable 여부)**: 이 2-layer 출력은 **사람이 읽을 수 있는 문장이 아니다**(32층 중 2층만
    써서 logit이 무의미 → 프롬프트를 잠깐 따라가다 토큰 garbage로 붕괴: 예 "...hiding␦ecs筒INALeck...").
    byte-identical은 **"같은 op·같은 수치, KV 소스만 다름 → 동일 출력"이라는 read-path 등가성**을 증명하는
    것이지 coherence 주장이 아니다. 사람이 읽을 수 있는 정상 출력의 동등성은 32-layer end-to-end run
    (`run_960_l1.log`, "As a digital AI...")에서 별도로 확인했다(아래 절).
- L1 수치가 context에 따라 증가(9.88→18.90→24.61)하며 DRAM(9.86→18.59→24.72)을 추종
  → full context를 연산한다는 증거(클램프됐다면 고정값으로 평평해짐).
- Hybrid(부분 L1 + DRAM fallback)도 동일 context에서 DRAM/L1과 ±2% 이내.
- DRAM 컨텍스트 스케일링 ≈ `3.9 + 0.0116 × tokens` µs.

### 3. 결론 (유효 근거 기반 재확인)

- **SDPA-decode op latency는 KV 저장 위치(전부 DRAM / 전부 L1 / 분할)와 무관하다 (±2% 이내).**
  KV read가 flash-attention 연산 뒤로 double-buffered prefetch되어 완전히 가려지므로(critical path 밖)
  op은 **compute-bound**다.
- 따라서 **L1 KV는 decode latency에서 DRAM을 이길 수 없다** (L1에 들어가는 context 범위에서).
- L1이 이길 수 있는 유일한 영역은 read time > compute time이 되는 매우 긴 context인데, 그 영역에서는
  KV가 애초에 L1에 안 들어간다 → 관건은 layout이 아니라 **capacity**(32-layer ~960 tok ceiling).
- end-to-end에서 본 l1_only의 ~3.5% steady-state 우위는 SDPA가 아니라 **DRAM KV write 생략**에서 온다(아래 절).

### 4. 측정 한계 / 후속

- 2-layer에서 full-resident 상한 ≈ 2,400 tok(코어당 interleaved KV ≈ 0.25 KiB/tok/layer)이라
  1,792까지만 측정. 그 이상은 L1 미적재 = capacity 벽 그 자체이므로 layout 비교 대상이 아님.
- NoC event profiler(`--collect-noc-traces`)는 현재 Blackhole 빌드에서 decode 중
  `TT_FATAL kernel.cpp:293`로 crash → read-busy margin 정량화 미완. 단 "read가 compute 밑에 숨는다"는
  결론은 §2의 위치-불변성으로 이미 직접 증명됨(정확한 여유 margin만 미측정).
- FPU vs SFPU 분리는 built-in per-RISC 카운터가 timestamp overflow로 손상 → compute kernel
  (`sdpa_flash_decode.cpp`/`compute_common.hpp`)의 matmul/softmax 구간을 `DeviceZoneScopedN`으로
  감싸 재컴파일 후 재측정 필요(미실행).

## SFPU vs FPU 분해 + read/compute overlap (device zone profiling, 2026-06-04)

§3에서 "compute-bound"는 확인됐고, 남은 질문 두 개를 device-zone(`DeviceZoneScopedN`)으로 직접 측정했다:
(A) attention compute 중 softmax(SFPU)가 병목인가, matmul(FPU)인가? (B) KV read(NCRISC)가 compute(TRISC)
밑에 정확히 얼마나 가려지나?

방법: SDPA decode 커널(`sdpa_flash_decode.cpp`)에 per-chunk zone 삽입 — FPU `QK_MM`/`PV_MM`,
SFPU `SM_NORM`(row-max+exp+row-sum)/`SM_RESCALE`(online-softmax rescale), envelope `CMP_CHUNK`.
reader(`dataflow_common.hpp::read_kv_mask_chunks`)에 `RD_K`/`RD_V`. `SDPA_PROFILE_ZONES` 가드.
DRAM, ctx 1792, 2-layer, 8 decode step. `profile_log_device.csv`를 `analyze_zones.py`로 집계
(per-core avg, 같은 코어의 NCRISC·TRISC는 동일 cycle base). DROPPED_ZONES 없음. 재현 명령은 아래 "재현".

### (A) FPU(matmul)가 지배적이다 — softmax는 병목이 **아니다**

compute는 3 TRISC로 파이프라인(UNPACK/MATH/PACK)되며 zone 시간은 스레드별 busy. 칩-wall은 가장 느린
스레드(여기선 TRISC_1/2 ≈ 8.45 µs/chunk)가 결정. per-chunk avg (µs):

| RISC | QK_MM(FPU) | PV_MM(FPU) | SM_NORM(SFPU) | SM_RESCALE(SFPU) | CMP_CHUNK | FPU% / SFPU% |
|---|---|---|---|---|---|---|
| TRISC_0 | 2.06 | 0.49 | 1.54 | 2.98 | 6.31 | 47% / 53% |
| TRISC_1 | 4.32 | 0.42 | 1.72 | 2.90 | 8.45 | 61% / 39% |
| TRISC_2 | 4.45 | 0.55 | 1.42 | 2.78 | 8.47 | 65% / 35% |

- wall을 결정하는 TRISC_1/2에서 **FPU matmul 61-65%, SFPU softmax 35-39%**. 단일 최대 zone은
  **QK^T matmul(`QK_MM`)** (math thread 4.3-4.4 µs). softmax는 2차 비용이다.
- 따라서 **softmax(SFPU)는 병목이 아니다. attention decode compute는 matmul-bound(특히 QK^T).**
- `SM_RESCALE`(online-softmax rescale)는 호출당 2.8-3.0 µs로 비싸지만 multi-chunk에서만(896/1920 호출)
  발생. `SM_NORM`(exp 포함)은 1.4-1.7 µs로 저렴.
- 함의: `EXP_APPROX_MODE`/math-approx로 softmax를 줄여도 상한은 attention compute의 ~35-40%이고,
  attention 자체가 decode step device 시간의 작은 비중이라 end-to-end 이득은 제한적. matmul이 바닥.

### (B) KV read는 compute 밑에 100% 가려진다

64개 attention 코어 전부에서 NCRISC read(`RD_K`+`RD_V`) 구간이 같은 코어의 TRISC compute
(`CMP_CHUNK` 3-TRISC union) 구간 **안에 100% 포함**된다(read-hidden fraction mean/min/max = 100.0%).
합계: read 8,506 µs vs compute(union) 16,357 µs → **margin 7,851 µs (compute가 read의 ~1.9배)**.
이는 §2의 L1==DRAM(위치-불변) 결론을 커널 타임라인 수준에서 직접 재확인한 것이다: read ⊆ compute, 여유 큼.

> **read 소스 명시: 이 측정의 read는 전부 DRAM read다** (`--l1_kv_mode dram`이므로 reader가
> `read_kv_mask_chunks`(DRAM 경로) 실행). 즉 "DRAM read조차 compute 밑에 완전히 숨는다"는 뜻 —
> 더 빠른 L1 read로 바꿔도 op latency가 안 줄어드는 이유의 직접 증거. (L1 read 타임라인은 32-layer
> l1_only 프로파일에서 별도 측정 예정 — §To-do.)

타임라인 간트(한 SDPA-decode op, 한 attention 코어, NCRISC DRAM read vs 3×TRISC compute):
`reprofile/zones_dram_1792/overlap_gantt.png` (`plot_overlap.py`로 생성). NCRISC read 밴드가
compute span 안에서 끝나고 compute의 SM_RESCALE 꼬리가 read 뒤로 더 이어지는 것이 보인다(read 숨김 + 여유).

### 재현

커널을 `SDPA_PROFILE_ZONES`로 계측(기본은 off — `sdpa_flash_decode.cpp`/`dataflow_common.hpp`의
`#define SDPA_PROFILE_ZONES 1` 주석 해제). build-key는 소스를 해시하지 않으므로 토글 후 반드시
`rm -rf ~/.cache/tt-metal-cache` (스크립트의 `wipe` 인자). 그 다음:
```bash
bash research_codes/documents/l1_kv_cache_cache/reprofile/run_zones.sh wipe   # 첫 회/커널 수정 후
bash research_codes/documents/l1_kv_cache_cache/reprofile/run_zones.sh        # 이후
# 분석: analyze_zones.py <report>/profile_log_device.csv (run_zones.sh가 자동 호출)
```
주의: tt-metal run을 강제 종료(kill)하면 device가 wedge되어 다음 run이 prefill에서 hang한다.
복구: `python_env/bin/tt-smi -r`. 원시 CSV/로그는 `reprofile/zones_dram_1792/`에 보존.

## L1_only 배포 config 확인 + L1 read 타임라인 + warm-up=compile 확정 (2026-06-04)

§A/§B는 DRAM 경로였다. 여기서는 **l1_only(L1 read)** 경로를 같은 방법으로 측정하고, 같은 context에서
DRAM과 직접 비교한다.

> **왜 2-layer인가 (32-layer zone 불가)**: 커스텀 zone marker는 코어당 marker 버퍼(250개)에 op 호출을
> 가로질러 누적된다 — 2-layer(SDPA 16회/step-set)는 ~192개로 들어가지만 **32-layer(256회)는 오버플로우 →
> start/end marker 불균형 → profiler post-proc abort**(`profiler.cpp:1575`)로 device CSV가 안 나온다(실측 확인).
> SDPA op·reader·compute는 **layer 수와 무관하게 동일**하므로(§2) per-op zone 결과는 2-layer가 32-layer를
> 그대로 대표한다. 32-layer 고유 지표(end-to-end tok/s, warm-up)는 zone 없이 측정. 또한 context는 큰 2의 거듭제곱
> 약수를 갖는 값이어야 한다(ctx 900 = 2²·225 → k_chunk 4 → 225 chunk → marker 폭주로 abort; **ctx 896 = 2⁷·7
> → k_chunk 128 → 7 chunk**로 해결). l1_only window 960이라 ctx 896은 full-resident(클램프 없음).

### 같은 context(896) DRAM vs L1 직접 비교 (2-layer, zones)

| 지표 | DRAM | L1_only |
|---|---|---|
| SDPA-decode op latency (per-core avg) | 17.23 µs | 17.33 µs (+0.6%) |
| FPU(matmul) / SFPU(softmax), wall TRISC_1/2 | 80–84% / 16–20% | 81–84% / 16–19% |
| K+V read per chunk | **4.16 µs** (RD_K 2.10 + RD_V 2.06, DRAM reader) | **4.72 µs** (RD_CHUNK, n-tier L1 reader) |
| read-hidden fraction (read ⊆ compute) | 100% (64/64 코어) | 100% (64/64 코어) |
| warm-up penalty | 없음 (iter1 = 31 ms) | iter1 991 ms + iter2 2259 ms |

- **op latency: L1 == DRAM(+0.6%)** — §2(2-layer, clean)·end-to-end와 일치. 배포 config(l1_only)에서도 compute-bound 재확인.
- **FPU/SFPU 분해는 L1·DRAM 동일** (matmul ~80%, softmax ~20%). softmax 병목 아님 재확인. (ctx 896은 코어당
  대개 1 chunk라 `SM_RESCALE`(multi-chunk 전용)이 없어 §A의 1792보다 FPU 비중이 더 높게 보인다 — 결론 동일.)
- **반직관 포인트: L1 read가 DRAM read보다 오히려 ~13% 느리다**(4.72 vs 4.16 µs/chunk). L1 메모리가 더 빨라도
  n-tier L1 reader의 **소프트웨어 오버헤드**(`find_tier` tier-dispatch, ring modular remap, goto ladder)가 단순
  DRAM reader보다 커서, read time 자체가 줄지 않는다. 그런데도 **둘 다 compute 밑에 100% 숨으므로 op latency는
  동일**. → "read는 lever가 아니다"를 한 번 더 못박는다(더 빠른 메모리는 물론, 더 느린 reader여도 latency 불변).

### read 타임라인(간트)
- DRAM read: `reprofile/zones_dram_1792/overlap_gantt.png`
- **L1 read**: `reprofile/zones2_l1only_896/overlap_gantt_l1.png` — NCRISC L1 read 밴드가 TRISC compute 안에서
  끝나고 PV_MM 꼬리가 read 뒤로 더 간다(read 숨김 + 여유). 생성: `plot_overlap.py <csv> <out.png>`.

### warm-up = compile 확정 (item 2)
2-layer l1_only run이 결정적: tier alloc(2개)·seed(2층)는 **합 ~0.1s**(16:55:59.6–16:56:00.6)인데도
**iter1 991 ms + iter2 2259 ms (~3.25s)**가 나온다. alloc+seed로 설명 불가 → 나머지는 **L1-경로 program의
first-use JIT 컴파일**(n-tier SDPA reader/compute + L1 ring write). 같은 ctx에서 **DRAM run은 warm-up이 전혀 없다**
(iter1부터 31 ms). 32-layer에선 여기에 alloc+seed(32× = ~3s)가 더해질 뿐, compile 성분(~2–3s)은 layer-무관.
→ 사용자 의심대로 warm-up은 순수 alloc+copy가 아니라 **compile 포함**. (해소책: program을 미리 워밍업하거나
persistent cache 재사용으로 first-use compile을 prefill 단계로 흡수.)

## End-to-end 실측 (32-layer 전체 모델, 비프로파일 run)

`simple_text_demo.py` batch-1 node, 32 layers 전체, 145-tok context, 200 decode tokens,
tracy 없이(순수 wall-clock, DEBUG 로깅) 측정. 로그: `run_dram_debug.log`(DRAM baseline,
DEBUG 재측정), `run_960_l1.log`(`--l1_kv_mode interleaved --l1_kv_only_mode
--l1_kv_window_size 960` → 실제 할당 992 tokens, full-prefill seed). 둘 다 같은 config.

| metric | DRAM baseline | L1 interleaved l1_only, 992 tok |
|---|---|---|
| TTFT (prefill→first token) | 59.91 ms | 58.15 ms |
| decode 프로그램 compile (1회성, iter 0) | 33.52 s | 33.48 s |
| warm-up iter 1 / 2 | 91 / 90 ms (패널티 없음) | 3165 / 2173 ms (L1 tier alloc+seed) |
| **steady-state decode (iter 3–199)** | 90.5 ms, **11.05 tok/s** (min 87 / max 99) | 87.4 ms, **11.44 tok/s** (min 85 / max 92) |
| avg decode (compile iter 제외, demo 지표) | 90.5 ms, 11.05 tok/s | 113.38 ms, 8.82 tok/s |

해석:
- **steady-state: L1 l1_only 11.44 tok/s vs DRAM 11.05 tok/s → L1이 ~3.5% 근소하게 빠르다.**
  이 차이는 SDPA가 아니다(§2에서 op latency는 동일). l1_only가 매 step **DRAM KV write를
  생략**(paged_update_cache가 DRAM 대신 L1로 write)하고 DRAM read도 안 하기 때문이며, write/traffic
  쪽의 작은 이득이다. 크기가 작아(~3ms/iter) noise와 경계 수준이지만 방향은 일관된다.
  → "L1은 DRAM을 못 이긴다"는 **SDPA read 경로**에 대한 결론이고, write 생략으로 인한 미세한 edge는 별개.
- DRAM은 **warm-up 패널티가 없다**(iter 1–2가 이미 steady). L1은 iter 1–2가 느려(3165 / 2173 ms,
  합 ~5.3s) 200-token 평균을 8.82 tok/s로 끌어내린다.
  - **warm-up 분해 정정(`run_960_l1.log` 타임스탬프 분석)**: 이전 "iter1–2 = 단순 alloc+seed"는 부정확.
    iter 0(33.5s) = decode program compile. **iter 1(3165ms)** = post-compile L1 tier alloc(32층, ~1s)
    + ~0.8s gap + full-prefill seed(32층 × ~41ms ≈ 1.3s) → 합 ~3.1s로 iter 1을 정확히 설명(seed는 iter 1
    안에서 끝남, 마지막 seed 로그 21:48:11.98). **iter 2(2173ms)는 alloc/seed로 설명 안 됨**(둘 다 iter 1에
    완료) → **L1 경로 전용 program의 first-use JIT 컴파일**(n-tier SDPA reader/compute + L1 ring write)로
    추정. iter 0 컴파일은 tier 할당 *이전* graph라 L1-경로 program은 첫 실제 decode(iter 1–2)에서 새로 컴파일됨.
    즉 사용자 의심대로 **warm-up에 compile이 포함**돼 있다(순수 alloc+copy 아님). **확정**: 위 "L1_only 배포
    config 확인" 절의 2-layer l1_only run에서 alloc+seed가 ~0.1s에 불과한데도 iter1+2가 ~3.25s 걸려 compile
    성분이 분리 증명됨.
  즉 L1 l1_only는 **짧은 생성에선 net 손해(8.82 vs 11.05), steady/긴 생성에선 근소 우위(11.44 vs 11.05)**.
  교차점은 warm-up ~5.3s를 step당 ~3ms 이득으로 회수 → 대략 1700+ tokens 이후.
- 992 tokens = 32-layer에서 달성된 L1 window(앞서 언급한 ~900–1000 ceiling 부근).

## To-do (갱신)

  - ~~memory read time DRAM/L1 명시 + 간트~~ **[완료 2026-06-04]** read 소스를 DRAM/L1로 라벨링(§"L1_only 배포
    config 확인", §B). 간트 2종: `reprofile/zones_dram_1792/overlap_gantt.png`(DRAM),
    `reprofile/zones2_l1only_896/overlap_gantt_l1.png`(L1). DRAM read 4.16 µs vs L1 read 4.72 µs/chunk, 둘 다 100% hidden.

  - ~~warm-up >2000ms 원인~~ **[완료 2026-06-04]** compile 포함 확정: 2-layer l1_only에서 alloc+seed ~0.1s인데도
    iter1+2 ~3.25s → 나머지는 L1-경로 program first-use JIT 컴파일. DRAM은 warm-up 없음. (§"warm-up = compile 확정")

  - ~~ctx 960(+32) l1_only 32-layer 프로파일~~ **[완료 2026-06-04, 단 caveat]** **32-layer zone은 불가**(marker
    버퍼가 op 호출 가로질러 누적 → 256회에서 오버플로우 → profiler abort). per-op zone은 layer-무관이라 **2-layer
    l1_only(ctx 896)로 대표 측정**: op latency L1==DRAM(17.33 vs 17.23 µs), FPU/SFPU 동일, L1 read 100% hidden.
    ctx는 896 사용(960은 2의 거듭제곱 약수가 작아 OK지만 900류는 chunk 폭주 주의). 32-layer 고유 지표(tok/s, warm-up)는
    end-to-end run으로 커버. (§"L1_only 배포 config 확인")

  - ~~SDPA profiling output human-readable?~~ **[완료 2026-06-04]** 2-layer 출력은 **human-readable 아님**(32층 중
    2층 → garbage). byte-identical은 read-path 수치 등가성 증명이지 coherence 주장 아님. coherent 출력 등가성은 32-layer
    end-to-end(run_960_l1)에서 별도 확인. (§2 정정)

  - **"Head sharding" 구현** — 단, 목적은 "DRAM을 이기는 것"이 아니라 sharded path를
    interleaved/DRAM **parity로 복귀**시켜 그 용량을 활용 가능하게 만드는 것. latency win은 기대 불가.
  - 진짜 win이 필요하면 **capacity 쪽**: KV quantization(int8/int4) 또는 multi-chip로 long-context를
    L1-resident로 만들어, read가 병목이 되는 영역(read time > compute time)을 L1으로 끌어오는 것.
  - (선택) NoC event profiler(`--collect-noc-traces`)로 read가 compute 밑에 얼마나 여유 있게
    숨는지(노출까지의 margin)를 정량화 — 위 결론에는 불필요하지만 crossover context 추정에 유용.

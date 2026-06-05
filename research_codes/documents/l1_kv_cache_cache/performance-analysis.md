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

  - **결론(재검증 반영): hybrid의 유일한 가치는 "용량(feasibility)"이고 "latency"가 아니다.** SDPA-decode가
    compute-bound이고 L1==DRAM이며(§2), compute·read가 둘 다 context에 선형이라 **read는 context와 무관하게 영구히
    숨는다**(§3 정정). 따라서 KV를 더 많이 L1에 올려도 decode latency는 안 줄고, "긴 context면 read>compute가 된다"는
    crossover는 context로는 안 온다. 따라서 hybrid 구현의 우선순위는 낮다(§To-do 참고).

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
- 따라서 **L1 KV는 decode latency에서 DRAM을 이길 수 없다**, 그리고 이는 **context 길이와 무관하게 영구적**이다.
- **정정 (scaling 논증; 이전 "매우 긴 context면 read>compute" 주장은 틀림)**: decode attention의 compute와 read는
  **둘 다 context N에 선형(O(N·d))**이라 비율이 N-불변이다(측정상 compute/read margin ~1.9–2.2×가 ctx 896·1792에서 일정).
  reader가 chunk 단위 double-buffer라 숨김 여부는 **per-chunk read vs per-chunk compute**가 결정 — 둘 다 chunk당 고정,
  N과 무관. context를 늘리면 "이미 숨은" chunk 반복만 늘 뿐 → read는 **영구히 숨는다**. DRAM 대역폭 포화도 context로는
  안 생긴다(단위시간당 바이트 = (bytes∝N)/(step time∝N) = N-불변).
- **read가 노출되는(= L1이 latency로 이길 수 있는) 유일한 조건은 context가 아니라 per-chunk compute:read 균형이
  뒤집힐 때**다: (1) decode matmul의 **Sq tile padding**(쿼리 1행을 32행 타일로 패딩 → effective compute 최대 ~32× 부풀림)
  제거 시 compute 급감 → intensity ~1 FLOP/byte로 read 지배 → L1 이득; (2) sharded congested layout(8-on-1 직렬화)
  로 read 노출(단 parity 회복이지 win 아님); (3) LM head/MLP의 DRAM 대역폭 경쟁으로 KV read starvation.
  (KV quant은 read를 더 싸게 만들어 **더 숨김** — capacity용이지 read-latency용 아님.)
- 즉 관건은 layout도 "긴 context"도 아니다. **capacity(multi-chip/quant)는 긴 context를 on-chip으로 돌리는
  feasibility lever**이고, read를 병목으로 만들어 latency를 이기게 하는 lever가 **아니다**.
- **[정정 2026-06-05] l1_only는 end-to-end에서도 DRAM을 못 이긴다**: "DRAM write 생략 ~3.5% edge"는 **철회**.
  write는 스킵되는 게 아니라 동일 비용으로 L1에 재배치된다(`PagedUpdateCache` DRAM·l1_only 동일 32회 × ~9.5 µs;
  op-overhead-bound). 실측(200-token)에선 오히려 **DRAM이 빠르다**(11.74 vs 11.24 tok/s) — n-tier reader 오버헤드 때문.
  자세한 건 아래 "End-to-end 실측" 절 해석 참고.

### 4. 측정 한계 / 후속

- **full-resident 상한은 layer 수에 반비례(∝ 1/layers)** — 모든 layer의 KV가 동시에 L1에 상주해야 하고,
  decode에서 모델 자체의 L1 사용(CB)은 layer 수와 거의 무관하므로 KV용 예산은 2-layer·32-layer가 비슷.
  per-core KV footprint = (K+V × n_kv_heads × head_dim × 2B) / 110 banks = (2×8×128×2)/110 ≈ **0.036 KiB/tok/layer/core**.
  32-layer × 960 tok × 0.036 ≈ **1.09 MB/core** → 측정된 ~960 ceiling과 일치(가용 ~1.1 MB).
  따라서 **2-layer 상한 ≈ 1.09 MB / (2 × 0.036 KiB) ≈ 15,000 tok = 16 × 960**(capacity-scaling 추정; 실측은 1,792까지만).
  - **정정**: 이전 "2-layer ≈ 2,400 tok / 0.25 KiB/tok/layer"는 **틀림**. 0.25는 alloc 로그의 "248 KiB/core"
    (`attention.py:561` = `tile_rows×tile_size×head_dim×2` = 한 head의 전체-token 슬랩, **110 bank로 나누지도
    n_kv_heads·K+V를 반영하지도 않음**)에서 잘못 끌어온 값이다 → 실제 per-core는 ~7× 작다. (그 로그 라인은 interleaved
    tier에 대해 "KiB/core" 라벨이 오해의 소지가 있음 — minor logging bug.)
  - per-op zone 결과는 layer-무관(§"왜 2-layer인가")이라 이 정정은 §2·§3 결론을 바꾸지 않는다.

- **장거리 context 실측 (1-layer, 2026-06-05): read는 24k까지도 영구 숨김 (compute-bound 유지)**.
  - **L1 full-resident는 ctx ≤ 4096에 구조적으로 묶인다**: ctx>4096은 chunked prefill 필요 → paged_attention(page_table)
    요구 → L1 KV 경로는 `not page_table` 게이트라 동시 불가(`generator.py:343` "page_table must be provided for chunked
    prefill"). 즉 L1 full-resident 상한은 **L1 capacity(~15k/~30k)가 아니라 prefill 경로(4096 단일 chunk)** 가 정한다.
    → L1-vs-DRAM 직접 비교는 **4096까지**(1-layer 4096: DRAM 42.31 vs L1 38.83 µs, n=8, noise 수준 + 2-layer 512/1024/1792 parity).
  - read 숨김을 더 긴 context에서 확인하려고 **DRAM-only(paged) 1-layer SDPA-decode latency 곡선** 측정:

  | ctx | 4,096 | 8,192 | 16,384 | 24,576 |
  |---|---|---|---|---|
  | SDPA-decode avg (µs) | 41.3 | 67.0 | 117.1 | 165.4 |

  step별 slope ≈ 0.0063 / 0.0061 / 0.0059 µs/tok로 **일정(미세 감소)** → **선형, supra-linear bend 없음**
  (≈ `16.5 + 0.0061 × tokens`). 즉 **24,576 tok까지 op는 compute-bound이고 read는 숨은 채**(노출 시작 안 함) →
  §3 "read 영구 숨김 / context crossover 없음"을 24k까지 실측 확인. slope가 오히려 줄어드는 것(sub-linear)은
  read 노출의 반대(고정 per-chunk 오버헤드의 amortization). (이건 paged DRAM op·1-layer라 절대값/slope는 §2 비-paged와
  다름 — 핵심은 곡선이 선형이라는 점. L1은 4096 초과를 못 담아 24k에서 L1==DRAM은 미확인.)
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
| whole-chunk read RD_CHUNK (K+mask+V, **공정 비교**) | **4.285 µs** | **4.687 µs** (L1 ~9% 느림) |
| final K barrier = **memory-wait** (RD_KBAR) | **0.029 µs (29 ns)** | **0.031 µs (31 ns)** |
| read-hidden fraction (read ⊆ compute) | 100% (64/64 코어) | 100% (64/64 코어) |
| warm-up penalty | 없음 (iter1 = 31 ms) | iter1 991 ms + iter2 2259 ms |

- **op latency: L1 == DRAM(+0.6%)** — §2(2-layer, clean)·end-to-end와 일치. 배포 config(l1_only)에서도 compute-bound 재확인.
- **FPU/SFPU 분해는 L1·DRAM 동일** (matmul ~80%, softmax ~20%). softmax 병목 아님 재확인. (ctx 896은 코어당
  대개 1 chunk라 `SM_RESCALE`(multi-chunk 전용)이 없어 §A의 1792보다 FPU 비중이 더 높게 보인다 — 결론 동일.)
- **"L1 read가 더 빨라야 하지 않나?"에 대한 정밀 답 (이전 "13%/tier-dispatch" 주장 정정)**:
  read는 **memory-bound가 아니라 NCRISC issue-bound**다. 최종 barrier(=실제 메모리 전송 대기)는 L1·DRAM 모두
  **~30 ns로 무시 가능**(RD_KBAR) — 즉 tile read들이 issue 루프와 완전히 overlap되어 barrier 시점엔 이미 다 도착해 있다.
  따라서 **L1 메모리가 더 빨라도 critical path에 줄일 memory-wait 자체가 없다.** read time(RD_CHUNK ~4.3–4.7 µs)은
  거의 전부 **NCRISC가 tile당 주소계산 + noc 명령을 issue하는 비용**이고, n-tier L1 reader가 이 비용이 더 크다
  (tile당 런타임 **modulo `% ring_tile_count`**(SW divide) + 주소 **곱셈 2회** + `find_tier` 분기 사다리). 이는
  **단일 tier에서도 발생**(tiering 무관 — 사용자 지적이 옳음, interleaved=1 tier 확인). 그래서 L1 reader가 **공정 비교로도
  ~9% 더 느리다**(4.687 vs 4.285 µs). 단 **둘 다 compute 밑 100% hidden이라 op latency는 불변**.
- **SW 병목 지점 / 최적화 타깃 (필요시; 단 hidden이라 latency 이득 0)**: (1) tile당 불변식
  `l1_kv_head_base * tier_size_tiles[0] * DHt`를 chunk 밖으로 hoist, (2) 연속 fresh window는 affine이므로 tile당
  modulo를 running-counter wrap-subtract로 제거, (3) full-resident no-wrap(l1_only 일반 케이스) 특수화로 ring 로직 skip.
  read가 노출되는 영역(매우 긴 context, 또는 sharded처럼 congested layout)에서만 의미. decode latency lever는 여전히
  read가 아니라 **capacity**.

### pure memory read latency probe (RD_LAT, 단일-tile isolated round-trip, 2026-06-05)

RD_KBAR(~30 ns)는 메모리 시간이 아니라 "이미 도착한 뒤의 barrier tail"이다(read가 issue 루프와 완전 overlap →
barrier 시점엔 데이터가 이미 도착). 순수 메모리 read 시간을 보려면 overlap을 없애야 하므로, 각 reader에 **tile 1개
read + 즉시 barrier**(zone `RD_LAT`) 프로브를 chunk마다 1회 삽입했다(throwaway — 실제 read 루프가 덮어씀).
DRAM reader는 `k_reader`(DRAM), n-tier reader는 `l1_k0_rd`(L1 tier)에서 읽는다. (기존 표준 microbench
`6_dram_offchip`/`old/noc/test_noc_read_*_l1`는 Blackhole 현행 빌드에서 stale로 실행 불가 — kernel API drift,
dispatch-core 충돌, `kernel.cpp:293` binary-not-found — 라 in-situ 프로브로 대체.)

| | DRAM | L1 |
|---|---|---|
| RD_LAT (1-tile read+barrier) | 0.412 µs | 0.336 µs |
| RD_KBAR (bare barrier 기준) | 0.033 µs | 0.034 µs |
| net round-trip (RD_LAT − barrier) | ~380 ns | ~302 ns |

- **L1이 per-access로 더 빠르다(사용자 premise 확인): round-trip ~76 ns(~20%) 낮음.** issue·barrier 고정비가
  delta에서 상쇄되므로 76 ns ≈ 순수 메모리 endpoint 접근시간 차.
- **격차가 작은 이유**: 단일-tile latency는 **NoC traversal 왕복(~300 ns, request→endpoint→response)**이 지배하고,
  메모리 array 접근시간 차(SRAM vs GDDR6)는 그 위 ~76 ns뿐. NoC 왕복은 L1·DRAM 공통.
- 이건 **latency**(직렬 1-access) 측정이지 bandwidth가 아니다. L1의 큰 이점은 bandwidth(병렬)인데, SDPA read는
  애초에 **NCRISC issue-bound**(파이프라인)라 latency도 bandwidth도 op latency를 안 바꾼다(§B: 실측 barrier ~30 ns).
- **결론**: L1 메모리는 측정상으로도 더 빠르지만(~20% lower access latency), (1) 격차가 NoC 지배로 작고, (2) read가
  compute 밑에 영구히 숨으므로(§3) decode latency에는 영향 없음. 즉 이 측정은 §3 결론을 바꾸지 않는다(informational).

### read 타임라인(간트)
- **공정 비교(같은 ctx 896, 같은 chunk 수)**:
  - DRAM read: `reprofile/zones2_dram_896/overlap_gantt_dram896.png`
  - L1 read: `reprofile/zones2_l1only_896/overlap_gantt_l1.png`
  - 둘 다 NCRISC read 밴드(DRAM 4.285 / L1 4.687 µs)가 TRISC compute(~5.9 µs) 안에서 끝나고 compute 꼬리가
    read 뒤로 더 이어진다(read 숨김 + 여유). L1 밴드가 약간 더 넓다(issue 오버헤드).
- (참고, ctx 1792 DRAM): `reprofile/zones_dram_1792/overlap_gantt.png`
- 생성: `plot_overlap.py <profile_log_device.csv> <out.png> [DRAM|L1]` (3번째 인자로 read 소스 라벨 지정).

### warm-up = compile **직접 증명** (item 2, 2026-06-05)

이전엔 "alloc+seed ~0.1s인데 iter1+2가 ~3.25s → 나머지는 compile"이라는 **추론**이었다. 이번에 **직접 측정**으로
증명: 실행 중 kernel 컴파일러 프로세스(`riscv-tt-elf-g++`/`cc1plus`) 개수를 0.15s 간격으로 timestamp 샘플링하고
각 iteration window에 버킷팅(`run_compile_probe.sh` + `compile_probe_parse.py`, non-tracy 2-layer ctx896 l1_only).

| iter | wall | compiler-busy | max 병렬 컴파일러 |
|---|---|---|---|
| 0 | 32.63 s | 95% | 108 |
| 1 | 0.93 s | 100% | 3 |
| 2 | 2.12 s | 83% | 6 |
| 3+ (steady) | 31 ms | 0% | 0 |

**warm-up(iter0–2) = wall 35.68s 중 컴파일러 busy ~33.81s (95%); steady(iter3+)는 0% 컴파일러 활동.** 즉 warm-up
시간은 **컴파일이다**(추론 아님): iter0 = decode 그래프 전체 컴파일(병렬 108 프로세스), iter1–2 = L1-경로 program
(n-tier SDPA + L1 ring write) first-use 컴파일(소수 프로세스). 일단 컴파일되면 in-process program cache로 steady step은
31 ms·컴파일러 0. (steady iter는 31 ms < 샘플 간격 0.15s라 steady window에 든 샘플은 2개뿐이지만 둘 다 0이고, 측정된
컴파일러-busy 33.8s가 전부 warm-up 안에 든다.)

- **cold/warm 영속 캐시 A/B는 무효였음**: 데모가 persistent on-disk kernel cache를 켜지 않아(`EnablePersistentKernelCache()`
  호출·env 없음) 매 프로세스가 재컴파일 → cold(iter0 32.68s/iter1 926/iter2 2109)와 warm(32.65s/949/2163)이 동일.
  이 동일성 자체가 "프로세스마다 재컴파일"을 뒷받침(컴파일러-활동 증명과 일치).
- DRAM run은 warm-up 없음(iter1부터 31 ms). 32-layer에선 alloc+seed(32× ≈ ~3s)가 더해질 뿐 compile 성분은 layer-무관.
- **제거 방법** (compile이 원인이므로): (1) timed loop 전에 L1-tier를 할당하고 **L1-경로 decode 1회를 untimed warmup**으로
  실행 → first-use 컴파일을 timed 구간 밖으로; (2) trace 경로는 capture 단계에서 컴파일되므로 **trace capture 전에 tier
  alloc+seed**; (3) `EnablePersistentKernelCache()` → 재실행 시 cross-process 재사용(첫 실행은 여전히 1회 컴파일).

### warm-up 제거 **구현+검증** (단일 cold run, 2026-06-05)

방법 (1)을 구현: `simple_text_demo.py`에 env-gated **untimed decode warmup** 추가(`DECODE_WARMUP_ITERS=N`,
기본 0=무변경). timed loop 직전에 `current_pos`/`out_tok`의 **clone**으로 N회 throwaway decode를 돌려 decode-graph +
L1-경로(n-tier SDPA + ring write) first-use 컴파일과 tier alloc+seed를 **untimed 구간에서** 끝낸다. clone이라 real state는
안 움직이고, warmup이 건드린 KV는 real decode가 읽기 전에 동일 값으로 덮어쓴다(deterministic argmax).

검증 (cold, cache wipe, 2-layer ctx896 l1_only, `DECODE_WARMUP_ITERS=4`):

| | iter0 | iter1 | iter2 | iter3+ |
|---|---|---|---|---|
| baseline | 32,681 ms | 926 ms | 2,109 ms | 31 ms |
| DECODE_WARMUP_ITERS=4 | **33 ms** | **30 ms** | **31 ms** | 31 ms |

"Starting decode warmup" → "Finished decode warmup" 사이 ~35.9s(전체 compile+alloc+seed)가 untimed로 흡수되고,
**timed loop은 iter0부터 steady(~31ms)**. 생성 토큰은 no-warmup baseline과 **byte-identical**(clone warmup이 출력 무손상).
→ warm-up을 단일 cold run 안에서 제거 확인(트레이스/영속캐시 없이). 실서빙엔 N=3~4면 충분(graph+alloc/seed+L1 compile 커버).

## 측정 변동성(±5% noise) + n-tier reader modulo 최적화 (2026-06-05)

### (a) DRAM↔L1 "순위 뒤집힘"의 원인 = run-to-run variance (clock 아님, dispatch jitter)

같은 명령으로 DRAM baseline 3회(32-layer, ctx896, non-tracy, `DECODE_WARMUP_ITERS=4`):
**85.8 / 86.4 / 89.6 ms** (run-to-run **~4.4% 변동**), run 내부도 84–93 ms(±5% jitter). 부하 중 AICLK는
**1350 MHz로 풀부스트**(idle 800)라 thermal throttle 아님 → **host dispatch jitter**(no-trace decode step은
dispatch-bound: device work는 소수, 호스트 큐/샘플링/LM-head가 다수).

single-run은 순위가 뒤집힌다(이전 results/ 로그 flip의 원인). 단 위 89.6은 **outlier**였다 — 아래 N회 평균에서
실제 run-to-run noise는 ~±1%로 작다.

### (a-2) N회 평균 비교 (alternating DRAM/L1, modulo-opt 적용, 2026-06-05) — **최종: tie within noise**

**N=4 (작은-표본 fluke)**: DRAM 85.33 / L1 88.85 ms → +4.1%, 분포 분리처럼 보였으나 그 DRAM 4회가 우연히
~85로 타이트했던 탓. **이 "reliable 4%, not a tie"는 아래 N=8로 철회됨.**

**N=8 (최종, 8회씩 교대)**:
| mode | mean | std | range |
|---|---|---|---|
| DRAM | **87.38 ms** | 1.94 | 84.8–90.0 |
| L1 (interleaved l1_only) | **89.29 ms** | 1.44 | 86.9–91.0 |

**L1 − DRAM = +1.91 ms (+2.2%), ~2σ, 분포 OVERLAP(86.9–90.0).** N=8에서 DRAM 실제 spread는 ~6%(84.8–90.0)다.
→ **L1은 DRAM에 신뢰성 있게 지지 않는다 = tie within noise**(기껏해야 유의성 경계의 ~2% lean, ±6% run-to-run noise에
묻힘). 이는 (a-3) 진단(어떤 L1 KV op도 gap을 안 짊어짐 — write 동일, read 숨김, dispatch 동일)과 일치 — 닫을 실제
gap이 처음부터 없었다. modulo-opt도 read가 이미 숨어 end-to-end엔 안 잡힌다.
### (a-3) per-layer host-dispatch 분해 (TT_L1_KV_PERF, 2026-06-05) — write-dispatch 가설 **반증**

32-layer L1 vs DRAM의 host-side per-section enqueue 시간(min=steady; avg/max는 warmup compile 오염이라 무시):

| section | DRAM min | L1 min | 차이 |
|---|---|---|---|
| KV write (`dram_kv_write` ↔ `adaptive_l1_kv_write`) | 0.052 ms | 0.054 ms | **동일** |
| `sdpa_call` (enqueue) | 0.077 ms | 0.076 ms | **동일** |
| `prepare_inputs_host` (per step) | 0.306 ms | 0.618 ms | **L1 +0.31 ms** |

write count도 동일(DRAM 1152; L1 = 1088 L1-write + 64 warmup DRAM-write = 1152). → **L1 ring write·SDPA read의
dispatch는 DRAM과 같다 — ~4%는 KV write/read dispatch가 아니다(가설 반증).** 유일하게 명확한 L1 추가비용은
`prepare_inputs_host` +0.31 ms/step(ring write-pos 계산 `_build_adaptive_l1_write_pos`) ≈ **0.4%**로, 4%를
설명하기엔 한 자릿수 작다. 즉 **~4%는 어떤 L1 KV op에도 국소화되지 않는다**(write/read/dispatch 모두 DRAM과 일치)
→ 제거할 큰 L1 오버헤드 없음. 잔여 ~4%는 (1) N=4 추정의 작은-표본 영향이거나 (2) 캡처 안 되는 미세 device-side
누적. **결론: L1은 dispatch/op 수준에서 DRAM과 동급이며, parity를 깨는 단일 lever는 없다.** 진짜 ~4%인지 tie인지는
larger-N(8–10회씩)으로만 확정 가능.

### (b) n-tier reader: interleaved에서 per-tile modulo 제거 (구현+검증)

n-tier reader `find_tier`의 per-tile 비용은 런타임 **modulo `% ring_tile_count`**(RISC-V SW divide, ~20–40 cyc/tile).
이는 tiering과 무관하게 1-tier에서도 돈다. **no-wrap(= interleaved l1_only full-resident)에서는 ring map이 항등이라
`flat_tile = gst`로 modulo를 건너뛴다**(`dataflow_common.hpp` `ring_wrapped` 플래그). modulo는 **실제로 wrap하는
ring(hybrid long-context)에서만** 유지 → sharded/hybrid 동작 불변(구조적으로 보존).
검증(cache wipe, 2-layer ctx896): l1_only 출력 **byte-identical to DRAM**(정확성 OK), steady DRAM 31.4 vs L1 32.1 ms
(~2%, **±5% noise 이내** → 회귀 없음). read가 compute 밑에 숨으므로 end-to-end 가속은 noise 이하지만, 구조적
per-tile 오버헤드와 context-growth 경향을 제거. (literal "interleaved를 다른 함수로 라우팅"은 미실시 — modulo skip이
비용을 이미 제거.)

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
- ~~**steady-state: L1 l1_only 11.44 vs DRAM 11.05 tok/s → L1 ~3.5% 빠르다 (DRAM write 생략)**~~
  **[정정 2026-06-05: 이 주장 철회]**. 이전 "L1이 write 생략으로 ~3.5% 빠르다"는 **틀렸다 — run-variance noise**.
  - **DRAM write 생략으로 인한 이득은 0이다(측정 확인)**: l1_only는 DRAM K/V write(`paged_update_cache`→DRAM 2회/layer)를
    skip하지만 `_write_adaptive_l1_tiers`가 **같은 `paged_update_cache`를 L1 tier에 동일 횟수 발행**한다. ops_perf에서
    `PagedUpdateCacheDeviceOperation`이 **DRAM·l1_only 둘 다 정확히 같은 count(32), 같은 비용(~9.5 µs/call)**. 1-token write는
    op-overhead-bound라 target(L1 vs DRAM)이 비용을 안 바꾼다. 전체 op profile도 거의 동일(DRAM 885 vs L1 889 ops, +4 Slice).
    → write는 "공짜로 스킵"되는 게 아니라 **동일 비용으로 L1에 재배치**될 뿐. write-skip은 speedup 원천이 **아니다**.
  - **실측(results/, 200-token, post-fix): DRAM이 더 빠르다.** `run_dram_baseline.log` 85.18 ms/11.74 tok/s(per-token FLAT
    86.30→86.08) vs `run_l1_interleaved_960.log` 88.98 ms/11.24 tok/s(per-token 증가 87.10→90.53). L1이 context에 따라
    느려지는 건 **n-tier reader의 tile당 NCRISC 오버헤드(modulo/곱셈, §"L1 read가 더 빨라야")가 context로 누적 노출**되기 때문.
  - 차이가 작아(~3–5%) **run마다 순위가 뒤집힌다**(128-token 요약 로그에선 L1이 근소 우위로 보이기도 함). 일관된 신호는
    "L1은 DRAM을 못 이긴다"이고, 오히려 **DRAM ≈ 또는 DRAM이 약간 빠르다**(write 이득 없음 + n-tier reader 오버헤드).
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
  즉 L1 l1_only는 **짧은 생성에선 net 손해(warm-up compile, 단 `DECODE_WARMUP_ITERS`로 제거 가능 §"warm-up 제거"),
  steady에선 DRAM와 동급이거나 약간 느림**(write 이득 없음 + n-tier reader 오버헤드). 이전의 "긴 생성 근소 우위"는 철회.
- 992 tokens = 32-layer에서 달성된 L1 window(앞서 언급한 ~900–1000 ceiling 부근).

## To-do (갱신)

  - ~~memory read time DRAM/L1 명시 + 간트~~ **[완료 2026-06-04]** read 소스를 DRAM/L1로 라벨링(§"L1_only 배포
    config 확인", §B). 간트 2종: `reprofile/zones_dram_1792/overlap_gantt.png`(DRAM),
    `reprofile/zones2_l1only_896/overlap_gantt_l1.png`(L1). 공정 비교(RD_CHUNK) DRAM 4.285 vs L1 4.687 µs/chunk(L1 ~9% 느림,
    원인은 issue-bound + n-tier reader SW 오버헤드; memory-wait는 양쪽 ~30 ns), 둘 다 100% hidden.

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
  - **정정**: "long-context면 read가 병목"은 틀림 — compute·read 둘 다 context에 선형이라 read는 영구 숨김(§3).
    capacity(KV quant int8/int4, multi-chip)는 **긴 context를 on-chip으로 돌릴 feasibility**용이지 decode latency를
    이기게 하는 lever가 아니다. read를 실제로 노출시키려면(= L1이 latency로 이길 유일한 길) context가 아니라
    **per-chunk compute:read 균형**을 바꿔야 한다 → 진짜 latency-lever 후보는 **decode matmul depadding**(Sq 1행을 32행
    타일로 패딩하는 비효율 제거; compute가 급감하면 read가 노출되어 L1이 의미). congested layout/cross-op DRAM 경쟁 해소도 보조.
  - (선택) NoC event profiler(`--collect-noc-traces`)로 read가 compute 밑에 얼마나 여유 있게
    숨는지(노출까지의 margin)를 정량화 — 위 결론에는 불필요하지만 crossover context 추정에 유용.

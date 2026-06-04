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
- DRAM은 **warm-up 패널티가 없다**(iter 1–2가 이미 steady). L1은 992 tokens × 32 layers tier
  할당+seed가 iter 1–2의 느린 구간(합 ~5.3s)으로 나타나 200-token 평균을 8.82 tok/s로 끌어내린다.
  즉 L1 l1_only는 **짧은 생성에선 net 손해(8.82 vs 11.05), steady/긴 생성에선 근소 우위(11.44 vs 11.05)**.
  교차점은 warm-up ~5.3s를 step당 ~3ms 이득으로 회수 → 대략 1700+ tokens 이후.
- 992 tokens = 32-layer에서 달성된 L1 window(앞서 언급한 ~900–1000 ceiling 부근).

## To-do (갱신)

  - **Hybrid(sharded+interleaved) 구현은 보류/우선순위 낮음.** §2에서 SDPA가 compute-bound이고
    L1==DRAM임이 확정됐으므로, 코어 여유를 더 짜내 capacity를 늘려도 decode latency 이득이 없다.
    hybrid의 가치는 순수 capacity뿐이고, dynamic branching 복잡성으로 오히려 느려질 위험이 있다.
  - ~~cluster-replicated~~ **제거**: head/block-sharding이 8-on-1 congestion을 제거하므로 불필요.
  - **"Head sharding" 구현** — 단, 목적은 "DRAM을 이기는 것"이 아니라 sharded path를
    interleaved/DRAM **parity로 복귀**시켜 그 용량을 활용 가능하게 만드는 것. latency win은 기대 불가.
  - 진짜 win이 필요하면 **capacity 쪽**: KV quantization(int8/int4) 또는 multi-chip로 long-context를
    L1-resident로 만들어, read가 병목이 되는 영역(read time > compute time)을 L1으로 끌어오는 것.
  - (선택) NoC event profiler(`--collect-noc-traces`)로 read가 compute 밑에 얼마나 여유 있게
    숨는지(노출까지의 margin)를 정량화 — 위 결론에는 불필요하지만 crossover context 추정에 유용.

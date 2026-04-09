# 연구 주제 전환: weight L1 caching -> KV Cache L1 Caching

Weight matrix 자체는 sparsity에 상관 없이 matmul 수행 시 모든 row가 1회씩 조회된다. 동일  layer 내에서는 단일 row가 다른 row보다 자주 읽히는 일이 발생하지 않는다. 따라서 dense 한(또는 output에 기여도가 높은) row를 우선적으로 caching한다고 해도 (서로 다른 weight block의 사용 빈도가 달라지는  MoE 같은 경우를 제외하면) 성능 이득이 없을 것으로 예상된다.

KV Cache: decode stage에서 다음 토큰을 추론하기 위해 참조되는 과거 토큰에 대한 정보를 저장하고 있는데, 토큰에 따라 중요도 및 output에 대한 기여도가 달라진다. 따라서 중요한 토큰들에 해당하는 KV 캐시 부분을 우선적으로 L1에 캐싱(Tiered KV Cache) 하고, 이들만으로 충분한 attention score를 낼 수 있을때 DRAM의 Cold KV Cache 접근을 생략한다면 그만큼 DRAM 접근 대비 L1 접근 비율이 높아지므로 충분히 캐싱에 의한 성능 이득을 볼 수 있다.

Decode stage에서 참조되는 KV Cache의 총 크기는 모델 자체보다도 커질 수 있지만, 런타임에 실시간으로 올라가는 KV Cache는 단일 레이어 분량이며, 10만 개의 과거 토큰 중 현재 단어를 예측하는 데 필요한 hot token는 전체의 5~10%도 되지 않는다(SnapKV (2024), H2O (NeurIPS 2023), Scissorhands (NeurIPS 2023)).

Tenstorrent의 210MB SRAM은 이를 기준으로 했을때 수천 개 토큰 분량의 KV Cache를 캐싱할 수 있는 양이므로 hot token을 커버할 수 있다.

## FlashAttention vs This

FlashAttention: 기존에 Attention 계산의 중간 결과값을 HBM에 저장 후 다시 불러오던 것을, ‘부분 Softmax 및 축적’으로 중간 결과 HBM 저장-로드 과정을 생략해서 Attention 계산의 전체 과정이 SRAM에서 이루어질 수 있도록 함. Q의 길이가 N (N >> 1)인 (=QK^T의 크기가 NxN) Prefill stage에서 성능 향상 효과가 크고, training 단계의 back propagation 시에도 사용 가능. 단 K와 V가 SRAM에 있는 lifetime은 기존 Attention 계산과 거의 차이가 없다. 부분 Attention 계산이 끝나고 나면 바로 다음 계산할 part로 교체된다


- 기존　Attention: Q, K 로드 -> QK^T 계산, HBM에 저장 -> HBM에서 QK^T　로드 -> Softmax(QK^T) 계산, HBM에 저장 -> HBM에서 Softmax(QK^T) 로드 , Softmax(QK^T)V 계산 -> HBM에 최종 결과 저장

- FlashAttention: (Q, K, V 로드 -> Online Softmax(QK^T)V 계산) 반복 -> HBM에 최종 결과 저장

- This: K와 V의 특정 부분을 계속 SRAM에 고정해놓고 재사용한다. Q 길이가 1인 작은 크기의 연산이 반복적으로 발생하는 decode stage에서 메모리 병목을 줄임으로써 성능 향상이 클 것으로 기대

## CB size 정정

Hardware panic 발생 시, 에러 메시지는 코어 8개 (0,0 ~ 7,0) 에서 메모리 영역이 겹친다는 내용이었음:

```
Statically allocated circular buffers in program 5 clash with L1 buffers on core range [(x=0,y=0) - (x=7,y=0)]. L1 buffer allocated at ... and static circular buffer region ends at 1249664
```

사실은 모든 코어가 CB를 SRAM의 85%에 할당하는 것이 아니라, 저 8개의 코어에서만 forward pass에서 축적된 CB 용량이 85%에 달해서 메모리가 부족했던 것

![Llama3 Architecture](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*_xNP7aBpcmcMk4tXJ-Z8Mw.png)

위 그림에서 블록 1개에 포함된 모든 연산(Attention, RMS Norm, SwiGLU)에 각각 해당하는 `Program` 이 캐시되면서 각 프로그램에서 할당한 CB가 각각 고정됨

실제 Attention 계산 시에 할당되는 CB 용량:

| CB | Data | Bytes |
|---|---|---|
| c0, c10 (Q) | (4+4) tiles × 2,048 B | 16,384 |
| c1 (K) | 32 tiles × 1,024 B | 32,768 |
| c2 (V) | 32 tiles × 1,024 B | 32,768 |
| c3 (mask) | 4 × 2,048 B | 8,192 |
| c19 (intermed) | 42 × 2,048 B | **86,016** |
| c16, c20, c23–c26 (out/im) | 5×4×2,048 B | 40,960 |
| c5–c7, c17–c18, c21–c22, c27–c31 (stats) | 11 × 2,048 B | 22,528 |
| c11, c12 (identity/zero) | 2 × 2,048 B | 4,096 |
| c24 (qk_im) | 4 × 2,048 B | 8,192 |
| **TOTAL** | | **253,952 B = 248 KiB** |

### SRAM usage in 130 Tensix cores

| y \ x | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0** | 1249664 (79.5%) | 1249664 (79.5%) | 1249664 (79.5%) | 1249664 (79.5%) | 1249664 (79.5%) | 1249664 (79.5%) | 1249664 (79.5%) | 1249664 (79.5%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) |
| **1** | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) |
| **2** | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) |
| **3** | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) |
| **4** | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) |
| **5** | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) |
| **6** | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) |
| **7** | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 766336 (48.7%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) |
| **8** | 722944 (46.0%) | 722944 (46.0%) | 722944 (46.0%) | 722944 (46.0%) | 722944 (46.0%) | 722944 (46.0%) | 722944 (46.0%) | 722944 (46.0%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) | 360832 (22.9%) |
| **9** | 203136 (12.9%) | 203136 (12.9%) | 203136 (12.9%) | 203136 (12.9%) | 203136 (12.9%) | 203136 (12.9%) | 203136 (12.9%) | 203136 (12.9%) | 194944 (12.4%) | 194944 (12.4%) | 194944 (12.4%) | 194944 (12.4%) | 194944 (12.4%) |

- 참고:  기존 (0,0) ~ (7,0)의 85%는 첫 `DEFAULT_UNRESERVED_BASE` 102,752 바이트를 제외한 1,470,112 바이트를 분모로 잡았을 때 계산된 수치임

## Methodology

- ~~먼저 Tenstorrent에서 전체 KV캐시를 안 쓰고 이미 Hot KV만을 선별해서 LLM inference를 수행하도록 하고 있는지 판별~~

- Hot KV를 DRAM에 100% 저장할 때 (기존 구현) vs Hot KV를 SRAM에 일부 또는 전부 저장할 때 속도, 성능 (생성된 텍스트 품질) 비교

    1. SRAM에 전부 안들어가면 남은 부분 버리고 (KV 캐시를 100% SRAM에서만 참조) LLM inference 수행했을 때의 속도, 성능 평가

    2. SRAM에 안 들어가는 부분을 기존처럼 DRAM에 저장함으로써 KV캐시를 SRAM과 DRAM에 분산하여 저장했을 때 성능 평가

## Results

KV Cache hit ratio
(l1_kv_window_size=256)

Total KV Tiles Read: 78848
L1 Hits:           63744 (80.84%)
DRAM Reads:        15104 (19.16%)

- Llama 3 레이어 수: 32

- Occurrences of DPRINT: 4,928

- Tokens Generated: 4,928 / 32 layers = 154 tokens

- Total K Tiles Read: 4,928 × 16 tiles per chunk = 78,848 K tiles

- 16 tiles = 4 ( Sk_chunk_t = 128 -> /32 = 4 sequence tiles ) x 4 ( head dimension )

- context window를 늘렸을때 SRAM 필요한 용량 계산, 실제 사용되는 context window 크기

    - context 길이 10만 토큰 기준, 5%~10% = 5,000 ~ 10,000개 토큰

- 8B 모델들 context 길이, 토큰 수

    - Llama 3.1 8B 1토큰당 KV 캐시 용량: 32 (레이어) * 8 (KV 헤드) * 128 (차원) * 2 (K와 V) * 1 (바이트, bfloat8_b 자료형) = 64KB

    - 2,000 토큰: 125 MB

    - 16,000 토큰: 1 GB

- 512토큰 분량의 KV 캐시를 130개 코어에 할당(INTERLEAVED): 코어당 약 258KB 필요 -> 실패 (이전에 레이어 개수 1개 환경이라 성공함)

- 실제 제한: 약 427토큰 (~215KB)

- (정정) Llama 3.1 8B 워크로드 실행 시 단일 Blackhole 디바이스 환경에서는 8x8 grid가 사용됨

- (참고) Llama 3.1 8B 모델 기준 정확한 코어 할당 구조 (decode stage)

    - KV헤드 개수: 8개

    - Q헤드 개수: 32개

    - KV헤드 하나당 8x1 row 할당됨 (Tensor Parallelism on a device)

    - Attention 계산 시, KV헤드 하나당 Q헤드 4개와 곱해짐

        - 이때 각 코어에서는 KV헤드와 Q헤드를 위해 Circular Buffer가 1.2MB 차지

## Remaining Tasks

- L1 layout INTERLEAVED -> SHARDED

- KV Cache l1 buffer의 가능한 할당 범위가 decode stage에서 정적 할당된 circular buffer 크기에 따라 제한되는 문제 해결

- *8x8 이외 공간에서 utilization 100% 뽑기*

- L1에 데이터가 비효율적으로 저장되는 문제 (이미 L1에 할당되어 있는 KV 캐시 엔트리가 circular buffer에 복사가 됨) 해결

- Circular Buffer를 계쏙 고정해놓지 말고 한 연산마다 가능한 최대의 양을 할당

- ~~KV Cache에서 hot token의 scale 조사~~

- ~~CB크기를 줄일 수 있는지 조사~~

- DRAM KV cache를 버렸을때의 성능 차이

- ~~tt-metal FlashAttention 문서~~

- 8x8 말고 다른 layout도 시도, 다른 parameter도 바꿔보기

- CB 크기를 줄이고 KV캐시 넣는 양을 늘렸을때 성능 차이

- CB 특정 용량 할당할때 Crash내서 콜스택 확인

- Sharding, Interleaved 말고

- ~~DRAM만 쓸 때 SRAM의 utilization 측정~~

    - 약 85% (8x8 만 사용중일 때)
    - 현재 tt-metal이 제공하는 API인 `get_memory_view()`, `dump_device_memory_state()` 로는 allocator로 할당된 영역만 덤프되고 정적 할당된 CB 영역이 반영되지 않아 정확한 측정 불가 (free space도 잘못 계산되어 나옴)

## References

- Efficient Streaming Language Models with Attention Sinks (ICLR 2024 / StreamingLLM)

    KV Cache 전체를 담으면(dense attention) HBM이 터지고, 연산 속도도 느려지기 때문에 “attention window” 를 도입해 일부만을 저장하는 것이 속도 저하를 막을 수 있다. 이렇게 해도 되는 이유는, 우선 KV 캐시에서 각 토큰에 대한 점수 편차가 크다. 중요도가 떨어져서 attention score가 매우 낮게 나오는 부분은 아예 버리고 계산을 생략해도 결과에 큰 영향이 없고, 점수가 크게 나오는 부분만 선별하여 계산해도 된다. 그런데 보통 점수가 크게 나오는 부분은 최근 토큰에 집중되어 있는 경향이 있기 때문에 이 attention window는 기본적으로 최근 N개의 토큰에 해당하는 영역으로 설정하는 것이 효율적이다. - Sparse Transformer (OpenAI, 2019), Longformer (AllenAI, 2020) & BigBird (Google, 2020)

    Attention 계산 과정에서 softmax 점수 총합을 1로 맞추기 위해 모델이 ‘남는 점수'를 버릴 sink가 필요하다고 한다. 지금까지 생성한 문장 전체의 중요도가 그렇게 높지 않은 경우에도 반드시 전체 총합을 1로 맞춰야 한다는 규칙 때문에 발생하는 현상이다. 그런데 일반적인 LLM 모델은 최초의 4개 토큰에 대한 KV cache를 sink로 지정(최근이거나 특별히 중요도가 높은 토큰이 아니어도 attention score가 높게 나옴)하는 경향이 있다.

    단순한 sliding window 방식으로 최근 토큰에 대한 KV Cache만 저장하는 방식은 window가 이동하며 sink로 사용할 최초의 4개 토큰이 없어져서 토큰 생성 시 성능(품질)이 안 좋아진다.

    여기서 Streaming LLM의 제안은 최초 4개의 토큰을 “attention sink”로 지정하여 KV Cache에 pinning 해두고, 나머지는 sliding window를 적용하면 성능 저하를 최소화하면서 빠른 추론을 이끌어낼 수 있다는 것이다.

    기존 모델은 그냥 KV캐시에 처음 입력되는 4개의 토큰을 sink로 지정하면 되고, 이에 더해서 모델의 pre-training이 가능한 환경일 경우 pre-training시 처음부터 sink용 토큰 을 {SINK}와 같은 형태로 지정해 주면 처음 4개 토큰의 중요도가 높을 경우 sink로 사용되는 사태를 방지해 더욱 성능을 끌어올릴 수 있다.

- Seesaw: Tiered KV Cache (CPU DRAM + GPU HBM)

- HiP(Hierarchically Pruned Attention, ICLR 2025): O(logT) 개의 Hot Token 선별 O(TlogT) 만에 하는 알고리즘 제시, 즉 Attention score가 뭐가 높게나올지 실제 계산 전에 미리 예측가능

- H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs (NeurIPS 2023) KV 캐시 다쓸필요없이 Heavy Hitter만 남겨도 성능이 유지된다

- FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU (ICML 2023) -> Tiered KV Cache buffering 제시 (GPU - CPU - SSD)

- MTDS (Multi-Tier Dynamic Storage, 2025): Cache hit probability 예측

- SnapKV (2024) : Long context 처리할 때 모델이 실제 필요한 정보는 극히 한정된 위치 (observation winodw)에 집중되고 양은 전체의 1~5%정도

- Scissorhands (NeurIPS 2023): 과거에 중요한 토큰들은 미래에도 중요하다는 것을 증명, KV 캐시 90% 삭제해도 모델 성능이 유지됨을 보임
FlashAttention, FlashDecoding)

- TurboQuant: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

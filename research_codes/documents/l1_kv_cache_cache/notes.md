## DRAM Write 측정

- l1-only의 3.5% 성능향상이 DRAM write 생략에 의한 것이라면, tt-nn에 의해 찍히는 로그 decode time에서 DRAM write 시간이 포함되어 있다는 뜻인데, 여기서 DRAM write 시간이 어느정도에 해당하는지 profiling할 수 있는가?

## 단순한 scale out은 절대로 compute time - memory read time crossover를 발생시키지 않는다.

- crossover를 발생시킬 수 있는 대안
    1.  multi-chip 시나리오에서 Tensor Parallelism 적용: 한 칩이 담당하는 어텐션 헤드 개수가 줄어들어, 연산량이 줄어든다. 다만 메모리 read command issue time 으로 인한 latency 하방 경직성은 주의해야 한다.

    2.  아키텍처 제언: SRAM을 깎고 ALU를 늘려야 한다 (그 많은 SRAM이 높은 utilization을 보이지 못하고 결국 순수 성능 한계로 인해 compute-bound 되는 참극이 발생하니, 향후 Tenstorrent 칩을 설계할 때 연산 코어 비중을 늘려야 한다 (SRAM 용량을 줄이더라도))

    3.  Batch-1 이 아닌 다중 사용자 환경 (Batch 32이상)에서 실험해 본다 (근데 batch size 늘어난다고 crossover 발생할지는 의문임)

    (c) scope/prototype the decode-matmul depad — the actual path to an L1 latency advantage.

## 단순한 scale out은 절대로 compute time - memory read time crossover를 발생시키지 않는다.

- crossover를 발생시킬 수 있는 대안
    1.  multi-chip 시나리오에서 Tensor Parallelism 적용: 기각. 단일 칩 연산량이 줄어들지만, 메모리 읽기 양도 줄어든다.

    2.  아키텍처 제언: SRAM을 깎고 ALU를 늘려야 한다 (그 많은 SRAM이 높은 utilization을 보이지 못하고 결국 순수 성능 한계로 인해 compute-bound 되는 참극이 발생하니, 향후 Tenstorrent 칩을 설계할 때 연산 코어 비중을 늘려야 한다 (SRAM 용량을 줄이더라도))

    3.  Batch-1 이 아닌 다중 사용자 환경 (Batch 32이상)에서 실험해 본다: 기각. Scale-out은 compute와 memory를 동시에 늘리기 때문에 compute-memory crossover를 발생시키지 못한다.

## NEW RESEARCH TOPIC - scope/prototype the decode-matmul depad

Sound instinct, and it is probably the single best batch-1 decode-latency lever available here. Two corrections to
  the framing before the details: the payoff is amortizing the dispatch-bound per-step overhead across k tokens,
  not "recovering wasted matmul rows" (at batch-1 those idle rows already cost ~nothing in wall-clock), and it is
  orthogonal to the L1-KV question — if anything it makes attention more compute-bound and hides L1 read further.
  Confidence: high on the mechanism and the attention reasoning; medium on whether this repo already supports spec
  decode (I have not checked).

  The mechanism you describe is correct. In speculative decoding the target model verifies k draft tokens in one
  forward pass, so the activation row dimension is M = k instead of 1. For k <= 32 that fits the single tile that
  batch-1 decode leaves 31/32 idle, so verifying k tokens costs ~the same matmul-tile work as decoding 1 in the
  linear ops (QKV proj, o_proj, MLP, LM head).

  Why it actually helps (and why the framing matters):
  1. Decode is dispatch-bound (~87 ms/step, device compute sub-ms). The real win is fewer sequential target steps
  per accepted token. At a typical ~2-3 accepted tokens/step you get ~2-3x tokens/sec because you amortize the fixed
  dispatch + sampling + LM-head cost over multiple tokens.
  2. The row-filling is the enabler, not the prize. At batch-1 the linear ops are weight-streaming/dispatch-bound,
  not compute-saturated, so the idle rows were already nearly free in wall-clock. The correct mental model is "the
  verify pass is nearly as cheap as a 1-token step, so harvest k tokens from it," not "reclaim wasted compute."

  Where it does not ride the same free-rows mechanism, and why it does not help the L1 thesis:
  3. SDPA's matmul rows are heads, not tokens, so it does not get filled by k tokens the way QKV does. Verifying k
  query positions is a k-row causal attention — that routes through the prefill/extend kernel, not the 1-token
  decode kernel, and it is genuinely more expensive. Crucially it reuses one KV read across k query rows, which
  raises attention arithmetic intensity (more compute per byte read), pushing attention further into compute-bound
  and hiding L1 read even more. So spec decode moves away from any read-bound regime where faster L1 could win.
  4. It is a different project from L1 KV, with real costs: a draft model or self-speculation heads
  (Medusa/EAGLE/n-gram to avoid a separate draft dispatch), speculative KV writes plus rollback of rejected tokens
  (which adds complexity specifically to the L1 ring-write path), acceptance-rate-dependent payoff, and kernel
  support for k-token "extend" attention.

  Net: pursue it as a decode-latency/throughput lever on its own merits — it is well-matched to this dispatch-bound,
  under-utilized-matmul regime — but file it under "fewer steps per token," not "make L1 beat DRAM." It leaves the
  L1-vs-DRAM verdict unchanged (still a tie) and slightly deepens the compute-boundedness that makes L1 read
  irrelevant to latency.

  I can check whether tt_transformers/simple_text_demo has any speculative-decode or Medusa/EAGLE path and scope
  what a verify-step would take here, if you want to take it further.
  The mechanism you describe is correct. In speculative decoding the target model verifies k draft tokens in one forward pass, so the activation row dimension is M = k instead of 1. For k <=
  32 that fits the single tile that batch-1 decode leaves 31/32 idle, so verifying k tokens costs ~the same matmul-tile work as decoding 1 in the linear ops (QKV proj, o_proj, MLP, LM head).

  Why it actually helps (and why the framing matters):
  1. Decode is dispatch-bound (~87 ms/step, device compute sub-ms). The real win is fewer sequential target steps per accepted token. At a typical ~2-3 accepted tokens/step you get ~2-3x
  tokens/sec because you amortize the fixed dispatch + sampling + LM-head cost over multiple tokens.
  2. The row-filling is the enabler, not the prize. At batch-1 the linear ops are weight-streaming/dispatch-bound, not compute-saturated, so the idle rows were already nearly free in
  wall-clock. The correct mental model is "the verify pass is nearly as cheap as a 1-token step, so harvest k tokens from it," not "reclaim wasted compute."

  Where it does not ride the same free-rows mechanism, and why it does not help the L1 thesis:
  3. SDPA's matmul rows are heads, not tokens, so it does not get filled by k tokens the way QKV does. Verifying k query positions is a k-row causal attention — that routes through the
  prefill/extend kernel, not the 1-token decode kernel, and it is genuinely more expensive. Crucially it reuses one KV read across k query rows, which raises attention arithmetic intensity
  (more compute per byte read), pushing attention further into compute-bound and hiding L1 read even more. So spec decode moves away from any read-bound regime where faster L1 could win.
  4. It is a different project from L1 KV, with real costs: a draft model or self-speculation heads (Medusa/EAGLE/n-gram to avoid a separate draft dispatch), speculative KV writes plus
  rollback of rejected tokens (which adds complexity specifically to the L1 ring-write path), acceptance-rate-dependent payoff, and kernel support for k-token "extend" attention.

  Net: pursue it as a decode-latency/throughput lever on its own merits — it is well-matched to this dispatch-bound, under-utilized-matmul regime — but file it under "fewer steps per token,"
  not "make L1 beat DRAM." It leaves the L1-vs-DRAM verdict unchanged (still a tie) and slightly deepens the compute-boundedness that makes L1 read irrelevant to latency.

  I can check whether tt_transformers/simple_text_demo has any speculative-decode or Medusa/EAGLE path and scope what a verify-step would take here, if you want to take it further.

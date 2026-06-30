# BH Llama-3.1-8B decode regression: on-device sampling (#31046)

Investigation date: 2026-06-29/30. Hardware: P150 (Blackhole), single chip, batch-1
latency config. Demo: `models/tt_transformers/demo/simple_text_demo.py -k
"performance and batch-1"`.

## TL;DR

- The decode throughput regression on Blackhole P150 (Llama-3.1-8B, batch-1) from
  ~35 t/s/u to ~21 t/s/u is caused by upstream commit **`58ac27a9125`
  "[TT-Transformers] Support on-device sampling" (#31046)**.
- It is **not** the user's L1-KV work, not the clock, not the host, not PCIe, not
  the LLK uplift. Those were all measured and ruled out.
- Measured on the same P150: parent commit `ca3c90bbb73` (#31750) = **35.49
  t/s/u**; `58ac27a9125` (#31046) = **21.51 t/s/u**. Clean step change at the
  direct parent boundary.
- Mechanism (high-confidence inference, not yet directly profiled): #31046 moves
  the per-token token-selection step from the host onto the device, but the
  on-device sampling op runs the full top-k/top-p machinery over the entire
  128,256-entry vocabulary **every decode step, even for greedy (temperature=0)
  decoding**. On Blackhole at the fixed 800 MHz AICLK that on-device op costs more
  per token (~18 ms) than the host round-trip it was meant to eliminate. Because
  decode runs as a single on-device trace, that op adds directly to per-token
  latency.

---

## 1. Background: where sampling sits in LLM inference

LLM text generation has two phases:

1. **Prefill**: the prompt (here 128 tokens) is run through the model once to
   populate the KV cache. Produces the logits for the last prompt position.
2. **Decode**: tokens are generated one at a time, autoregressively. Each decode
   step does a forward pass for a single position (seq_len = 1) and produces a
   new token, which is fed back as the input for the next step.

Each forward pass (prefill last-position or decode step) ends at the **LM head**,
a matmul `hidden_state @ W_unembed` that produces **logits**: one score per
vocabulary token. For Llama-3.1-8B the vocab is **128,256**, so the logits tensor
for batch-1 is shape `[1, 128256]` (= 4008 tiles of 32 along the vocab axis).

To turn logits into the next token you need a **token-selection (sampling)** step.
This runs **once per generated token**, at the very end of every decode iteration
(and once after prefill for the first token). It is on the critical path of every
output token.

### Greedy / argmax

"Greedy decoding" = pick the single highest-scoring token:

```
next_token = argmax(logits)          # over the 128,256 vocab axis
```

This is what the **batch-1 latency benchmark uses** (`sampling_params =
{"temperature": 0, "top_p": 0.08, "top_k": 32}`; temperature 0 means greedy). It
is deterministic and is the standard setting for measuring single-user latency.

### Stochastic sampling (temperature / top-k / top-p)

For non-deterministic generation you instead:

1. scale logits by `1/temperature`,
2. restrict the candidate set to the **top-k** highest logits (here k = 32),
3. further restrict to the smallest set whose probability mass ≥ **top-p** (nucleus,
   here p = 0.08),
4. softmax over the survivors and draw a sample.

Top-k and top-p both require finding the largest elements over the full 128,256
vocab — a (partial) sort / repeated reduction over a large axis. This is much more
work than a plain argmax.

The important subtlety: **the demo passes a non-None `sampling_params` even for the
greedy (temp=0) case.** Whether that means the full top-k/top-p kernel runs or a
cheap argmax shortcut runs depends entirely on the sampling implementation — and
that is exactly what #31046 changed.

---

## 2. Host sampling vs on-device sampling

### Host sampling (before #31046)

- Device computes logits `[1, 128256]` on-chip.
- Logits are **copied back to host** (device DRAM -> host over PCIe).
- Host (Python/torch, on the CPU) computes argmax / top-k / top-p and picks the
  token id.
- The chosen token id (a single integer) is **written back to the device** as the
  input for the next decode step.

Cost profile: the argmax/sort itself is cheap on a fast x86 CPU. The dominant cost
is the **host<->device round-trip and synchronization every token**: the device
must finish, stall while the host reads logits, computes, and writes the token
back, then resume. This serializes host and device and **breaks the pure on-device
decode trace** (the loop cannot be a single replayed trace because a host step sits
in the middle of every iteration).

### On-device sampling (#31046, the change)

- The argmax / top-k / top-p is computed **in a device kernel**, on the Tensix
  cores, directly from the on-chip logits.
- Only the resulting token id stays on device (or a tiny readback), so there is
  **no per-token logits readback and no host stall**.
- The entire decode step, including sampling, fits **inside the captured trace**.

In principle this is the faster design: it removes the PCIe logits transfer and the
host-device sync, and lets the whole decode loop run as one on-device trace. That
is why the change was made.

---

## 3. Why it made decode *slower* on Blackhole (the core question)

The expectation ("removing host communication must be faster") is reasonable but
incomplete. Sampling is not free work that disappears — it **moves from the host
CPU onto the Tensix cores**. The net effect is:

```
delta_per_token  =  (device cost of the on-device sampling op)
                  - (host round-trip + host-compute cost it replaced)
```

On this configuration that delta is **positive (~+18 ms/token)**, i.e. the device
op is more expensive than the round-trip it eliminated. The reasons:

1. **The sampling op operates over the full 128,256-vocab logits on-device.**
   Top-k=32 + top-p selection requires (partial) sorting / repeated max-reductions
   across 4008 tiles. On the host this is microseconds of optimized C/torch; on
   the Tensix grid at a **fixed 800 MHz** AICLK (Blackhole P150 runs decode at 800
   MHz by firmware design, not throttling) an unoptimized full-vocab selection
   kernel is expensive. Blackhole bring-up software is explicitly "under active
   development", and this op was not yet tuned for BH.

2. **Greedy (temp=0) still pays for the full machinery.** Because the demo passes
   `sampling_params` with `top_k=32, top_p=0.08`, the on-device path runs the
   top-k/top-p selection even though temperature 0 is mathematically just an
   argmax. A direct on-device argmax would be a single reduction; the top-k/top-p
   path is far heavier. So the "greedy latency benchmark" is actually exercising
   the expensive sampling kernel.

3. **Decode is trace-bound, so the op adds directly to latency.** Earlier in the
   investigation we established that this decode runs as a single on-device trace
   with negligible per-token host work, and the chip is compute/op bound at 800
   MHz (not host-, dispatch-, PCIe-, or memory-bandwidth limited with headroom).
   In that regime, **every op in the traced decode step adds its full device time
   to the per-token latency** — there is nothing to hide it behind. Adding a heavy
   sampling op therefore shows up 1:1 in t/s/u.

4. **The host round-trip it replaced was relatively cheap here.** For batch-1 the
   logits readback is ~256 KB/token over PCIe Gen5 x16 (microseconds of transfer),
   and the sync overhead, while real, was smaller than the ~18 ms the on-device op
   now costs. So trading the round-trip for the op was a net loss on BH.

Net: the change is a good idea in principle (and likely a win on Wormhole or with a
tuned BH kernel, or at large batch where the round-trip dominates), but on BH P150
batch-1 with an untuned full-vocab top-k/top-p kernel it regressed decode ~1.6x.

> Confidence: the **attribution** to #31046 is certain (bisected + isolation, see
> §4). The **mechanism** in this section is a high-confidence inference from the
> code change scope (`tt_sampling.py` +413, decode-path rewrite) and the trace-
> bound/compute-bound model established earlier. It has **not** been confirmed by a
> direct per-op device profile, because the device profiler post-processor
> (`process_device_log.py`) fails to parse this BH build's output (buffer-wrap
> assertion) and the Tracy capture flow could not be launched. The decisive
> remaining confirmation is to profile the sampling op's device-time in isolation,
> or to A/B `sampling_params=None` (host argmax) vs on-device on the complete
> Nov-5 base code (see §6).

---

## 4. Evidence chain (how the root cause was established)

All measurements on the same P150, canonical `performance batch-1` config
(prefill-128 prompts, max_seq_len 1024, paged attention, trace on, all 32 layers).

| Build / commit | date | t/s/u | note |
|---|---|---|---|
| v0.64.0-dev20251030 (`dda3d05c6e1`, release branch) | Oct 30 | **34.4** | fast reference |
| main merge-base `eb29ebf1c63` (Sept 5) | Sep 5 | **35.08** | bisect good endpoint |
| `ca3c90bbb73` (#31750, parent of culprit) | late Oct | **35.49** | last good |
| **`58ac27a9125` (#31046 on-device sampling)** | late Oct | **21.51** | **first bad** |
| Nov-5 base `e47fe9a3417` (l1-kv-cache upstream base) | Nov 5 | **21.3** | slow |
| current `upstream` HEAD `82ca47adc7c` | Jun 2026 | **~22** | still not fixed |

Steps:

1. **Ruled out environment.** AICLK fixed at 800 MHz by BH firmware (both boards,
   not throttled: 49 C, 42 W); PCIe Gen5 x16; 1 GB hugepages present; decode runs
   in trace mode with on-device sampling, so it is device/op bound, not host- or
   dispatch-bound. Host governor / NUMA irrelevant to a trace-bound decode.

2. **Ruled out the user's L1-KV work.** The Nov-5 main base (without any L1-KV
   changes) is already 21.3, and current `upstream` (no L1 changes) is ~22. The
   regression exists without the L1-KV code; the L1-KV branch merely inherits a
   regressed baseline. Decode in default mode uses stock paged DRAM KV (the L1
   path is opt-in behind `--l1_kv_mode`).

3. **Confirmed it is a code regression, not a release-vs-main artifact.** v0.64
   (release branch, forked from main ~Sept 5) is fast; main after Sept 5 is slow.
   Same board, same clock, same config -> a commit between Sept 5 and Nov 5
   regressed it.

4. **Bisected** `eb29ebf1c63..e47fe9a3417` on main. All commits up to
   `ca3c90bbb73` tested good (~35). The suspect band could not be bisected cleanly
   because #31046 also introduced a demo-breaking bug (`argmax_on_device`
   NameError in `generator.py::_decode_forward_trace_text`, a half-finished rename
   to `sampling_on_device`) that makes the demo fail on #31046 and many later
   commits -> those steps were skipped.

5. **Ruled out the LLK uplift `#31907`** (submodule `1e5c228`->`1078754`) by
   isolation: swapping only the LLK submodule to the slow pointer on the good
   commit `ca3c90` kept it at **35.45** t/s/u (not 21). The bisect merely walked
   through that commit. (The LLK changes are SFPU reduce + unpack tilize/untilize;
   the matmul FPU microkernel is unchanged.)

6. **Isolated #31046.** Built `58ac27a9125`, applied the `argmax_on_device ->
   sampling_on_device` fix to make the demo run, measured **21.51 t/s/u**. Its
   direct parent `ca3c90bbb73` is **35.49**. No commits between them. Therefore
   #31046 is the first bad commit.

### Reproduction worktrees (left in place)

- `/home/masterjunmo/codes/tt-metal-v064-cmp` — v0.64 (35), python_env on 3.10.
- `/home/masterjunmo/codes/tt-metal-nov5-base` — Nov-5 base (21); currently checked
  out at `58ac27` with throwaway demo/generator edits (the argmax fix and a
  `sampling_params=None` experiment).

Build notes for these (BH, old tags): build the `ttnn` cmake target (not `install`
— the `2d_big_mesh_cabling_gen` scaleout tool fails on a missing generated
`cluster_config.pb.h`); build against Python 3.10 (pyenv 3.10.19), not the
pyenv-default 3.13, or the old pip pins (Pillow 10.3.0 etc.) fail to build;
reconfigure cmake with explicit `Python3_EXECUTABLE/INCLUDE_DIR/LIBRARY`.

---

## 5. Why v0.64 (the fast reference) is unaffected

`v0.64.0-dev20251030` was cut from a release branch that forked from main around
Sept 5, **before** #31046 landed (late Oct). So v0.64 uses the older host-side (or
older/cheaper) sampling path and stays at ~34-35. The fast/slow split lines up
exactly with the presence of #31046.

(The v0.64 run log does show a "Pre-compiling sampling path" line, i.e. it has
*some* device-side sampling support, but it is the pre-#31046 implementation that
does not carry this cost.)

---

## 6. Recommendations / workarounds

In order of preference:

1. **Use host-side argmax for greedy decode.** Setting `sampling_params=None` for
   the batch-1 latency path routes token selection back to the host, which should
   recover ~35 t/s/u. This failed to test at `58ac27` itself due to that commit's
   other in-development bugs (a `NoneType` subscript in the demo), but on the
   complete Nov-5 base / current code the None path is fully implemented and should
   work. **This is the quickest way to un-regress your L1-KV benchmarks.** (Not yet
   validated end-to-end — pending one run on the Nov-5 base.)

2. **Add / use a cheap on-device argmax shortcut for temp=0.** The expensive part
   is the top-k/top-p selection. When `temperature==0`, the sampling op should
   short-circuit to a single argmax reduction instead of running top-k=32 +
   top-p=0.08 over the full vocab. Worth checking whether the current sampling op
   already has such a path and why the demo config (which sets top_k/top_p) doesn't
   hit it.

3. **Optimize the on-device sampling kernel for Blackhole** (the real upstream
   fix): parallelize the full-vocab top-k/top-p reduction across the Tensix grid.

4. **For L1-KV benchmarking specifically:** baseline against `ca3c90bbb73` or the
   v0.64 line (both ~35), or disable on-device sampling, so the L1-vs-DRAM KV
   comparison is not confounded by the sampling regression. Any prior L1-KV decode
   numbers (~21 t/s/u) were measured against this regressed baseline and should be
   re-interpreted: the L1-KV work is at parity with a *regressed* DRAM baseline,
   not with the true ~35 t/s/u capability.

5. **File upstream.** The regression is still present on current `main` (~22
   t/s/u), so it was never fixed. Report against #31046: "BH P150 Llama-3.1-8B
   batch-1 decode regressed 35 -> 21 t/s/u; on-device sampling op dominates
   per-token latency for greedy decode."

---

## 7. Open items (to reach 100% certainty on mechanism)

- Validate workaround #1 (`sampling_params=None`) on the complete Nov-5 base code
  -> expect ~35.
- Get a per-op device time for the sampling op (needs a working profiler path on
  BH; the standard `process_device_log.py` is broken here — see the existing
  `DEVICE_PROFILING_LIMITATIONS.md` / `PROFILER_BUFFER_OVERFLOW_FIX.md` notes).
- Confirm whether a temp=0 argmax fast-path exists in the current sampling op and
  why the demo's `{temp:0, top_k:32, top_p:0.08}` config does not take it.

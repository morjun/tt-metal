# Plan: census every parameter, make the fused path observable, record provenance, then multi-hypothesis drafting

> **Stage labels, expanded** (they are local names, not standard terms):
> **A0** archival task 0 = the provenance record (`MEASUREMENT_RECORD.md`); **A1** = re-measure anything
> tagged with two configs; **A2** = the acceptance histogram of `m`; **Stage P** = *Path*, make the
> spec-decode route observable then re-measure — **P1** code changes, **P2** the three-arm matrix, **P3**
> maximise the winning route; **Stage B0** = the L1 budget as a design variable; **Stage B** = the
> weight-pinning knapsack; **Stage C** = multi-hypothesis drafting. **PLI** = per-layer inputs, **CME** =
> Centroid Masked Embedding, **`m`** = accepted draft count, **`K`** = `draft_len`.
>
> **Config tags renamed 2026-08-20** to `CFG-<harness>-<route>-tp<N>-b<batch>`. `CFG-E2E2` read as
> "tensor-parallel 2", which it never was — every E2E config here is **tp=1, batch=1**. Mapping:
> `CFG-E2E1`→`CFG-E2E-host-tp1-b1`, `CFG-E2E2`→`CFG-E2E-fused-tp1-b1`, `CFG-STD1`→`CFG-DRAFT-tp1`,
> `CFG-STD2`→`CFG-DRAFT-tp2` (the only tp=2 config), `CFG-PLAIN`→`CFG-PLAIN-tp1-b1`.

Sixth revision. Rev 5 was audited claim-by-claim against source; most of it held, but **its central
premise did not** and is corrected here (see Context §2). Otherwise unchanged in intent: **a complete
parameter census**, **on-device PLI default-on**, the provenance document, tp=1 preferred, op-level fixes
deferred as engineering, and Stage C built around the multi-hypothesis drafting idea.

> **Audit corrections applied in rev 6.** `GEMMA4_SPEC_FUSED_PLI_DEV` was **never inert** — rev 5's
> Stage P was built on that false claim and its one-line "gate fix" was a regression. The real defect is
> reporting, not routing. Also corrected: four wrong line citations, the census count (145 → 144), Stage
> C's EV table and core budget (both self-contradictory), and A0's target repo. Every other citation in
> this plan was verified against source and stands.

---

## WHERE WE ARE (2026-08-21)

| stage | state | evidence |
|---|---|---|
| **A0** provenance record | **DONE** | `MEASUREMENT_RECORD.md`, ~1300 lines, `lab-meeting-notes@be2e174` |
| **P1** route + metric + PLI defaults | **DONE** | `6806855687d` (routes/defaults), `78273b48375` (host-loop steady timer) |
| **P2** three-route matrix | **DONE** | plain 25.61 / host-loop 22.83 / **fused-batched 67.78** tok/s/u, canonical prompt, 3 interleaved rounds. §6.1 |
| **P3** maximise the fused route | **NOT STARTED** | the drafter-scope tuner re-test and the fused op census are both still open |
| **A1** disambiguate two-config numbers | **PARTIAL** | the host-loop phase split is measured (§6.7: draft 43.09 / verify 48.06 / accept 6.34 ms per 98.1 ms iter). The `d(ms/iter)/d(draft_len)` sweep is **not** done |
| **A2** acceptance histogram of `m` | **NOT STARTED** | only the mean exists (1.19/3 on the canonical prompt) |
| **B0/B/C** budget, knapsack, multi-hypothesis | **NOT STARTED** | and B's premise just changed — see below |

**Unplanned work that landed anyway, because measuring forced it:** the specdec instrumentation port
(`8bf8be6d1d1`), the L1-pinning budget sweep on the fused route (§6.4 — 2/4 MB run, 8 MB+ crashes, and the
ceiling is **not** route-specific), the RMSNorm path-cost crossover (§6.5), the tuner's acceptance cost
(§6.6), and a code guard refusing the two-alternating-traces hang (`885bbbae681`) after it wedged both
boards.

### The one premise change that matters for Stage B

**RETRACTED: "the drafter's negative result is bounded by exposure and is final."** The
bandwidth-exposure model gives a ~3.4-5 us/step ceiling; **measured 5-class composed pinning is
+9.9 us/step**, roughly 3x above it. An observation above a ceiling means the ceiling is not binding, so
a second and currently unidentified variable is present (locality/latency, NoC contention relief in a
composed step, `IN1_SHARDED=1` changing program structure, transient address map).

What survives is the *measurement*: drafter L1 pinning is **e2e 0%** (20 paired rounds, CI ±0.42%). What
does not survive is the *explanation* and the *ceiling*.

**Consequences for this plan:**

1. **Stage B's drafter branch is reopened**, not closed. It was going to be dismissed on the exposure
   argument; that argument is void.
2. **A new blocking item, ahead of B0**: profile the +9.9 us (eager execution + `ttnn.graph`; traced ops
   cannot be attributed with tracy). Until it is attributed, no drafter-side placement ceiling is
   quotable — including a negative one.
3. **The "target is ~7x the drafter's prize" comparison is provisional**, since its denominator is the
   retracted drafter estimate.
4. Stage B's DP-vs-budget curve still costs no device time and is still the right way to rank B against
   C — but its drafter rows now carry an explicit unknown rather than a settled zero.


## Context — two provenance failures, and the second was misdiagnosed

**1. A capacity number with no source.** You asked where `~448 KB/core` came from. Tracing it exposed
exactly the hazard you predicted: **the docs contain four different capacity numbers and I used one of them
without saying which.**

| number | what it actually is | source | usable today? |
|--:|---|---|---|
| **320 KB/core** | LOCKSTEP allowance at the **default** SDPA cap (16) | `test_per_bank_headroom`; model `1496 − 782 (CB high-water) − ~500 (transient)` | **YES — this is the real budget** |
| **448 KB/core** | LOCKSTEP allowance at `GEMMA4_SDPA_MAX_CORES=8`; costs +0.6% latency at ctx2048, +2.9% at 8192 | `test_per_bank_headroom`, 64 KB scan granularity, matches the predicted +144 KB | YES, only if that knob is set |
| **~448 KB** | *coincidence* — the **computed cost** of the 7.75 MB the drafter actually pinned, flagged in-doc as an open inconsistency because it exceeds the 320 KB probe | ASSISTANT_L1_WEIGHTS §6.2 note | **not a budget at all** |
| **1024 KB** | physically free L1 on the **78 quiet cores**, default grids | §6.1 right-sizing table | **NO under LOCKSTEP** |
| **1323 KB** | same, with both SDPA grids right-sized to 4x4 | same table | **NO under LOCKSTEP** — this is the HYBRID/tiered prize |

**Your ~1 MB recollection is right** — it is the 1024/1323 KB row, physically free L1 on the cores that
carry only the 472 KB sliding-SDPA stack rather than the 782 KB global one. Under LOCKSTEP it is
unreachable, because the budget is `top − max_over_programs(CB high-water)`, i.e. set by the **worst**
core and charged to all 110. That gap — ~24 MB/device — is precisely what HYBRID + per-core allocation
would unlock, and it is why the tiered idea is the only thing that raises the ceiling.

**And I had used the wrong one.** My target-feasibility table assumed 448 KB. At the **default 320 KB**:

| shape (tp=1, per-device) | KB/core | fits 320? | fits 448? |
|---|--:|:--:|:--:|
| gate/up wide 1536x12288 | 576 | no | no |
| down wide 12288x1536 | 768 | no | no |
| wqkv global 1536x5120 | 384 | **no** | yes |
| down narrow 6144x1536 | 384 | **no** | yes |
| gate/up narrow 1536x6144 | 288 | yes | yes |
| o_proj global 4096x1536 | 256 | yes | yes |
| wqkv sliding 1536x2560 | 192 | yes | yes |
| o_proj sliding 2048x1536 | 128 | yes | yes |

So at tp=1 and the default cap only **4 of 8** target shapes are individually feasible, and the two
highest-value ones are not — worse than I said. Two more numbers move with the budget choice, which is
the whole argument for writing the provenance record before any more analysis is layered on top.

**2. A run whose path and whose metric are both unrecorded.** Your `path=host` question is still the more
expensive of the two findings, but rev 5's diagnosis of it was wrong and is retracted here.

**Retracted: "`GEMMA4_SPEC_FUSED_PLI_DEV=1` was inert."** It is false. The flag is read in **two** places,
not one. Rev 5 checked only `spec_decode.py:1215` (inside `generate_fused`) and concluded the flag could
never fire. It is also read at `spec_decode.py:1966`, inside `generate()` — the host-loop entry the demo
actually calls:

```
generate()                                                            spec_decode.py:1965
  if self._use_trace:
    if greedy and self.target_needs_host_pli and self._fused_pli_device:      # :1966
        self.target.init_pli_device_weights()                                 # :1978 (already hoisted)
        return generate_batched(...) -> _generate_fused_traced_batched        # :1994
                                     -> _fused_body_batched                   # :1616
                                     -> ttnn_packed_verify_forward            # :1665
```

So with the flag set, E2B has always run the **fused batched single trace through the packed verify** —
precisely the configuration rev 5 proposed to "unblock". The doc rev 5 cited says so directly: the 31.77
tok/s/u run (`SPECDECODE_E2B_TRACE_AND_VERIFY.md:240-252`) was invoked with `GEMMA4_SPEC_FUSED_PLI_DEV=1`
and **no** `GEMMA4_SPEC_FUSED=1`, and the text reads *"`GEMMA4_SPEC_FUSED_PLI_DEV=1` is the switch. Drop
it and you get the host loop (~18.4 tok/s/u)."*

**What is actually broken is reporting, and it is subtler and worse.** `generate()` reroutes silently, so:

1. **The `path=` line is blind to it.** `text_demo_v2.py:812-820` prints the *demo-level* `use_fused`, so
   a run executing the fused batched trace still logs `path=host, trace=True`. That is the line rev 5 read
   and misread.
2. **The throughput metric silently changes meaning.** `_generate_fused_traced_batched` does set
   `_last_fused_setup_s` (`:1769`) and `_last_fused_replay_s` (`:1873`), but the demo reads them only
   `if use_fused` (`text_demo_v2.py:844-845`). On the reroute `use_fused` is False, so
   `steady_elapsed = elapsed` — **wall time including the 4.38 GiB PLI upload and trace capture**, while
   every other fused number on record is steady-state. With 24 tokens reading 1.65 tok/s/u and 500 reading
   32.0, that is not a rounding difference.

Rev 5's *conclusion* survives, for a sharper reason: **whether this session's 52.14 tok/s/u is a host-loop
steady-state number or a fused-batched wall-clock number is decided entirely by whether
`GEMMA4_SPEC_FUSED_PLI_DEV` was exported — and neither the log line nor the metric records it.** The
ranking of the two paths is unknown, and so is which path the baseline itself was measured on.

> **ANSWERED by P2 (below).** It was the **fused batched path on the wall-clock metric**. The host loop
> measures 20.72 tok/s/u steady and cannot reach 52; the fused path's wall figure measures 48.0-48.4 in
> this config, which is the same quantity 52.14 was. Steady-state, that path is **62.46**.

Hence the two additions to this revision: **an exhaustive, mechanically-generated census of all 151
parameters** so no knob is ever again assumed-set-but-inert (or, as here, assumed-inert-but-set), and
**on-device PLI default-on with the effective route and its metric actually logged**, so the fused path is
identifiable in the record, measured, and then maximised.

---

## A0 — `~/codes/lab-meeting-notes/documents/dflash/MEASUREMENT_RECORD.md` (do this first)

> **Repo note.** `documents/dflash/` is **not** in the tt-metal worktree — it lives in the separate
> `lab-meeting-notes` repo. Every other path in this plan is tt-metal-relative; these are not.

A single document whose only job is: **every number, its source, its configuration, and its status.** No
argument, no narrative — those live in the existing reports. Sections:

0. **The complete parameter census** — every knob a user can set, none omitted. Enumerated
   mechanically, not from memory, so the list is provably complete:

   ```bash
   # env vars: 151 in the union of the two branches
   grep -rhoE '(os\.environ\.get|os\.getenv)\(\s*"[A-Z0-9_]+"|os\.environ\[\s*"[A-Z0-9_]+"' \
       models/demos/gemma4 | grep -oE '"[A-Z0-9_]+"' | tr -d '"' | sort -u
   # pytest CLI options
   grep -rn "addoption" models/demos/gemma4 conftest.py
   ```

   Counts found: **127 on `gemma4-specdec`, 144 on `gemma4-assistant-l1-weights`, 151 in the union.**
   The 24 l1w-only ones are the L1/tuning work (`GEMMA4_TUNE_MATMULS`, `GEMMA4_WEIGHTS_IN_L1`,
   `GEMMA4_L1_*`, `GEMMA4_SDPA_MAX_CORES`, `GEMMA4_GATHER_IN0`, `GEMMA4_CME_*`, `GEMMA4_PRECISION`, …).
   The 7 specdec-only ones (`GEMMA4_ROW2_*`, `GEMMA4_ITER_LOG`, `GEMMA4_REPLAY_SNAP*`) came from
   **uncommitted** row-2 instrumentation in that worktree. **Done** — ported and committed in
   `8bf8be6d1d1`, so l1w now reads 151 and the specdec-only set is empty; the census is reproducible from
   this branch alone. (`gemma4-specdec`'s HEAD is the merge-base of this branch, so there was never
   anything else to bring.) That commit also carried a real `GEMMA4_STAGE_DIFF` bug fix: the reference
   chain was picked by `flat.shape[0] == 1`, which misfiles plain decode for every head-width tensor —
   any earlier stage-diff output for those tags is void.

   Each row carries: **name, read site (`file:line`), type/legal values, current default, scope
   (model / spec-decode / L1-placement / prefill / test-only / debug), effect, and status** — one of
   `production` (a real deployment knob), `experiment` (used to produce a recorded number),
   `debug` (instrumentation), or `dead` (no longer wired to anything). Plus the non-env parameters:
   pytest options (`--speculative`, `--spec-draft-len`, `--skip-model-load`, and the root `conftest.py`
   set), the `@pytest.mark.parametrize` sampling dicts (`temperature`, `top_p`, `top_k`) at
   `text_demo_v2.py:180-320`, and the constructor arguments (`draft_len`, `max_seq_len`, `batch_size`,
   `matmul_tuner`, `weight_placement`).

   **Every default that this plan changes is called out explicitly in its own subsection** (the three
   PLI knobs, and `GEMMA4_TUNE_MATMULS` which already flipped in `bcfda5d0861`), with the before/after
   measurement beside it.

1. **Configurations.** The named CFG rows (below), each with mesh shape, device count, tp, harness, and
   every fixed parameter. Nothing in the record may cite a number without a CFG tag.
2. **The performance trajectory**, original tt-metal → today, one row per change, each with: what changed,
   the measured before/after, the config, the commit, and whether it is landed/default-on. This is the
   "big picture" table — from the inherited 3.7 tok/s/u through the CME digit gather, the second gather
   (197x), the unpadded argmax (17.9x), activation chaining (−9.1% backbone), the matmul tuner, the fused
   trace, to today's 52.14 tok/s/u.
3. **Capacity numbers**, exactly the table above — the four-way disambiguation, so `448` is never quoted
   bare again.
4. **Measured constants**: DRAM bandwidth 444.7 GB/s peak (and the 512 spec / 717-implied figures marked
   withdrawn), per-op floor ~5.7 us, exposed fraction 0-17%/63%, per-core charge formula
   `bytes / num_shard_cores`.
5. **Results**, each with config and status: pinning (drafter scope +0.88%, e2e 0%), target tuning
   (+6.5-7.6%), the op censuses, acceptance figures.
6. **Known-ambiguous / needs re-measurement** — an explicit list, so nothing ambiguous is silently reused.
7. **Instrument defects**, since they invalidate numbers: the profiler clock skew (537/885 rows, column
   x=11), the two-copies-of-448 confusion, the `GEMMA4_GATHER_IN0` truthiness bug, the mixed
   `ms/iter` vs `ms/step` units.

**Rule the document establishes:** a number without a config tag and a source is not citable.

---

## Configurations

| name | mesh | devices | tp | harness | fixed params |
|---|---|---|--:|---|---|
| **CFG-E2E-host-tp1-b1** | 1x1 | `TT_VISIBLE_DEVICES=0`, `MESH_DEVICE=P150` | 1 | `text_demo_v2.py -k test_demo_spec_decode` | real target + drafter + packed verify, `draft_len=3`, `max_seq_len=1024`, 500 tokens, greedy, one fixed prompt. **Selector: `GEMMA4_SPEC_FUSED_PLI_DEV=0`** — the traced host loop, packed verify with host PLI. The session's 52.14 tok/s/u is *claimed* to be this config; P2 must confirm it, since the flag's value at the time was not recorded (Context §2) |
| **CFG-E2E-fused-tp1-b1** | 1x1 | same | 1 | same | identical to CFG-E2E-host-tp1-b1 except **`GEMMA4_SPEC_FUSED_PLI_DEV=1`**: `generate()` reroutes to the fused batched single trace with on-device PLI (`spec_decode.py:1966`). Reachable **today** — the default flip only makes it the default, and the route/metric fix makes it identifiable. Every re-measurement below is repeated in it. **Not** selected by `GEMMA4_SPEC_FUSED=1`, which picks the wrong body for E2B (see P1) |
| **CFG-PLAIN-tp1-b1** | 1x1 | same | 1 | `text_demo_v2.py -k test_demo` (no `--speculative`) | plain decode, same prompt / token count / `max_seq_len`. The vanilla denominator for every speedup claim; device PLI on, so it matches the spec paths' PLI implementation |
| **CFG-DRAFT-tp1** | 1x1 | `TT_VISIBLE_DEVICES=0` | 1 | `test_assistant_standalone_l1.py -k 1x1` | drafter alone on `_TargetStub` |
| **CFG-DRAFT-tp2** | 1x2 | `TT_VISIBLE_DEVICES=0,1` | 2 | same file, `-k 1x2` | drafter alone, ctx512, K=3, 50 reps, `GEMMA4_TUNE_MATMULS=1` |
| **CFG-PV** | factory default | per shape | per shape | `test_packed_verify.py` | `trace_region_size=256M`, `draft_len = 4 x mesh_width − 1` (`:188`) |

### `trace` and `fused` are two different knobs — and `path=` reports neither reliably

You asked what `path=host` meant given the trace is on. Both were true; they are independent knobs. But
the log line is also **not authoritative about which iteration structure ran**, which is what rev 5 got
wrong. Verified from source, not from the log:

- **Tracing was ON.** `GEMMA4_SPEC_TRACE=1` → `trace=True`. The drafter and the verify each replay metal
  traces off persistent I/O buffers, so CFG-E2E-host-tp1-b1 was not host-dispatch bound. `path=host` was *intended*
  to describe the **iteration structure** — drafts round-trip to the CPU each iteration so PLI can be
  rebuilt from the token ids — not the absence of tracing.
- **But `path=` prints the demo-level `use_fused` only** (`text_demo_v2.py:812-820`). The demo's gate
  (`:796-805`) consults `spec.target_needs_host_pli` and sets `use_fused = False` for E2B — and then
  `generate()` reroutes to the fused batched trace anyway when `_fused_pli_device` is set
  (`spec_decode.py:1966`). So `path=host` is printed in **both** cases and distinguishes nothing for a PLI
  target. The flag is read at `:1215` *and* at `:1966`; rev 5 saw only the first.

**The last recorded fused measurement is 31.77 tok/s/u against 18.4 for the host loop — 1.73x**
(`SPECDECODE_E2B_TRACE_AND_VERIFY.md:240-252`, which calls that flag "the single most useful A/B here").
Note the command recorded there sets `GEMMA4_SPEC_FUSED_PLI_DEV=1` and *not* `GEMMA4_SPEC_FUSED=1` — which
is the direct evidence that the flag alone has always selected the fused batched path.

**Both figures are stale, in the same direction.** That 31.77/18.4 pair predates the 33 commits now on
`gemma4-assistant-l1-weights` (CME digit gather 2.57x, the second gather 197x, unpadded argmax 5.7x,
activation chaining −9.1% backbone, target matmul tuning +6.5-7.6%). Those carried the **host** loop from
18.4 to 52.14 tok/s/u. The fused path has been re-measured against exactly **one** of them (the CME digit
fix: 32.07 → 41.33). So the fused path's present speed is **unknown**; and because the metric itself
switches from steady-state to wall-clock on the silent reroute (Context §2), so is the question of which
path the 52.14 was measured on. That is now the first measurement, not a footnote.

### Consequence: on-device PLI goes default-on and the effective route becomes recorded

Per instruction. **Three** distinct PLI knobs exist, all three default off today:

| knob | site | what it does | flip to |
|---|---|---|---|
| `GEMMA4_SPEC_FUSED_PLI_DEV` | set `spec_decode.py:159`; read `:1215` **and `:1966`** | PLI on device inside the fused trace. At `:1966` it also *selects* the fused batched path for a PLI target — this is the CFG-E2E-host-tp1-b1/E2E2 selector | **on** |
| `GEMMA4_SPEC_PLI_DEV` | `spec_decode.py:162` | on-device PLI in the **host-loop** packed verify | **on** |
| `GEMMA4_DECODE_PLI_DEV` | **`model.py:2154`** | on-device PLI in **plain decode** — the parity route, and "slightly faster" (`7d5f31821c2`) | **on** |

All three must flip together, because device PLI is **not bit-identical to host PLI**: PCC 0.9999947,
max|diff| = 0.0625 = one bf16 ULP, because the device runs the projection in bf16 where the host uses fp32
(**`model.py:727-729`**). Flipping only the spec side would silently compare two different PLI implementations
in every spec-vs-plain number. Flipping all three restores parity *and* is the faster configuration. The
fp32 alternative is already measured and rejected: PCC 0.99999774 but **31.77 → 20.30 tok/s/u**.

Two costs to record rather than hide:
- The **4.38 GiB** `embed_tokens_per_layer` table ([262144, 35x256] bf16) becomes an unconditional DRAM
  allocation for E2B. It was *assumed* to be a blocker and measured not to be on a 32 GB p150a
  (**`model.py:696-699`**) — but it now competes with the KV cache, so **DRAM headroom at `max_seq_len`
  1024 and 4096 must be measured, not assumed** (**`model.py:1181`** already flags ~2 GiB of KV at 4096).
- Upload + trace capture amortise over the run: 24 tokens reads 1.65 tok/s/u, 500 reads 32.0. **No perf
  claim below 400 generated tokens** (`SPECDECODE_E2B_TRACE_AND_VERIFY.md:277`).

Also required, or the flip fails loudly: `init_pli_device_weights` is **lazy**, and a 4.38 GiB host write
*during* trace capture is illegal and "fails silently rather than loudly" (**`spec_decode.py:1974-1978`**).
The batched fused path already hoists it at `:1978`; with `GEMMA4_SPEC_PLI_DEV` and `GEMMA4_DECODE_PLI_DEV`
on, only the host-loop and plain-decode capture sites still need the same hoist.

**Open correctness item, stated plainly.** The fused path still diverges from plain decode: first differing
KV position **128** (a paged block boundary), visible token divergence ~135 (`SPECDECODE_ROOT_CAUSE.md:36,38`).
It is a **wrong verify row of 1-ULP origin, not a dropped token** (`1b768fd79c0`); it traces to the packed
verify passing no `cur_pos` to SDPA, and the fix is blocked upstream (`is_causal=False` ignores `cur_pos`,
`is_causal=True` forbids an explicit mask). So defaulting to fused ships a path that is numerically
self-consistent but **not bit-identical to plain decode past ~128 positions**. It does not block performance
measurement, but it is quoted next to every fused number.

**Stage C is unaffected either way.** Multi-hypothesis drafting needs a host-side "which drafter guessed `m`
correctly" decision each iteration, which the host loop already provides. If fused wins on speed, Stage C
inherits fused-vs-host as an open design question instead of a settled one.

`tp = mesh_device.shape[1] if num_devices > 1 else 1` (`test_assistant_standalone_l1.py:193`) — **tp=1 is
one device**; tp=2 is a 1x2 mesh with weights *tensor*-split (`tp_axis=1`), hence halved per-device N and
the `ccl_allreduce` after `o_proj`/`down_proj`. Not data parallelism (`MeshConfig` carries `dp`
separately, `config.py:35-47`).

**tp is a fixed deployment hyperparameter, not a search dimension.** Given a mesh, the search space is the
weight subset alone. **Prefer CFG-E2E-host-tp1-b1 (tp=1) for everything; use a tp=2 config only where tp=1 is
infeasible**, and say so explicitly when you do. Legal `draft_len` is `{3,7,11}` at tp=1, `{7,15,23}` at
tp=2 (`(H_local x P) % 32 == 0`, `attention/decode.py:647`).

---

## Stage P — flip on-device PLI to default, make the effective route and its metric observable, and re-measure

> **P1 LANDED in `6806855687d`.** The line numbers quoted in this stage are the **pre-P1** ones, which is
> what makes the diff readable. Post-P1 they are: `GEMMA4_SPEC_FUSED_PLI_DEV` `spec_decode.py:168`,
> `GEMMA4_SPEC_PLI_DEV` `:174`, the `generate()` reroute `:2047`, the host-loop PLI hoist `:2149`,
> `generate_fused`'s guard `:1233`, and `GEMMA4_DECODE_PLI_DEV` `model.py:2158`.

Runs **with** A0, because A0 cannot record a trustworthy performance trajectory while neither the route a
run took nor the meaning of its throughput number is recorded.

### P1 — the code changes (small, and all in existing code)

| file | change |
|---|---|
| `demo/text_demo_v2.py:812-820` | **log the effective route.** Have `generate()` set `spec._last_route` (`"fused-batched"` / `"host-loop"`) at the point the decision is actually made, and print *that*, not the demo-level `use_fused`. Today the demo prints `path=host` for both, so a PLI target rerouted at `spec_decode.py:1966` is indistinguishable from a genuine host loop |
| `demo/text_demo_v2.py:844-845` | **fix the metric attribution.** Gate `setup_elapsed`/`steady_elapsed` on `_last_fused_setup_s` having been *set this run*, not on `use_fused`. `_generate_fused_traced_batched` already sets it (`:1769`, `:1873`), but the demo ignores it on the reroute and reports **wall** time including the 4.38 GiB upload and capture, where every other fused number is steady-state |
| `tt/spec_decode.py:159,162` | `GEMMA4_SPEC_FUSED_PLI_DEV` and `GEMMA4_SPEC_PLI_DEV` default `"0"` → `"1"`. This is now the *whole* of what makes fused the default — no demo-gate change is needed or wanted |
| `tt/model.py:2154` | `GEMMA4_DECODE_PLI_DEV` default on (keep `and self.hidden_size_per_layer_input`, so non-PLI checkpoints are untouched) |
| `tt/spec_decode.py`, `tt/model.py` | hoist `init_pli_device_weights()` ahead of the **host-loop and plain-decode** capture sites — a lazy 4.38 GiB write inside a capture fails silently. `:1978` already covers the batched fused path |

**Do not touch the demo gate at `:799`.** Rev 5 proposed adding
`and not getattr(spec, "_fused_pli_device", False)` there to force `use_fused=True` for E2B. That is a
**regression**: the demo would then call `spec.generate_fused()` (`:825`) → `_generate_fused_traced`
(`:1388`) → `_fused_body` (`:1278`) → `ttnn_verify_forward` (`model.py:1381`) — the **batch-dim** verify,
which re-loads KV per candidate and has no `pli_on_device` parameter at all. Only `_fused_body_batched`
routes through `ttnn_packed_verify_forward`, which is exactly why `generate()` sends PLI targets there
instead (`spec_decode.py:1211-1213`, `:1966`).

`GEMMA4_SPEC_FUSED=1/0` stays as an explicit override, but it is **not** the E2B fused selector — for a
PLI target it forces the wrong body, per the paragraph above. Each of the three PLI knobs stays
individually settable to `0` — the flip changes defaults, it does not remove control.

### P2 — the re-measurement matrix (this is the deliverable)

Three paths in **one** code state, one prompt, one token count, ≥400 tokens, run 1 discarded, interleaved
arms, medians reported — and **the new `route=` line asserted per arm** rather than the environment
trusted, which is the whole point of P1:

| arm | selector | what it answers |
|---|---|---|
| plain decode | CFG-PLAIN-tp1-b1, `-k "test_demo and not spec"` | the vanilla denominator |
| traced host loop | CFG-E2E-host-tp1-b1, **`GEMMA4_SPEC_FUSED_PLI_DEV=0`** | is 52.14 tok/s/u actually this path? The session did not record the flag, so this is a re-establishment, not a reproduction |
| **fused batched single trace** | CFG-E2E-fused-tp1-b1, **`GEMMA4_SPEC_FUSED_PLI_DEV=1`** | the unknown — last seen at 31.77 (pre-fix) and 41.33 (CME fix only), and last measured on a metric that may have been wall-clock |

`GEMMA4_SPEC_FUSED=1` is **not** the CFG-E2E-fused-tp1-b1 selector; for a PLI target it forces the batch-dim verify
(see P1). The selector is `GEMMA4_SPEC_FUSED_PLI_DEV`, which is what the recorded 31.77 run used.

### P2 RESULT — measured 2026-08-19 on `78273b48375`

500 generated tokens, one fixed prompt in all arms, `max_seq_len=1024`, `draft_len=3`, greedy, tp=1,
`TT_VISIBLE_DEVICES=0`, `MESH_DEVICE=P150`, `GEMMA4_SPEC_TRACE=1`. Three interleaved rounds, order reversed
per round, **median** reported, every quoted run at `JIT cache stats: 100.0%`. All steady-state.

| arm | route (asserted, not assumed) | tok/s/u | ms/token | mean `m`/K | vs plain |
|---|---|--:|--:|--:|--:|
| CFG-PLAIN-tp1-b1 | plain decode, batch-1 | **26.15** | 38.24 | — | 1.00x |
| CFG-E2E-host-tp1-b1 | `host-loop-traced` | **20.72** | 48.26 | 1.02/3 | **0.79x** |
| CFG-E2E-fused-tp1-b1 | `fused-batched-traced` | **62.46** | 16.01 | 1.02/3 | **2.39x** |

Runs: E2E2 62.70 / 62.38 / 62.46 (spread 0.5%); E2E1 20.72 / 21.32 / 20.42; plain 26.24 / 25.84 / 23.78.

**Three findings, in order of consequence.**

1. **Fused beats host by 3.01x**, not the historical 1.73x — and the host loop is **slower than plain
   decode** (0.79x). Speculation on the host loop is a net loss at this acceptance rate; only the fused
   batched path pays. Acceptance is identical on both (1.02/3), so this is purely iteration structure.
2. **The 52.14 tok/s/u baseline was almost certainly never the host loop.** The host loop cannot reach it
   — it tops out at ~20.7 here. What the *old* code printed for a rerouted E2B run is the wall-clock
   figure, and that measures **48.0 / 48.4 / 48.2** in this config. 52.14 is that quantity on a different
   prompt, not a host-loop number. So the baseline was the fused batched path all along, reported on the
   wrong metric — which is exactly the failure mode Context §2 describes, now confirmed by measurement
   rather than inferred.
3. **The mean acceptance here is 1.02/3, not 1.20/3.** Prompt-dependent, and it lowers the chain-model
   `p` to ~0.55. A2's histogram must be measured on the same prompt as any Stage C decision.

Cost side, also measured: fused setup (4.38 GiB PLI upload + trace capture) is **1.48-1.53 s**, amortised
over ~8 s of steady decode at 500 tokens — so the ">=400 tokens" rule holds and is not conservative. The
traced host loop's lazy verify capture costs **~1.0 s**, visible as a first iteration of 1102 ms against
92.8 ms/iter for the rest (11.9x).

**Stop-gate: not triggered.** CFG-E2E-fused-tp1-b1 wins decisively, so the flipped defaults stand.

Report tok/s/u, ms/token, mean accepted `m`/K, and the spec-vs-plain speedup for both spec paths. Also
measure, and record in A0: **DRAM headroom with the 4.38 GiB table resident** at `max_seq_len` 1024 and
4096, and the **fixed setup cost** (PLI upload + capture) separately from steady-state throughput, since
short runs otherwise misattribute it — and since, pre-P1, the reroute silently folded it into the headline
number.

### P3 — then maximise the fused path

Only after P2 says where it stands, and in this order:

1. **Re-point the L1/tuning work at it.** `GEMMA4_TUNE_MATMULS` currently defaults to the **target** scope
   only (`DEFAULT_SCOPES = frozenset({"target"})`); the fused iteration runs the drafter chain in-graph, so
   the drafter scope is worth re-testing there even though it was ~0% on the host loop.
2. **Census the fused trace's ops** the way the drafter step was censused — the fused body is one program,
   so a single tracy capture attributes the whole iteration, which the host loop could never give us.
3. **Then** the deferred op fixes become re-rankable against real fused numbers (TopK at 336 µs on one core
   is 30% of a *drafter* step; its share of a fused iteration is unmeasured).

**Stop-gate.** If CFG-E2E-fused-tp1-b1 is slower than CFG-E2E-host-tp1-b1 in the current code state, record that with the numbers
and keep CFG-E2E-host-tp1-b1 as the measurement baseline — but keep the route logging and the metric fix regardless
of which path wins, because an unrecorded route and a silently-switching metric are bugs either way.
Whether the *defaults* stay flipped is then a judgement call the numbers decide.

---

## A1 — re-measure everything whose configuration is ambiguous

Anything tagged with two configs is not a measurement. The known cases:

| quantity | current status | fix |
|---|---|---|
| **drafter share of the iteration** | "~8.1%" from a **CFG-DRAFT-tp2** step time (3.385 ms/iter) divided by a **CFG-E2E-host-tp1-b1** iteration (41.72 ms). Two configs, so it is **not a measurement**. My earlier "21%" was a worse inference from the same muddle. | Measure in **CFG-E2E-host-tp1-b1 only**: `d(ms/iter)/d(draft_len)` over `{3,7,11}`. Three points give the slope and its linearity |
| **acceptance distribution** | only the **mean** is recorded (1.20/3, CFG-E2E-host-tp1-b1) | log the **per-iteration histogram of `m`** — required by Stage C, see below |
| **exposed fraction** | 0-17% drafter / 63% target, from isolated single matmuls at **CFG-DRAFT-tp1** | fine as-is, but record that it is isolated-op, not in-model |
| **the +9.9 us cause** | real and reproducible (CFG-DRAFT-tp2, 11x its floor) but **unattributed** | discriminators, then tracy on the 14 affected calls (707 ns each) |

Everything else in the record must carry one CFG tag or move to the needs-re-measurement list.

**A1 and A2 run in whichever path P2 declares fastest, and are reported for both.** The drafter share in
particular is *structurally* different between them: on the host loop the K draft steps are K separate
traced dispatches with a host round-trip each, while in the fused iteration they chain in-graph. So
`d(ms/iter)/d(draft_len)` is a different quantity on each path, and quoting one number for "the drafter
share" without the path is the same category of error as the CFG-DRAFT-tp2/CFG-E2E-host-tp1-b1 mixture it replaces.

---

## A2 — the acceptance histogram of `m` (prerequisite for Stage C, near-free)

We have only `E[m] = 1.20/3`. Stage C's design depends on the **distribution**, and the demo already
computes accepted counts per iteration — this is a logging change plus one CFG-E2E-host-tp1-b1 run.

Why it matters, from a chain model fitted to `E[m] = p + p² + p³ = 1.20` → `p ≈ 0.607`:

| m | modelled P(m) | cumulative |
|--:|--:|--:|
| 0 | **39.3%** | 39.3% |
| 1 | 23.9% | 63.2% |
| 2 | 14.4% | 77.6% |
| 3 | **22.4%** | 100% |

**The distribution is bimodal** — `m=0` is the single most likely outcome and `m=3` is second. That
matters for your idea (below), and it is a *model*, not a measurement, which is exactly why the histogram
must be measured before any hypothesis set is chosen.

---

## Stage C — multi-hypothesis drafting (your idea), now the primary research direction

**The idea, and why it is better than what I had.** The blocker for async was a triple read-after-write:
draft `i+1` needs verify `i`'s committed token, the hidden **row `m`** (where `m` *is* the accept count),
and the KV writes. My proposed workaround was to draft from a *stale* seed — which risks acceptance
collapse, and the repo records 0.53 live vs 0.98 for fresh-vs-stale seeding, so that risk is
open-ended and could erase the entire gain.

**Your idea removes that risk instead of pricing it.** Run several drafters concurrently, each assuming a
*different* value of `m`, each therefore working from an **exact** seed for its own hypothesis; when the
verify lands, keep the one that guessed right. There is no stale-seed degradation at all — acceptance per
hypothesis is unchanged. The open-ended risk becomes a **bounded coverage cost**, which is measurable.
That is a materially better trade and it is the right basis for the design.

**One correction to the specific hypothesis choice.** You proposed covering `m=1` and `m=2`. Per the model
above that covers only **~38%**, because `m=0` is the mode (~39%) and `m=3` is second (~22%) — the chain
structure makes the distribution bimodal, not centred on the mean of 1.2. Better 2-drafter covers:
`{0,1}` ≈ **63%** or `{0,3}` ≈ **62%**. So the idea is structurally right and the hypothesis set should
be picked from the measured histogram (A2), not from the mean.

**Expected value — every number in this table is PROVISIONAL.** The ceiling "hiding the drafter is worth
~+8.9% (iteration 41.72 → 38.3 ms)" is just `41.72 − 3.385`, and that 3.385 ms is the **CFG-DRAFT-tp2** drafter
step that A1 above declares **"not a measurement"** because it is divided into a CFG-E2E-host-tp1-b1 iteration. So
the whole column inherits a quantity this plan voids two sections earlier. Recompute it from A1's
`d(ms/iter)/d(draft_len)` in CFG-E2E-host-tp1-b1 (and again in CFG-E2E-fused-tp1-b1 — the drafter's structural share differs
between the paths) before any of it is used to rank C against B.

| drafters | best cover | modelled coverage | E[gain] *(provisional)* |
|--:|---|--:|--:|
| 1 (stale seed) | — | 100% but degraded acceptance | 0 to +8.9%, open-ended risk |
| **2** | `{0,1}` | **~63%** | **~+5.6%** |
| 3 | `{0,1,3}` | ~86% | ~+7.6% |

**Core budget decides how many drafters fit — but the naive count is the wrong test.** At ~16 cores per
drafter partition, 2 drafters = 32 cores leaving **78**, and 3 = 48 leaving **62**. Rev 5 compared those
against "the target's largest 64-core decode grid", which contradicts B0.1 below: the **sliding** SDPA
*claims* the full 11x10 = **110** cores (`attention/decode.py:318` defaults its grid to the whole device
grid), so on claimed area *no* drafter fits. The usable argument is **claimed vs active**: SDPA claims 32
or 110 cores and **computes on 16 either way** (`ASSISTANT_L1_WEIGHTS.md:244,255,863`). So the question is
not "do 78 cores exceed a grid size" but "does carving 32 cores away collide with the 16 that are actually
active" — which is exactly what B0.1's right-size-**and-offset** lever is for, and it must be measured,
not asserted. `K=3` with 2 drafters remains the natural design point; its feasibility is now a B0.1
deliverable rather than an assumption.

Work items, in order:

1. **A2's histogram** — pick the hypothesis set from data.
2. **The cheap sub-device validation** — two sub-devices, one CQ, pre-captured traces. Sub-devices give
   concurrency on a **single** CQ (`SubDevices.md:22,97`); do **not** use 2 CQs — putting the drafter on
   its own command queue **hung one iteration earlier**, a correctness failure rather than a slower
   measurement (`spec_decode.py:2085-2086`). Proven in-tree: two programs on two sub-devices
   (`dispatch_program/test_sub_device.cpp:194-229`) and traces with sub-devices
   (`dispatch_trace/test_sub_device.cpp:26-96`).
3. **Partition plumbing** — every gemma4 grid is hardcoded at origin (0,0) and the SDPA grid knobs take a
   *size*, so they shrink but never offset. Partition **once at startup**
   (`load_sub_device_manager` needs empty local allocators, `SubDevices.md:57`).
4. **The KV race** — the drafter reads exactly the DRAM pages the verify writes at `c..c+K`, with zero
   synchronisation today, and DRAM is outside sub-device (L1-only) isolation. Needs a GlobalSemaphore or
   event fence. With N drafters reading concurrently this gets harder, not easier.
5. **Gaps to fill** — paged KV cache ops have **no** sub-grid support; ttnn tensors cannot allocate from a
   sub-device allocator (`SubDevices.md:87`); `worker_cores(type, sub_device_id)` is unbound in Python.

---

## Stage B0 — the budget is a design variable: tiered residency under HYBRID

**The budget must not be taken as 320 or 448 KB.** Those are *LOCKSTEP-max* numbers. The relevant
quantity is `TOP − own_CB_high_water − transient`, which under HYBRID is **per core**:

| regime | CB high-water seen | budget/core | physically free |
|---|--:|--:|--:|
| LOCKSTEP, default (max = SDPA global 782) | 782 | **214-320** | 714 |
| LOCKSTEP, `GEMMA4_SDPA_MAX_CORES=8` (max 638) | 638 | **358-448** | 858 |
| **HYBRID, quiet cores, default grids** (sliding SDPA 472) | 472 | **~524** | 1024 |
| **HYBRID, quiet cores, SDPA right-sized 4x4** (173) | 173 | **~823** | 1323 |

(The 320/448 pair are the measured LOCKSTEP allowances; the 214/358 pair is the same model with the
~500 KB transient. The spread between them *is* the open 320-vs-448 inconsistency, which is why the
transient term must be measured — see B0.2.)

**Why this changes the problem qualitatively, not just quantitatively.** At 320 or 448 KB the two
highest-value target weights are simply infeasible. At ~823 KB both become placeable **for the first
time**:

| target weight (tp=1) | KB/core | 320 | 448 | 524 | **823** | us saved |
|---|--:|:--:|:--:|:--:|:--:|--:|
| gate/up wide 1536x12288 | 576 | — | — | — | **Y** | **90.4** |
| down wide 12288x1536 | 768 | — | — | — | **Y** | **57.4** |
| wqkv global 1536x5120 | 384 | — | Y | Y | Y | 38.8 |
| down narrow 6144x1536 | 384 | — | Y | Y | Y | 28.6 |
| gate/up narrow 1536x6144 | 288 | Y | Y | Y | Y | 43.4 |
| o_proj global 4096x1536 | 256 | Y | Y | Y | Y | 19.0 |

So the knapsack's **item set is a function of the budget**, and the budget is something we can attack.
That makes B0 part of the knapsack work, not a precondition to it.

### B0.1 The grid-overlap problem, and the lever nobody has pulled

Every gemma4 matmul grid is anchored at **(0,0)** (`matmul_tuning.py:185`, `weight_placement.py:368`),
and the SDPA **global** grid is 8x4 = `[0-0 – 7-3]` — so an 8x8 matmul rectangle **completely contains**
the SDPA block. The weights need headroom on exactly the cores with the worst CB high-water. Worse, the
**sliding** SDPA claims the full 11x10 = 110 cores, so its 472 KB covers *every* core.

The unpulled lever: **right-size AND offset both SDPA grids off the matmul rectangle.** At B=1 only 16
cores compute and the latency sweep showed even a 4-core grid matches, so a 4x4 block placed away from the
matmul region is free of compute cost. This is expressible — `SDPAProgramConfig.sub_core_grids` is a
`CoreRangeSet` (honoured at `sdpa_decode_program_factory.cpp:174-178`, cardinality must equal
`compute_with_storage_grid_size`), unlike the current `GEMMA4_SDPA_*_GRID` knobs which take a **size** and
therefore can only shrink, never move.

**Do not guess the resulting budget — recompute the per-core high-water map.** `test_cb_high_water`
already emits exactly that map, keyed by `(op, grid)`. Run it under three placements: default; both SDPA
grids right-sized; right-sized *and* offset. Note some 1-core ops sit at (0,0) and will then set the
local ceiling — `RotaryEmbedding` is **500.9 KB on 1 core** `[0-0]` and `LayerNorm` 296.9 KB — so the map,
not a single number, is the answer.

### B0.2 Measure the transient peak, per core

The `~500 KB` transient term is the least-measured quantity in the whole model, and it is what makes
320-vs-448 irreconcilable. Under HYBRID it becomes per-core too. Measure it from `buffer_allocate` graph
nodes (`graph_processor.cpp:224-274` carries address / size / num_cores / max_size_per_bank). **No budget
claim in Stage B is quotable until this exists.**

### B0.3 The two blockers that decide whether the tiered budget is reachable

Both confirmed in-source, and the second is the expensive one:

1. **HYBRID frees space for CBs, not for lockstep tensors.** `deps_map[AllocatorID{0}]` lists every
   per-bank allocator (`l1_banking_allocator.cpp:124-128`), so a normally-allocated sharded tensor still
   subtracts the **union** of per-core occupied ranges — i.e. HYBRID alone buys nothing for weights. The
   tensor must itself be per-core allocated (`experimental_set_per_core_allocation(True)`, wired end to
   end per `SRAM_WEIGHT_PINNING.md` §17.5).
2. **No stock op can consume a per-core buffer.** `set_globally_allocated_address`
   (`circular_buffer_config.cpp:219-231`) stores a **single scalar** from `Buffer::address()`, which for a
   per-core buffer returns `cores[0]`'s address — so handing one to `ttnn.matmul` **silently broadcasts
   core 0's address to every core, with no error**. That is the same class of silent-corruption bug as the
   shard-grid mismatch already found and fixed. The known solution is a custom op emitting one
   CBDescriptor per core (`deepseek_v3_b1`'s `ExpertKernel`), which is real C++ work.

**Sequencing consequence.** B0.1 and B0.2 are cheap measurements that size the prize; blocker (2) is the
gate on collecting it. So: measure first, and only commit to the custom op if the map says the budget
genuinely reaches ~600-820 KB/core on the matmul grid — because that, and only that, is what admits the
90.4 us and 57.4 us items.

## Stage B — the knapsack, kept, and run at tp=1 first

Problem statement: for a **fixed** config, choose the weight subset `S` (drafter **and** target) to hold in
L1 WIDTH_SHARDED maximising throughput subject to `Σ charge(w) ≤ budget_per_core`, where
`charge(w) = bytes(w) / cores(w)` and the objective is **not** `Σ value(w)` — it must be evaluated.

Reinstated because both of my grounds for dropping it were wrong: non-additivity kills the *formulation*,
not the problem (a best subset still exists; it just has to be measured), and **the oracle already
exists** — `_relocate` + `GEMMA4_AB_ONLY` evaluates an arbitrary subset against a model built once at
~1.5 s and a 0.01% floor, i.e. ~2000 evaluations/hour.

1. **Establish the item set and the additive bound, at CFG-E2E-host-tp1-b1, for each budget B0 establishes** —
   per-layer tensor counts (`use_double_wide_mlp` splits gate/up/down at layer 15: 15 narrow, 20 wide),
   per-core charge per tensor, value per shape measured **in that config**, and the exact 0/1 DP optimum.
   Run the DP at **every** budget from B0's ladder (214/320/358/448/~524/~823 KB/core) — the item set
   changes with the budget, so the DP curve *vs* budget is the real deliverable, and it tells you exactly
   what the custom-op work in B0.3 would be buying.
2. **Unblock the target** — one weight cannot carry both a prefill- and decode-compatible layout (why
   `GEMMA4_L1_SCOPES` defaults to `draft`). Instead of extending the tuner to emit prefill-valid configs
   for a sharded in1, **keep two copies**: DRAM original for prefill, L1 copy for decode, selected on
   `Mt == 1`. Without this the target contributes zero items.
3. **Search** with the existing oracle, seeded from the DP optimum, hunting the non-additive gains greedy
   cannot see. Per config; never pool across configs.
4. **Deliverable split decided by step 1's ceiling** — above ~2% e2e, the number is the contribution;
   below, the *structure* is (is the objective submodular? do pairwise interactions predict the composite?
   can a surrogate trained on N subsets predict the rest?). Deciding after step 1 costs nothing.

**If tp=1's feasible set proves too thin to search** (4 of 8 shapes at 320 KB, and the drafter's own
weights are 96 KB/core each), record that as the finding and *then* consider a tp=2 config as a separate,
explicitly-labelled experiment — not as a substitute.

---

## Deferred: op-level fixes

`TopK` (336 us, 30% of the drafter step), the 46 layout conversions, and the 17 CCL ops are **deferred**.
They are engineering that raises the baseline rather than research contribution, and they are not on the
critical path of either Stage B or Stage C. Recorded in A0's trajectory table with their measured sizes so
the opportunity is not lost. Note one is nearly free if ever wanted: activation chaining is already
implemented and measured at −9.1% backbone and is a default flip.

The verify/target op census is **not** in this category — it is measurement, not a fix, and it stays as a
Stage B/C input (it tells us where the 92% goes and whether the verify is flat in K, which Stage C's
K-scaling estimate assumes).

---

## Files

| file | change |
|---|---|
| *(lab-meeting-notes)* `documents/dflash/MEASUREMENT_RECORD.md` | **A0 — DONE**, `lab-meeting-notes@8b1b1cb`. The provenance record: configs, the P2 three-path matrix, 11 withdrawn/re-attributed numbers, the four-way capacity disambiguation, instrument defects, the do-not-reuse list, and §9's 151-entry census. **Note the repo**: `documents/dflash/` is in `~/codes/lab-meeting-notes`, not in tt-metal |
| `models/demos/gemma4/tt/spec_decode.py` | **P1** — `GEMMA4_SPEC_FUSED_PLI_DEV` (`:159`) and `GEMMA4_SPEC_PLI_DEV` (`:162`) default on; set `_last_route` where the routing decision is made (`:1966`, `generate_fused`, `generate_batched`); hoist `init_pli_device_weights` ahead of the host-loop capture site (`:1978` already covers the batched fused path). **A2** — log the per-iteration `m` histogram, not just the mean |
| `models/demos/gemma4/tt/model.py` | **P1** — `GEMMA4_DECODE_PLI_DEV` (**`:2154`**) default on, still guarded by `hidden_size_per_layer_input` |
| `models/demos/gemma4/demo/text_demo_v2.py` | **P1** — print the effective `route=` from `spec._last_route` (`:812-820`), and gate the steady-state metric on `_last_fused_setup_s` rather than `use_fused` (`:844-845`). **Leave the gate at `:799` alone** — forcing `use_fused=True` for a PLI target selects the batch-dim verify and is a regression |
| `models/demos/gemma4/tests/unit/test_packed_verify.py` | P1 regression gate — `test_pli_device_matches_host` must still hold at PCC 0.9999947 after the flip |
| (no code) | A1, A2, P2 — sweeps in CFG-PLAIN-tp1-b1 / CFG-E2E-host-tp1-b1 / CFG-E2E-fused-tp1-b1: `draft_len ∈ {3,7,11}` for the share, `m`-histogram, the three-path matrix |
| `models/demos/gemma4/tests/unit/test_packed_verify.py` | verify census beside `test_packed_verify_traced_pli_matches_eager` (`:562`) — **not** `test_packed_verify_batch_perf` (`:281`), which skips for PLI/E2B |
| `models/demos/gemma4/tt/attention/decode.py` | B0.1 — SDPA grids via `sub_core_grids` (a `CoreRangeSet`, so it can **offset**) instead of `compute_with_storage_grid_size` (a size, shrink-only) |
| `models/demos/gemma4/tests/unit/test_assistant_standalone_l1.py` | B0.1/B0.2 — per-core CB high-water map under the three placements; per-core transient peak from `buffer_allocate` nodes |
| `models/demos/gemma4/tt/model.py`, `tt/weight_placement.py` | Stage B step 2 — two-copy prefill/decode weights; B0.3 — `experimental_set_per_core_allocation` path |
| *(new C++, only if B0 justifies it)* | B0.3 — a matmul that emits one CBDescriptor **per core**, since `set_globally_allocated_address` stores a single scalar (`deepseek_v3_b1`'s `ExpertKernel` pattern) |
| *(lab-meeting-notes)* `documents/dflash/ASSISTANT_L1_WEIGHTS.md`, `SRAM_WEIGHT_PINNING.md` | correct the bare `448 KB` citations to name which budget |

**Ranking of Stage B vs Stage C is deliberately left to B0.** C's ceiling is ~+5.6% (2 drafters at ~63%
coverage) and reasonably well bounded. B's is *not* — it was <0.5% under the LOCKSTEP budget, but the
tiered budget admits the 90.4 us and 57.4 us items that no previous estimate could include, so the
DP-vs-budget curve (B step 1) is what decides which of the two leads. That curve costs no device time.

## Verification

```bash
cd ~/codes/tt-metal-gemma4-l1w
source ~/codes/tt-metal-gemma4-specdec/python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD:$PWD/ttnn:$PWD/tools ARCH_NAME=blackhole
export HF_MODEL=google/gemma-4-E2B-it GEMMA4_ASSISTANT_MODEL=google/gemma-4-E2B-it-assistant
unset GEMMA4_GATHER_IN0 GEMMA4_WEIGHTS_IN_L1 GEMMA4_SHARD_ACTIVATIONS

export TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 HF_HUB_OFFLINE=1
export GEMMA4_SPEC_TRACE=1 GEMMA4_MAX_SEQ_LEN=1024 GEMMA4_MAX_NEW_TOKENS=500
# ONE prompt and ONE token count across all three arms. Comparing across prompts or lengths is
# exactly how the earlier 1.32x overclaim happened (SPECDECODE_E2B_TRACE_AND_VERIFY.md gotchas).
export GEMMA4_PROMPT="$LONG" GEMMA4_SPEC_PROMPT="$LONG"
D=models/demos/gemma4/demo/text_demo_v2.py

# P2 — the three-path matrix. The selector is GEMMA4_SPEC_FUSED_PLI_DEV, NOT GEMMA4_SPEC_FUSED:
# for a PLI target the latter forces generate_fused -> _fused_body -> the batch-dim verify, which
# has no pli_on_device at all. Post-flip the knobs need no export; set them explicitly anyway so
# the log records the intent, and assert the NEW route= line rather than trusting the env.
GEMMA4_SPEC_FUSED_PLI_DEV=0 pytest -svv $D -k test_demo_spec_decode  # CFG-E2E-host-tp1-b1  expect route=host-loop
GEMMA4_SPEC_FUSED_PLI_DEV=1 pytest -svv $D -k test_demo_spec_decode  # CFG-E2E-fused-tp1-b1  expect route=fused-batched
pytest -svv $D -k "test_demo and not spec"                           # CFG-PLAIN-tp1-b1
# Post-flip sanity: the default (nothing exported) must now log route=fused-batched.
pytest -svv $D -k test_demo_spec_decode | grep -E "route=(fused-batched|host-loop)"

# A1 + A2 — repeated in BOTH spec paths; d(ms/iter)/d(draft_len) differs structurally between them.
for K in 3 7 11; do for F in 0 1; do
  GEMMA4_SPEC_DRAFT_LEN=$K GEMMA4_SPEC_FUSED_PLI_DEV=$F pytest -svv $D -k test_demo_spec_decode
done; done
```

Interleave arms, reverse order per round, **discard run 1** (JIT cold reads 4.63 vs 41.28 tok/s/u), quote
only `JIT cache stats: N/N (100.0%)`, and report medians. Never quote a fused number below 400 generated
tokens — the 4.38 GiB upload plus capture has not amortised (24 tokens reads 1.65 tok/s/u).

If a run hangs the board wedges and every later run fails with `Read 0xffffffff over PCIe`; recover with
`/home/masterjunmo/.tenstorrent-venv/bin/tt-smi -r`.

Regressions that must stay at their documented baselines:

**MEASURED 2026-08-19, both arms, on the P1 tree.** The previous comments in this block were wrong for two
of the four suites; these are the numbers actually observed, and the "post" column is the flipped default
while "pre" is `GEMMA4_SPEC_PLI_DEV=0 GEMMA4_DECODE_PLI_DEV=0 GEMMA4_SPEC_FUSED_PLI_DEV=0`.

| suite | pre-flip | post-flip | moved? |
|---|---|---|:--:|
| `test_packed_verify.py -k "…pli…"` (P1's own gate) | — | **3 passed** | — |
| `test_model.py` | 1 failed / 8 passed / 1 skipped | **identical** | no |
| `test_spec_decode.py` + `test_masked_embedding.py` | 3 failed / 13 passed / 13 skipped | **identical** | no |
| `test_assistant_standalone_l1.py -k 1x2` | — | **29 passed / 1 failed** | — (drafter-only; cannot see the PLI knobs) |

```bash
# NOTE --timeout=1800. pytest.ini's default is 300 s, and test_profile_eager_step exceeds it
# during model build; without the override the run does not "fail", it DUMPS CORE, which is
# easy to misread as a device wedge.
pytest -q --timeout=1800 models/demos/gemma4/tests/unit/test_assistant_standalone_l1.py -k 1x2
pytest -q models/demos/gemma4/tests/unit/test_model.py
pytest -q models/demos/gemma4/tests/unit/test_spec_decode.py models/demos/gemma4/tests/unit/test_masked_embedding.py
# P1's own gate — the PLI default flip must not move device-vs-host PLI parity
pytest -q models/demos/gemma4/tests/unit/test_packed_verify.py \
  -k "test_pli_device_matches_host or test_packed_verify_traced_pli_matches_eager or test_packed_verify_writes_correct_kv"
```

Corrections this measurement forced, recorded rather than quietly folded in: `test_model.py` is a
**10-test** suite failing **1**, not "12 passed / the same 3"; and the spec pair is **13 passed / 3
failed / 13 skipped**, not "20 passed / the same 6".

**A default flip changes what these baselines mean.** Each of the three suites must be run *before* the
flip and *after* it, and any test whose result moves is recorded in A0 as caused by the flip — not folded
into "the same N pre-existing failures". The three PLI knobs are now the defaults, so
`GEMMA4_SPEC_PLI_DEV=0 GEMMA4_DECODE_PLI_DEV=0 GEMMA4_SPEC_FUSED_PLI_DEV=0` becomes the way to reproduce
any pre-flip number.

**The "failing PCCs" note was also stale.** `test_model.py`'s single failure is **not** a PCC failure at
all: `test_full_model_decode[blackhole-1x1]` raises
`ValueError: Model has per-layer inputs configured but input_ids_torch/embeds_torch are missing`
(`model.py:638`) — the test never supplies PLI inputs. It fails identically with the knobs at 0 and at 1,
so it is structural and pre-existing. The recorded PCCs **0.9368017506152909 / 0.905595082490069** (and
`0.9459935351921065` for `GEMMA4_TUNE_MATMULS=0`) belong to some earlier tree state and are **not**
reproducible here; they go on A0's needs-re-measurement list rather than being quoted again. The
clean-tree marker is "the same 1 fails, with that ValueError", not any PCC value.

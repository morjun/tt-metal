# Gather capture inspection — 2026-09-09

This is a localization worksheet, not a causal verdict. Item 19 remains deferred.

## Existing graph evidence

Reproduce without accessing hardware:

```sh
python research_codes/mm_profiling/compare_gather_graphs.py generated/gather_diagnosis --output generated/gather_qualification_20260909/legacy_graph_comparison.json
```

The report hashes its six source graphs. Comparison requires identical device-op
name prefixes before aligning entries by position. Within each arm, K3's first
three steps have identical recorded operation arguments in K4 and K5. Multicast
also has identical input metadata; ring K4 differs at 276 of 534 input-metadata
entries, while ring K5 does not. Examples: initial temporary DRAM allocations
shift from 6,026,880 to 6,223,488 bytes; some norm-weight addresses shift 65,536
bytes. These are address differences, not evidence of changed tensor values.

Only one untimed graph process exists per arm/K. Graph mode executes the body
eagerly after setup, not inside `begin_trace_capture`. Therefore this observation
cannot establish that the same address pattern exists in the timed traces,
that it is repeatable, or that it causes the K4 timing dip. Confirm actual
captured addresses and independent-process reproducibility before intervention.
Recorded argument equality does not prove equal compiled programs/runtime args.

## Actual trace metadata required

Source inspection found concrete places to collect the missing evidence:

- `tt_metal/distributed/fd_mesh_command_queue.cpp`, `record_end`: immediately
  after `SimpleTraceAllocator::allocate_trace_programs`, record each trace node's
  ordinal, program identity, worker count, binary/nonbinary configuration sizes
  and addresses, `send_binary`, `sync_count`, `stall_first`, and
  `stall_before_program`. Use stable program/kernel content identities across
  processes, not raw numeric program IDs alone.
- In the same function, record prefetcher cache decisions after their assignment:
  cache residency, offset and binary size. Dump the captured node's `rta_data`
  and `cb_configs_payloads`, rather than the mutable program's latest arguments.
- `tt_metal/distributed/mesh_trace.cpp`, `populate_mesh_buffer`: preserve each
  range's actual `mesh_trace_data.data` before page padding, alongside total
  descriptor bytes and padded allocation. Preserve raw bytes and command
  boundaries; a payload hash alone cannot localize differences.
- Associate nodes with verified operation/step/layer identities. Confirm the
  expected inventory against actual nodes before assigning names to timings.

Why this is relevant: `simple_trace_allocator.cpp` scans future program uses,
places reusable versus last-use binaries differently, and inserts synchronization
when allocations or launch-message slots would conflict. These decisions can
change with K without changing the operation-count slope. This is a testable
candidate, not evidence that it causes the regression. The inspected allocator
and mesh-command-queue sources match the specdec source tree backing the shared
build, but that alone does not prove the build's exact source provenance.

`TT_METAL_DISPATCH_DATA_COLLECTION=1` is an existing host-side inventory option.
Its `dispatch_data.txt` aggregates per-program transactions and kernel groups;
it does not supply the ordered captured-node decisions above. It also writes a
fixed filename at process exit. Use only in an isolated diagnostic working
context, preserve each file, and never substitute its totals for trace payloads.

## Instrument qualification

Keep timing campaign sources/build unchanged. Build any required dump extension
in a distinct output directory; `build_Release` and the Python extension here
point into the specdec checkout. Never rebuild through those symlinks. A dump
must leave the captured graph and commands unchanged, with diagnostic timing
separately qualified. No mid-run profiler drain. Require complete explicit replay
boundaries and operation identities for later device profiling.

After localization, predeclare a narrow intervention and expected affected
intervals, validate correctness, and require an unprofiled response plus reversal.

Additional command-boundary hook: in `record_end`, sample the host bypass-data
length immediately before and after `write_program_command_sequence` for each
node, then preserve the final stream including dummy GO and `exec_buf_end`.
Record final `sync_count` after the dummy-worker-count offset. These byte ranges
provide verified command ownership; a largest-gap algorithm does not. Command
ranges still do not provide device execution timestamps without an independent
qualified instrument.


## Actual-capture graph result — two completed rounds

Source: `generated/gather_capture_inspection_20260909_metadata_only/`, twelve
completed cells, K3/4/5 × both arms × two counterbalanced rounds. Reports:
`comparison_round0.json` and `comparison_round1.json`. The canonical body and
ledger functions remained AST-identical to the completed timing campaign.

| Round | Arm | First temporary DRAM address K3 / K4 / K5 |
|---:|---|---|
| 0 | multicast | 6223488 / 6223488 / 6223488 |
| 0 | ring | 6026880 / 6223488 / 6026880 |
| 1 | multicast | 6026880 / 6026880 / 6026880 |
| 1 | ring | 6223488 / 6223488 / 6223488 |

Every within-arm K3-prefix comparison has identical recorded operation arguments.
Round 0 repeats the 276 ring-K4 input-metadata differences; round 1 has none.
Thus the address pattern is not deterministically tied to K4 in these captures.
This does not exclude address sensitivity or prove metadata recording is neutral;
these are untimed diagnostics, not simultaneous addresses from the 60 timing cells.
No actual dispatch payloads, binary-residency decisions or wait commands were
extracted **by the graph inspection**. Do not label §3B complete.

### Dispatch write payloads and binary residency — measured, MEASUREMENT_RECORD §6P.29

`TT_METAL_DISPATCH_DATA_COLLECTION=1` dumps `dispatch_data.txt` to the working
directory. It records **dispatch-core-to-worker transfer payloads**, not
host-device traffic: the call sites are inside command assembly beside the NOC
multicast descriptor construction (`impl/program/dispatch.cpp:896`), and
`RecordDispatchData` is a guarded host `std::map` increment
(`impl/dispatch/data_collection.cpp:20-30`) that issues no device work. The
bookkeeping is host-side; the quantity is on-device dispatch traffic, enumerated
when the command stream is assembled. Assembly occurs at trace capture and never
during replay, so replayed traffic is unperturbed. The flag costs host CPU time
at assembly, so use it only for untimed cells. The flag and the
output path are present in the loaded build. Six untimed `validate`-mode cells,
K3/4/5 x both arms, one fresh process each, campaign lock held:

- Binary bytes per processor, CB-config, semaphore and runtime-argument bytes are
  **byte-identical at K=3, 4 and 5** within an arm; program counts are 69
  multicast and 70 ring at every K. K4 is not distinguished by any of them.
- Between arms: ring writes 27,696 fewer binary bytes in total (only
  `TENSIX_DM_1` writes more, +784), +784 runtime-argument bytes, +96 CB-config
  bytes, −8 semaphore bytes, and one additional unique program.

Two limitations that keep §3B open:

1. Every program reports **"Ran 0 time(s)"**. The collector records writes at
   program creation, not per enqueue, so it describes the program set and its
   capture-time payloads and **not the replayed command stream**. Ordered
   commands and any wait/stall commands in the TRACE buffer remain unmeasured,
   and K-dependence must live in the enqueue sequence this instrument reports as
   zero.
2. Program identifiers are per-process counters. They cannot be matched between
   arms, so "+1 program" is a count only; which program differs was not
   established. Matching by kernel-group signature would be required.

`dump_cqs` (`tt_metal/impl/dispatch/debug_tools.cpp`, via the `watcher_dump`
tool) dumps host issue/completion queues. That is the wrong instrument for a
replay served from the TRACE DRAM buffer by the prefetcher.

### Ordered command stream with waits — BLOCKED, blocker located

The observable exists in the source and is not reachable from this build.

**Where it is.** `write_program_command_sequence`
(`tt_metal/impl/program/dispatch.cpp:2894`) logs, per program write and in
enqueue order:

- `Stall First: {}, Stall Before Program: {}, Send Binary: {}` (`:2902-2910`) —
  the two synchronization decisions and the binary-residency decision;
- `One-shot mode: {}, Fetch size: {} bytes` (`:2918`);
- `========== Finished Writing Program Command Sequence ==========` (`:3006`).

That is precisely the ordered stream with waits and per-program binary residency
that §3B still lacks. The calls are `LOG_TRACE_LAZY`, defined at `:82` as a
**runtime** spdlog level check, so they are compiled into Release and gated only
by log level.

**Why it is blocked.** The gate cannot be opened in the loaded build.
`TT_LOGGER_LEVEL`, `TT_LOGGER_TYPES` and `TT_LOGGER_FILE` are present as strings
in `libtt_metal.so`, but setting `TT_LOGGER_LEVEL` to `debug` or to `trace`
produces **identical output** — 24 lines at `info`/`warning` only, on a trivial
device op with a file sink. The documented set is `fatal|info|error|debug`
(`tt_metal/tools/mem_bench/README.md:16`), and neither value moved the level, so
no `LOG_TRACE_LAZY` site can fire. A `TT_LOGGER_TYPES=Dispatch` filter made no
difference.

**The other two routes are also closed without a code change.**

1. `trace_buffer->desc->ordered_trace_data` holds the staged command words, and
   `MeshTrace::populate_mesh_buffer` (`tt_metal/distributed/mesh_trace.cpp:94-102`)
   computes `unpadded_data_size` against the page-rounded `padded_data_size` —
   which would also settle the actual-payload-versus-allocated-bytes question.
   Reaching it needs `MeshDevice::get_mesh_trace`
   (`api/tt-metalium/mesh_device.hpp:173`), which is C++ only: Python exposes
   `begin_trace_capture`, `end_trace_capture`, `execute_trace`, `release_trace`
   and `MeshTraceId`, and nothing that returns a descriptor.
2. Reading the TRACE DRAM buffer back and parsing it has no Python entry point.

**RESOLVED 2026-09-09 by an isolated build — see MEASUREMENT_RECORD.md §6P.30.**

Worktree `tt-metal-cmdprobe`, detached at `84147125934`, own submodules, own
`python_env` (3.10.19) and own `build_Release`. Nothing is symlinked, so the
shared build is untouched. `build_metal.sh` was run only with that venv active,
after asserting `which python3` resolves inside it — the pyenv shim produces a
3.13-ABI `_ttnn.so` that will not load. Build took 18 minutes and exited 0.

Two env-gated accessors were added (59 lines, inert unless set):

| Variable | Site | Records |
|---|---|---|
| `GEMMA4_CMDSTREAM_OUT` | end of `write_program_command_sequence`, `impl/program/dispatch.cpp` | one line per program write, in enqueue order: the stall/binary flags and every sub-sequence byte size, including `wait_barrier` |
| `GEMMA4_TRACEBYTES_OUT` | `MeshTrace::populate_mesh_buffer`, `distributed/mesh_trace.cpp` | unpadded vs padded payload bytes, page size, per device range |

`fmt` is not included in `mesh_trace.cpp`, so that probe uses plain `ostream <<`
rather than relying on a transitive include. Both writers are mutex-guarded with
lazily-opened static append streams so concurrent assembly cannot interleave
partial lines.

**Result: every observable is affine in K and none distinguishes K4.** Program
writes step by exactly 680 (multicast) and 696 (ring) per unit K; the ring adds
exactly 16 writes per step, which is the 8 reshard operations counted twice by
`validate` mode. `wait_barrier` is 64 bytes on all 16,632 writes in both arms at
every K. Binary residency is real and varies (380-637 `send_binary=false` per
cell) and scales with K like everything else. Actual trace payload is now known
and replaces the legacy page-rounded `GetMemoryView` figure. The only K4
asymmetry is that the multicast arm pads 7,104 bytes versus 512/384 at K3/K5 and
sits one 8 KB page above linear interpolation — far too small to be a candidate
for an 11.9 us differential, and reported only because it is the sole asymmetry
found.

The captured-program branch is therefore exhausted for the dip. What remains is
runtime consumption of the recorded commands during replay, which none of these
instruments observes.

Reproduce: `scratchpad/dispatch_probe.sh <K> <arm> <outdir>`, which sets the flag,
runs one diagnostic cell, and moves `dispatch_data.txt` into the output directory.
The dump is written to the process working directory and will be overwritten by
the next run if it is not moved.

A concrete source of process-varying construction order exists:
`models/demos/gemma4/tt/model.py:create_rope_caches` iterates
`set(hf_config.layer_types)`. String-set order can vary with Python's hash seed.
This is a candidate explanation for address variation, not a measured cause of
the addresses or the timing penalty. Future manifests now record the explicit
Python hash seed (null means unspecified) and actual RoPE-cache construction
order. Control/record this order before any address-layout intervention.

### Failed first attempt, preserved

`generated/gather_capture_inspection_20260909/` contains the failed first cell
and `FAILURE.json`. Python I/O graph recording tried a forbidden device read
inside trace capture; the test failed and teardown stalled. Only the owned
pytest process was terminated with SIGTERM; no device reset was performed.
The retry used a new directory and disabled Python I/O recording after beginning
the graph, preserving C++ graph metadata. All twelve retry cells completed and
closed normally. This was not a profiler drain; the completed timing data is
unaffected. The diagnostic mode retains that fix.

## Replay profiling — the 1,461 cap is raised, but a cap remains

`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` is an **environment variable**
(`llrt/rtoptions.cpp:1001`, default 1000), not a compile-time constant, and it is
present in **both** the shared and the isolated build. The 1,461-launch ceiling
recorded since the earlier dispatch-profiling work therefore never required a
rebuild to lift; only the command-stream accessors did.

It sizes the profiler DRAM bank per RISC
(`impl/profiler/profiler_state_manager.cpp:19-40`) and feeds the JIT define
`PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC`. Changing its value therefore triggers
a **full kernel recompile** on the next run; the value is part of the build key,
so there is no stale-cache hazard, but budget several minutes for the first run
at each new value. Use a private `TT_METAL_CACHE` so this never rewrites the JIT
cache the other worktrees share.

**At count=6000, K3/4/5 x both arms all returned exactly 8,720 go-signal
ZONE_STARTs.** Identical across K is a hard cap, not data: the dispatch RISC's
buffer fills and later markers are dropped. Every run also logged
`Profiler DRAM buffers were full, markers were dropped` for worker RISCs, with
the warning count rising with K (58 at K3, 293 at K4, 363 at K5).

Consequences for anyone using these captures:

1. **Worker-kernel zones are truncated.** These captures support launch-interval
   analysis only, never kernel-window analysis. The `absence of dropped markers`
   gate is **not** met for worker zones.
2. **The dispatch stream is capped too** at count=6000, so a K sweep taken there
   is not comparable across K — each cell holds the same number of launches
   rather than the number its K implies.
3. `trace id` and `trace id counter` remain **empty** on dispatch rows, so replay
   segmentation is still count-based, not joinable by field.
4. Much of the budget is spent before the interesting part: the harness preserves
   the ledger's initial DRAM capture, which replays 20 times before relocation.
   Raising the count, or shortening that history for an untimed diagnostic, are
   the two ways to fit complete traced replays — the second changes the capture
   history the campaign otherwise holds fixed.

## Route 1: dispatch-only profiling is NOT available — source verdict

**Checked 2026-09-10 in the built worktree's source.** Worker instrumentation
**cannot** be disabled while retaining dispatch timestamps.

The gate is `tt_metal/tools/profiler/kernel_profiler.hpp:44-45`:

```
#if defined(PROFILE_KERNEL) && \
    (!defined(DISPATCH_KERNEL) || (defined(DISPATCH_KERNEL) && (PROFILE_KERNEL & PROFILER_OPT_DO_DISPATCH_CORES)))
```

For a **worker** kernel `DISPATCH_KERNEL` is undefined, so the condition collapses
to `defined(PROFILE_KERNEL)` — instrumentation is compiled in whenever profiling is
enabled at all. And `jit_build/build.cpp:176-189` composes the value as
`profiler_options = 1` with the option bits OR-ed on top, so bit 0 is unconditional
and `PROFILER_OPT_DO_DISPATCH_CORES` (bit 1) only **adds** dispatch cores. Requesting
`--profile-dispatch-cores` therefore yields `PROFILE_KERNEL=3`, which is worker
profiling **plus** dispatch, never dispatch alone. This confirms that asking for
dispatch profiling does not disable worker profiling.

**The one lever that does reduce Tensix marker volume** is
`PROFILER_OPT_DO_TRACE_ONLY` (bit 2), reachable via the environment variable
`TT_METAL_PROFILER_TRACE_TRACKING`, which is present in the built library. It sets
`TRACE_ON_TENSIX` (`:69-73`), and in that mode `profileScopeGuaranteed` emits markers
only when the `TRACE_REPLAY_STATUS` control word permits (`:751-780`), rather than on
every invocation — which is precisely the overflow source. It also pins
`myRiscID = 0` (`:107-111`) instead of the per-hardware-thread index, changing the
per-RISC data layout.

**Two risks to verify before relying on it, neither yet tested.**

1. **It may gate the measurement itself.** Dispatch runs on Tensix worker cores on
   this part, so a Tensix-scoped trace gate could suppress the very
   `SEND_GO_SIGNAL` markers the analysis needs. Compiled zones and **actually
   emitted records** must both be checked, not just the define.
2. **`myRiscID = 0` collapses per-RISC slots.** Whether records remain separable per
   RISC, or overwrite one another, is unverified.

It changes a JIT define, so the first run at each new setting recompiles all kernels.

**Consequence for the plan:** dispatch-only is not expressible with the existing
option bits, so route 1 as originally framed is closed. The nearest available
substitute is trace-only marker gating, which must itself be qualified — compiled
zones, emitted records, and a profiler-off control at the chosen replay count —
before any further matrix is collected.

## Trace-only qualification: FAILS gate 2. Route 1 is closed.

**2026-09-10.** The substitute for dispatch-only profiling was
`PROFILER_OPT_DO_TRACE_ONLY`. It fails on emitted records, and the failure is a
hard abort rather than a degradation.

**Activation failure — wrong variable first, and it failed silently.** The evidence
that settles this is the **configuration/code-path check** (which field the variable
writes, and the absent `-DPROFILE_KERNEL` change). Absent recompilation and equal CSV
size are consistent with inactivity but do not prove it on their own. `TT_METAL_PROFILER_TRACE_TRACKING`
sets `profiler_trace_tracking`; the field feeding `PROFILER_OPT_DO_TRACE_ONLY` is
`get_profiler_trace_only()` -> `profiler_trace_profiler`, set by
**`TT_METAL_TRACE_PROFILER`**. Two similar names, and the wrong one is inert.
Evidence it was inert: **zero kernel recompiles** and no `-DPROFILE_KERNEL` in the
log, while the two device CSVs came out the **same size** with different content.
Comparing only emitted records would have read that as "trace-only suppressed
nothing" instead of "trace-only never ran". Preserved under `tt/wrong_var/`.

Both flags are conditional on `profiler_enabled` already being true, so parse order
could silently drop either request. `TT_METAL_DEVICE_PROFILER` is enum 125 and
`TT_METAL_TRACE_PROFILER` is 132, so the ordering is safe here — but it is the same
class of silent no-op.

**With the correct variable: compiled zones PASS, emitted records FAIL.**

| Check | Result |
|---|---|
| Compiled zones | **PASS** — 255 kernel recompiles, `-DPROFILE_KERNEL=7` = base \| DISPATCH_CORES \| TRACE_ONLY |
| Test execution | ran; harness recorded 2 replays, `trace_us=3322.3`, `valid_clock=true` |
| Emitted records | **FAIL — profiler aborts** |
| Setup-marker suppression | not reached |
| Dispatch GO markers retained | not reached |
| Core/RISC attribution | **FAIL, and it is the cause** |
| Independent replay boundaries | not reached |

```
TT_FATAL profiler.cpp:2149: start_marker_it->marker_name == marker.marker_name
  Start and end marker names do not match.
  CQ-DISPATCH-SUBORDINATE ZONE_END, RISC: TENSIX_RISC_AGG, id 43138
  cq_dispatch_subordinate.cpp:609
```

**The abort proves marker pairing failed; it does not by itself prove the cause.**
`TENSIX_RISC_AGG` is the aggregated identity that `myRiscID = 0` produces under
trace-only, and the failing record reports exactly that identity, so aggregation is a
**strong suspected mechanism** — but **missing records or zone-boundary handling could
equally leave an end marker unpaired**, and the raw stream was not analysed to
separate them. Treat the mechanism as suspected. The predicted risk — that a
Tensix-scoped gate would damage the dispatch markers the analysis needs — did occur,
in attribution rather than suppression form, and the decision to stop using this path
does not depend on which mechanism is responsible.

**Teardown stalled** exactly as the earlier recorded incident: pytest aborted, tracy
hung for ~1h24m. Only `masterjunmo` processes were SIGTERMed; **no device reset**,
and nothing belonging to the co-user was signalled. Capture preserved under
`tt/gate2_failed/` with `FAILURE.json`.

**Occupancy gate worked and found something endpoints would have missed.** 199
samples across the cell: **no foreign device holders at any point**, but foreign CPU
spiked to **1426%** (load 6.15 against a 0.20 baseline) in three samples over ~30
seconds. Device exclusivity held; host CPU exclusivity did not. Note the sampler
polls at 10 s, so it bounds contamination loosely and would miss a shorter spike.

**Verdict: route 1 is closed.** Dispatch-only is not expressible with the option
bits, and the only marker-volume lever corrupts dispatch attribution. Per the agreed
sequence, the next work is **host-side command capture** — actual bytes, wait
targets, configuration addresses, program identities — with interventions evaluated
on the existing unprofiled 400-replay harness. Do not shorten capture history or fit
an overhead correction as a substitute; each needs its own validation first.

## Host-side command capture: gates passed, hypothesis tested, hypothesis rejected

**2026-09-10.** The route the profiler-path failure sent us to. Full result in
MEASUREMENT_RECORD.md 6P.34.

### Probe extension

Two locations, as scoped:

- `FDMeshCommandQueue::record_end` — per program: identity, byte range within the
  trace stream (taken as the growth of the sysmem bypass buffer across each
  `write_program_command_sequence`), `sync_count`, the stall flags, `send_binary`,
  and the prefetcher-cache residency fields. Plus a `trace_begin` record with
  trace sequence, mesh device and CQ id, a terminator range, and a range total.
- `MeshTrace::populate_mesh_buffer` — the exact **unpadded** command stream before
  page padding, keyed to the same trace sequence.

Each `record_end` writes its own file, so the initial DRAM capture and the measured
pinned capture are separable by construction.

### Gates

| Gate | Result |
|---|---|
| Separate DRAM and pinned trace identity | **PASS** — 2 traces, 2 streams per capture, tagged with device and CQ |
| Complete byte accounting | **PASS** — preamble + programs + terminator equals the range total exactly, all traces; `.bin` size = manifest unpadded bytes = words x 4 |
| Reproducibility, controlled construction order | **PASS** — with `PYTHONHASHSEED=0`, two fresh processes give **byte-identical streams and identical manifests** |
| Probe neutrality (declared in advance: median paired shift <= 3.0 us, range <= 10.0) | **PASS** — median **−0.02 us/trace**, range **0.48 us**, K=3, 400 replays, 4 paired rounds |
| Profiler-off control reproduces the qualified baseline | **PASS** — median 3164.15 against 6P.28's 3164.042, within **0.11 us** |

The address variation recorded earlier was a **Python hash-seed effect**: with the
seed pinned it does not occur. Raw addresses are preserved in the dump; any
normalized view is additional, never a replacement.

### What the capture showed, and what the intervention did to it

Residency decisions differ **only in the final step** — the first K−1 steps are
byte-identical at every K (verified K=3→4, 4→5, 5→6, both arms) — and the arms
diverge in opposite directions at K=4. Rank order across K=3/4/5 matched the timing
penalty exactly.

**Tested and rejected.** Forcing the whole final step to send its binaries
(`GEMMA4_FORCE_SEND_TAIL`) removed the divergence by construction and **did not
shrink the dip; it grew 13.33 -> 16.47 us**. Against predeclared thresholds this is
NULL. Baseline cells reproduced the qualified values, so the reversal gate passed
and the null is trustworthy.

**New observation, needing its own confirmation:** the intervention makes the ring
send +12/+10/+5 more binaries than multicast at K=3/4/5. Timing responds at
**0.39 and 0.31 us per differential send at K=3 and K=5**, and at **0.01 at K=4**,
whose three rounds straddle zero. **K=4 absorbs added binary work its neighbours pay
for.** That is a negative result about responsiveness from a single lever and three
rounds — not a mechanism, and possibly a property of the lever rather than of K=4.

### Scoped repeat — MEASUREMENT_RECORD.md 6P.35

`GEMMA4_FORCE_SEND_TRACE=1` restricts the override to the measured pinned trace.
Verified before timing: the initial DRAM trace is **byte-identical** (MD5) in both
arms with 0 programs changed, and output validation passes (3 blocking replays
bit-identical against eager). 6P.34's scope defect is eliminated.

Full audit of the change: **only `send_binary`, `bytes`, `pc_is_cached` and
`pc_offset` differ**, in 60 (multicast) / 70 (ring) final-step programs.
`sync_count`, both stall flags, program and runtime ids, worker counts and launch
write pointers are identical in every program of every trace. First differing raw
byte falls inside the final step in both arms.

72 cells, 6 counterbalanced rounds, none flagged. **The dip survives and deepens:
13.88 -> 17.19 us, a within-round change of +3.27 [+2.62, +3.84] excluding zero.**
Baseline reproduces 6P.26's 13.74, so the reversal gate passes.

**Hypothesis closed:** the final-step skipped-send pattern does not explain the dip
through this intervention. **Not closed:** binary residency generally, the
captured-program branch, or anything about K=4 responsiveness -- the K=4
differential of +0.55 is near-common-mode, both arms paying ~34.7-35.1 us.

### Standing cautions carried forward

Equal byte counts never closed this question, and a differing command never
established causality — the intervention is what settled it, and it settled it
against the hypothesis. Item 19 remains deferred.

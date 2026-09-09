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

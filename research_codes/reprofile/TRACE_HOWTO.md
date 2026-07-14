# Enabling / disabling decode TRACE for the batch-1 perf benchmark

Trace replays a captured command stream and removes the per-step host dispatch that dominates
no-trace batch-1 decode. Measured on this P150 (32-layer, single card): trace ON ~45 ms/step
(~22 tok/s) vs trace OFF ~109 ms/step (~9 tok/s). The "old branch 11 vs upstream 22" gap is
entirely this setting.

Always run single-card to avoid the 2-card fabric-router timeout:
`TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150`

The two branches control trace in INCOMPATIBLE ways. Check `git rev-parse --abbrev-ref HEAD` first.

---

## upstream branch — CLI flags (no source edit, no rebuild)

- `models/tt_transformers/demo/conftest.py:55-56`:
  `--enable_trace` (store_true) and `--disable_trace` (store_false, dest=enable_trace).
- `models/tt_transformers/demo/simple_text_demo.py:935-937` honors the flag; batch-1 default = ON.

Trace ON (default; flag optional):
```
TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 TT_LOGGER_LEVEL=Info \
  ./python_env/bin/pytest models/tt_transformers/demo/simple_text_demo.py \
    -k "performance and batch-1"
```
Trace OFF:
```
TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 TT_LOGGER_LEVEL=Info \
  ./python_env/bin/pytest models/tt_transformers/demo/simple_text_demo.py \
    -k "performance and batch-1" --disable_trace
```
Do NOT pass both flags (they share dest `enable_trace`).

---

## l1-kv-cache branch — edit ONE source line (CLI cannot enable trace)

Trace is HARDCODED OFF and the CLI is overridden:
- `simple_text_demo.py:930`  `enable_trace = request.config.getoption("--enable_trace") or enable_trace`
- `simple_text_demo.py:931`  `enable_trace = False  # FORECE DISABLE FOR DPRINT DEBUGGING`  <-- overrides line 930
- batch-1 parametrize default (line 441) is also `False`.
- conftest (line ~60) defines only `--enable_trace` as `action="store", type=bool` (no `--disable_trace`
  -> "unrecognized argument"). That `type=bool` flag is also buggy (`bool("False")` is True), so the
  CLI is NOT a reliable control. Control trace by editing line 931.

DISABLE trace (this is the default): do nothing.

ENABLE trace: edit `models/tt_transformers/demo/simple_text_demo.py:931`:
```
-    enable_trace = False  # FORECE DISABLE FOR DPRINT DEBUGGING
+    enable_trace = True   # trace enabled for perf measurement
```
Then run (no rebuild needed — Python-only change; the first run recompiles the trace-path kernels):
```
TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 TT_LOGGER_LEVEL=Info \
  ./python_env/bin/pytest models/tt_transformers/demo/simple_text_demo.py \
    -k "performance and batch-1"
```
Notes:
- `device_params` on this branch already has `trace_region_size: 70000000`, so trace capture has a
  region; `fabric_config: False` is fine on a single chip (fabric is inter-chip only).
- Revert line 931 to restore the original DPRINT-debug behavior.
- If you hit a stale-kernel error after toggling, wipe `~/.cache/tt-metal-cache` and rerun.

---

## Reading the result (both branches)

The demo logs per step `Iteration N: Xms` and a `Y tok/s` line. Average the steady-state iterations
(skip the first ~5 warmup/compile steps): `tok/s ~= 1000 / mean_ms`.
Expected: trace ON ~22 tok/s (~45 ms/step), trace OFF ~9-11 tok/s (~90-110 ms/step).

Automated A/B (upstream only, uses `--disable_trace`): `reprofile/run_trace_ab.sh`.

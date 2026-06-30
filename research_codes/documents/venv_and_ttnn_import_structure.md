# venv structure, ttnn import, and the role of TT_METAL_HOME

Written 2026-06-30, explaining the shared-venv setup created for the tt-metal
worktrees (`tt-metal`, `tt-metal-l1kv`, plus throwaway comparison worktrees) and
why it behaves differently from the per-worktree venvs that `create_venv.sh`
produces.

## TL;DR

- `ttnn` is **not** a normal pip library. It is a hybrid: pure-Python sources
  (`ttnn/ttnn/*.py`) **plus** a compiled extension `_ttnn.so` that dynamically
  links the rest of the built C++ stack (`_ttnncpp.so`, `libtt_metal.so`,
  `libdevice.so`) from that worktree's `build_Release/`, **plus** a runtime
  dependency on the repo's kernel sources/firmware. Importing it ties you to a
  specific built checkout, unlike a self-contained wheel in `site-packages`.
- `create_venv.sh` installs ttnn with `pip install -e .` (editable). That does
  not copy ttnn into `site-packages`; it writes `.pth` files that put the
  worktree's **absolute** source paths on `sys.path`. So `import ttnn` resolves
  to that one worktree with **no environment variable needed** — the path is
  hard-baked.
- The **shared venv** (`~/codes/tt-metal-venv`) deliberately has **no** editable
  ttnn install (that is what lets one venv serve worktrees with different ttnn
  code). Instead a `sitecustomize.py` adds the ttnn source paths at interpreter
  startup, reading the worktree location from **`TT_METAL_HOME`**. That is why
  the shared venv requires `TT_METAL_HOME` and the per-worktree venvs did not:
  it is the *selector* that tells the shared interpreter which worktree's ttnn to
  import.
- Separately, tt-metal's C++ runtime needs a "root dir" to find kernels/firmware.
  In this version that is resolved by `TT_METAL_INSTALL_ROOT` (compile-time) ->
  `TT_METAL_RUNTIME_ROOT` (env) -> a current-working-directory fallback — **not**
  `TT_METAL_HOME`. `TT_METAL_HOME` proper is used for profiler output paths, graph
  reports, some tools, and the `tt-run` launcher.

---

## 1. Is `import ttnn` only dependent on `_ttnn.so`? Is it not like a pip library?

Correct — it is not like a normal pip library, and `_ttnn.so` is only one of
several pieces. A normal pip package (`pip install numpy`) copies a self-contained
wheel into `site-packages/`, and `import numpy` finds it there with no external
state. `ttnn` is different on every axis:

1. **Two-layer package.** `import ttnn` loads the Python package at
   `ttnn/ttnn/__init__.py`, which in turn imports the compiled extension
   `ttnn._ttnn` (the file `ttnn/ttnn/_ttnn.so`, ~34 MB, built from the repo's C++
   via nanobind).

2. **The extension drags in the whole built C++ stack.** `_ttnn.so` is not
   self-contained; `ldd` shows it dynamically links, via an RPATH baked to *that
   worktree's* build dir:
   - `_ttnncpp.so`  (the C++ ttnn ops)
   - `libtt_metal.so`  (the Metalium runtime)
   - `libdevice.so`  (UMD, user-mode driver)
   all under `<worktree>/build_Release/...`. So importing ttnn from worktree X
   pulls in X's compiled libraries specifically.

3. **Runtime needs the repo's kernel sources + firmware.** Device kernels are
   JIT-compiled at runtime from source files in the repo tree, and firmware/
   runtime artifacts are read from a "root dir" (see §4). So a working `ttnn`
   needs the built `build_Release/` *and* the source/runtime tree of a real
   checkout, not just the `.so`.

Net: an importable `ttnn` is bound to a fully built repo worktree (sources +
`build_Release` + runtime root), which is exactly why the worktrees each needed
their own build, and why a single shared *wheel* is not how this works.

---

## 2. How `import ttnn` is resolved: pip vs editable vs shared venv

### (a) Normal pip install (not how tt-metal works)
Wheel copied into `site-packages/ttnn/`; `import ttnn` finds it on the default
`sys.path`. No repo, no env.

### (b) Editable install — what `create_venv.sh` does (`pip install -e .`)
tt-metal's `setup.py` (`EditableWheel`) writes, into the venv's `site-packages`:
- `ttnn-custom.pth` containing three **absolute** paths:
  ```
  /home/masterjunmo/codes/tt-metal
  /home/masterjunmo/codes/tt-metal/ttnn
  /home/masterjunmo/codes/tt-metal/tools
  ```
- `__editable__.ttnn-*.pth` + a finder, and a `ttnn-*.dist-info` (so pip records
  it as installed).

Python reads `.pth` files at startup and appends their lines to `sys.path`. So the
worktree's source dirs are on the path permanently, and `import ttnn` loads
`<worktree>/ttnn/ttnn/__init__.py`. The `_ttnn.so` lives at
`<worktree>/ttnn/ttnn/_ttnn.so` (placed by the build/install). **No environment
variable is consulted** — the worktree path is hard-coded in the `.pth`. This is
why the per-worktree venvs "just worked" without `TT_METAL_HOME`.

Consequence: a venv created this way is pinned to one worktree. If you point a
second worktree at the same venv, `import ttnn` still loads the first worktree's
sources (the `.pth` wins regardless of your current directory). That is the exact
reason a single editable venv cannot be shared across worktrees with different
ttnn code.

### (c) Shared venv — `~/codes/tt-metal-venv`
To serve multiple worktrees from one venv, it is built **without** the editable
ttnn install (no `ttnn-custom.pth`, no pin). It contains only the dependencies
(`requirements-dev.txt` + ttnn's declared deps such as `graphviz`,`numpy`,
`networkx`, `pandas`, `seaborn`, ...). To make `import ttnn` work, a
`sitecustomize.py` (auto-run by Python at startup) injects the ttnn source paths,
chosen from `TT_METAL_HOME`:
```python
# ~/codes/tt-metal-venv/lib/python3.10/site-packages/sitecustomize.py
import os, sys
_h = os.environ.get("TT_METAL_HOME")
if _h:
    for _p in (os.path.join(_h, "ttnn"), _h, os.path.join(_h, "tools")):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
```
So with `TT_METAL_HOME=<worktree>`, the shared interpreter imports that worktree's
ttnn and (via the `_ttnn.so` there) that worktree's compiled stack. Without
`TT_METAL_HOME`, `sitecustomize` adds nothing and `import ttnn` fails. **This is
the dominant reason the shared venv requires `TT_METAL_HOME`** — it replaces the
hard-coded `.pth` selector with an env-var selector.

(Alternative design: `sitecustomize` could instead derive the worktree from the
current directory — e.g. if `./ttnn/ttnn` exists — which would remove the
`TT_METAL_HOME` requirement when you run from a worktree root. We keyed on
`TT_METAL_HOME` because it is explicit and matches how the worktree build/run
scripts already set it.)

---

## 3. Why the previous (per-worktree) venv did not need TT_METAL_HOME

Two independent reasons, both already covered:
1. **Python import**: the editable `ttnn-custom.pth` hard-coded the worktree path,
   so `import ttnn` needed no env (§2b).
2. **C++ root dir**: resolved without `TT_METAL_HOME` (§4) via the build's baked
   install root or the cwd fallback.

So a plain `cd <worktree> && ./python_env/bin/pytest ...` worked with no
`TT_METAL_HOME`. The original demo runs in this investigation did exactly that.

---

## 4. What TT_METAL_HOME actually does (and what it does NOT do)

It is easy to assume `TT_METAL_HOME` is "the" root that finds kernels. In this
version of the code it is not — there are two separate concepts:

### Core runtime root dir (kernels, firmware, runtime libs)
Resolved in `tt_metal/llrt/rtoptions.cpp` (`RunTimeOptions` ctor), in order:
1. `TT_METAL_INSTALL_ROOT` — a **compile-time** `#define` baked into the build; if
   that directory exists, it is used (rtoptions.cpp ~L309-313).
2. `TT_METAL_RUNTIME_ROOT` — **environment** override (rtoptions.cpp L256, L317).
3. `RunTimeOptions::set_root_dir()` — programmatic override.
4. **cwd fallback**: if the current working directory contains a `tt_metal/`
   subdir, use the cwd (rtoptions.cpp L324-331).
5. else FATAL.

Note: the env var here is `TT_METAL_RUNTIME_ROOT`, **not** `TT_METAL_HOME`. For an
editable build, step 1 (baked install root) or step 4 (cwd) usually satisfies it,
which is why kernels are found even with no env set, as long as you run from the
worktree or use its build.

### Where TT_METAL_HOME is genuinely read
Grep across the tree shows `TT_METAL_HOME` is used for ancillary paths, not the
core kernel root:
- `tt_metal/impl/profiler/profiler_paths.hpp:28` — profiler output location
  (`getenv("TT_METAL_HOME")`). This is why our **device-profiler** runs needed
  `TT_METAL_HOME` set.
- `ttnn/core/graph/graph_processor.cpp:115` — graph-report output.
- `ttnn/ttnn/distributed/ttrun.py` (several) — the multi-process `tt-run` launcher.
- `tt_metal/tools/memset.py` — asserts it is set.

So `TT_METAL_HOME` matters for profiling, graph reports, tools, and distributed
runs. In our **shared venv** it takes on the *additional* job of selecting the
Python ttnn (§2c), which is the part you hit day-to-day.

Practical implication: when using the shared venv, set `TT_METAL_HOME=<worktree>`.
That one variable then (a) selects the worktree's Python ttnn via sitecustomize,
and (b) also satisfies the profiler/graph/tools paths. The core kernel root is
still resolved by the baked install root / cwd, so running from the worktree root
keeps that happy too.

---

## 5. Practical usage

Per-worktree editable venv (what `create_venv.sh` makes) — unchanged, simplest,
no env needed, but ~3.5 GB each and pinned to its worktree:
```
cd <worktree> && ./python_env/bin/pytest ...
```

Shared venv (`~/codes/tt-metal-venv`, one 3.2 GB copy for all worktrees):
1. Point the worktree at it once: `ln -sfn ~/codes/tt-metal-venv <worktree>/python_env`
   (already done for `tt-metal-l1kv`).
2. Build that worktree so `<worktree>/ttnn/ttnn/_ttnn.so` exists (the build target
   `ttnn` produces `build_Release/ttnn/_ttnn.so`; symlink or install it into
   `ttnn/ttnn/`).
3. Run with `TT_METAL_HOME` set:
   ```
   cd <worktree>
   TT_METAL_HOME=$PWD TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 ./python_env/bin/pytest ...
   ```

The main `tt-metal` (upstream) worktree was intentionally left on its own editable
venv so its existing `TT_METAL_HOME`-free workflow keeps working. To switch it to
the shared venv, replace `tt-metal/python_env` with a symlink to
`~/codes/tt-metal-venv` and start setting `TT_METAL_HOME` in its run commands.

---

## 6. Why not just share the editable venv directly?

Because the editable `.pth` hard-codes one worktree's absolute paths into
`sys.path`, so `import ttnn` from any other worktree using that venv would still
load the first worktree's sources and its `_ttnn.so` — silently wrong (mismatched
code, possibly mismatched ABI). The deps-only + `sitecustomize` approach is what
makes one venv safely serve multiple worktrees, at the cost of requiring
`TT_METAL_HOME`.

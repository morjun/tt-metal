"""Opt-in address capture for the 12B fused speculative iteration.

The graph must bracket the *actual* Metal trace capture. NORMAL graph capture
dispatches work and, with the cached-CB graph hook, records resolved CB ranges
for both newly compiled and cached programs. It is deliberately separate from
uninstrumented timing runs.
"""

import json
import os
import inspect
import subprocess
from pathlib import Path

import ttnn


def enabled():
    return bool(os.getenv("GEMMA4_LMHEAD_PROFILE_OUT"))


def _path(suffix):
    root = Path(os.environ["GEMMA4_LMHEAD_PROFILE_OUT"])
    root.mkdir(parents=True, exist_ok=True)
    return root / suffix


def _owners(roots):
    names = {}
    visited = set()

    def walk(obj, path, depth):
        if depth > 9:
            return
        if isinstance(obj, ttnn.Tensor):
            try:
                device_tensor = ttnn.get_device_tensors(obj)[0]
                if device_tensor.memory_config().buffer_type == ttnn.BufferType.L1:
                    names.setdefault(str(device_tensor.buffer_address()), []).append(path)
            except (RuntimeError, AttributeError, IndexError):
                pass
        elif isinstance(obj, dict):
            if id(obj) in visited:
                return
            visited.add(id(obj))
            for key, value in obj.items():
                walk(value, f"{path}.{key}", depth + 1)
        elif isinstance(obj, (list, tuple)):
            if id(obj) in visited:
                return
            visited.add(id(obj))
            for i, value in enumerate(obj):
                walk(value, f"{path}[{i}]", depth + 1)
        elif type(obj).__module__.startswith("models.demos.gemma4"):
            if id(obj) in visited:
                return
            visited.add(id(obj))
            for key, value in vars(obj).items():
                walk(value, f"{path}.{key}", depth + 1)

    for name, obj in roots.items():
        walk(obj, name, 0)
    return names


def snapshot(mesh_device, stage, owners=None):
    if not enabled():
        return
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    buffers = {}
    for b in ttnn._ttnn.reports.get_buffers(mesh_device):
        if b.buffer_type == ttnn.BufferType.L1:
            buffers[int(b.address)] = {
                "layout": str(b.buffer_layout),
                "max_size_per_bank": int(b.max_size_per_bank),
            }
    # Aggregate page spans before serialization. A uniform head has 35K pages;
    # its physical mapping reduces to 110 short interval lists.
    pages = {}
    for p in ttnn._ttnn.reports.get_buffer_pages(mesh_device):
        if p.buffer_type != ttnn.BufferType.L1:
            continue
        key = (int(p.address), int(p.core_x), int(p.core_y))
        pages.setdefault(key, []).append((int(p.page_address), int(p.page_address + p.page_size)))
    physical = {}
    for (address, x, y), spans in pages.items():
        merged = []
        for lo, hi in sorted(spans):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        physical.setdefault(str(address), {})[f"{x},{y}"] = merged
    data = {
        "stage": stage,
        "base": int(ttnn.get_allocator_base_address(mesh_device, ttnn.BufferType.L1)),
        "num_banks": int(view.num_banks),
        "total_bytes_per_bank": int(view.total_bytes_per_bank),
        "allocated_bytes_per_bank": int(view.total_bytes_allocated_per_bank),
        "free_bytes_per_bank": int(view.total_bytes_free_per_bank),
        "largest_free_bytes_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
        "blocks": view.block_table,
        "buffers": buffers,
        "physical_pages": physical,
        "owners": _owners(owners or {}),
    }
    _path(f"{stage}.l1.json").write_text(json.dumps(data, indent=2) + "\n")


def begin():
    if enabled():
        global _original_linear, _original_matmul, _calls
        _calls = []
        _original_linear, _original_matmul = ttnn.linear, ttnn.matmul

        def wrap(op, original):
            def record(*args, **kwargs):
                frame = inspect.currentframe().f_back
                source, function = "<unknown>", "<unknown>"
                while frame is not None:
                    filename = frame.f_code.co_filename
                    if "/models/demos/gemma4/" in filename and not filename.endswith("lm_head_l1_capture.py"):
                        source, function = f"{filename}:{frame.f_lineno}", frame.f_code.co_name
                        break
                    frame = frame.f_back
                del frame
                _calls.append(
                    {
                        "op": op,
                        "source": source,
                        "function": function,
                        "input_shape": list(args[0].shape) if len(args) > 0 else None,
                        "weight_shape": list(args[1].shape) if len(args) > 1 else None,
                        "requested_program_config": repr(kwargs.get("program_config")),
                    }
                )
                return original(*args, **kwargs)

            return record

        ttnn.linear = wrap("linear", _original_linear)
        ttnn.matmul = wrap("matmul", _original_matmul)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        # Argument formatting can read tensors while Metal trace capture forbids
        # device reads. C++ graph nodes still carry tensor shapes and addresses.
        ttnn.graph.disable_python_io_recording()


def end(stage="iteration"):
    if enabled():
        try:
            graph = ttnn.graph.end_graph_capture()
        finally:
            ttnn.linear, ttnn.matmul = _original_linear, _original_matmul
        _path(f"{stage}.graph.json").write_text(json.dumps(graph) + "\n")
        _path(f"{stage}.calls.json").write_text(json.dumps(_calls, indent=2) + "\n")
        _path("manifest.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "cols": int(os.getenv("GEMMA4_LMHEAD_L1_COLS", "0")),
                    "tune_matmuls": os.getenv("GEMMA4_TUNE_MATMULS", "0"),
                    "target_tune_matmuls": os.getenv("GEMMA4_TUNE_TARGET_MATMULS", ""),
                    "route": os.getenv("GEMMA4_SPEC_ROUTE", "auto"),
                    "draft_len": int(os.getenv("GEMMA4_SPEC_DRAFT_LEN", "3")),
                    "git_revision": subprocess.check_output(
                        ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "loaded_ttnn_library": str(Path(ttnn._ttnn.__file__).resolve()),
                    "visible_devices": os.getenv("TT_VISIBLE_DEVICES"),
                },
                indent=2,
            )
            + "\n"
        )

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-module dtype/precision overrides for Gemma4.

Reads precision_overrides.json and resolves a (model variant, mesh shape) tuple
into a {module_name: ttnn.DataType} mapping. Modules without an override use a
caller-supplied default (typically bfloat16).

The JSON file is the single source of truth for per-system precision tweaks
(e.g. dropping shared_mlp to bfp8 on Gemma4-31B at 1x2 to fit DRAM). New entries
are added there rather than in code.
"""

import json
import os

import ttnn

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "precision_overrides.json")

# Module names that may be overridden — keep in sync with the JSON schema and
# with the constructors that accept these kwargs (Gemma4Model and below).
KNOWN_MODULES = ("shared_mlp", "attention", "experts", "router", "lm_head", "embedding")

_DTYPE_BY_NAME = {
    "bf16": ttnn.bfloat16,
    "bfloat16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfloat8_b": ttnn.bfloat8_b,
    "fp32": ttnn.float32,
    "float32": ttnn.float32,
}


def dtype_to_str(dtype):
    """Short stable string for cache-filename suffixes ("bf16" / "bfp8" / "fp32").

    Cache filenames embed the dtype string so flipping a module's dtype in
    precision_overrides.json doesn't reuse a stale cached tensor at the
    previous precision.
    """
    if dtype == ttnn.bfloat16:
        return "bf16"
    if dtype == ttnn.bfloat8_b:
        return "bfp8"
    if dtype == ttnn.float32:
        return "fp32"
    raise ValueError(f"No cache-suffix mapping for dtype {dtype}")


class Gemma4Precision:
    """Per-module dtype mapping. Construct via ``Gemma4Precision.load(...)``
    or directly with ``Gemma4Precision({...})``."""

    def __init__(self, overrides=None):
        self._overrides = dict(overrides) if overrides else {}

    def get(self, module_name, default=ttnn.bfloat16):
        return self._overrides.get(module_name, default)

    def __repr__(self):
        return f"Gemma4Precision({self._overrides!r})"

    @classmethod
    def load(cls, model_path, mesh_shape):
        """Resolve overrides for the given (model, mesh).

        model_path: full path to the HF checkpoint; we key on the basename.
        mesh_shape: (rows, cols) tuple, formatted as "RxC" for the JSON key.
        """
        model_key = os.path.basename(str(model_path).rstrip("/"))
        mesh_key = f"{mesh_shape[0]}x{mesh_shape[1]}"

        try:
            with open(_PATH) as f:
                table = json.load(f)
        except FileNotFoundError:
            table = {}

        # Mesh-specific override wins over "default"; a model absent from the
        # table simply starts empty (the env overlay below still applies).
        model_entry = table.get(model_key) or {}
        raw = model_entry.get(mesh_key) or model_entry.get("default") or {}
        resolved = {}
        for k, v in raw.items():
            if k not in KNOWN_MODULES:
                continue  # ignore unknown / future keys silently
            if v not in _DTYPE_BY_NAME:
                raise ValueError(
                    f"precision_overrides.json[{model_key}][{mesh_key}][{k}]={v!r} — "
                    f"unknown dtype; expected one of {sorted(_DTYPE_BY_NAME)}"
                )
            resolved[k] = _DTYPE_BY_NAME[v]

        # GEMMA4_PRECISION overlays the JSON, e.g.
        #   GEMMA4_PRECISION="shared_mlp=bfp8,attention=bfp8"
        # so a precision A/B does not require editing the shared config file.
        # Cache filenames already carry the dtype suffix, so flipping back and
        # forth re-caches rather than reusing a stale tensor.
        raw_env = (os.getenv("GEMMA4_PRECISION") or "").strip()
        for item in (i.strip() for i in raw_env.split(",") if i.strip()):
            k, _, v = item.partition("=")
            k, v = k.strip(), v.strip()
            if k not in KNOWN_MODULES:
                raise ValueError(f"GEMMA4_PRECISION: unknown module {k!r}; expected one of {sorted(KNOWN_MODULES)}")
            if v not in _DTYPE_BY_NAME:
                raise ValueError(f"GEMMA4_PRECISION: {k}={v!r} — expected one of {sorted(_DTYPE_BY_NAME)}")
            resolved[k] = _DTYPE_BY_NAME[v]
        if raw_env:
            import loguru

            loguru.logger.info(f"[precision] GEMMA4_PRECISION overlay -> {resolved}")
        return cls(resolved)

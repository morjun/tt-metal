#!/usr/bin/env python3
"""Counterbalanced, uninstrumented 12B B=1 K=3 500-token E2E cells.

Uses an isolated Python package overlay pointing at build_codex_current. Every
cell is a fresh pytest process and an identical environment except tuner scope
and head columns. The profiler output knob is explicitly removed.
"""
import hashlib
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OVERLAY = Path("/tmp/lmhead_ttnn_overlay_e2e")


def setup_overlay():
    package = OVERLAY / "ttnn"
    if package.exists():
        shutil.rmtree(package)
    OVERLAY.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO / "ttnn/ttnn", package, symlinks=True)
    (package / "_ttnn.so").unlink()
    (package / "_ttnn.so").symlink_to(REPO / "build_codex_current/ttnn/_ttnn.so")


def parse(log):
    def get(pattern, cast=float):
        match = re.search(pattern, log)
        return cast(match.group(1)) if match else None

    text = re.search(r"== SPEC-DECODE GENERATION ==\n(.*?)\n\s*\d{4}-.*=== Speculative decoding metrics ===", log, re.S)
    return {
        "route": get(r"Spec-decode route=([^ ]+)", str),
        "generated_tokens": get(r"generated tokens: (\d+)", int),
        "accepted_per_3": get(r"mean accepted ([\d.]+)/3"),
        "iterations": get(r"Verify iterations: (\d+)", int),
        "ms_per_iteration": get(r"Verify iterations: \d+ \(([\d.]+) ms/iter\)"),
        "tokens_per_s": get(r"Decode: [\d.]+ ms/token @ ([\d.]+) tok/s/user"),
        "text_sha256": hashlib.sha256(text.group(1).strip().encode()).hexdigest() if text else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cols", type=int, default=35200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.output or REPO / "generated/lmhead_l1_profile" / f"e2e_500_{args.cols}"
    root.mkdir(parents=True, exist_ok=True)
    setup_overlay()
    order = [
        ("tuned", 0),
        ("tuned", args.cols),
        ("tuned", args.cols),
        ("tuned", 0),
        ("default", 0),
        ("default", args.cols),
        ("default", args.cols),
        ("default", 0),
    ]
    results = []
    for index, (scope, cols) in enumerate(order):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("GEMMA4_") and not k.startswith("PLI_") and k not in ("HF_MODEL", "MESH_DEVICE")
        }
        env.update(
            {
                "TT_METAL_RUNTIME_ROOT": str(REPO),
                "TT_METAL_HOME": str(REPO),
                "PYTHONPATH": f"{OVERLAY}:{REPO}",
                "TT_VISIBLE_DEVICES": "0",
                "MESH_DEVICE": "P150",
                "HF_MODEL": "google/gemma-4-12B-it",
                "GEMMA4_ASSISTANT_MODEL": "google/gemma-4-12B-it-assistant",
                "GEMMA4_SPEC_ROUTE": "fused-batch-dim",
                "GEMMA4_SPEC_TRACE": "1",
                "GEMMA4_LMHEAD_L1_COLS": str(cols),
                "GEMMA4_MAX_NEW_TOKENS": "500",
                "GEMMA4_MAX_SEQ_LEN": "4096",
                "GEMMA4_SPEC_PROMPT": (
                    "Write a comprehensive chapter on the history of computing from the abacus to modern computers. "
                    "Include detailed sections on mechanical calculators, early electronic computers, programming "
                    "languages, microprocessors, personal computers, the internet, and contemporary AI. Continue with "
                    "examples and historical context until you have written at least 2,000 words."
                ),
            }
        )
        if scope == "tuned":
            env["GEMMA4_TUNE_MATMULS"] = "1"
        name = f"{index:02d}_{scope}_{cols}"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "models/demos/gemma4/demo/text_demo_v2.py",
            "-k",
            "test_demo_spec_decode",
            "--tt-arch",
            "blackhole",
        ]
        print("running", name, flush=True)
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (root / f"{name}.log").write_text(proc.stdout)
        metrics = parse(proc.stdout)
        row = {
            "index": index,
            "scope": scope,
            "cols": cols,
            "exit_code": proc.returncode,
            "log": f"{name}.log",
            **metrics,
        }
        results.append(row)
        (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(row, flush=True)
        if proc.returncode != 0 or metrics["generated_tokens"] != 500 or metrics["route"] != "fused-traced":
            raise SystemExit(f"cell {name} failed its route/token gate")
    for scope in ("tuned", "default"):
        subset = [r for r in results if r["scope"] == scope]
        texts = {r["text_sha256"] for r in subset}
        accepts = {r["accepted_per_3"] for r in subset}
        iterations = {r["iterations"] for r in subset}
        if len(texts) != 1 or len(accepts) != 1 or len(iterations) != 1:
            raise SystemExit(f"{scope}: text, acceptance or iteration mismatch")
        paired = [
            subset[1]["ms_per_iteration"] - subset[0]["ms_per_iteration"],
            subset[2]["ms_per_iteration"] - subset[3]["ms_per_iteration"],
        ]
        print(scope, "paired split-minus-base ms/iteration", paired, flush=True)


if __name__ == "__main__":
    main()

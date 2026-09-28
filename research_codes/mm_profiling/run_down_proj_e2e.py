#!/usr/bin/env python3
"""End-to-end A/B of the drafter's down_proj L1 pin, with and without gather_in0, on the 12B demo.

Arms (drafter only; the target stays as in every other e2e number):
    dram_mcast   baseline, every weight in DRAM
    shard_mcast  L0.down_proj pinned L1 width-sharded  (GEMMA4_WEIGHTS_IN_L1=sharded, _L1_ONLY, _L1_LAYERS)
    shard_ring   the same + gather_in0 ring            (+ GEMMA4_GATHER_IN0=1)

Same protocol as the lm_head e2e campaign: text_demo_v2.py::test_demo_spec_decode, 12B + 12B
assistant, 1x1, bf16, tuned matmuls, canonical prompt, 500 new tokens, K=3, traced. One fresh
process per cell; round 0 is a discarded warm-up (JIT cold); rounds rotate the arm order so each
arm sits in each position equally often. Resumable: finished cells are skipped.

    python research_codes/mm_profiling/run_down_proj_e2e.py --rounds 6
    python research_codes/mm_profiling/run_down_proj_e2e.py --csv
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PROMPT = (
    "Write a detailed technical explanation of how a modern CPU instruction pipeline works. Cover fetch, "
    "decode, execute, memory access, and writeback in depth, explain hazards and forwarding, and give "
    "concrete examples throughout."
)
PIN = dict(GEMMA4_WEIGHTS_IN_L1="sharded", GEMMA4_L1_ONLY="down_proj", GEMMA4_L1_LAYERS="0")
ARMS = {
    "dram_mcast": dict(GEMMA4_GATHER_IN0="0"),
    "shard_mcast": dict(PIN, GEMMA4_GATHER_IN0="0"),
    "shard_ring": dict(PIN, GEMMA4_GATHER_IN0="1"),
}
TEST = "models/demos/gemma4/demo/text_demo_v2.py::test_demo_spec_decode"
RE = {
    "ms_per_iter": r"Verify iterations: (\d+) \(([\d.]+) ms/iter\)",
    "tok_s_u": r"Decode: [\d.]+ ms/token @ ([\d.]+) tok/s/user",
    "accept": r"mean accepted ([\d.]+)/(\d+)",
    "pinned": r"\[placement:draft\].*L1=([\d.]+) MB/device over (\d+) tensors",
}


def parse(log):
    t = log.read_text(errors="replace")
    it = re.search(RE["ms_per_iter"], t)
    tok = re.search(RE["tok_s_u"], t)
    acc = re.search(RE["accept"], t)
    pin = re.findall(RE["pinned"], t)
    gen = re.search(r"== SPEC-DECODE GENERATION ==\n(.*?)\n.*?=== Speculative decoding metrics", t, re.S)
    if not (it and tok and acc and gen):
        raise RuntimeError(f"{log}: could not parse the demo's result lines")
    return dict(
        n_iters=int(it.group(1)),
        ms_per_iter=float(it.group(2)),
        tok_s_u=float(tok.group(1)),
        accept=float(acc.group(1)),
        l1_mb=float(pin[-1][0]) if pin else 0.0,
        l1_tensors=int(pin[-1][1]) if pin else 0,
        text_sha=hashlib.sha256(gen.group(1).encode()).hexdigest()[:16],
    )


def order(r):
    names = list(ARMS)
    k = r % 3
    rot = names[k:] + names[:k]
    return rot if r % 2 == 0 else list(reversed(rot))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--output", type=Path, default=Path("generated/down_proj_e2e"))
    p.add_argument("--csv", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output if args.output.is_absolute() else root / args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.csv:
        rows = []
        for f in sorted(out.glob("r*_s*_*.log")):
            r, s, arm = f.stem.split("_", 2)
            if (out / f"{f.stem}.done.json").exists():
                rows.append(dict(round=r[1:], slot=s[1:], arm=arm, warmup=r == "r0", **parse(f)))
        path = Path(__file__).with_name("down_proj_e2e_cells.csv")
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda d: (int(d["round"]), int(d["slot"]))))
        print(f"wrote {len(rows)} rows to {path}")
        return
    lock = open("/tmp/gemma4-gather-device0.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GEMMA4_")}
    env.update(
        TT_METAL_HOME=str(root),
        PYTHONPATH=f"{root}:{root}/ttnn:{root}/tools",
        ARCH_NAME="blackhole",
        TT_VISIBLE_DEVICES="0",
        MESH_DEVICE="P150",
        HF_HUB_OFFLINE="1",
        HF_MODEL="google/gemma-4-12B-it",
        GEMMA4_ASSISTANT_MODEL="google/gemma-4-12B-it-assistant",
        GEMMA4_TUNE_MATMULS="1",
        GEMMA4_SHARD_ACTIVATIONS="0",
        GEMMA4_SPEC_TRACE="1",
        GEMMA4_MAX_SEQ_LEN="1024",
        GEMMA4_MAX_NEW_TOKENS="500",
        GEMMA4_PROMPT=PROMPT,
        GEMMA4_SPEC_PROMPT=PROMPT,
    )
    for r in range(args.rounds + 1):  # r0 = warm-up, discarded
        for slot, arm in enumerate(order(r)):
            name = f"r{r}_s{slot}_{arm}"
            log, done = out / f"{name}.log", out / f"{name}.done.json"
            if done.exists():
                continue
            print(f"START {name}", flush=True)
            start = time.monotonic()
            with log.open("w") as fh:
                rc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-s", "-q", "--timeout=3000", TEST],
                    cwd=root,
                    env=dict(env, **ARMS[arm]),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    timeout=3100,
                ).returncode
            if rc:
                raise RuntimeError(f"Cell failed: {name}; campaign stopped, inspect {log}")
            res = parse(log)
            want = (0, 0) if arm == "dram_mcast" else (16.0, 1)
            if (round(res["l1_mb"]), res["l1_tensors"]) != want:
                raise RuntimeError(f"{name}: drafter L1 placement {res['l1_mb']} MB / {res['l1_tensors']} tensors")
            done.write_text(json.dumps(dict(res, seconds=time.monotonic() - start)))
            print(f"DONE {name}: {res['ms_per_iter']:.2f} ms/iter, accept {res['accept']:.2f}", flush=True)


if __name__ == "__main__":
    main()

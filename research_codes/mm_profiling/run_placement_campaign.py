#!/usr/bin/env python3
"""12B down_proj placement x delivery campaign: 6 arms, one toolchain, matched ring count.

Arms = {DRAM, L1-interleaved, L1-width-sharded} x {mcast, ring}. Every ring arm rings the
same 3 matmuls per K=3 trace (L0.down_proj x K): the DRAM+ring arm uses GEMMA4_GATHER_LAYERS
so it does not also ring the 21 other DRAM weights §6.5's arm did. One fresh process per
cell, never two live traces. Rounds are a cyclic 6x6 Latin square, then its mirror, so each
arm sits in each position exactly twice per 12 rounds. Resumable: finished cells are skipped.

    python research_codes/mm_profiling/run_placement_campaign.py --stage validate
    python research_codes/mm_profiling/run_placement_campaign.py --stage timing --rounds 12
    python research_codes/mm_profiling/run_placement_campaign.py --stage csv
"""
import argparse
import csv
import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ARMS = {
    "dram_mcast": dict(GEMMA4_L1_RELOC_LAYERS="none", GEMMA4_GATHER_IN0="0"),
    "dram_ring": dict(
        GEMMA4_L1_RELOC_LAYERS="none",
        GEMMA4_GATHER_IN0="1",
        GEMMA4_GATHER_DRAM_WEIGHT="1",
        GEMMA4_GATHER_LAYERS="0",
    ),
    "il_mcast": dict(GEMMA4_L1_RELOC_LAYERS="0", GEMMA4_L1_PLACEMENT="interleaved", GEMMA4_GATHER_IN0="0"),
    "il_ring": dict(
        GEMMA4_L1_RELOC_LAYERS="0",
        GEMMA4_L1_PLACEMENT="interleaved",
        GEMMA4_GATHER_IN0="1",
        GEMMA4_GATHER_L1_INTERLEAVED="1",
    ),
    "shard_mcast": dict(GEMMA4_L1_RELOC_LAYERS="0", GEMMA4_L1_PLACEMENT="sharded", GEMMA4_GATHER_IN0="0"),
    "shard_ring": dict(GEMMA4_L1_RELOC_LAYERS="0", GEMMA4_L1_PLACEMENT="sharded", GEMMA4_GATHER_IN0="1"),
}
# (alloc_per_bank, ring plans per K=3 trace) each arm must show, or the cell is mislabelled.
EXPECT = {
    "dram_mcast": (0, 0),
    "dram_ring": (0, 3),
    "il_mcast": (153600, 0),
    "il_ring": (153600, 3),
    "shard_mcast": (524288, 0),
    "shard_ring": (524288, 3),
}
TEST = "models/demos/gemma4/tests/unit/test_assistant_standalone_l1.py::test_gather_matched_trace"


def _order(r):
    names = list(ARMS)
    rot = names[r % 6 :] + names[: r % 6]
    return rot if r < 6 else list(reversed(rot))


def _clock(row, key):
    snap = row.get(key)
    if isinstance(snap, list):
        snap = snap[-1]
    info = (snap or {}).get("device_info") or [{}]
    t = info[0].get("telemetry") or {}
    return t.get("aiclk"), t.get("asic_temperature")


def _sfpi(root):
    """The SFPI version the kernels are actually compiled with (what the runtime PROVIDES,
    as check_sfpi_toolchain.sh reads it), not what the source pins."""
    gpp = root / "runtime" / "sfpi" / "compiler" / "bin" / "riscv-tt-elf-g++"
    m = re.search(r"sfpi:([0-9.]+)", subprocess.check_output([str(gpp), "--version"], text=True))
    return m.group(1) if m else "unknown"


def run_cell(root, env, out, name, arm, mode, replays, retries):
    target, done = out / f"{name}.jsonl", out / f"{name}.done.json"
    if done.exists():
        return
    if target.exists():
        raise RuntimeError(f"Incomplete previous cell: preserve and rename {target} before retry")
    child = dict(
        env,
        **ARMS[arm],
        GEMMA4_LEDGER_K="3",
        GEMMA4_DIAG_WARMUP="3",
        GEMMA4_DIAG_OUT=str(target.resolve()),
        GEMMA4_DIAG_MODE=mode,
        GEMMA4_DIAG_REPLAYS=str(replays),
    )
    start = time.monotonic()
    for attempt in range(retries + 1):
        print(f"START {name}" + (f" (attempt {attempt})" if attempt else ""), flush=True)
        with (out / f"{name}.log").open("w") as log:
            rc = subprocess.run(
                [sys.executable, "-m", "pytest", "-s", "-q", "--timeout=1800", TEST],
                cwd=root,
                env=child,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1900,
            ).returncode
        if rc:
            raise RuntimeError(f"Cell failed: {name}; campaign stopped, inspect its log")
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        cap = next(r for r in rows if r["event"] == "capture")
        want_alloc, want_ring = EXPECT[arm]
        got = (cap["l1_state"]["alloc_per_bank"], cap["observed_ring"])
        if got != (want_alloc, want_ring):
            raise RuntimeError(f"{name}: placement gate failed, (alloc, ring) = {got}, want {EXPECT[arm]}")
        if all(r.get("valid_clock", True) for r in rows):
            break
        # Preserve the rejected capture; never overwrite raw data.
        target.rename(out / f"{name}.rejected{attempt}.jsonl")
        (out / f"{name}.log").rename(out / f"{name}.rejected{attempt}.log")
        print(f"REJECT {name} attempt {attempt}: clock moved", flush=True)
    else:
        (out / f"{name}.excluded.json").write_text(json.dumps({"reason": "clock_changed"}))
        print(f"EXCLUDE {name}: clock moved on every attempt", flush=True)
        return
    done.write_text(json.dumps({"seconds": time.monotonic() - start}))
    print(f"DONE {name}: {time.monotonic() - start:.1f}s", flush=True)


def write_csv(root, out):
    fields = "round,slot,arm,trace_us,replays,valid_clock,aiclk_pre,aiclk_post,temp_pre,temp_post,"
    fields += "alloc_per_bank,ring_plans,commit,harness_sha256,sfpi,warmup_cell"
    rows = []
    for f in sorted(out.glob("timing_*.jsonl")):
        if ".rejected" in f.name:
            continue
        _, r, slot, arm = f.stem.split("_", 3)
        recs = [json.loads(line) for line in f.read_text().splitlines()]
        cap = next(x for x in recs if x["event"] == "capture")
        for t in (x for x in recs if x["event"] == "timing"):
            (cp, tp), (cq, tq) = _clock(t, "pre"), _clock(t, "post")
            rows.append(
                dict(
                    round=r[1:],
                    slot=slot[1:],
                    arm=arm,
                    trace_us=t["trace_us"],
                    replays=t["replays"],
                    valid_clock=t["valid_clock"],
                    aiclk_pre=cp,
                    aiclk_post=cq,
                    temp_pre=tp,
                    temp_post=tq,
                    alloc_per_bank=cap["l1_state"]["alloc_per_bank"],
                    ring_plans=cap["observed_ring"],
                    commit=t["commit"],
                    harness_sha256=t["harness_sha256"],
                    sfpi=t["environment"].get("GEMMA4_SFPI"),
                    warmup_cell=r == "rW",
                )
            )
    path = Path(__file__).with_name("placement6_cells.csv")
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields.split(","))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda d: (d["round"] != "W", d["round"].zfill(3), int(d["slot"]))))
    print(f"wrote {len(rows)} rows to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["validate", "timing", "csv"], required=True)
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--replays", type=int, default=400)
    p.add_argument("--clock-retries", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("generated/placement6"))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = (root / args.output) if not args.output.is_absolute() else args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "csv":
        return write_csv(root, out)
    lock = open("/tmp/gemma4-gather-device0.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GEMMA4_")}
    env.update(
        TT_METAL_HOME=str(root),
        PYTHONPATH=f"{root}:{root}/ttnn:{root}/tools",
        MPLCONFIGDIR="/tmp/gemma4-mpl",
        ARCH_NAME="blackhole",
        TT_VISIBLE_DEVICES="0",
        MESH_DEVICE="P150",
        HF_HUB_OFFLINE="1",
        GEMMA4_ASSISTANT_MODEL="google/gemma-4-12B-it-assistant",
        GEMMA4_TUNE_MATMULS="1",
        GEMMA4_SHARD_ACTIVATIONS="0",
        GEMMA4_SFPI=_sfpi(root),
    )
    if args.stage == "validate":
        for arm in ARMS:
            run_cell(root, env, out, f"validate_{arm}", arm, "validate", 1, 0)
        return
    for arm in _order(0):  # one discarded warm-up cell per arm (JIT cold)
        run_cell(root, env, out, f"timing_rW_s{list(ARMS).index(arm)}_{arm}", arm, "timing", args.replays, 0)
    for r in range(args.rounds):
        for slot, arm in enumerate(_order(r)):
            run_cell(root, env, out, f"timing_r{r}_s{slot}_{arm}", arm, "timing", args.replays, args.clock_retries)


if __name__ == "__main__":
    main()

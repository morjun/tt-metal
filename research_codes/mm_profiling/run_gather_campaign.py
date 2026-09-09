#!/usr/bin/env python3
"""Sequential, resumable fresh-process campaign; never alternates live traces."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _clock(row):
    """The cell's AICLK, from its post-batch snapshot. None when telemetry is absent --
    missing telemetry is not evidence the clock held (GATHER_DIAGNOSIS.md)."""
    post = row.get("post")
    if not isinstance(post, dict):
        return None
    info = post.get("device_info") or [{}]
    return (info[0].get("telemetry") or {}).get("aiclk") if isinstance(info[0], dict) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        choices=["validate", "baseline", "replays", "warmup", "same", "graph", "capture_graph"],
        required=True,
    )
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--ks", type=int, nargs="+")
    p.add_argument("--replays", type=int, nargs="+", help="Override timed batch counts; reverse order on odd rounds")
    p.add_argument("--output", type=Path, default=Path("generated/gather_diagnosis"))
    # A cell whose AICLK moves during its measured batch is contaminated: 1350 -> 1343 MHz
    # is 0.52%, which on a K=3 trace is ~16.5 us -- the size of the effect being measured
    # (MEASUREMENT_RECORD.md 7.0). The clock cannot be pinned on this platform: tt-smi has
    # no clock control, and the driver's `power_policy` parameter is read-only at runtime
    # AND governs only the 800 MHz idle floor, not the loaded droop this guard catches.
    # So the recovery is to re-run the cell and, failing that, exclude it -- never to
    # accept it. `abort` stays the default so existing behaviour is unchanged.
    p.add_argument("--on-clock-change", choices=["abort", "retry"], default="abort")
    p.add_argument("--clock-retries", type=int, default=2)
    args = p.parse_args()
    if args.rounds < 1 or any(n < 1 for n in (args.replays or []) + (args.ks or [])):
        p.error("rounds, K and replay counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = open("/tmp/gemma4-gather-device0.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    # Remove inherited experiment settings; record effective settings in child.
    env = {k: v for k, v in env.items() if not k.startswith("GEMMA4_")}
    env.update(
        TT_METAL_HOME=str(root),
        PYTHONPATH=f"{root}:{root}/ttnn:{root}/tools",
        MPLCONFIGDIR="/tmp/gemma4-mpl",
        ARCH_NAME="blackhole",
        TT_VISIBLE_DEVICES="0",
        MESH_DEVICE="P150",
        HF_HUB_OFFLINE="1",
        GEMMA4_ASSISTANT_MODEL="google/gemma-4-E2B-it-assistant",
        GEMMA4_TUNE_MATMULS="1",
        GEMMA4_SHARD_ACTIVATIONS="0",
    )
    ks = args.ks or (
        {
            "validate": [1, 3],
            "baseline": [1, 2, 3, 4, 6, 8],
            "replays": [1, 2, 3, 8],
            "warmup": [1, 3],
            "same": [1, 3],
            "graph": [1, 2, 3, 8],
            "capture_graph": [3, 4, 5],
        }[args.stage]
    )
    rounds = 1 if args.stage in {"validate", "graph"} else args.rounds
    settings = dict(stage=args.stage, ks=ks, replays=args.replays, warmup_order="counterbalanced_by_round_v1")
    settings_path = args.output / f"{args.stage}.settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise RuntimeError("Campaign settings changed; use a new output directory")
    elif any(args.output.glob(f"{args.stage}_*.jsonl")):
        raise RuntimeError("Legacy campaign has no settings manifest; use a new output directory")
    else:
        settings_path.write_text(json.dumps(settings, indent=2))
    for r in range(rounds):
        counts = args.replays or ([20, 100, 400] if args.stage == "replays" else [20])
        counts = counts if r % 2 == 0 else list(reversed(counts))
        warmups = ([3, 20] if r % 2 == 0 else [20, 3]) if args.stage == "warmup" else [3]
        for k in ks if r % 2 == 0 else list(reversed(ks)):
            arms = ["0", "1"] if r % 2 == 0 else ["1", "0"]
            if args.stage == "same":
                arms = ["0", "0"] if r % 2 == 0 else ["1", "1"]
            for slot, arm in enumerate(arms):
                for warmup in warmups:
                    name = f"{args.stage}_r{r}_k{k}_a{arm}_s{slot}_w{warmup}"
                    target = args.output / (name + ".jsonl")
                    done = args.output / (name + ".done.json")
                    if done.exists():
                        continue
                    if target.exists():
                        raise RuntimeError(f"Incomplete previous cell: preserve and rename {target} before retry")
                    child = dict(
                        env,
                        GEMMA4_GATHER_IN0=arm,
                        GEMMA4_LEDGER_K=str(k),
                        GEMMA4_DIAG_WARMUP=str(warmup),
                        GEMMA4_DIAG_OUT=str(target.resolve()),
                        GEMMA4_DIAG_MODE=args.stage
                        if args.stage in {"validate", "graph", "capture_graph"}
                        else "timing",
                        GEMMA4_DIAG_REPLAYS=",".join(map(str, counts)),
                    )
                    start = time.monotonic()
                    attempts = []
                    for attempt in range(args.clock_retries + 1 if args.on_clock_change == "retry" else 1):
                        print(f"START {name}" + (f" (attempt {attempt})" if attempt else ""), flush=True)
                        with (args.output / (name + ".log")).open("w") as log:
                            result = subprocess.run(
                                [
                                    sys.executable,
                                    "-m",
                                    "pytest",
                                    "-s",
                                    "-q",
                                    "--timeout=1800",
                                    "models/demos/gemma4/tests/unit/test_assistant_standalone_l1.py::test_gather_matched_trace",
                                ],
                                cwd=root,
                                env=child,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=1900,
                            )
                        if result.returncode:
                            raise RuntimeError(f"Cell failed: {name}; campaign stopped, inspect its log")
                        records = [json.loads(line) for line in target.read_text().splitlines()]
                        bad = [row for row in records if not row.get("valid_clock", True)]
                        if not bad:
                            break
                        if args.on_clock_change == "abort":
                            raise RuntimeError(f"Clock changed in {name}; campaign stopped for qualification")
                        # Preserve the rejected capture; never overwrite raw data.
                        kept = args.output / f"{name}.rejected{attempt}.jsonl"
                        target.rename(kept)
                        (args.output / (name + ".log")).rename(args.output / f"{name}.rejected{attempt}.log")
                        attempts.append(
                            {
                                "attempt": attempt,
                                "kept": kept.name,
                                "clocks": sorted({_clock(row) for row in records if _clock(row)}),
                            }
                        )
                        print(f"REJECT {name} attempt {attempt}: clock moved; preserved as {kept.name}", flush=True)
                    else:
                        (args.output / (name + ".excluded.json")).write_text(
                            json.dumps(
                                {"reason": "clock_changed", "attempts": attempts, "seconds": time.monotonic() - start},
                                indent=2,
                            )
                        )
                        print(f"EXCLUDE {name}: clock moved on every attempt; continuing", flush=True)
                        continue
                    done.write_text(
                        json.dumps(
                            {"seconds": time.monotonic() - start, "returncode": 0, "rejected_attempts": attempts}
                        )
                    )
                    print(
                        f"DONE {name}: {time.monotonic() - start:.1f}s"
                        + (f" after {len(attempts)} clock rejection(s)" if attempts else ""),
                        flush=True,
                    )


if __name__ == "__main__":
    main()

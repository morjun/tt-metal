"""Host-only qualification helpers for the matched gather investigation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def snapshot():
    result = subprocess.run(
        [os.path.expanduser("~/.tenstorrent-venv/bin/tt-smi"), "-s", "--snapshot_no_tty"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    start = result.stdout.find("{")
    data = json.loads(result.stdout[start:])
    if not data.get("device_info"):
        raise RuntimeError("No device telemetry")
    return data


def readings(data, bus_id):
    matches = [d for d in data["device_info"] if d["board_info"]["bus_id"] == bus_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one device at {bus_id}")
    d = matches[0]
    t = d["telemetry"]
    return float(t["aiclk"]), float(t["asic_temperature"])


def stable(samples, bus_id):
    values = [readings(s, bus_id) for s in samples]
    return (
        len(values) >= 3
        and len({c for c, _ in values[-3:]}) == 1
        and max(t for _, t in values[-3:]) - min(t for _, t in values[-3:]) <= 1.0
    )


def stabilize(bus_id):
    """Idle outside the timed batch, retaining every qualification sample."""
    time.sleep(10)
    deadline = time.monotonic() + 120
    samples = []
    while time.monotonic() < deadline:
        samples.append(snapshot())
        if stable(samples, bus_id):
            return samples
        time.sleep(2)
    raise RuntimeError("Device did not stabilize within 120 seconds")


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    source = Path("models/demos/gemma4/tests/unit/test_assistant_standalone_l1.py")
    return {
        "commit": git("rev-parse", "HEAD"),
        "status": git("status", "--short"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "harness_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("GEMMA4_", "TT_METAL_", "TT_VISIBLE", "ARCH_", "MESH_", "HF_"))
            or k in {"PYTHONPATH", "VIRTUAL_ENV"}
        },
    }


def append_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())

"""CPU checks; run with pytest --confcutdir=research_codes/mm_profiling."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_codes.mm_profiling.analyze_bursts import bursts
from research_codes.mm_profiling.gather_measurement import readings, stable
from research_codes.mm_profiling.analyze_gather_campaign import bootstrap, paired_contrast, wallclock_reports


def telemetry(clock=800, temp=50):
    return {
        "device_info": [
            {"board_info": {"bus_id": "board"}, "telemetry": {"aiclk": str(clock), "asic_temperature": str(temp)}}
        ]
    }


def test_thermal_gate():
    assert stable([telemetry(), telemetry(temp=50.5), telemetry()], "board")
    assert not stable([telemetry(), telemetry(1350), telemetry()], "board")
    assert not stable([telemetry(), telemetry(temp=52), telemetry()], "board")
    with pytest.raises(RuntimeError):
        readings(telemetry(), "wrong-board")


def test_long_stall_is_not_a_boundary():
    ts = [1, 5, 335, 340, 350, 355, 685, 690]
    assert bursts(ts, 1, 4) == ([], [])
    actual, _ = bursts(ts, 1, 4, [[0, 4], [4, 8]])
    assert actual == [ts[:4], ts[4:]]


@pytest.mark.parametrize("ranges", [[[0, 4], [4, 9]], [[0, 4], [3, 7]], [[0, 3]]])
def test_invalid_capture_ranges(ranges):
    with pytest.raises(ValueError):
        bursts(list(range(8)), 1, 4, ranges)


def test_bootstrap_uses_independent_samples():
    assert bootstrap([23]) is None
    assert bootstrap([23, 23, 23]) == [23, 23]
    assert bootstrap([20, 25, 23, 21]) == bootstrap([20, 25, 23, 21])


def test_contrast_matches_rounds_before_subtraction():
    left = {"samples": [(0, 100), (1, 20), (2, 30), (9, 999)]}
    right = {"samples": [(2, 29), (0, 99), (1, 19)]}
    result = paired_contrast(left, right, "test")
    assert result["rounds"] == [0, 1, 2]
    assert result["median_delta_us"] == 1
    assert result["ci95_us"] == [1, 1]


def test_absolute_contrasts_require_common_rounds_settings_and_clocks():
    def arms(t, clock=1350):
        return {0: (t, clock), 1: (t + 10, clock)}

    result = wallclock_reports(
        {
            (3, 400, 3): {0: arms(300), 1: arms(3000), 2: arms(9000)},
            (4, 400, 3): {0: arms(410), 1: arms(3100, 1343)},
            (5, 400, 3): {0: arms(500), 1: arms(3200)},
            (6, 20, 3): {0: arms(600)},
        }
    )
    assert result["absolute_wallclock"][0]["mcast"]["process_rounds"] == 3
    inc = result["within_arm_k_increments"][0]
    assert inc["samples"] == [(0, 110)]
    assert inc["dropped_cross_k_clock_rounds"] == [1]
    assert all(r["ks"] != [5, 6] for r in result["within_arm_k_increments"])
    curve = result["within_arm_midpoint_residuals"][0]
    assert curve["samples"] == [(0, 10)]
    assert curve["ci95_us"] is None


def test_launcher_controls_survive_environment_cleanup(tmp_path, monkeypatch):
    from research_codes.mm_profiling import run_gather_campaign as launcher

    calls = []

    def run(argv, **kwargs):
        env = kwargs["env"]
        calls.append((env["GEMMA4_GATHER_IN0"], env["GEMMA4_DIAG_WARMUP"], env["GEMMA4_DIAG_REPLAYS"]))
        Path(env["GEMMA4_DIAG_OUT"]).write_text(json.dumps({"event": "timing", "valid_clock": True}) + "\n")
        return SimpleNamespace(returncode=0)

    args = [
        "launcher",
        "--stage",
        "warmup",
        "--ks",
        "3",
        "--replays",
        "400",
        "--rounds",
        "2",
        "--output",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher.fcntl, "flock", lambda *args: None)
    monkeypatch.setenv("GEMMA4_DIAG_REPLAYS", "7")
    launcher.main()
    assert calls == [
        ("0", "3", "400"),
        ("0", "20", "400"),
        ("1", "3", "400"),
        ("1", "20", "400"),
        ("1", "20", "400"),
        ("1", "3", "400"),
        ("0", "20", "400"),
        ("0", "3", "400"),
    ]
    monkeypatch.setattr(sys, "argv", ["100" if x == "400" else x for x in args])
    with pytest.raises(RuntimeError, match="settings changed"):
        launcher.main()


def test_canonical_body_preserves_recurrence_and_ownership():
    # Execute the real factory without importing device-dependent pytest fixtures.
    path = Path(__file__).resolve().parents[2] / "models/demos/gemma4/tests/unit/test_assistant_standalone_l1.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_make_fused_k_body")
    calls, freed = [], []

    class Tensor:
        def __init__(self, name):
            self.name = name

        def deallocate(self, force):
            freed.append(self.name)

    def step(tok, hidden, *args, **kwargs):
        calls.append((tok.name, hidden.name))
        i = len(calls)
        return Tensor(f"logits{i}"), Tensor(f"hidden{i}")

    ns = {
        "ttnn": SimpleNamespace(reshape=lambda x, shape: x),
        "_argmax_token": lambda assistant, logits, rows: Tensor(logits.name.replace("logits", "token")),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    rig = dict(
        assistant=SimpleNamespace(step=step),
        token=Tensor("seed"),
        hidden=Tensor("initial"),
        shared_kv=None,
        page_tables=None,
        pos_uint32=None,
        pos_int32=None,
    )
    body = ns["_make_fused_k_body"](rig, 3)
    assert calls == []
    outputs = body()
    assert calls == [("seed", "initial"), ("token1", "hidden1"), ("token2", "hidden2")]
    assert freed == ["logits1", "logits2", "logits3"]
    assert [x.name for x in outputs] == ["token3", "hidden3"]
    body()
    assert calls[3] == ("seed", "initial")

#!/usr/bin/env python3
"""Paired process-level estimates. No eager/profiled residual arithmetic."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np


def bootstrap(values, seed=7):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    estimates = np.median(rng.choice(values, (10000, len(values)), replace=True), axis=1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def paired_contrast(left, right, label):
    """Difference of round-matched arm deltas; never subtract marginal medians."""
    a, b = dict(left["samples"]), dict(right["samples"])
    rounds = sorted(a.keys() & b.keys())
    values = [a[r] - b[r] for r in rounds]
    return dict(
        label=label,
        rounds=rounds,
        pairs=len(values),
        median_delta_us=float(np.median(values)) if values else None,
        ci95_us=bootstrap(values),
        samples=list(zip(rounds, values)),
    )


def _clock(row):
    """The cell's AICLK from its post-batch snapshot, or None when telemetry is absent.
    `valid_clock` says the clock held WITHIN a cell; it does not say two cells ran at the
    same clock. A 1350 vs 1343 pair is 0.52% = ~16.5 us on a K=3 trace, the size of the
    effect (MEASUREMENT_RECORD.md 7.0), and it does NOT cancel in the paired delta."""
    post = row.get("post")
    if not isinstance(post, dict):
        return None
    info = post.get("device_info") or [{}]
    return (info[0].get("telemetry") or {}).get("aiclk") if isinstance(info[0], dict) else None


def summarize_times(samples):
    values = [v for _, v in samples]
    return dict(
        process_rounds=len(samples),
        samples=samples,
        median_us=float(np.median(values)) if values else None,
        ci95_us=bootstrap(values),
        range_us=[min(values), max(values)] if values else None,
    )


def wallclock_reports(accepted):
    """Use accepted A/B pairs, and equal clocks across K for absolute contrasts."""
    absolute, increments, curvature = [], [], []
    for (k, n, w), rounds in sorted(accepted.items()):
        row = dict(k=k, replays=n, warmup=w)
        for arm, name in enumerate(("mcast", "ring")):
            row[name] = summarize_times([(r, arms[arm][0]) for r, arms in sorted(rounds.items())])
        absolute.append(row)
        for kind, ks, coefficients, target in (
            ("increment", [k, k + 1], [-1, 1], increments),
            ("midpoint_residual", [k - 1, k, k + 1], [-0.5, 1, -0.5], curvature),
        ):
            if not all((other, n, w) in accepted for other in ks):
                continue
            groups = [accepted[other, n, w] for other in ks]
            common = sorted(set.intersection(*(set(g) for g in groups)))
            for arm, name in enumerate(("mcast", "ring")):
                samples, dropped = [], []
                for r in common:
                    if len({g[r][arm][1] for g in groups}) != 1:
                        dropped.append(r)
                        continue
                    samples.append((r, sum(c * g[r][arm][0] for c, g in zip(coefficients, groups))))
                target.append(
                    dict(
                        kind=kind,
                        ks=ks,
                        arm=name,
                        replays=n,
                        warmup=w,
                        dropped_cross_k_clock_rounds=dropped,
                        **summarize_times(samples),
                    )
                )
    return dict(
        absolute_wallclock=absolute, within_arm_k_increments=increments, within_arm_midpoint_residuals=curvature
    )


def markdown_report(result):
    def number(value):
        return "—" if value is None else f"{value:,.3f}"

    def interval(value):
        return "—" if value is None else f"[{number(value[0])}, {number(value[1])}]"

    lines = [
        f"# {result['stage']} wall-clock report",
        "",
        "Units: µs per complete K-step trace. Host replay loop plus final synchronization; setup and warm-up excluded.",
        "Arm medians use the same accepted process pairs as paired deltas. Difference of arm medians need not equal median paired delta.",
        "",
        "| K | Replays | Warm-ups | Pairs | Multicast [95% CI] | Gather [95% CI] | Paired gather−multicast [95% CI] |",
        "|---:|---:|---:|---:|---|---|---|",
    ]
    indexed = {(r["k"], r["replays"], r["warmup"]): r for r in result["rows"]}
    for row in result.get("absolute_wallclock", []):
        delta = indexed[row["k"], row["replays"], row["warmup"]]
        cells = [f"{number(row[a]['median_us'])} {interval(row[a]['ci95_us'])}" for a in ("mcast", "ring")]
        lines.append(
            f"| {row['k']} | {row['replays']} | {row['warmup']} | {delta['pairs']} | "
            + " | ".join(cells)
            + f" | {number(delta['median_delta_us'])} {interval(delta['ci95_us'])} |"
        )
    for key, title in (
        ("within_arm_k_increments", "Within-arm K increments"),
        ("within_arm_midpoint_residuals", "Within-arm midpoint residuals"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "Computed within matched process rounds at equal reported clocks and identical replay/warm-up settings.",
            "These compare separately captured traces; they are not timestamps of individual steps or causal decompositions.",
            "",
            "| Arm | K values | Replays | Warm-ups | Rounds | Median [95% CI] | Clock-excluded rounds |",
            "|---|---|---:|---:|---:|---|---|",
        ]
        for row in result.get(key, []):
            lines.append(
                f"| {row['arm']} | {row['ks']} | {row['replays']} | {row['warmup']} | {row['process_rounds']} | "
                f"{number(row['median_us'])} {interval(row['ci95_us'])} | {row['dropped_cross_k_clock_rounds']} |"
            )
    lines += [
        "",
        "Increments are T(K+1)−T(K). Midpoint residuals are T(K)−[T(K−1)+T(K+1)]/2.",
        "Intervals bootstrap independent process rounds; they do not bound systematic measurement error.",
        "",
    ]
    return "\n".join(lines)


def analyze(root, stage):
    pairs = defaultdict(dict)
    for path in sorted(root.glob(f"{stage}_r*_k*_a*_s*_w*.jsonl")):
        if not path.with_suffix(".done.json").exists():
            continue
        match = re.search(r"_r(\d+)_k(\d+)_a([01])_s(\d+)_w(\d+)", path.name)
        r, k, arm, slot, warmup = map(int, match.groups())
        for row in map(json.loads, path.read_text().splitlines()):
            if row["event"] == "timing" and row["valid_clock"]:
                pairs[(k, row["replays"], warmup, r)][slot if stage == "same" else arm] = (row["trace_us"], _clock(row))
    groups = defaultdict(list)
    accepted = defaultdict(dict)
    dropped = []
    for (k, n, w, r), arms in sorted(pairs.items()):
        if set(arms) != {0, 1}:
            continue
        clocks = {arms[0][1], arms[1][1]}
        # Both members must have run at ONE reported clock. Unknown telemetry is not
        # evidence the clocks matched, so a None drops the pair too.
        if len(clocks) != 1 or None in clocks:
            dropped.append({"k": k, "replays": n, "warmup": w, "round": r, "clocks_mhz": [arms[0][1], arms[1][1]]})
            continue
        groups[(k, n, w)].append((r, arms[1][0] - arms[0][0]))
        accepted[(k, n, w)][r] = arms
    rows = []
    for (k, n, w), samples in sorted(groups.items()):
        values = [v for _, v in samples]
        rows.append(
            dict(
                k=k,
                replays=n,
                warmup=w,
                pairs=len(values),
                median_delta_us=float(np.median(values)),
                range_us=[min(values), max(values)],
                ci95_us=bootstrap(values),
                samples=samples,
            )
        )
    result = {
        "stage": stage,
        "rows": rows,
        "units": "us per complete trace",
        "dropped_clock_mismatch": dropped,
        "excluded_cells": sorted(p.name for p in root.glob(f"{stage}_*.excluded.json")),
    }
    if stage != "same":
        result.update(wallclock_reports(accepted))
    else:
        result[
            "absolute_wallclock_note"
        ] = "Same-arm controls alternate physical arms by round; slot times are not mcast/ring arms"
    contrasts = []
    indexed = {(row["k"], row["replays"], row["warmup"]): row for row in rows}
    for k, n, w in indexed:
        left = indexed[k, n, w]
        if stage == "baseline" and k == 4:
            for other in (3, 5):
                if (other, n, w) in indexed:
                    contrasts.append(
                        paired_contrast(left, indexed[other, n, w], f"delta_K4-minus-K{other}; N={n}; warmup={w}")
                    )
        if stage == "warmup" and w == 20 and (k, n, 3) in indexed:
            contrasts.append(paired_contrast(left, indexed[k, n, 3], f"delta_warmup20-minus-warmup3; K={k}; N={n}"))
    result["paired_contrasts"] = contrasts
    if stage == "baseline" and len({(row["replays"], row["warmup"]) for row in rows}) > 1:
        result["descriptive_models_skipped"] = "Multiple replay/warm-up settings; do not mix them in a K fit"
    elif stage == "baseline" and len(rows) >= 3:
        # Resample complete round blocks to preserve cross-K run-order structure.
        by_k = {row["k"]: dict(row["samples"]) for row in rows}
        common = sorted(set.intersection(*(set(v) for v in by_k.values())))
        if len(common) >= 2:
            ks = np.array(sorted(by_k), dtype=float)
            ys = np.array([[by_k[int(k)][r] for k in ks] for r in common])
            models = {
                "proportional": ks[:, None],
                "affine": np.column_stack([np.ones(len(ks)), ks]),
                "step_plus_slope": np.column_stack([ks >= 2, ks]),
            }
            fits = {}
            rng = np.random.default_rng(7)
            indices = rng.integers(len(common), size=(2000, len(common)))
            for name, x in models.items():
                y = np.median(ys, axis=0)
                beta = np.linalg.lstsq(x, y, rcond=None)[0]
                bs = np.array([np.linalg.lstsq(x, np.median(ys[i], axis=0), rcond=None)[0] for i in indices])
                fits[name] = dict(
                    coefficients=beta.tolist(),
                    ci95=np.quantile(bs, [0.025, 0.975], axis=0).tolist(),
                    residuals_us=(y - x @ beta).tolist(),
                )
            result["descriptive_models_not_mechanisms"] = fits
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--stage", default="baseline")
    parser.add_argument("--markdown", type=Path, help="Also write an absolute/paired wall-clock report")
    args = parser.parse_args()
    result = analyze(args.root, args.stage)
    if args.markdown:
        args.markdown.write_text(markdown_report(result))
    print(json.dumps(result, indent=2))

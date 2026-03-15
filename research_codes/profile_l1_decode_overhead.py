import argparse
import json
import os
import subprocess
from pathlib import Path


IMPORTANT_KEYS = [
    "decode.prepare_inputs_host",
    "decode.host_to_device",
    "decode.transform_inputs_device",
    "decode.model_forward",
    "decode.dram_kv_write",
    "decode.l1_clone_path",
    "decode.l1_index_path",
    "decode.l1_kv_write",
    "decode.l1_sharded_view",
    "decode.sdpa_call",
    "decode.output_all_gather",
    "decode.output_untilize",
    "decode.output_to_dram",
    "decode.output_readback",
    "decode.output_postprocess",
    "decode.expected_l1_hit_ratio",
]


def run_command(command, report_path, cwd):
    env = os.environ.copy()
    env["TT_L1_KV_PERF"] = "1"
    env["TT_L1_KV_PERF_REPORT"] = str(report_path)
    completed = subprocess.run(command, shell=True, cwd=cwd, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {command}")


def load_report(path):
    with open(path, "r") as f:
        return json.load(f)["stats"]


def get_avg_ms(stats, key):
    bucket = stats.get(key)
    if not bucket:
        return None
    return bucket.get("avg_ms")


def write_summary(path, dual_stats, dram_stats):
    lines = ["# L1 KV Decode Breakdown", ""]
    lines.append("| Metric | Dual-source avg ms | DRAM-only avg ms | Delta ms |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in IMPORTANT_KEYS:
        dual = get_avg_ms(dual_stats, key)
        dram = get_avg_ms(dram_stats, key)
        if key == "decode.expected_l1_hit_ratio":
            dual_ratio = dual_stats.get(key, {}).get("avg")
            dram_ratio = dram_stats.get(key, {}).get("avg")
            dual_str = "-" if dual_ratio is None else f"{dual_ratio:.3f}"
            dram_str = "-" if dram_ratio is None else f"{dram_ratio:.3f}"
            delta_str = "-"
            if dual_ratio is not None and dram_ratio is not None:
                delta_str = f"{dual_ratio - dram_ratio:+.3f}"
            lines.append(f"| `{key}` | {dual_str} | {dram_str} | {delta_str} |")
            continue
        if dual is None and dram is None:
            continue
        dual_str = "-" if dual is None else f"{dual:.3f}"
        dram_str = "-" if dram is None else f"{dram:.3f}"
        delta_str = "-"
        if dual is not None and dram is not None:
            delta_str = f"{dual - dram:+.3f}"
        lines.append(f"| `{key}` | {dual_str} | {dram_str} | {delta_str} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run matched dual-source vs DRAM-only decode benchmarks.")
    parser.add_argument("--dual-source-cmd", required=True, help="Shell command for the dual-source L1 KV run.")
    parser.add_argument("--dram-only-cmd", required=True, help="Shell command for the DRAM-only baseline run.")
    parser.add_argument("--working-directory", default=".", help="Working directory for both commands.")
    parser.add_argument("--output-dir", default="research_codes/l1_kv_perf", help="Directory for reports.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dual_report = output_dir / "dual_source_report.json"
    dram_report = output_dir / "dram_only_report.json"

    run_command(args.dual_source_cmd, dual_report, args.working_directory)
    run_command(args.dram_only_cmd, dram_report, args.working_directory)

    dual_stats = load_report(dual_report)
    dram_stats = load_report(dram_report)

    summary_path = output_dir / "summary.md"
    write_summary(summary_path, dual_stats, dram_stats)
    print(summary_path)


if __name__ == "__main__":
    main()

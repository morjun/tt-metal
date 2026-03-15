import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


DTYPE_BYTES = {
    "bfloat16": 2.0,
    "bfloat8_b": 1.0,
    "bfloat4_b": 0.5,
}


def align_down(value, multiple):
    if multiple <= 0:
        return value
    return (value // multiple) * multiple


def run_with_snapshot(command, snapshot_path, cwd):
    completed = subprocess.run(
        f"{command} --l1_memory_view_path {shlex.quote(str(snapshot_path))}",
        shell=True,
        cwd=cwd,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {command}")


def load_snapshot(path, label):
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported snapshot payload in {path}")
    for item in payload:
        if item.get("label") == label:
            return item
    raise ValueError(f"Snapshot label '{label}' not found in {path}")


def summarize_window(snapshot, args):
    bytes_per_token_per_layer = (
        2 * args.batch_size_per_device_group * args.num_local_kv_heads * args.head_dim * DTYPE_BYTES[args.kv_dtype]
    )
    bytes_per_token_all_layers = bytes_per_token_per_layer * args.num_layers
    largest_interleavable_free = snapshot["largest_interleavable_free_bytes_estimate"]
    safe_interleavable_free = int(largest_interleavable_free * args.safety_margin)
    total_free_bytes = snapshot["chip_total_free_bytes"]

    max_total_tokens_from_free = align_down(int(total_free_bytes // bytes_per_token_all_layers), args.tile_size)
    max_total_tokens_from_interleavable = align_down(
        int(largest_interleavable_free // bytes_per_token_all_layers), args.tile_size
    )
    max_total_tokens_from_safe_interleavable = align_down(
        int(safe_interleavable_free // bytes_per_token_all_layers), args.tile_size
    )

    return {
        "bytes_per_kv_token_per_layer": bytes_per_token_per_layer,
        "bytes_per_kv_token_all_layers": bytes_per_token_all_layers,
        "max_total_l1_tokens_from_total_free_bytes": max_total_tokens_from_free,
        "max_total_l1_tokens_from_largest_interleavable_free": max_total_tokens_from_interleavable,
        "max_total_l1_tokens_from_safe_interleavable_free": max_total_tokens_from_safe_interleavable,
        "max_window_size_from_total_free_bytes": max(max_total_tokens_from_free - args.l1_kv_sink_size, 0),
        "max_window_size_from_largest_interleavable_free": max(
            max_total_tokens_from_interleavable - args.l1_kv_sink_size, 0
        ),
        "max_window_size_from_safe_interleavable_free": max(
            max_total_tokens_from_safe_interleavable - args.l1_kv_sink_size, 0
        ),
    }


def format_pct(value):
    return f"{value:.2f}%"


def write_summary(path, dual_snapshot, dram_snapshot, dual_window, dram_window):
    lines = [
        "# L1 Memory Usage Comparison",
        "",
        "| Metric | Dual-source | DRAM-only | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]

    metric_rows = [
        ("Per-bank allocated bytes", "total_bytes_allocated_per_bank", False),
        ("Per-bank free bytes", "total_bytes_free_per_bank", False),
        ("Largest contiguous free bytes per bank", "largest_contiguous_bytes_free_per_bank", False),
        ("Largest interleavable free bytes estimate", "largest_interleavable_free_bytes_estimate", False),
        ("Chip-total allocated bytes", "chip_total_allocated_bytes", False),
        ("Chip-total free bytes", "chip_total_free_bytes", False),
        ("Per-bank allocated %", "per_bank_allocated_pct", True),
        ("Per-bank free %", "per_bank_free_pct", True),
        ("Per-bank largest contiguous free %", "per_bank_largest_contiguous_free_pct", True),
    ]

    for label, key, is_pct in metric_rows:
        dual_value = dual_snapshot[key]
        dram_value = dram_snapshot[key]
        dual_str = format_pct(dual_value) if is_pct else f"{dual_value}"
        dram_str = format_pct(dram_value) if is_pct else f"{dram_value}"
        delta_value = dual_value - dram_value
        delta_str = format_pct(delta_value) if is_pct else f"{delta_value:+}"
        lines.append(f"| {label} | {dual_str} | {dram_str} | {delta_str} |")

    lines.extend(
        [
            "",
            "## Window Estimate",
            "",
            "| Metric | Dual-source | DRAM-only |",
            "| --- | ---: | ---: |",
            f"| Bytes per KV token per layer | {dual_window['bytes_per_kv_token_per_layer']:.1f} | {dram_window['bytes_per_kv_token_per_layer']:.1f} |",
            f"| Bytes per KV token across all layers | {dual_window['bytes_per_kv_token_all_layers']:.1f} | {dram_window['bytes_per_kv_token_all_layers']:.1f} |",
            f"| Max `l1_kv_window_size` from total free bytes | {dual_window['max_window_size_from_total_free_bytes']} | {dram_window['max_window_size_from_total_free_bytes']} |",
            f"| Max `l1_kv_window_size` from largest interleavable free | {dual_window['max_window_size_from_largest_interleavable_free']} | {dram_window['max_window_size_from_largest_interleavable_free']} |",
            f"| Max `l1_kv_window_size` from safe interleavable free | {dual_window['max_window_size_from_safe_interleavable_free']} | {dram_window['max_window_size_from_safe_interleavable_free']} |",
            "",
            "The `largest interleavable free` estimate is usually the most realistic upper bound.",
            "The `safe interleavable free` number applies the configured safety margin and is the best starting point for experiments.",
        ]
    )

    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run matched dual-source vs DRAM-only L1 memory-view comparisons.")
    parser.add_argument("--dual-source-cmd", required=True, help="Shell command for the dual-source run.")
    parser.add_argument("--dram-only-cmd", required=True, help="Shell command for the DRAM-only run.")
    parser.add_argument("--working-directory", default=".", help="Working directory for both commands.")
    parser.add_argument("--output-dir", default="research_codes/l1_memory_compare", help="Directory for outputs.")
    parser.add_argument("--snapshot-label", default="after_model_load", help="Snapshot label to compare.")
    parser.add_argument("--batch-size-per-device-group", type=int, default=1)
    parser.add_argument("--num-local-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--kv-dtype", choices=sorted(DTYPE_BYTES), default="bfloat8_b")
    parser.add_argument("--l1-kv-sink-size", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.9,
        help="Scale factor applied to the largest-interleavable-free estimate before converting to tokens.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dual_snapshot_path = output_dir / "dual_source_memory_view.json"
    dram_snapshot_path = output_dir / "dram_only_memory_view.json"

    run_with_snapshot(args.dual_source_cmd, dual_snapshot_path, args.working_directory)
    run_with_snapshot(args.dram_only_cmd, dram_snapshot_path, args.working_directory)

    dual_snapshot = load_snapshot(dual_snapshot_path, args.snapshot_label)
    dram_snapshot = load_snapshot(dram_snapshot_path, args.snapshot_label)

    dual_window = summarize_window(dual_snapshot, args)
    dram_window = summarize_window(dram_snapshot, args)

    report = {
        "inputs": vars(args),
        "dual_source": {"snapshot": dual_snapshot, "window_estimate": dual_window},
        "dram_only": {"snapshot": dram_snapshot, "window_estimate": dram_window},
    }

    json_path = output_dir / "comparison.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_summary(md_path, dual_snapshot, dram_snapshot, dual_window, dram_window)
    print(md_path)


if __name__ == "__main__":
    main()

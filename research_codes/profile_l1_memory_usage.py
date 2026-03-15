import argparse
import json
import os
import re
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


def run_command_capture(command, cwd):
    completed = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.returncode, completed.stdout


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


def parse_probe_values(raw_values):
    if not raw_values:
        return []
    values = []
    for part in raw_values.split(","):
        token = part.strip()
        if token:
            values.append(int(token))
    return values


def probe_real_window_limit(command_template, probe_values, cwd, total_bytes_per_bank):
    if not command_template or not probe_values:
        return None

    clash_pattern = re.compile(
        r"Statically allocated circular buffers.*L1 buffer allocated at (\d+) and static circular buffer region ends at (\d+)"
    )
    results = []

    for window in probe_values:
        command = command_template.format(window=window)
        returncode, output = run_command_capture(command, cwd)
        probe = {
            "window": window,
            "returncode": returncode,
            "passed": returncode == 0,
            "failure_reason": None,
        }
        match = clash_pattern.search(output)
        if match:
            l1_buffer_address = int(match.group(1))
            static_cb_end = int(match.group(2))
            probe.update(
                {
                    "failure_reason": "static_cb_l1_clash",
                    "l1_buffer_allocated_address_per_bank": l1_buffer_address,
                    "static_circular_buffer_end_address_per_bank": static_cb_end,
                    "static_cb_occupied_bytes_per_bank": static_cb_end,
                    "static_cb_occupied_pct_per_bank": 100.0 * static_cb_end / total_bytes_per_bank,
                    "effective_headroom_bytes_per_bank": max(total_bytes_per_bank - static_cb_end, 0),
                    "effective_headroom_pct_per_bank": 100.0
                    * max(total_bytes_per_bank - static_cb_end, 0)
                    / total_bytes_per_bank,
                    "requested_l1_buffer_bytes_per_bank": max(total_bytes_per_bank - l1_buffer_address, 0),
                    "requested_l1_buffer_pct_per_bank": 100.0
                    * max(total_bytes_per_bank - l1_buffer_address, 0)
                    / total_bytes_per_bank,
                    "overlap_bytes_per_bank": max(static_cb_end - l1_buffer_address, 0),
                    "overlap_pct_per_bank": 100.0 * max(static_cb_end - l1_buffer_address, 0) / total_bytes_per_bank,
                }
            )
        elif returncode != 0:
            probe["failure_reason"] = "other_failure"
        results.append(probe)

    passed_windows = [item["window"] for item in results if item["passed"]]
    failed_windows = [item["window"] for item in results if not item["passed"]]
    first_clash = next((item for item in results if item.get("failure_reason") == "static_cb_l1_clash"), None)

    return {
        "probe_values": probe_values,
        "max_passing_window": max(passed_windows) if passed_windows else None,
        "min_failing_window": min(failed_windows) if failed_windows else None,
        "first_static_cb_clash": first_clash,
        "results": results,
    }


def write_summary(path, dual_snapshot, dram_snapshot, dual_window, dram_window, real_window_probe):
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

    if dual_snapshot.get("allocator_num_allocated_blocks", 0) or dram_snapshot.get("allocator_num_allocated_blocks", 0):
        metric_rows.extend(
            [
                ("Allocator top-down reserved bytes per bank", "allocator_top_down_reserved_bytes_per_bank", False),
                (
                    "Allocator highest allocated end address per bank",
                    "allocator_highest_allocated_end_address_per_bank",
                    False,
                ),
                ("Allocator largest block bytes per bank", "allocator_largest_allocated_block_bytes_per_bank", False),
            ]
        )

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
            "These bounds come from allocator-visible `get_memory_view()` state only.",
            "They can significantly overestimate the real decode-time limit because static circular buffers are typically not allocator-managed.",
        ]
    )

    if dual_snapshot.get("captures_allocator_state_only"):
        lines.extend(
            [
                "",
                "## Allocator Caveat",
                "",
                "- `get_memory_view()` reports allocator-managed L1 state.",
                "- Static circular buffers are typically not allocator-managed and are not fully reflected in these free-space numbers.",
                f"- Snapshot note: `{dual_snapshot.get('allocator_note', '')}`",
            ]
        )

    if real_window_probe:
        lines.extend(
            [
                "",
                "## Real Window Probe",
                "",
                f"- Max passing `l1_kv_window_size`: `{real_window_probe['max_passing_window']}`",
                f"- Min failing `l1_kv_window_size`: `{real_window_probe['min_failing_window']}`",
            ]
        )
        first_clash = real_window_probe.get("first_static_cb_clash")
        if first_clash:
            lines.extend(
                [
                    f"- First observed failure reason: `{first_clash['failure_reason']}`",
                    f"- Static CB end address on constraining cores: `{first_clash['static_circular_buffer_end_address_per_bank']}`",
                    f"- Inferred runtime SRAM already occupied by static CBs on constraining cores: `{first_clash['static_cb_occupied_bytes_per_bank']}` bytes ({first_clash['static_cb_occupied_pct_per_bank']:.2f}%)",
                    f"- Requested L1 buffer start address on constraining cores: `{first_clash['l1_buffer_allocated_address_per_bank']}`",
                    f"- Real top-of-bank headroom on constraining cores: `{first_clash['effective_headroom_bytes_per_bank']}` bytes ({first_clash['effective_headroom_pct_per_bank']:.2f}%)",
                    f"- Requested buffer bytes on constraining cores: `{first_clash['requested_l1_buffer_bytes_per_bank']}` bytes ({first_clash['requested_l1_buffer_pct_per_bank']:.2f}%)",
                    f"- Observed overlap on constraining cores: `{first_clash['overlap_bytes_per_bank']}` bytes ({first_clash['overlap_pct_per_bank']:.2f}%)",
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
    parser.add_argument(
        "--real-window-cmd-template",
        default=None,
        help="Optional shell command template used to probe the real pass/fail window. Use {window} as the placeholder.",
    )
    parser.add_argument(
        "--real-window-values",
        default="",
        help="Comma-separated l1_kv_window_size values to probe with --real-window-cmd-template.",
    )
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
    real_window_probe = probe_real_window_limit(
        args.real_window_cmd_template,
        parse_probe_values(args.real_window_values),
        args.working_directory,
        dual_snapshot["total_bytes_per_bank"],
    )

    report = {
        "inputs": vars(args),
        "dual_source": {"snapshot": dual_snapshot, "window_estimate": dual_window},
        "dram_only": {"snapshot": dram_snapshot, "window_estimate": dram_window},
        "real_window_probe": real_window_probe,
    }

    json_path = output_dir / "comparison.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_summary(md_path, dual_snapshot, dram_snapshot, dual_window, dram_window, real_window_probe)
    print(md_path)


if __name__ == "__main__":
    main()

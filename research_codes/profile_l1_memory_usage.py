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
            "These bounds come from allocator-visible state (from `dump_device_memory_state()` CSVs) only.",
            "They can significantly overestimate the real decode-time limit because static circular buffers are typically not allocator-managed.",
        ]
    )

    if dual_snapshot.get("captures_allocator_state_only"):
        lines.extend(
            [
                "",
                "## Allocator Caveat",
                "",
                "- L1 state is read from `dump_device_memory_state()` CSV reports (allocator-managed).",
                "- Static circular buffers are typically not allocator-managed and are not fully reflected in these free-space numbers.",
                f"- Snapshot note: `{dual_snapshot.get('allocator_note', '')}`",
            ]
        )

    if real_window_probe:
        first_clash = real_window_probe.get("first_static_cb_clash")
        total_per_bank = dual_snapshot.get("total_bytes_per_bank", 1470080)

        lines.extend(
            [
                "",
                "## Combined Usage (Allocator vs Real Runtime)",
                "",
                "Single compact view of allocator-visible state vs inferred runtime bottleneck-bank usage.",
                "",
                "| View | Source | Occupied (bytes) | Occupied (%) | Free/Headroom (bytes) | Free/Headroom (%) |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        dual_occ = dual_snapshot.get("total_bytes_allocated_per_bank", 0)
        dram_occ = dram_snapshot.get("total_bytes_allocated_per_bank", 0)
        dual_free = dual_snapshot.get("total_bytes_free_per_bank", 0)
        dram_free = dram_snapshot.get("total_bytes_free_per_bank", 0)
        lines.append(
            f"| Allocator (Dual-source) | `dump_device_memory_state()` | {dual_occ} | {format_pct(100.0 * dual_occ / total_per_bank)} | {dual_free} | {format_pct(100.0 * dual_free / total_per_bank)} |"
        )
        lines.append(
            f"| Allocator (DRAM-only) | `dump_device_memory_state()` | {dram_occ} | {format_pct(100.0 * dram_occ / total_per_bank)} | {dram_free} | {format_pct(100.0 * dram_free / total_per_bank)} |"
        )
        if first_clash:
            cb_occ = first_clash["static_cb_occupied_bytes_per_bank"]
            headroom = first_clash["effective_headroom_bytes_per_bank"]
            lines.append(
                f"| **Real runtime (bottleneck cores)** | Pass/fail probe + clash parse | **{cb_occ}** | **{format_pct(first_clash['static_cb_occupied_pct_per_bank'])}** | **{headroom}** | **{format_pct(first_clash['effective_headroom_pct_per_bank'])}** |"
            )
        lines.extend(
            [
                "",
                "**Explanation:** Allocator rows show what `dump_device_memory_state()` reports (allocator-managed blocks only). "
                "The real runtime row is inferred from the first failing `l1_kv_window_size` probe: when the L1 KV buffer "
                "clashes with static circular buffers, we parse the error to get the static CB end address. That address "
                "is the actual SRAM already occupied on the constraining decode cores (8×1 bottleneck range). The headroom "
                "is the remainder—the only space available for the L1 KV mirror.",
                "",
                "## Real Window Probe",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Max passing `l1_kv_window_size` | `{real_window_probe['max_passing_window']}` |",
                f"| Min failing `l1_kv_window_size` | `{real_window_probe['min_failing_window']}` |",
            ]
        )
        if first_clash:
            lines.extend(
                [
                    f"| First failure reason | `{first_clash['failure_reason']}` |",
                    f"| Static CB end address (bottleneck cores) | `{first_clash['static_circular_buffer_end_address_per_bank']}` bytes |",
                    f"| Inferred SRAM occupied by static CBs | `{first_clash['static_cb_occupied_bytes_per_bank']}` bytes ({first_clash['static_cb_occupied_pct_per_bank']:.2f}%) |",
                    f"| L1 buffer start address (requested) | `{first_clash['l1_buffer_allocated_address_per_bank']}` |",
                    f"| Real headroom on bottleneck cores | `{first_clash['effective_headroom_bytes_per_bank']}` bytes ({first_clash['effective_headroom_pct_per_bank']:.2f}%) |",
                    f"| Requested L1 buffer size | `{first_clash['requested_l1_buffer_bytes_per_bank']}` bytes ({first_clash['requested_l1_buffer_pct_per_bank']:.2f}%) |",
                    f"| Overlap (clash region) | `{first_clash['overlap_bytes_per_bank']}` bytes ({first_clash['overlap_pct_per_bank']:.2f}%) |",
                    "",
                    "**Why the allocator view misleads:** The dump only tracks allocator-managed blocks. "
                    "Static circular buffers (CBs) used by decode kernels are typically not allocator-managed, so they "
                    "do not appear in the allocator's free-space numbers. DRAM-only mode shows ~99.7% free from the "
                    "allocator's perspective, but the real bottleneck cores have ~85% of their L1 already occupied by "
                    "static CBs, leaving only ~15% headroom for the L1 KV mirror.",
                    "",
                    f"**Interpretation:** At window size {first_clash['window']}, the L1 KV mirror requires "
                    f"{first_clash['requested_l1_buffer_bytes_per_bank']:,} bytes per bank on the constraining cores. "
                    f"The allocator places it starting at address {first_clash['l1_buffer_allocated_address_per_bank']:,} "
                    f"(top-down). Static CBs already occupy 0–{first_clash['static_cb_occupied_bytes_per_bank']:,}, so the "
                    f"requested buffer overlaps the static region by {first_clash['overlap_bytes_per_bank']:,} bytes, "
                    f"causing the clash. The real upper bound is {real_window_probe['max_passing_window']} tokens—the "
                    "largest window that passes the actual workload. The exact per-bank footprint depends on "
                    "interleaving and which cores hold the L1 KV cache; the probe confirms the limit.",
                ]
            )
        lines.append("")

        # Compile-time CB size section: from bottleneck_core_memory.json if present, else from probe
        bottleneck_json = path.parent / "bottleneck_core_memory.json"
        cb_bytes = first_clash["static_cb_occupied_bytes_per_bank"] if first_clash else None
        cb_source = "probe_clash"
        cb_breakdown = None
        if bottleneck_json.exists():
            try:
                bn = json.loads(bottleneck_json.read_text())
                if bn.get("static_cb_bytes_per_core") is not None:
                    cb_bytes = bn["static_cb_bytes_per_core"]
                    cb_source = bn.get("static_cb_source", "probe_clash")
                cb_breakdown = bn.get("cb_breakdown")
            except (json.JSONDecodeError, KeyError):
                pass
        if cb_bytes is not None:
            total_per_bank = total_per_bank or dual_snapshot.get("total_bytes_per_bank", 1470080)
            cb_pct = round(100.0 * cb_bytes / total_per_bank, 2)
            lines.extend(
                [
                    "## Compile-time CB size (debug log)",
                    "",
                    "Static CB size per core from debug log (when available) or from probe clash.",
                    "",
                    "| Metric | Value | Source |",
                    "| --- | ---: | --- |",
                    f"| Total static CB size per core (bytes) | {cb_bytes:,} | {cb_source} |",
                    f"| % of L1 per bank | {cb_pct}% | — |",
                    "",
                ]
            )
            if cb_breakdown:
                lines.append("Per-CB sizes (bytes per core):")
                for k in sorted(cb_breakdown.keys(), key=lambda x: (len(x), x)):
                    lines.append(f"- {k}: {cb_breakdown[k]:,}")
                lines.append("")
            lines.extend(
                [
                    "To refresh from debug log: run demo with `TT_LOGGER_LEVEL=Debug`, save log, then:",
                    "`python research_codes/parse_sdpa_cb_memory.py --log <log> --comparison <comparison.json> --output-json "
                    + str(path.parent / "bottleneck_core_memory.json")
                    + "`",
                    "",
                ]
            )

        # Example commands
        out_dir = path.parent
        comp_path = out_dir / "comparison.json"
        bottleneck_path = out_dir / "bottleneck_core_memory.json"
        lines.extend(
            [
                "## Example commands",
                "",
                "**Regenerate this summary from existing `comparison.json`** (no re-run of inference):",
                "",
                f"```bash\npython research_codes/profile_l1_memory_usage.py --from-json {comp_path}\n```",
                "",
                "To reflect **compile-time CB size from debug logs**: run the demo with `TT_LOGGER_LEVEL=Debug`, save log, then:",
                f"`python research_codes/parse_sdpa_cb_memory.py --log <log> --comparison {comp_path} --output-json {bottleneck_path}`, then regenerate the summary again.",
                "",
                "**Full run** (creates comparison.json and summary from scratch; runs dual-source, DRAM-only, and real-window probe):",
                "",
                "```bash",
                "python research_codes/profile_l1_memory_usage.py \\",
                "  --dual-source-cmd 'pytest models/tt_transformers/demo/simple_text_demo.py -k \"performance and batch-1\" --max_generated_tokens 2 --stop_at_eos 0 --l1_kv_window_size 128' \\",
                "  --dram-only-cmd 'pytest models/tt_transformers/demo/simple_text_demo.py -k \"performance and batch-1\" --max_generated_tokens 2 --stop_at_eos 0 --l1_kv_window_size 0' \\",
                "  --working-directory . \\",
                f"  --output-dir {out_dir} \\",
                "  --snapshot-label after_inference \\",
                "  --real-window-cmd-template 'pytest models/tt_transformers/demo/simple_text_demo.py -k \"performance and batch-1\" --max_generated_tokens 2 --stop_at_eos 0 --l1_kv_window_size {window}' \\",
                "  --real-window-values 512,544 \\",
                "  --num-local-kv-heads 8 \\",
                "  --num-layers 32",
                "```",
                "",
                "Run from repo root.",
                "",
            ]
        )

    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run matched dual-source vs DRAM-only L1 memory-view comparisons.")
    parser.add_argument(
        "--from-json",
        metavar="PATH",
        help="Regenerate summary.md from existing comparison.json (skips running commands).",
    )
    parser.add_argument("--dual-source-cmd", help="Shell command for the dual-source run.")
    parser.add_argument("--dram-only-cmd", help="Shell command for the DRAM-only run.")
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
    parser.add_argument("--num-local-kv-heads", type=int, help="Required when not using --from-json.")
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

    if args.from_json:
        json_path = Path(args.from_json)
        if not json_path.is_file():
            raise FileNotFoundError(f"comparison.json not found: {json_path}")
        report = json.loads(json_path.read_text())
        output_dir = json_path.parent
        dual_snapshot = report["dual_source"]["snapshot"]
        dram_snapshot = report["dram_only"]["snapshot"]
        dual_window = report["dual_source"]["window_estimate"]
        dram_window = report["dram_only"]["window_estimate"]
        real_window_probe = report.get("real_window_probe")
        inputs = report.get("inputs", {})
        for k, v in inputs.items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)
    else:
        if not args.dual_source_cmd or not args.dram_only_cmd:
            parser.error("--dual-source-cmd and --dram-only-cmd are required when not using --from-json")
        if args.num_local_kv_heads is None:
            parser.error("--num-local-kv-heads is required when not using --from-json")
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
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    md_path = output_dir / "summary.md"
    write_summary(md_path, dual_snapshot, dram_snapshot, dual_window, dram_window, real_window_probe)
    print(md_path)


if __name__ == "__main__":
    main()

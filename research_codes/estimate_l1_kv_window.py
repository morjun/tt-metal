import argparse
import json
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


def load_snapshot(path, label):
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported snapshot payload in {path}")

    if label is None:
        return payload[-1]

    for item in payload:
        if item.get("label") == label:
            return item
    raise ValueError(f"Snapshot label '{label}' not found in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate practical L1 KV mirror/window limits from a simple_text_demo memory-view snapshot."
    )
    parser.add_argument("--memory-view-json", required=True, help="Path written by --l1_memory_view_path")
    parser.add_argument(
        "--snapshot-label",
        default="after_inference",
        help="Which snapshot to analyze (default: after_inference, use after_model_load for pre-decode headroom)",
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
        default=1.0,
        help="Scale largest-interleavable-free bytes by this factor for a more conservative estimate",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    snapshot = load_snapshot(args.memory_view_json, args.snapshot_label)
    bytes_per_elem = DTYPE_BYTES[args.kv_dtype]
    bytes_per_token_per_layer = (
        2 * args.batch_size_per_device_group * args.num_local_kv_heads * args.head_dim * bytes_per_elem
    )
    bytes_per_token_all_layers = bytes_per_token_per_layer * args.num_layers

    chip_total_allocatable = snapshot["chip_total_allocatable_bytes"]
    chip_total_free = snapshot["chip_total_free_bytes"]
    largest_interleavable_free = snapshot["largest_interleavable_free_bytes_estimate"]
    safe_interleavable_free = int(largest_interleavable_free * args.safety_margin)

    theoretical_total_tokens = align_down(int(chip_total_allocatable // bytes_per_token_all_layers), args.tile_size)
    total_free_total_tokens = align_down(int(chip_total_free // bytes_per_token_all_layers), args.tile_size)
    interleavable_total_tokens = align_down(
        int(largest_interleavable_free // bytes_per_token_all_layers), args.tile_size
    )
    safe_interleavable_total_tokens = align_down(
        int(safe_interleavable_free // bytes_per_token_all_layers), args.tile_size
    )

    result = {
        "snapshot_label": snapshot.get("label"),
        "num_banks": snapshot["num_banks"],
        "total_bytes_per_bank": snapshot["total_bytes_per_bank"],
        "largest_contiguous_bytes_free_per_bank": snapshot["largest_contiguous_bytes_free_per_bank"],
        "largest_interleavable_free_bytes_estimate": largest_interleavable_free,
        "safe_interleavable_free_bytes_estimate": safe_interleavable_free,
        "chip_total_allocatable_bytes": chip_total_allocatable,
        "chip_total_free_bytes": chip_total_free,
        "bytes_per_kv_token_per_layer": bytes_per_token_per_layer,
        "bytes_per_kv_token_all_layers": bytes_per_token_all_layers,
        "theoretical_total_l1_tokens_no_other_usage": theoretical_total_tokens,
        "max_total_l1_tokens_from_total_free_bytes": total_free_total_tokens,
        "max_total_l1_tokens_from_largest_interleavable_free": interleavable_total_tokens,
        "max_total_l1_tokens_from_safe_interleavable_free": safe_interleavable_total_tokens,
        "max_window_size_from_total_free_bytes": max(total_free_total_tokens - args.l1_kv_sink_size, 0),
        "max_window_size_from_largest_interleavable_free": max(interleavable_total_tokens - args.l1_kv_sink_size, 0),
        "max_window_size_from_safe_interleavable_free": max(safe_interleavable_total_tokens - args.l1_kv_sink_size, 0),
        "inputs": vars(args),
    }

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if args.output_md:
        lines = [
            "# L1 KV Window Estimate",
            "",
            f"- Snapshot label: `{result['snapshot_label']}`",
            f"- Per-device local KV bytes per token per layer: `{bytes_per_token_per_layer:.1f} B`",
            f"- Per-device local KV bytes per token across all layers: `{bytes_per_token_all_layers:.1f} B`",
            f"- Chip-total allocatable L1: `{chip_total_allocatable}` bytes",
            f"- Chip-total free L1 at snapshot: `{chip_total_free}` bytes",
            f"- Largest interleavable free L1 estimate: `{largest_interleavable_free}` bytes",
            f"- Safe interleavable free L1 estimate: `{safe_interleavable_free}` bytes",
            f"- Theoretical total L1 tokens with no other L1 usage: `{theoretical_total_tokens}`",
            f"- Max total L1 tokens from total free bytes: `{total_free_total_tokens}`",
            f"- Max total L1 tokens from largest interleavable free: `{interleavable_total_tokens}`",
            f"- Max total L1 tokens from safe interleavable free: `{safe_interleavable_total_tokens}`",
            f"- Max `l1_kv_window_size` from total free bytes: `{result['max_window_size_from_total_free_bytes']}`",
            f"- Max `l1_kv_window_size` from largest interleavable free: `{result['max_window_size_from_largest_interleavable_free']}`",
            f"- Max `l1_kv_window_size` from safe interleavable free: `{result['max_window_size_from_safe_interleavable_free']}`",
        ]
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text("\n".join(lines) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

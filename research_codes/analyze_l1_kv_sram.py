import argparse
import csv
import json
import math
from pathlib import Path


DTYPE_BYTES = {
    "bfloat16": 2.0,
    "bfloat8_b": 1.0,
    "bfloat4_b": 0.5,
}


def mib(value):
    return value / (1024.0 * 1024.0)


def load_hit_ratio(csv_path):
    if not csv_path:
        return None
    total_l1 = 0
    total_dram = 0
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k_l1_key = "k_l1" if "k_l1" in row else "k_l1_hits"
            v_l1_key = "v_l1" if "v_l1" in row else "v_l1_hits"
            k_dram_key = "k_dram" if "k_dram" in row else "k_dram_reads"
            v_dram_key = "v_dram" if "v_dram" in row else "v_dram_reads"
            total_l1 += int(row[k_l1_key]) + int(row[v_l1_key])
            total_dram += int(row[k_dram_key]) + int(row[v_dram_key])
    total = total_l1 + total_dram
    if total == 0:
        return None
    return total_l1 / total


def main():
    parser = argparse.ArgumentParser(description="Estimate persistent and transient SRAM pressure for L1 KV decode.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--num-q-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--l1-kv-window-size", type=int, default=256)
    parser.add_argument("--l1-kv-sink-size", type=int, default=0)
    parser.add_argument("--k-chunk-size", type=int, default=512)
    parser.add_argument("--kv-dtype", choices=sorted(DTYPE_BYTES), default="bfloat8_b")
    parser.add_argument("--dprint-chunks-csv", default=None)
    parser.add_argument("--output-json", default="research_codes/l1_kv_sram_report.json")
    parser.add_argument("--output-md", default="research_codes/l1_kv_sram_report.md")
    args = parser.parse_args()

    bytes_per_elem = DTYPE_BYTES[args.kv_dtype]
    sink = args.l1_kv_sink_size
    ring = args.l1_kv_window_size
    total_tokens = sink + ring
    persistent_bytes = 2 * args.batch_size * args.num_kv_heads * total_tokens * args.head_dim * bytes_per_elem

    dht = args.head_dim // 32
    pnh_t = max(1, math.ceil(args.num_q_heads / 32))
    sk_chunk_t = args.k_chunk_size // 32
    element_tile_bytes = int(32 * 32 * bytes_per_elem)

    k_tiles = sk_chunk_t * dht * 2
    v_tiles = sk_chunk_t * dht * 2
    q_tiles = pnh_t * dht
    qk_tiles = pnh_t * sk_chunk_t
    out_tiles = pnh_t * dht
    transient_tile_bytes = (k_tiles + v_tiles + q_tiles + qk_tiles + out_tiles) * element_tile_bytes

    observed_hit_ratio = load_hit_ratio(args.dprint_chunks_csv)
    effective_hot_bytes = None if observed_hit_ratio is None else persistent_bytes * observed_hit_ratio

    report = {
        "inputs": vars(args),
        "dram_only_persistent_l1_kv_bytes": 0,
        "dram_only_persistent_l1_kv_mib": 0.0,
        "persistent_l1_kv_bytes": persistent_bytes,
        "persistent_l1_kv_mib": mib(persistent_bytes),
        "persistent_l1_kv_over_dram_only_bytes": persistent_bytes,
        "persistent_l1_kv_over_dram_only_mib": mib(persistent_bytes),
        "transient_sdpa_cb_bytes_estimate": transient_tile_bytes,
        "transient_sdpa_cb_mib_estimate": mib(transient_tile_bytes),
        "observed_hit_ratio": observed_hit_ratio,
        "effective_hot_bytes_estimate": effective_hot_bytes,
        "effective_hot_mib_estimate": None if effective_hot_bytes is None else mib(effective_hot_bytes),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# L1 KV SRAM Analysis",
        "",
        f"- DRAM-only persistent KV mirror in L1: `{0.0:.3f} MiB`",
        f"- Persistent L1 KV mirror: `{mib(persistent_bytes):.3f} MiB`",
        f"- Extra persistent SRAM over DRAM-only baseline: `{mib(persistent_bytes):.3f} MiB`",
        f"- Estimated transient SDPA CB pressure: `{mib(transient_tile_bytes):.3f} MiB`",
    ]
    if observed_hit_ratio is not None:
        lines.append(f"- Observed L1 hit ratio from DPRINT: `{observed_hit_ratio:.3%}`")
        lines.append(f"- Effective hot-set bytes served from L1: `{mib(effective_hot_bytes):.3f} MiB`")
    lines.append("")
    lines.append("This report uses the current DPRINT hit ratio as a baseline for future sharded/zero-copy work.")
    lines.append(
        "The hit ratio tells you how much of the decode working set is actually hot enough to justify keeping in SRAM."
    )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()

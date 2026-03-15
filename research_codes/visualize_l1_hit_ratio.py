import argparse
import csv
import html
import re
from collections import defaultdict
from pathlib import Path


LINE_RE = re.compile(
    r"\(x=(?P<x>\d+),y=(?P<y>\d+)\):.*?CHUNK\s+(?P<chunk>\d+)\s+\|\s+"
    r"K\(L1:(?P<k_l1>\d+)\s+DRAM:(?P<k_dram>\d+)\)\s+\|\s+"
    r"V\(L1:(?P<v_l1>\d+)\s+DRAM:(?P<v_dram>\d+)\)"
)


def safe_ratio(hits, dram):
    total = hits + dram
    return (hits / total) if total else 0.0


def bucket_color(value, low, high):
    if high <= low:
        norm = 1.0
    else:
        norm = max(0.0, min(1.0, (value - low) / (high - low)))
    red = int(245 - norm * 160)
    green = int(245 - norm * 20)
    blue = int(245 - norm * 180)
    return f"rgb({red}, {green}, {blue})"


def build_heatmap_table(title, row_labels, col_labels, values, value_fmt):
    flat_values = [values[row][col] for row in row_labels for col in col_labels]
    low = min(flat_values) if flat_values else 0.0
    high = max(flat_values) if flat_values else 1.0

    parts = [f"<h2>{html.escape(title)}</h2>", '<table class="heatmap">']
    parts.append("<tr><th></th>" + "".join(f"<th>{html.escape(str(col))}</th>" for col in col_labels) + "</tr>")
    for row in row_labels:
        parts.append(f"<tr><th>{html.escape(str(row))}</th>")
        for col in col_labels:
            value = values[row][col]
            color = bucket_color(value, low, high)
            parts.append(f'<td style="background:{color}">{html.escape(value_fmt(value))}</td>')
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def parse_log(log_path):
    core_totals = defaultdict(lambda: {"k_l1": 0, "k_dram": 0, "v_l1": 0, "v_dram": 0})
    chunk_totals = defaultdict(lambda: {"k_l1": 0, "k_dram": 0, "v_l1": 0, "v_dram": 0})
    core_chunk_totals = defaultdict(lambda: {"k_l1": 0, "k_dram": 0, "v_l1": 0, "v_dram": 0})
    overall = {"k_l1": 0, "k_dram": 0, "v_l1": 0, "v_dram": 0}

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LINE_RE.search(line)
            if not match:
                continue

            core = f"{match.group('x')},{match.group('y')}"
            chunk = int(match.group("chunk"))
            values = {
                "k_l1": int(match.group("k_l1")),
                "k_dram": int(match.group("k_dram")),
                "v_l1": int(match.group("v_l1")),
                "v_dram": int(match.group("v_dram")),
            }
            for key, value in values.items():
                overall[key] += value
                core_totals[core][key] += value
                chunk_totals[chunk][key] += value
                core_chunk_totals[(core, chunk)][key] += value

    return overall, core_totals, chunk_totals, core_chunk_totals


def write_core_csv(csv_path, core_totals):
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["core", "k_l1_hits", "k_dram_reads", "k_hit_ratio", "v_l1_hits", "v_dram_reads", "v_hit_ratio"]
        )
        for core in sorted(core_totals):
            stats = core_totals[core]
            writer.writerow(
                [
                    core,
                    stats["k_l1"],
                    stats["k_dram"],
                    safe_ratio(stats["k_l1"], stats["k_dram"]),
                    stats["v_l1"],
                    stats["v_dram"],
                    safe_ratio(stats["v_l1"], stats["v_dram"]),
                ]
            )


def write_chunk_csv(csv_path, chunk_totals):
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["chunk", "k_l1_hits", "k_dram_reads", "k_hit_ratio", "v_l1_hits", "v_dram_reads", "v_hit_ratio"]
        )
        for chunk in sorted(chunk_totals):
            stats = chunk_totals[chunk]
            writer.writerow(
                [
                    chunk,
                    stats["k_l1"],
                    stats["k_dram"],
                    safe_ratio(stats["k_l1"], stats["k_dram"]),
                    stats["v_l1"],
                    stats["v_dram"],
                    safe_ratio(stats["v_l1"], stats["v_dram"]),
                ]
            )


def write_html(html_path, log_path, overall, core_totals, chunk_totals, core_chunk_totals):
    cores = sorted(core_totals)
    chunks = sorted(chunk_totals)

    core_k_values = {
        core: {"hit_ratio": safe_ratio(stats["k_l1"], stats["k_dram"])} for core, stats in core_totals.items()
    }
    core_v_values = {
        core: {"hit_ratio": safe_ratio(stats["v_l1"], stats["v_dram"])} for core, stats in core_totals.items()
    }
    chunk_k_values = {
        chunk: {"hit_ratio": safe_ratio(stats["k_l1"], stats["k_dram"])} for chunk, stats in chunk_totals.items()
    }
    chunk_v_values = {
        chunk: {"hit_ratio": safe_ratio(stats["v_l1"], stats["v_dram"])} for chunk, stats in chunk_totals.items()
    }

    core_chunk_k = defaultdict(dict)
    core_chunk_v = defaultdict(dict)
    for core in cores:
        for chunk in chunks:
            stats = core_chunk_totals[(core, chunk)]
            core_chunk_k[core][chunk] = safe_ratio(stats["k_l1"], stats["k_dram"])
            core_chunk_v[core][chunk] = safe_ratio(stats["v_l1"], stats["v_dram"])

    k_total = overall["k_l1"] + overall["k_dram"]
    v_total = overall["v_l1"] + overall["v_dram"]
    summary_rows = [
        ("K total tiles", str(k_total)),
        ("K L1 hit ratio", f"{safe_ratio(overall['k_l1'], overall['k_dram']) * 100:.2f}%"),
        ("V total tiles", str(v_total)),
        ("V L1 hit ratio", f"{safe_ratio(overall['v_l1'], overall['v_dram']) * 100:.2f}%"),
        ("Cores seen", str(len(cores))),
        ("Chunks seen", str(len(chunks))),
    ]

    body = [
        "<!doctype html><html><head><meta charset='utf-8'><title>L1 Hit Ratio</title>",
        "<style>body{font-family:monospace;margin:24px;background:#111;color:#eee} h1,h2{margin:0 0 12px} ",
        "table{border-collapse:collapse;margin:12px 0 28px;max-width:100%} th,td{border:1px solid #333;padding:6px 8px;text-align:center} ",
        "th{background:#1d1d1d;position:sticky;top:0} .summary td,.summary th{text-align:left} .heatmap td{min-width:48px}</style></head><body>",
        f"<h1>L1 Hit Ratio Report</h1><p>Source log: <code>{html.escape(str(log_path))}</code></p>",
        "<table class='summary'>",
    ]
    for key, value in summary_rows:
        body.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>")
    body.append("</table>")
    body.append(
        build_heatmap_table(
            "Per-Core K Hit Ratio",
            ["K"],
            cores,
            {"K": {core: core_k_values[core]["hit_ratio"] for core in cores}},
            lambda v: f"{v * 100:.1f}%",
        )
    )
    body.append(
        build_heatmap_table(
            "Per-Core V Hit Ratio",
            ["V"],
            cores,
            {"V": {core: core_v_values[core]["hit_ratio"] for core in cores}},
            lambda v: f"{v * 100:.1f}%",
        )
    )
    body.append(
        build_heatmap_table(
            "Per-Chunk K Hit Ratio",
            ["K"],
            chunks,
            {"K": {chunk: chunk_k_values[chunk]["hit_ratio"] for chunk in chunks}},
            lambda v: f"{v * 100:.1f}%",
        )
    )
    body.append(
        build_heatmap_table(
            "Per-Chunk V Hit Ratio",
            ["V"],
            chunks,
            {"V": {chunk: chunk_v_values[chunk]["hit_ratio"] for chunk in chunks}},
            lambda v: f"{v * 100:.1f}%",
        )
    )
    body.append(
        build_heatmap_table("Core x Chunk K Hit Ratio", cores, chunks, core_chunk_k, lambda v: f"{v * 100:.0f}%")
    )
    body.append(
        build_heatmap_table("Core x Chunk V Hit Ratio", cores, chunks, core_chunk_v, lambda v: f"{v * 100:.0f}%")
    )
    body.append("</body></html>")
    html_path.write_text("".join(body), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Parse and visualize L1-vs-DRAM decode hit ratios from DPRINT logs.")
    parser.add_argument("log_file")
    parser.add_argument("--html", type=Path, help="Optional HTML report output path.")
    parser.add_argument("--csv-prefix", type=Path, help="Optional prefix for CSV summary files.")
    args = parser.parse_args()

    overall, core_totals, chunk_totals, core_chunk_totals = parse_log(args.log_file)
    total_reads = overall["k_l1"] + overall["k_dram"] + overall["v_l1"] + overall["v_dram"]
    if total_reads == 0:
        print(
            "No decode DPRINT hit lines found. Make sure DEBUG_PRINT is enabled in sdpa_decode and the log captures stdout."
        )
        return 1

    print(f"K total tiles: {overall['k_l1'] + overall['k_dram']}")
    print(f"K L1 hits:     {overall['k_l1']} ({safe_ratio(overall['k_l1'], overall['k_dram']) * 100:.2f}%)")
    print(f"K DRAM reads:  {overall['k_dram']} ({(1 - safe_ratio(overall['k_l1'], overall['k_dram'])) * 100:.2f}%)")
    print(f"V total tiles: {overall['v_l1'] + overall['v_dram']}")
    print(f"V L1 hits:     {overall['v_l1']} ({safe_ratio(overall['v_l1'], overall['v_dram']) * 100:.2f}%)")
    print(f"V DRAM reads:  {overall['v_dram']} ({(1 - safe_ratio(overall['v_l1'], overall['v_dram'])) * 100:.2f}%)")
    print(f"Cores seen:    {len(core_totals)}")
    print(f"Chunks seen:   {len(chunk_totals)}")

    if args.csv_prefix:
        write_core_csv(args.csv_prefix.with_name(args.csv_prefix.name + "_cores.csv"), core_totals)
        write_chunk_csv(args.csv_prefix.with_name(args.csv_prefix.name + "_chunks.csv"), chunk_totals)
        print(f"Wrote CSV summaries with prefix {args.csv_prefix}")

    if args.html:
        write_html(args.html, args.log_file, overall, core_totals, chunk_totals, core_chunk_totals)
        print(f"Wrote HTML report to {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Parse SDPA decode circular buffer (CB) sizes from TT_LOGGER_LEVEL=Debug output
and/or from comparison.json probe data to report exact bottleneck-core memory usage.

Usage:
  # From debug log (run demo with TT_LOGGER_LEVEL=Debug and redirect to file):
  python research_codes/parse_sdpa_cb_memory.py --log /path/to/debug_log.txt

  # From existing comparison.json (uses probe static_cb_occupied if no log):
  python research_codes/parse_sdpa_cb_memory.py --comparison research_codes/l1_memory_compare_full_model_corrected/comparison.json

  # Both: log provides CB size, comparison provides total_per_bank and optional clash details
  python research_codes/parse_sdpa_cb_memory.py --log debug.txt --comparison comparison.json
"""

import argparse
import json
import re
from pathlib import Path


# Default total L1 bytes per bank (per core) for Blackhole P150 from allocator/soc
DEFAULT_TOTAL_BYTES_PER_BANK = 1_470_080


def parse_cb_total_from_log(log_path: Path) -> int | None:
    """Extract 'SDPA decode total static CB size per core (bytes): N' from log."""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    m = re.search(
        r"SDPA decode total static CB size per core \(bytes\):\s*(\d+)",
        text,
    )
    return int(m.group(1)) if m else None


def parse_cb_breakdown_from_log(log_path: Path) -> dict[str, int] | None:
    """Extract per-CB sizes from 'SDPA decode CB sizes (bytes per core): c0=...' line."""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    m = re.search(
        r"SDPA decode CB sizes \(bytes per core\):\s*(.+)",
        text,
    )
    if not m:
        return None
    parts = m.group(1).strip().split()
    out = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            try:
                out[k.strip()] = int(v.strip().rstrip(","))
            except ValueError:
                pass
    return out if out else None


def load_comparison(comparison_path: Path) -> dict | None:
    """Load comparison.json and return dict with total_bytes_per_bank and real_window_probe."""
    if not comparison_path.exists():
        return None
    data = json.loads(comparison_path.read_text())
    dual = data.get("dual_source", {})
    snapshot = dual.get("snapshot", {})
    total_per_bank = snapshot.get("total_bytes_per_bank", DEFAULT_TOTAL_BYTES_PER_BANK)
    probe = data.get("real_window_probe", {})
    first_clash = probe.get("first_static_cb_clash") if probe else None
    return {
        "total_bytes_per_bank": total_per_bank,
        "static_cb_occupied_bytes_per_bank": (
            first_clash.get("static_cb_occupied_bytes_per_bank") if first_clash else None
        ),
        "effective_headroom_bytes_per_bank": (
            first_clash.get("effective_headroom_bytes_per_bank") if first_clash else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Parse SDPA decode CB sizes and report bottleneck-core memory usage.")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Path to debug log (TT_LOGGER_LEVEL=Debug run).",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=None,
        help="Path to comparison.json (for total_per_bank and/or probe static CB end).",
    )
    parser.add_argument(
        "--total-per-bank",
        type=int,
        default=None,
        help="Override total L1 bytes per bank (default: from comparison or %s)." % DEFAULT_TOTAL_BYTES_PER_BANK,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write summary to JSON file.",
    )
    args = parser.parse_args()

    total_per_bank = args.total_per_bank or DEFAULT_TOTAL_BYTES_PER_BANK
    comp = load_comparison(args.comparison) if args.comparison else None
    if comp:
        total_per_bank = comp["total_bytes_per_bank"]

    cb_total_from_log = parse_cb_total_from_log(args.log) if args.log else None
    cb_breakdown = parse_cb_breakdown_from_log(args.log) if args.log else None

    # Best estimate of static CB size on bottleneck core: from debug log, else from probe
    static_cb_bytes = cb_total_from_log
    if static_cb_bytes is None and comp and comp.get("static_cb_occupied_bytes_per_bank") is not None:
        static_cb_bytes = comp["static_cb_occupied_bytes_per_bank"]

    result = {
        "total_bytes_per_bank": total_per_bank,
        "static_cb_bytes_per_core": static_cb_bytes,
        "static_cb_source": (
            "debug_log"
            if cb_total_from_log is not None
            else ("probe_clash" if comp and comp.get("static_cb_occupied_bytes_per_bank") is not None else None)
        ),
        "cb_breakdown": cb_breakdown,
        "effective_headroom_bytes": (
            comp.get("effective_headroom_bytes_per_bank")
            if comp
            else (total_per_bank - static_cb_bytes if static_cb_bytes is not None else None)
        ),
    }

    if static_cb_bytes is not None and total_per_bank > 0:
        result["static_cb_pct"] = round(100.0 * static_cb_bytes / total_per_bank, 2)
        result["headroom_pct"] = round(100.0 * (total_per_bank - static_cb_bytes) / total_per_bank, 2)
    if result.get("effective_headroom_bytes") is not None and total_per_bank > 0:
        result["effective_headroom_pct"] = round(100.0 * result["effective_headroom_bytes"] / total_per_bank, 2)

    # Print report
    print("Bottleneck-core L1 memory usage (exact CB + allocator view)")
    print("=" * 60)
    print(f"Total L1 bytes per bank (per core): {total_per_bank:,}")
    if static_cb_bytes is not None:
        print(f"Static CB size per core (bytes):   {static_cb_bytes:,}  (source: {result['static_cb_source']})")
        print(f"Static CB % of bank:               {result.get('static_cb_pct', 'N/A')}%")
        print(f"Headroom after CB (bytes):        {total_per_bank - static_cb_bytes:,}")
        print(f"Headroom % of bank:                {result.get('headroom_pct', 'N/A')}%")
    else:
        print("Static CB size:                    (not found in log or comparison)")
    if result.get("effective_headroom_bytes") is not None:
        print(
            f"Effective headroom (from probe):   {result['effective_headroom_bytes']:,}  ({result.get('effective_headroom_pct', 'N/A')}%)"
        )
    if cb_breakdown:
        print("\nPer-CB sizes (bytes per core):")
        for k in sorted(cb_breakdown.keys(), key=lambda x: (len(x), x)):
            print(f"  {k}: {cb_breakdown[k]:,}")
        print(f"  Sum: {sum(cb_breakdown.values()):,}")
    print()
    print("Note: 'Bank' for L1 SRAM = one compute core's L1. See document for bank vs core.")
    if not static_cb_bytes and not comp:
        print("Hint: Run with TT_LOGGER_LEVEL=Debug and --log, or pass --comparison with probe data.")
    print()

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

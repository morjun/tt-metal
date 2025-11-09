#!/usr/bin/env python3
"""
Parse tracy_ops_data.csv to extract operation information
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional


def parse_tracy_ops_data(csv_path: Path) -> Dict[int, Dict]:
    """
    Parse tracy_ops_data.csv to extract operation information by global_call_count.

    Returns:
        Dict mapping global_call_count to operation data
    """
    if not csv_path.exists():
        return {}

    with open(csv_path, "r") as f:
        content = f.read()

    lines = content.split("\n")
    ops_by_global_call_count = {}
    current_op = None
    current_json_lines = []
    in_json = False

    for line in lines[1:]:  # Skip header
        if not line.strip():
            continue

        # Check if line starts with TT_DNN_DEVICE_OP
        if "TT_DNN_DEVICE_OP" in line:
            # Extract global_call_count from the line
            # Format: `TT_DNN_DEVICE_OP: "OpName", hash, device_id, global_call_count ->
            match = re.search(r"(\d+)\s*->", line)
            if match:
                global_call_count = int(match.group(1))
                # Extract op name
                op_name_match = re.search(r'"([^"]+)"', line)
                op_name = op_name_match.group(1) if op_name_match else "Unknown"
                current_op = {"global_call_count": global_call_count, "op_name": op_name, "json_lines": []}
                in_json = True
        elif in_json and current_op:
            current_op["json_lines"].append(line)
            if line.strip() == "}":
                # End of JSON block
                try:
                    json_str = "\n".join(current_op["json_lines"])
                    op_data = json.loads(json_str)
                    current_op["data"] = op_data
                    ops_by_global_call_count[current_op["global_call_count"]] = current_op
                except Exception:
                    pass
                in_json = False
                current_op = None

    return ops_by_global_call_count


if __name__ == "__main__":
    csv_path = Path("generated/profiler/.logs/tracy_ops_data.csv")
    ops = parse_tracy_ops_data(csv_path)
    print(f"Found {len(ops)} operations")
    for gcc in sorted(ops.keys())[:10]:
        op = ops[gcc]
        op_code = op.get("data", {}).get("op_code", "N/A")
        print(f"  global_call_count={gcc}: op_name={op['op_name']}, op_code={op_code}")

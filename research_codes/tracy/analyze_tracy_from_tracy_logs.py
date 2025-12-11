#!/usr/bin/env python3
"""Deterministic mapping from TRACY host zones to device run_host_ids using
tracy_ops_times.csv and tracy_ops_data.csv, then aggregate device cycles from
profile_log_device.csv per forward pass and per component (BRISC/NCRISC/TRISC).

Usage: run from repo root or adjust paths below.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "tracy_output" / ".logs"


def parse_tracy_ops_times(path):
    zones = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                start_ns = int(r["ns_since_start"])
                dur_ns = int(r["exec_time_ns"]) if r["exec_time_ns"] else 0
            except Exception:
                continue
            zones.append(
                {
                    "name": r.get("name") or r.get("zone_name"),
                    "src_file": r.get("src_file"),
                    "start_ns": start_ns,
                    "end_ns": start_ns + dur_ns,
                    "dur_ns": dur_ns,
                }
            )
    return zones


def parse_tracy_ops_data(path):
    data = Path(path).read_text()
    # Find patterns like `...`;<timestamp>
    pattern = re.compile(r"`(.*?)`;\s*(\d+)", re.DOTALL)
    events = []
    for m in pattern.finditer(data):
        block = m.group(1)
        ts = int(m.group(2))
        header = block.splitlines()[0] if block.splitlines() else ""
        if header.startswith("TT_SIGNPOST:"):
            sp = header.split("TT_SIGNPOST:", 1)[1].strip()
            events.append({"type": "signpost", "name": sp, "ts": ts})
            continue
        # header often like: TT_DNN_DEVICE_OP: "Matmul", <hash>, 0, <global_call> ->
        m2 = re.match(r"TT_DNN_DEVICE_OP:\s*\"(?P<op>[^\"]+)\".*?,\s*(?P<count>\d+)\s*->", header)
        if m2:
            op = m2.group("op")
            global_call = int(m2.group("count"))
            # fallback: try to parse explicit global_call_count from block
            m3 = re.search(r'"global_call_count"\s*:\s*(\d+)', block)
            if m3:
                global_call = int(m3.group(1))
            events.append({"type": "device_op", "op": op, "global_call_count": global_call, "ts": ts})
            continue
        # ignore other blocks
    events.sort(key=lambda e: e["ts"])
    return events


def parse_profile_log_device(path):
    # returns list of dict rows; header present in first line
    rows = []
    with open(path, "r", newline="") as f:
        header = f.readline()
        # Use csv with header mapping by splitting the known header
        reader = csv.reader(f)
        for r in reader:
            if not r or len(r) < 12:
                continue
            try:
                time_cycles = int(r[5])
                run_host_id = int(r[7]) if r[7] else None
            except Exception:
                continue
            zone_name = r[10]
            rows.append({"time_cycles": time_cycles, "run_host_id": run_host_id, "zone_name": zone_name, "raw": r})
    return rows


def map_zones_to_run_ids(zones, ops_events):
    # For each host Tracy zone (host-side timing), collect device_op global_call_counts
    # whose timestamp falls within the zone's start..end window.
    idx = 0
    n = len(ops_events)
    zone_map = {}
    for z in zones:
        s, e = z["start_ns"], z["end_ns"]
        matched = set()
        # advance idx until events are >= s - small safety
        while idx < n and ops_events[idx]["ts"] < s - 1:
            idx += 1
        j = idx
        while j < n and ops_events[j]["ts"] <= e + 1:
            ev = ops_events[j]
            if ev["type"] == "device_op":
                matched.add(ev["global_call_count"])
            j += 1
        zone_map[z["name"]] = sorted(matched)
    return zone_map


def aggregate_device_cycles(prof_rows, run_ids_for_zone):
    # prof_rows: list of rows with time_cycles, run_host_id, zone_name
    # run_ids_for_zone: set/list of run_ids
    comp_cycles = defaultdict(int)
    times = []
    for r in prof_rows:
        if r["run_host_id"] in run_ids_for_zone:
            zn = r["zone_name"].upper() if r["zone_name"] else "UNKNOWN"
            # normalize to BRISC/NCRISC/TRISC
            if "BRISC" in zn:
                comp = "BRISC"
            elif "NCRISC" in zn:
                comp = "NCRISC"
            elif "TRISC" in zn or "TENSIX" in zn or "TENSIX_COMPUTE" in zn:
                comp = "TRISC"
            else:
                comp = "OTHER"
            comp_cycles[comp] += 1  # count rows (placeholder) - better to aggregate actual cycles per row if available
            times.append(r["time_cycles"])
    # Note: profile_log_device rows are sampled per zone event; we can compute wall-clock window from min/max cycles
    if times:
        minc, maxc = min(times), max(times)
    else:
        minc = maxc = None
    return comp_cycles, minc, maxc


def run():
    ops_times_f = LOG_DIR / "tracy_ops_times.csv"
    ops_data_f = LOG_DIR / "tracy_ops_data.csv"
    prof_f = LOG_DIR / "profile_log_device.csv"
    print("Reading host Tracy ops times...", ops_times_f)
    zones = parse_tracy_ops_times(ops_times_f)
    print(f"Found {len(zones)} host zones (examples):", [z["name"] for z in zones[:5]])
    print("Reading ops data (device ops + signposts)...", ops_data_f)
    ops_events = parse_tracy_ops_data(ops_data_f)
    print(f"Found {len(ops_events)} ops events (device_ops+signpost). Sample: ", ops_events[:5])
    print("Mapping host zones to device run_host_ids using timestamps...")
    zone_map = map_zones_to_run_ids(zones, ops_events)
    # show mapping for forward passes (heuristic: name contains 'Forward' or 'MiniBatch')
    forward_zones = {
        k: v for k, v in zone_map.items() if k and ("Forward" in k or "MiniBatch" in k or "LargeBatch" in k)
    }
    print("Forward zones mapped (name -> run_ids):")
    for k, v in forward_zones.items():
        print(f"  {k}: {v}")

    print("Reading device profile log...", prof_f)
    prof_rows = parse_profile_log_device(prof_f)
    print(f"Loaded {len(prof_rows)} device profile rows")

    # For each forward zone, aggregate device cycles across matched run_ids
    chip_freq_mhz = 1350.0
    for name, run_ids in forward_zones.items():
        if not run_ids:
            print(f"Zone {name}: no mapped run_ids")
            continue
        run_ids_set = set(run_ids)
        comp_counts, minc, maxc = aggregate_device_cycles(prof_rows, run_ids_set)
        print(f"\nZone: {name}")
        print(f"  mapped run_ids: {sorted(run_ids)}")
        total_counts = sum(comp_counts.values())
        for comp, cnt in comp_counts.items():
            pct = 100.0 * cnt / total_counts if total_counts else 0.0
            print(f"   {comp}: rows={cnt} ({pct:.1f}%)")
        if minc is not None and maxc is not None:
            wall_ms = (maxc - minc) / (chip_freq_mhz * 1e3)
            print(f"  device wall-clock approx: {wall_ms:.3f} ms (cycles {minc}->{maxc})")


if __name__ == "__main__":
    run()

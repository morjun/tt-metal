import csv
import sys
from collections import defaultdict


def analyze(filename):
    print(f"--- Analyzing {filename} ---")
    events = defaultdict(list)

    try:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 12:
                    continue
                # key: (core_time, risc, zone_name)
                # But we just want average duration per zone
                try:
                    risc = row[3]
                    if "TRISC_0" not in risc:
                        continue
                    # row[5] is time
                    time = int(row[5])
                    name = row[10]
                    ztype = row[11]

                    if ztype == "ZONE_START":
                        events[(risc, name)].append({"start": time, "end": None})
                    elif ztype == "ZONE_END":
                        # Find last start
                        l = events[(risc, name)]
                        if l and l[-1]["end"] is None:
                            l[-1]["end"] = time
                except:
                    pass
    except FileNotFoundError:
        print("File not found")
        return

    for k, intervals in events.items():
        risc, name = k
        if name not in ["CB-WAIT-FRONT", "BATCH-ITERATION", "GEMM-PROCESSING"]:
            continue

        durs = [i["end"] - i["start"] for i in intervals if i["end"] is not None]
        if not durs:
            continue
        avg = sum(durs) / len(durs)
        print(f"{risc} {name}: Count={len(durs)}, AvgDuration={avg:.0f} cycles")


analyze("generated/profiler/.logs/profile_log_device.csv")
# analyze("research_codes/profile_log_device_new_zones_deactivated.csv")
analyze("research_codes/profile_log_device_minimized_w0m1_gemm_nosharding.csv")

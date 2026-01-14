import argparse
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys


def parse_log(file_path):
    timer_data = []
    # Regex to capture timestamp, timer name, and duration
    # Example: 2026-01-12 08:46:59.311 | info | Metal | [Timer] to_device: 960 us (timer.hpp:21)
    regex = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+\|.*\[Timer\]\s+(.*?):\s+(\d+)\s+us"

    with open(file_path, "r") as f:
        for line in f:
            match = re.search(regex, line)
            if match:
                timestamp_str = match.group(1)
                name = match.group(2)
                duration_us = int(match.group(3))

                end_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                start_time = end_time - timedelta(microseconds=duration_us)

                timer_data.append({"name": name, "start": start_time, "end": end_time, "duration_us": duration_us})
    return timer_data


def visualize_timeline(timer_data, output_file):
    if not timer_data:
        print("No timer data found.")
        return

    # Sort by start time
    timer_data.sort(key=lambda x: x["start"])

    # Normalize start times to 0 relative to the first evnet
    base_time = timer_data[0]["start"]

    # Prepare data
    unique_names = sorted(list(set(d["name"] for d in timer_data)))
    # Use a colormap
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % 10) for i, name in enumerate(unique_names)}

    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot each bar
    for i, d in enumerate(timer_data):
        start_offset_ms = (d["start"] - base_time).total_seconds() * 1000
        duration_ms = d["duration_us"] / 1000.0
        color = colors[d["name"]]

        ax.barh(i, duration_ms, left=start_offset_ms, align="center", color=color, edgecolor="black", height=0.8)

    # Formatting
    ax.invert_yaxis()  # Top-to-bottom
    ax.set_yticks([])  # Remove Y-axis labels
    ax.set_xlabel("Time (ms)")
    ax.set_title("Timeline of ttnn::Timer Events")

    # Create legend handles
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[name]) for name in unique_names]
    ax.legend(handles, unique_names, title="Timer Events", loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Timeline saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ttnn::Timer logs.")
    parser.add_argument("log_file", help="Path to the log file")
    parser.add_argument("--output", default="timeline_visualization.png", help="Output image file")

    args = parser.parse_args()

    data = parse_log(args.log_file)
    visualize_timeline(data, args.output)

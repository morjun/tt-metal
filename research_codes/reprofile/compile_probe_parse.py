#!/usr/bin/env python3
"""Bucket kernel-compiler subprocess activity into per-iteration windows.
args: <demo_log> <sampler_log>
demo_log lines: 'YYYY-MM-DD HH:MM:SS.mmm | ... Iteration N: Xms @ ...'
sampler lines:  '<epoch.ns> <num_compiler_procs>'
For each iteration, window = [end_ts - dur, end_ts]; report compiler busy-fraction + max.
If iter0/1/2 are busy and steady (iter3+) is idle, the warm-up time IS compilation.
"""
import sys, re, datetime

demo_log, samp_log = sys.argv[1], sys.argv[2]

iters = []  # (n, start_epoch, end_epoch, dur_s)
pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*Iteration (\d+): (\d+)ms")
for ln in open(demo_log, errors="ignore"):
    m = pat.search(ln)
    if not m:
        continue
    ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
    n = int(m.group(2))
    dur = int(m.group(3)) / 1000.0
    iters.append((n, ts - dur, ts, dur))

samples = []  # (epoch, count)
for ln in open(samp_log, errors="ignore"):
    p = ln.split()
    if len(p) == 2:
        try:
            samples.append((float(p[0]), int(p[1])))
        except ValueError:
            pass

if not iters:
    print("  (no Iteration lines parsed)")
    sys.exit(0)
if not samples:
    print("  (no sampler data)")
    sys.exit(0)

print(f"  {'iter':>4} {'dur(s)':>8} {'#samp':>6} {'maxC':>5} {'busy%':>6}  verdict")
total_busy_compile = 0.0
for n, s, e, dur in iters:
    win = [c for (t, c) in samples if s <= t <= e]
    if not win:
        # window shorter than sample interval; mark n/a
        print(f"  {n:>4} {dur:>8.3f} {0:>6} {'-':>5} {'-':>6}  (window < sample interval)")
        continue
    busy = sum(1 for c in win if c > 0)
    frac = busy / len(win)
    maxc = max(win)
    busy_secs = frac * dur
    if n >= 3:
        total_busy_compile += 0.0
    print(
        f"  {n:>4} {dur:>8.3f} {len(win):>6} {maxc:>5} {100*frac:>5.0f}%  "
        f"{'COMPILING' if frac > 0.3 else ('steady/no-compile' if maxc == 0 else 'partial')}"
    )


# summary: compile-busy seconds in warm-up (iter0-2) vs steady (iter3+)
def busy_secs(lo, hi):
    tot = 0.0
    for n, s, e, dur in iters:
        if not (lo <= n <= hi):
            continue
        win = [c for (t, c) in samples if s <= t <= e]
        if win:
            tot += (sum(1 for c in win if c > 0) / len(win)) * dur
    return tot


warm = busy_secs(0, 2)
steady = busy_secs(3, 99)
warm_wall = sum(d for n, _, _, d in iters if n <= 2)
steady_wall = sum(d for n, _, _, d in iters if n >= 3)
print(f"\n  warm-up (iter0-2): wall {warm_wall:.2f}s, compiler-busy ~{warm:.2f}s")
print(f"  steady (iter3+):   wall {steady_wall:.2f}s, compiler-busy ~{steady:.2f}s")
print(
    f"  => warm-up is {100*warm/warm_wall:.0f}% compiler-busy; steady is "
    f"{(100*steady/steady_wall if steady_wall else 0):.0f}% compiler-busy"
)

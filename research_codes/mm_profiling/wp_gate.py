#!/usr/bin/env python3
"""Pre-timing audit gate for the wait-placement experiment (6P.39).

Required before any timing: unchanged DRAM trace, unchanged binary/config
addresses and sends, unchanged existing required synchronization, and exactly the
intended extra commands.
"""
import sys, os, subprocess, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configwait_map import load_cell, decision_tuple

root = sys.argv[1]
COND = ["ringbase", "bottomup", "early", "late"]
INJECT = {"ringbase": [], "bottomup": [], "early": [413, 449], "late": [445, 481]}
STALL_BYTES = 64  # measured: pid 57 is 2176 B with a wait, 2112 B without

cells = {}
for c in COND:
    d = os.path.join(root, f"r0_{c}")
    if os.path.exists(os.path.join(d, "alloc.jsonl")):
        cells[c] = (load_cell(d), d)

print("conditions captured:", sorted(cells))
if "bottomup" not in cells:
    raise SystemExit("bottomup reference missing")

ok = True

print("\nA. output correctness and injection log")
for c in COND:
    if c not in cells:
        print(f"  {c:9s} NOT CAPTURED")
        ok = False
        continue
    log = open(os.path.join(cells[c][1], "run.log")).read()
    n_inj = log.count("GEMMA4_ADD_WAITS[")
    passed = "1 passed" in log
    want = len(INJECT[c])
    good = passed and n_inj == want
    ok &= good
    print(f"  {c:9s} passed={passed}  injection log lines={n_inj} (expected {want})" + ("" if good else "   <-- FAIL"))

print("\nB. initial DRAM trace unchanged vs the bottom-up reference")
ref = os.path.join(cells["bottomup"][1], "stream0.bin")
for c in COND:
    if c not in cells:
        continue
    same = subprocess.call(["cmp", "-s", ref, os.path.join(cells[c][1], "stream0.bin")]) == 0
    ok &= same
    print(f"  {c:9s} stream0 identical to bottomup: {same}" + ("" if same else "   <-- FAIL"))

print("\nC. against bottom-up: only the intended nodes change, and only their stall flag")
bu = cells["bottomup"][0]
bup = bu[max(bu)]["programs"]
for c in ("early", "late"):
    if c not in cells:
        continue
    cc = cells[c][0]
    pp = cc[max(cc)]["programs"]
    if len(pp) != len(bup):
        print(f"  {c}: trace length {len(pp)} != {len(bup)}   <-- FAIL")
        ok = False
        continue
    changed, bad = [], []
    for x, y in zip(bup, pp):
        if decision_tuple(x) == decision_tuple(y):
            continue
        changed.append(x["idx"])
        # the only permitted change is gaining stall_before_program plus its target
        permitted = (
            y["stall_before_program"] == 1
            and x["stall_before_program"] == 0
            and x["stall_first"] == y["stall_first"] == 0
            and x["send_binary"] == y["send_binary"]
            and [r["nb_off"] for r in x["regions"]] == [r["nb_off"] for r in y["regions"]]
            and [r["bin_off"] for r in x["regions"]] == [r["bin_off"] for r in y["regions"]]
        )
        if not permitted:
            bad.append(x["idx"])
    good = changed == INJECT[c] and not bad
    ok &= good
    print(
        f"  {c:9s} nodes changed: {changed} (expected {INJECT[c]}); "
        f"changes that are not purely an added wait: {bad}" + ("" if good else "   <-- FAIL")
    )
    for i in INJECT[c]:
        print(
            f"      node {i}: target {pp[i]['sync_count']}  "
            f"(bottom-up had stall={bup[i]['stall_before_program']}, target {bup[i]['sync_count']})"
        )

print("\nD. existing required synchronization is untouched")
for c in ("early", "late"):
    if c not in cells:
        continue
    cc = cells[c][0]
    pp = cc[max(cc)]["programs"]
    diff = [
        x["idx"]
        for x, y in zip(bup, pp)
        if x["idx"] not in INJECT[c]
        and (x["stall_first"], x["stall_before_program"], x["sync_count"])
        != (y["stall_first"], y["stall_before_program"], y["sync_count"])
    ]
    ok &= not diff
    print(f"  {c:9s} other nodes whose wait changed: {diff}" + ("" if not diff else "   <-- FAIL"))

print("\nE. exactly the intended extra command bytes")
base_bytes = sum(p["bytes"] for p in bup)
for c in COND:
    if c not in cells:
        continue
    cc = cells[c][0]
    pp = cc[max(cc)]["programs"]
    tot = sum(p["bytes"] for p in pp)
    want = base_bytes + STALL_BYTES * len(INJECT[c]) if c != "ringbase" else None
    if want is None:
        print(f"  {c:9s} program command bytes {tot} (original allocator, not comparable)")
    else:
        good = tot == want
        ok &= good
        print(
            f"  {c:9s} program command bytes {tot}, bottom-up {base_bytes}, "
            f"delta {tot-base_bytes} (expected {STALL_BYTES*len(INJECT[c])})" + ("" if good else "   <-- FAIL")
        )

print(f"\nGATE: {'PASS -- timing may proceed' if ok else 'FAIL -- do not time; report the actual changes'}")

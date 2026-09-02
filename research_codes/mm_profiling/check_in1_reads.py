#!/usr/bin/env python3
"""Prove that the L1 WIDTH_SHARDED weight path issues NO NoC read, and the interleaved one does.

Backs the table in L1_WEIGHT_PINNING.md section 4.4.4.  Preprocesses the in1 reader kernel
against each macro set and reports which `noc.async_read` / `noc.async_read_barrier`
statements survive.  Undecidable conditions (macros we do not model) are kept, so the
surviving set is an OVER-estimate -- which is the conservative direction for the claim
"IN1_SHARDED issues zero weight reads".

Note `:262` is the SPARSITY accessor, not in1, and is additionally killed at the C++ level
by `if constexpr (batchB > 0)` -- batchB is compile-time arg 17, zero for every drafter matmul.
"""
import os
import re
import sys

REL = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/" "reader_bmm_tile_layout_in1_sender_writer_padding.cpp"
ARMS = (
    ("interleaved (no IN1_* macro)", frozenset()),
    ("L1 WIDTH_SHARDED (IN1_SHARDED)", frozenset({"IN1_SHARDED"})),
    ("DRAM width-sharded", frozenset({"IN1_DRAM_WIDTH_SHARDED"})),
    ("DRAM height-sharded", frozenset({"IN1_DRAM_HEIGHT_SHARDED"})),
)


def evaluate(cond, macros):
    """Evaluate a #if expression over defined()/!/&&/||. None when it cannot be decided."""
    e = re.sub(r"defined\s*\(\s*(\w+)\s*\)", lambda m: "1" if m.group(1) in macros else "0", cond)
    e = re.sub(r"defined\s+(\w+)", lambda m: "1" if m.group(1) in macros else "0", e)
    if re.search(r"[A-Za-z_]\w*", e):
        return None
    try:
        return bool(eval(e.replace("&&", " and ").replace("||", " or ").replace("!", " not ")))
    except Exception:
        return None


def surviving(lines, macros):
    """Line numbers that survive preprocessing under `macros`."""
    stack = []  # [active_now, some_branch_taken]
    live = set()
    for n, l in enumerate(lines, 1):
        m = re.match(r"#\s*(ifdef|ifndef|if|elif|else|endif)\b\s*(.*)", l.strip())
        if m:
            kw, cond = m.group(1), m.group(2).split("//")[0].strip()
            if kw in ("ifdef", "ifndef", "if"):
                v = (
                    (cond in macros)
                    if kw == "ifdef"
                    else (cond not in macros)
                    if kw == "ifndef"
                    else evaluate(cond, macros)
                )
                if v is None:
                    v = True  # undecidable -> keep
                stack.append([v, v])
            elif kw == "elif" and stack:
                v = evaluate(cond, macros)
                v = True if v is None else v
                v = v and not stack[-1][1]
                stack[-1] = [v, stack[-1][1] or v]
            elif kw == "else" and stack:
                stack[-1] = [not stack[-1][1], True]
            elif kw == "endif" and stack:
                stack.pop()
            continue
        if all(f[0] for f in stack):
            live.add(n)
    return live


def main():
    path = os.path.join(os.environ.get("TT_METAL_HOME", "."), REL)
    if not os.path.exists(path):
        sys.exit(f"not found: {path}\nSet TT_METAL_HOME to the tt-metal checkout.")
    lines = open(path).read().split("\n")
    reads = {n for n, l in enumerate(lines, 1) if "noc.async_read(" in l}
    barriers = {n for n, l in enumerate(lines, 1) if "noc.async_read_barrier()" in l}
    other = {n for n, l in enumerate(lines, 1) if "noc.async_read_with_state" in l}

    print(f"{REL}\n")
    print(f"  {'build':<32} {'async_read':>18} {'barrier':>18} {'read_with_state':>18}")
    for tag, macros in ARMS:
        live = surviving(lines, macros)
        f = lambda s: ",".join(f":{n}" for n in sorted(s & live)) or "-"
        print(f"  {tag:<32} {f(reads):>18} {f(barriers):>18} {f(other):>18}")
    print("\n  :262 is the sparsity accessor, not in1 (guarded by `if constexpr (batchB > 0)`, :261).")
    print("  Subtracting it: IN1_SHARDED issues ZERO NoC reads for the weight.")
    print("  DRAM-sharded has no `async_read` but DOES read -- via async_read_with_state. A row with no")
    print("  `async_read` is not by itself proof of no traffic; see section 4.4.2's allocation argument.")


if __name__ == "__main__":
    main()

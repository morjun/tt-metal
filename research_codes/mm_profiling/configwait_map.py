#!/usr/bin/env python3
"""Configuration-wait map.

Joins the allocator record (GEMMA4_ALLOCMAP_OUT) with the trace-stream record
(GEMMA4_TRACEDUMP_DIR) for one (K, arm) capture, and compares corresponding
programs across K and across the two gather arms.

The join key is (trace_idx, program index). alloc_begin carries trace_idx;
trace<N>.jsonl carries the same ordinal as N. Both are asserted to describe a
single device range, since a second range would make the ordinals ambiguous.

Nothing here infers timing. Every column is a decision the host made while
recording the trace.
"""
import json, sys, os
from collections import defaultdict


def load_cell(d):
    """Return {trace_idx: {'programs': [merged records], 'meta': {...}}}."""
    alloc = defaultdict(dict)  # trace_idx -> idx -> record
    alloc_meta = {}
    apath = os.path.join(d, "alloc.jsonl")
    seq_to_trace = {}
    with open(apath) as f:
        for line in f:
            r = json.loads(line)
            if r["event"] == "alloc_begin":
                seq_to_trace[r["alloc_seq"]] = r["trace_idx"]
                alloc_meta[r["trace_idx"]] = r
            elif r["event"] == "program":
                t = seq_to_trace[r["alloc_seq"]]
                if r["idx"] in alloc[t]:
                    raise SystemExit(
                        f"{d}: duplicate program idx {r['idx']} in trace {t}; "
                        "more than one device range or sub-device -- the join key is ambiguous"
                    )
                alloc[t][r["idx"]] = r

    out = {}
    n = 0
    while os.path.exists(os.path.join(d, f"trace{n}.jsonl")):
        progs, meta = [], {}
        with open(os.path.join(d, f"trace{n}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r["event"] == "trace_begin":
                    meta = r
                elif r["event"] == "program":
                    if r["range_idx"] != 0:
                        raise SystemExit(f"{d}/trace{n}: range_idx {r['range_idx']} != 0")
                    progs.append(r)
        merged = []
        for p in progs:
            a = alloc[n].get(p["idx"])
            if a is None:
                raise SystemExit(f"{d}/trace{n}: no allocator record for idx {p['idx']}")
            if a["program_id"] != p["program_id"]:
                raise SystemExit(f"{d}/trace{n}: program_id mismatch at idx {p['idx']}")
            m = dict(a)
            m.update({k: v for k, v in p.items() if k not in ("event", "idx", "program_id")})
            merged.append(m)
        out[n] = {"programs": merged, "trace_meta": meta, "alloc_meta": alloc_meta.get(n, {})}
        n += 1
    if not out:
        raise SystemExit(f"{d}: no trace*.jsonl")
    return out


DEPS = ["dep_nonbinary_reuse", "dep_binary_reuse", "dep_fixed_addr", "dep_alloc_reset", "dep_launch_buffer"]


def dep_source(p):
    """Which raw dependency equals the combined index the wait was built from."""
    c = p["combined"]
    if c is None:
        return "none"
    hits = [d[4:] for d in DEPS if p[d] == c]
    return "+".join(hits) if hits else "?"


def summarize(name, cell, trace_idx):
    ps = cell[trace_idx]["programs"]
    print(f"\n=== {name}  trace {trace_idx}  n={len(ps)} ===")
    tally = defaultdict(int)
    for p in ps:
        tally["send_binary"] += p["send_binary"]
        tally["stall_first"] += p["stall_first"]
        tally["stall_before_program"] += p["stall_before_program"]
        tally["suppressed"] += p["suppressed_by_last_stall"]
        tally["pc_queried"] += p["pc_queried"]
        tally["pc_hit"] += p["pc_q_hit"]
        tally["pc_wrapped"] += p["pc_q_wrapped"]
        tally["pc_emptied"] += p["pc_q_emptied"]
        tally["pc_evictions"] += p["pc_q_evictions"]
        tally["pc_reset_here"] += p["pc_reset_here"]
        tally["alloc_reset_events"] += p["alloc_reset_events"]
        for d in DEPS:
            if p[d] is not None:
                tally["fired_" + d[4:]] += 1
        tally["src_" + dep_source(p)] += 1
        for r in p["regions"]:
            tally["nb_evicted"] += r["nb_evicted"]
            tally["bin_evicted"] += r["bin_evicted"]
            tally["bin_" + r["bin_path"]] += 1
    for k in sorted(tally):
        print(f"  {k:32s} {tally[k]}")
    return tally


def print_window(name, cell, trace_idx, lo, hi):
    ps = cell[trace_idx]["programs"]
    print(f"\n--- {name} trace {trace_idx} programs [{lo}:{hi}) ---")
    hdr = "idx  pid  sendb natsb pcq hit wrp evc off  |  nbR  bR  fA  aR  lB | comb src" "            wait      stall"
    print(hdr)
    for p in ps[lo:hi]:
        d = lambda k: ("-" if p[k] is None else str(p[k]))
        st = "first" if p["stall_first"] else ("before" if p["stall_before_program"] else "-")
        print(
            f"{p['idx']:4d} {p['program_id']:4d} "
            f"{p['send_binary']:5d} {p['send_binary_natural']:5d} "
            f"{p['pc_queried']:3d} {p['pc_q_hit']:3d} {p['pc_q_wrapped']:3d} "
            f"{p['pc_q_evictions']:3d} {p['pc_q_offset_blocks']:4d} | "
            f"{d('dep_nonbinary_reuse'):>4s} {d('dep_binary_reuse'):>3s} "
            f"{d('dep_fixed_addr'):>3s} {d('dep_alloc_reset'):>3s} {d('dep_launch_buffer'):>3s} | "
            f"{d('combined'):>4s} {dep_source(p):<14s} {p['wait_target']:8d} {st}"
        )


def compare(cells, trace_idx, field_fn, label):
    """Per-index comparison of a scalar across cells; prints only differing rows."""
    names = list(cells)
    lens = {n: len(cells[n][trace_idx]["programs"]) for n in names}
    print(f"\n### {label} -- per-index differences (lengths: {lens})")
    m = min(lens.values())
    diffs = 0
    for i in range(m):
        vals = {n: field_fn(cells[n][trace_idx]["programs"][i]) for n in names}
        if len(set(map(str, vals.values()))) > 1:
            diffs += 1
            if diffs <= 40:
                pid = cells[names[0]][trace_idx]["programs"][i]["program_id"]
                print(f"  idx {i:4d} pid {pid:4d}: " + "  ".join(f"{n}={vals[n]}" for n in names))
    print(f"  ({diffs} differing indices out of {m})")


def steps_of(cell, trace_idx, K):
    """Split the pinned trace into per-draft-step slices.

    The drafter emits a fixed number of ops per step, so ops_per_step is derived
    from the trace length rather than assumed: a ring trace and a DRAM trace have
    different per-step counts, and slicing one with the other's number was an
    earlier error.
    """
    ps = cell[trace_idx]["programs"]
    n = len(ps)
    if n % K != 0:
        return None, n
    return n // K, n


def step_boundary_report(name, cell, trace_idx, K):
    per, n = steps_of(cell, trace_idx, K)
    print(f"\n--- {name} trace {trace_idx}: n={n} K={K} ops/step={per} ---")
    if per is None:
        return
    ps = cell[trace_idx]["programs"]
    for s in range(K):
        sl = ps[s * per : (s + 1) * per]
        sb = sum(p["send_binary"] for p in sl)
        pcq = sum(p["pc_queried"] for p in sl)
        hit = sum(p["pc_q_hit"] for p in sl)
        wrp = sum(p["pc_q_wrapped"] for p in sl)
        ev = sum(p["pc_q_evictions"] for p in sl)
        stalls = sum(p["stall_first"] + p["stall_before_program"] for p in sl)
        nbev = sum(r["nb_evicted"] for p in sl for r in p["regions"])
        bev = sum(r["bin_evicted"] for p in sl for r in p["regions"])
        srcs = defaultdict(int)
        for p in sl:
            srcs[dep_source(p)] += 1
        print(
            f"  step {s}: send_binary={sb:4d} pc_query={pcq:4d} hit={hit:4d} wrap={wrp:3d} "
            f"pc_evict={ev:4d} stalls={stalls:4d} nb_evict={nbev:4d} bin_evict={bev:4d} "
            f"src={dict(sorted(srcs.items()))}"
        )


# The decision fields compared position-by-position across K. `wait_target` is
# excluded from the equality test because it is a cumulative worker count that
# necessarily grows with K; `wait_lag` (how many programs back the wait reaches)
# is the K-invariant form of the same quantity.
DECISION_FIELDS = [
    "send_binary",
    "send_binary_natural",
    "stall_first",
    "stall_before_program",
    "suppressed_by_last_stall",
    "pc_queried",
    "pc_q_hit",
    "pc_q_wrapped",
    "pc_q_emptied",
    "pc_q_evictions",
    "pc_q_offset_blocks",
    "pc_reset_here",
    "alloc_reset_events",
]


def decision_tuple(p):
    v = [p[f] for f in DECISION_FIELDS]
    v.append(dep_source(p))
    v.append(None if p["combined"] is None else p["idx"] - p["combined"])  # wait_lag
    v.append(tuple((r["nb_off"], r["bin_off"], r["bin_path"], r["nb_evicted"], r["bin_evicted"]) for r in p["regions"]))
    return tuple(v)


def decision_dict(p):
    d = {f: p[f] for f in DECISION_FIELDS}
    d["src"] = dep_source(p)
    d["wait_lag"] = None if p["combined"] is None else p["idx"] - p["combined"]
    for r in p["regions"]:
        d[f"ct{r['core_type']}_nb_off"] = r["nb_off"]
        d[f"ct{r['core_type']}_bin_off"] = r["bin_off"]
        d[f"ct{r['core_type']}_bin_path"] = r["bin_path"]
        d[f"ct{r['core_type']}_nb_evicted"] = r["nb_evicted"]
        d[f"ct{r['core_type']}_bin_evicted"] = r["bin_evicted"]
    return d


def step_slice(cell, K, step):
    """Programs of one draft step of the measured (last) trace, or None."""
    t = max(cell)
    ps = cell[t]["programs"]
    if len(ps) % K:
        return None
    per = len(ps) // K
    return ps[step * per : (step + 1) * per]


def cross_K_step(cells, arm, Ks, which, label):
    """Compare one step position across K within a single arm, position by position.

    `which` is 0 for the first step or -1 for the final step. Within an arm the
    number of ops per step is constant across K, so position i is the same op.
    """
    present = [K for K in Ks if f"K{K}a{arm}" in cells]
    if len(present) < 2:
        return
    sl = {}
    for K in present:
        c = cells[f"K{K}a{arm}"]
        s = step_slice(c, K, K - 1 if which == -1 else 0)
        if s is None:
            print(f"  arm{arm} K={K}: trace length not divisible by K -- skipped")
            return
        sl[K] = s
    n = min(len(v) for v in sl.values())
    if len({len(v) for v in sl.values()}) > 1:
        print(
            f"  arm{arm}: ops/step differs across K {[len(sl[K]) for K in present]} -- " "positions are not comparable"
        )
        return
    print(f"\n### {label}: arm{arm}, {'final' if which == -1 else 'first'} step, " f"K in {present}, {n} positions")
    ndiff = 0
    odd_K4 = []
    for i in range(n):
        tuples = {K: decision_tuple(sl[K][i]) for K in present}
        if len(set(tuples.values())) == 1:
            continue
        ndiff += 1
        # A position where K=4 differs from every other K, and the others agree.
        others = [K for K in present if K != 4]
        if (
            4 in present
            and len(others) >= 2
            and len({tuples[K] for K in others}) == 1
            and tuples[4] != tuples[others[0]]
        ):
            odd_K4.append(i)
    print(f"  positions differing across K: {ndiff}/{n}")
    print(f"  positions where K=4 alone differs: {len(odd_K4)}" + (f" -> {odd_K4[:30]}" if odd_K4 else ""))
    for i in odd_K4[:12]:
        dd = {K: decision_dict(sl[K][i]) for K in present}
        keys = [k for k in dd[present[0]] if len({str(dd[K][k]) for K in present}) > 1]
        pid = sl[present[0]][i]["program_id"]
        print(
            f"    pos {i:4d} pid {pid:4d}: "
            + "; ".join(f"{k}=" + "/".join(f"K{K}:{dd[K][k]}" for K in present) for k in keys)
        )


def arm_delta_by_K(cells, Ks):
    """Per-step aggregates for both arms at each K, so the arm difference can be
    read at the same step position across K."""
    print("\n### per-step aggregates, both arms")
    hdr = (
        f"{'cell':8s} {'step':>4s} {'n':>4s} {'sendb':>6s} {'pcq':>5s} {'hit':>5s} {'wrap':>5s} "
        f"{'stallF':>6s} {'stallB':>6s} {'supp':>5s} {'nbEv':>5s} {'binEv':>5s} {'waitlag_med':>11s}"
    )
    print(hdr)
    for K in Ks:
        for a in (0, 1):
            name = f"K{K}a{a}"
            if name not in cells:
                continue
            c = cells[name]
            for s in range(K):
                sl = step_slice(c, K, s)
                if sl is None:
                    continue
                lags = sorted(p["idx"] - p["combined"] for p in sl if p["combined"] is not None)
                med = lags[len(lags) // 2] if lags else -1
                print(
                    f"{name:8s} {s:4d} {len(sl):4d} "
                    f"{sum(p['send_binary'] for p in sl):6d} "
                    f"{sum(p['pc_queried'] for p in sl):5d} "
                    f"{sum(p['pc_q_hit'] for p in sl):5d} "
                    f"{sum(p['pc_q_wrapped'] for p in sl):5d} "
                    f"{sum(p['stall_first'] for p in sl):6d} "
                    f"{sum(p['stall_before_program'] for p in sl):6d} "
                    f"{sum(p['suppressed_by_last_stall'] for p in sl):5d} "
                    f"{sum(r['nb_evicted'] for p in sl for r in p['regions']):5d} "
                    f"{sum(r['bin_evicted'] for p in sl for r in p['regions']):5d} "
                    f"{med:11d}"
                )


def replay_boundary(cells, Ks):
    """The wrap from the last recorded program back to the first on the next replay.

    Only the trace's own head and tail carry it: the recorded stream is byte-fixed,
    so the boundary's cost is whatever the head's wait and the tail's residency
    leave behind.
    """
    print("\n### replay boundary (last 3 and first 3 programs of the measured trace)")
    for K in Ks:
        for a in (0, 1):
            name = f"K{K}a{a}"
            if name not in cells:
                continue
            c = cells[name]
            ps = c[max(c)]["programs"]
            for tag, sel in (("tail", ps[-3:]), ("head", ps[:3])):
                for p in sel:
                    lag = "-" if p["combined"] is None else p["idx"] - p["combined"]
                    print(
                        f"  {name:8s} {tag} idx={p['idx']:4d} pid={p['program_id']:4d} "
                        f"sendb={p['send_binary']} nat={p['send_binary_natural']} "
                        f"pcq={p['pc_queried']} hit={p['pc_q_hit']} "
                        f"stallF={p['stall_first']} stallB={p['stall_before_program']} "
                        f"src={dep_source(p)} lag={lag} wait={p['wait_target']}"
                    )


def last_step_effect(cells, arm, Ks):
    """The effect of a step being LAST, isolated at each K.

    Steps 0..K-2 are decision-identical across K (verified separately), so step
    K-1 of the K-trace and step K-1 of the (K+1)-trace begin from the same
    allocator state. Their difference is exactly what "no successor" changes.
    This is the only comparison in which a K-to-K change is not confounded by a
    different prefix.
    """
    print(f"\n### last-step effect, arm{arm}: step K-1 as FINAL (trace K) vs as INTERIOR (trace K+1)")
    hdr = (
        f"{'K':>2s} {'n':>4s} {'diffpos':>7s} {'sendb F/I':>11s} {'stallF F/I':>11s} "
        f"{'stallB F/I':>11s} {'supp F/I':>9s} {'nbEv F/I':>11s} {'binEv F/I':>11s} {'lagsum F/I':>13s}"
    )
    print(hdr)
    rows = {}
    for K in Ks:
        a, b = f"K{K}a{arm}", f"K{K+1}a{arm}"
        if a not in cells or b not in cells:
            continue
        fin = step_slice(cells[a], K, K - 1)
        inter = step_slice(cells[b], K + 1, K - 1)
        if fin is None or inter is None or len(fin) != len(inter):
            print(f"{K:2d}  incomparable")
            continue
        diff = [i for i in range(len(fin)) if decision_tuple(fin[i]) != decision_tuple(inter[i])]
        agg = lambda sl, f: sum(f(p) for p in sl)
        lag = lambda sl: sum((p["idx"] - p["combined"]) for p in sl if p["combined"] is not None)
        row = dict(
            K=K,
            n=len(fin),
            diffpos=len(diff),
            sendb=(agg(fin, lambda p: p["send_binary"]), agg(inter, lambda p: p["send_binary"])),
            stallF=(agg(fin, lambda p: p["stall_first"]), agg(inter, lambda p: p["stall_first"])),
            stallB=(agg(fin, lambda p: p["stall_before_program"]), agg(inter, lambda p: p["stall_before_program"])),
            supp=(
                agg(fin, lambda p: p["suppressed_by_last_stall"]),
                agg(inter, lambda p: p["suppressed_by_last_stall"]),
            ),
            nbEv=(
                agg(fin, lambda p: sum(r["nb_evicted"] for r in p["regions"])),
                agg(inter, lambda p: sum(r["nb_evicted"] for r in p["regions"])),
            ),
            binEv=(
                agg(fin, lambda p: sum(r["bin_evicted"] for r in p["regions"])),
                agg(inter, lambda p: sum(r["bin_evicted"] for r in p["regions"])),
            ),
            lag=(lag(fin), lag(inter)),
            diff_idx=diff,
        )
        rows[K] = row
        f = lambda t: f"{t[0]}/{t[1]}"
        print(
            f"{K:2d} {row['n']:4d} {row['diffpos']:7d} {f(row['sendb']):>11s} "
            f"{f(row['stallF']):>11s} {f(row['stallB']):>11s} {f(row['supp']):>9s} "
            f"{f(row['nbEv']):>11s} {f(row['binEv']):>11s} {f(row['lag']):>13s}"
        )
    return rows


def last_step_field_breakdown(cells, arm, K):
    """Which fields the last-step effect actually touches, at one K."""
    a, b = f"K{K}a{arm}", f"K{K+1}a{arm}"
    if a not in cells or b not in cells:
        return
    fin = step_slice(cells[a], K, K - 1)
    inter = step_slice(cells[b], K + 1, K - 1)
    tally = defaultdict(int)
    for i in range(len(fin)):
        df, di = decision_dict(fin[i]), decision_dict(inter[i])
        for k in df:
            if str(df[k]) != str(di[k]):
                tally[k] += 1
    print(f"\n  arm{arm} K={K}: fields changed by being last (count of positions, n={len(fin)})")
    for k in sorted(tally, key=lambda x: -tally[x]):
        print(f"    {k:24s} {tally[k]}")


if __name__ == "__main__":
    root = sys.argv[1]
    cells = {}
    for k in range(3, 11):
        for a in (0, 1):
            d = os.path.join(root, f"k{k}_a{a}")
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "alloc.jsonl")):
                cells[f"K{k}a{a}"] = load_cell(d)
    if not cells:
        raise SystemExit("no cells")
    for n, c in cells.items():
        for t in c:
            summarize(n, c, t)
        K = int(n[1])
        # trace 0 is the harness's initial DRAM capture; the measured pinned capture
        # is the last one recorded.
        step_boundary_report(n, c, max(c), K)
    print("\n" + "=" * 78)
    Ks = sorted({int(n[1]) for n in cells})
    arm_delta_by_K(cells, Ks)
    replay_boundary(cells, Ks)
    for arm in (0, 1):
        cross_K_step(cells, arm, Ks, -1, "cross-K decision comparison")
        cross_K_step(cells, arm, Ks, 0, "cross-K decision comparison")
    for arm in (0, 1):
        last_step_effect(cells, arm, Ks)
        for K in Ks:
            last_step_field_breakdown(cells, arm, K)

#!/usr/bin/env python3
"""Figures for the 12B drafter contributions: argmax pad removal, L1 pinning, gather_in0.

Every number is copied verbatim from
lab-meeting-notes/documents/gemma4-specdec/gemma4-12b/MEASUREMENT_RECORD.md (section cited
beside each literal) or, for the placement x delivery figures, read from
placement6_cells.csv (run_placement_campaign.py). Medians and observed min-max only, no CIs,
per the record's convention. Nothing is dropped: flagged cells are drawn and annotated.

    python research_codes/mm_profiling/plot_contributions.py [--out DIR] [--only fig1,fig6]
"""
import argparse
import csv
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Reference palette, light mode (dataviz references/palette.md). Colour follows the entity:
# mcast is always slot 1, the gather_in0 ring always slot 2, wherever they appear.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
BLUE_L, ORANGE_L = "#9ec5f4", "#f5b394"  # lighter steps of the same hues (secondary segments)
MCAST, RING = BLUE, ORANGE

DEFAULT_OUT = Path.home() / "codes/lab-meeting-notes/documents/gemma4-specdec/gemma4-12b/figures"
CELLS = Path(__file__).with_name("placement6_cells.csv")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
        "hatch.color": SURFACE,
        "hatch.linewidth": 1.2,
    }
)
BAR = dict(edgecolor=SURFACE, linewidth=2)  # the 2px surface gap between touching fills


def caption(fig, text):
    fig.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=7.5, color=MUTED, wrap=True)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


def sg(v, nd=2):
    """Signed number with a typographic minus."""
    return f"{v:+.{nd}f}".replace("-", "\u2212")


def med_range(xs):
    return statistics.median(xs), min(xs), max(xs)


# --------------------------------------------------------------------------- Fig 1
def fig1_argmax(out):
    # §4.4, test_fused_draft_k_steps, 12B, 1x1, bf16, K=3, ctx 512
    parts = [("backbone (4 layers)", 1.644, 1.644, BLUE), ("dense lm_head", 1.411, 1.411, ORANGE)]
    parts.append(("argmax_token_id", 1.586, 0.102, AQUA))
    # §4.4 end-to-end A/B, 3 counterbalanced rounds (ms/iter)
    fast, pad = [101.21, 101.29, 101.32], [105.94, 105.81, 104.46]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
    rows = ["after (rows==1 fast path)", "before (pad 1→32 rows)"]
    totals = {1: 4.642, 2: 3.157}  # the record's measured TOTALs; the rounded parts sum 1 us off
    for i, idx in enumerate((2, 1)):
        left = 0
        for label, pre, post, c in parts:
            v = post if idx == 2 else pre
            a.barh(i, v, left=left, height=0.5, color=c, **BAR)
            if v > 0.3:
                a.text(left + v / 2, i, f"{v:.3f}", ha="center", va="center", color="white", fontsize=9)
            left += v
        a.text(left + 0.05, i, f"{totals[idx]:.3f} ms", va="center", color=INK, fontweight="semibold")
    a.text(3.10, -0.3, "argmax 0.102", fontsize=8, color=INK2, va="top")
    a.set_yticks([0, 1], rows)
    a.set_xlim(0, 5.4)
    a.set_xlabel("ms per drafter step")
    a.grid(axis="y", visible=False)
    a.set_title("Drafter step: −1.485 ms (−32.0%); argmax share 34.2% → 3.2%")
    a.legend(
        handles=[Patch(color=c, label=lbl) for lbl, _, _, c in parts],
        loc="upper center",
        bbox_to_anchor=(0.45, -0.22),
        ncol=3,
    )

    for x, ys, c in ((0, pad, INK2), (1, fast, BLUE)):
        m = statistics.median(ys)
        b.scatter([x] * len(ys), ys, s=40, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        b.hlines(m, x - 0.25, x + 0.25, color=INK, linewidth=2)
        b.text(x + 0.3, m, f"{m:.2f}", va="center", color=INK)
    b.set_xticks([0, 1], ["pad (before)", "fast (after)"])
    b.set_xlim(-0.5, 1.8)
    b.set_ylabel("ms / iteration")
    b.grid(axis="x", visible=False)
    b.set_title("End to end: −4.27% [−4.46, −3.01]")
    b.text(
        0.5, 102.6, "26.26 vs 25.14 tok/s/u (+4.46%)\nacceptance identical: 1.66/3", ha="center", fontsize=8, color=INK2
    )
    caption(
        fig,
        "12B + 12B-assistant, 1x1, bf16, K=3. Step: test_fused_draft_k_steps, ctx 512 (§4.4). "
        "E2E: test_demo_spec_decode, canonical prompt, 500 tok, 3 counterbalanced rounds; dots = rounds, "
        "bar = median. r3 pad = 104.46 is unexplained and kept (§4.4 caveat).",
    )
    fig.subplots_adjust(bottom=0.3, wspace=0.3)
    save(fig, out, "fig1_argmax_fix")


# --------------------------------------------------------------------------- Fig 6
def fig6_ring_k_per_core(out):
    # isolated-op us, HiFi2, SFPI 7.67.0. down_proj: §6.8 (mcast = L1-interleaved activation at
    # the tuner's in0_block_w=8). lm_head: §5.5.7 op-level table, second run.
    rows = [
        ("down_proj\nK=8192, 32 cores\n8 K-tiles/core", 40.22, 12.44, 15.07),
        ("lm_head slice 10,240 cols\nK=1024, 32 cores\n1 K-tile/core", 21.43, 22.00, 26.57),
        ("lm_head slice 35,200 cols\nK=1024, 110 cores\n32 cores × 1 tile, 78 empty", 41.09, 70.73, 77.51),
    ]
    fig = plt.figure(figsize=(11, 6.2))
    # strip widths proportional to core count, so no empty axis reads as empty cores
    gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1], width_ratios=[34, 112], hspace=0.75, wspace=0.08)
    a = fig.add_subplot(gs[0, :])
    w = 0.34
    for i, (lbl, mc, ring, ring_conv) in enumerate(rows):
        a.bar(i - w / 2 - 0.01, mc, w, color=MCAST, **BAR)
        a.bar(i + w / 2 + 0.01, ring, w, color=RING, **BAR)
        a.bar(i + w / 2 + 0.01, ring_conv - ring, w, bottom=ring, color=ORANGE_L, **BAR)
        a.text(i - w / 2, mc + 1.2, f"{mc:.2f}", ha="center", color=INK, fontsize=9)
        a.text(i + w / 2, ring_conv + 1.2, f"{ring:.2f} (+{ring_conv - ring:.2f})", ha="center", color=INK, fontsize=9)
        r = ring / mc
        verdict = f"ring {1 / r:.2f}× faster" if r < 0.9 else ("tie" if r < 1.1 else f"ring {r:.2f}× slower")
        a.text(
            i,
            max(mc, ring_conv) + 9,
            f"ring alone / mcast = {r:.2f}×\n{verdict}",
            ha="center",
            color=INK,
            fontsize=9,
            fontweight="semibold",
        )
    a.set_xticks(range(3), [r[0] for r in rows], fontsize=8.5)
    a.set_ylim(0, 100)
    a.set_ylabel("μs per matmul (isolated op)")
    a.grid(axis="x", visible=False)
    a.set_title("gather_in0 pays only when K is thick per core: it wins on down_proj, loses on the thin-K lm_head")
    a.legend(
        handles=[
            Patch(color=MCAST, label="1D mcast (tuner config)"),
            Patch(color=RING, label="gather_in0 ring alone (in0 pre-sharded)"),
            Patch(color=ORANGE_L, label="+ in0 reshard + sharded_to_interleaved"),
        ],
        loc="upper left",
        ncol=1,
        fontsize=8.5,
    )

    # Panel B: K-tile occupancy of each ring core (the ring K-shards in0 across its cores)
    for col, (title, tiles) in enumerate(
        (("down_proj: Kt=256 / 32 cores", [8] * 32), ("lm_head: Kt=32 / 110 cores", [1] * 32 + [0] * 78))
    ):
        b = fig.add_subplot(gs[1, col])
        b.bar(
            range(len(tiles)),
            [max(t, 0.12) for t in tiles],
            0.8,
            color=[RING if t else GRID for t in tiles],
            linewidth=0,
        )
        b.set_title(title, fontsize=9.5)
        b.set_ylim(0, 9)
        b.set_yticks([0, 1, 8])
        b.set_xlim(-1, len(tiles))
        b.set_xlabel("ring core index")
        b.grid(axis="x", visible=False)
        if col == 0:
            b.set_ylabel("K tiles held")
        else:
            b.text(71, 2.2, "78 cores hold no K,\nbut still sit in the ring", ha="center", fontsize=8.5, color=INK2)
            b.set_yticklabels([])
    caption(
        fig,
        "12B, 1x1, bf16, HiFi2 + packer_l1_acc, SFPI 7.67.0; isolated-op, weight pinned L1 width-sharded. "
        "down_proj: test_down_proj_gather_blocks (§6.8, random weight, mcast with L1-interleaved activation). "
        "lm_head: test_lm_head_gather (§5.5.7, real weights). The ring is not bit-exact vs mcast on lm_head "
        "(max|d| 0.078). Step level: split+mcast −0.171 ms vs split+ring −0.135 ms (§5.5.7).",
    )
    fig.subplots_adjust(bottom=0.14)
    save(fig, out, "fig6_ring_k_per_core")


# --------------------------------------------------------------------------- Fig 6b
def fig6b_mcast_steps(out):
    # §6.8, 12B down_proj shape, 32 cores, per_core_N=1, isolated-op
    blk = [1, 2, 4, 8, 16, 32]
    dram = [221.87, 119.45, 66.95, 41.19, 28.80, 23.74]
    l1 = [218.57, 116.29, 65.21, 40.22, 28.18, 23.63]
    steps = [256 // b for b in blk]
    fig, a = plt.subplots(figsize=(7.5, 4.3))
    a.plot(steps, dram, color=MCAST, lw=2, marker="o", ms=6, mec=SURFACE, mew=2, label="mcast, DRAM activation")
    a.plot(steps, l1, color=VIOLET, lw=2, marker="o", ms=6, mec=SURFACE, mew=2, label="mcast, L1 activation")
    a.scatter([32], [12.44], s=90, color=RING, edgecolor=SURFACE, linewidth=2, zorder=4, label="gather_in0 ring alone")
    a.scatter(
        [32], [15.07], s=60, color=ORANGE_L, edgecolor=SURFACE, linewidth=2, zorder=4, label="ring + reshard + S2I"
    )
    a.annotate(
        "tuner's width: in0_block_w=8, 32 K-steps\nmcast 40.22 μs vs ring 12.44 μs (+2.63 conversions)",
        xy=(34, 13),
        xytext=(48, 8),
        fontsize=8.5,
        color=INK2,
        va="center",
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=1),
    )
    a.set_xscale("log", base=2)
    a.set_xticks(steps, [f"{s}\n(blk {b})" for s, b in zip(steps, blk)])
    a.set_xlabel("K-steps per matmul (in0_block_w)")
    a.set_ylabel("μs per matmul")
    a.set_ylim(0, 235)
    a.set_title("mcast pays ~0.8 μs per K-step, whatever a step carries; the ring does the same 32 steps in 12.4 μs")
    a.legend(loc="upper left", fontsize=8.5)
    caption(
        fig,
        "test_down_proj_gather_blocks (§6.8): 12B down_proj K=8192, N=1024, 8x4 cores, weight pinned L1 "
        "width-sharded, HiFi2 + packer_l1_acc, one run, isolated-op, SFPI 7.67.0.",
    )
    fig.subplots_adjust(bottom=0.2)
    save(fig, out, "fig6b_mcast_steps_vs_ring")


# --------------------------------------------------------------------------- Fig 6c
def fig6c_lm_head_split(out):
    # §5.5.7 (HiFi2, current code). Op level (second run), step level, e2e paired deltas.
    op = [
        ("full head\n(DRAM, automatic)", 1412.84, INK2),
        ("35,200 cols in L1\n+ DRAM tail, mcast", 1222.48, MCAST),
        ("35,200 cols in L1\n+ DRAM tail, ring", 1264.39, RING),
    ]
    e2e = {
        ("tuned", "split (mcast)"): [-0.44, -0.54, +0.77, -0.48, -0.65, -0.63],
        ("tuned", "split (ring)"): [-0.48, -0.64, +0.89, -0.31, -0.80, -0.66],
        ("untuned", "split (mcast)"): [-0.83, -0.52, -0.50],
        ("untuned", "split (ring)"): [-0.71, -0.30, -0.55],
    }
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for i, (lbl, v, c) in enumerate(op):
        a.bar(i, v, 0.55, color=c, **BAR)
        a.text(
            i, v + 15, f"{v:.1f}" + ("" if i == 0 else f"\n({sg(v - op[0][1], 1)})"), ha="center", color=INK, fontsize=9
        )
    a.set_xticks(range(3), [o[0] for o in op], fontsize=8.5)
    a.set_ylim(1000, 1500)
    a.set_ylabel("μs, lm_head matmul (op level)")
    a.grid(axis="x", visible=False)
    a.set_title("lm_head column slice pinned in L1")
    a.text(-0.45, 1485, "axis starts at 1000 μs", fontsize=7.5, color=MUTED, ha="left")

    for i, ((drafter, arm), ds) in enumerate(e2e.items()):
        c = MCAST if "mcast" in arm else RING
        b.scatter([i] * len(ds), ds, s=40, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        m = statistics.median(ds)
        b.hlines(m, i - 0.25, i + 0.25, color=INK, lw=2)
        b.text(i + 0.28, m, sg(m), va="center", color=INK, fontsize=9)
    b.axhline(0, color=AXIS, lw=1)
    b.set_xticks(range(4), [f"{d}\n{a_}" for d, a_ in e2e], fontsize=8.5)
    b.set_ylabel("paired Δ ms / iteration vs base")
    b.grid(axis="x", visible=False)
    b.set_title("End to end: −0.51 ms/iter tuned, −0.62 default")
    b.annotate(
        "round 3, both arms: base = 99.04, an\nunexplained low outlier (kept, §5.5.7 flag)",
        xy=(0.08, 0.77),
        xytext=(1.35, 0.62),
        fontsize=8,
        color=INK2,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=1),
    )
    caption(
        fig,
        "§5.5.7, 12B, 1x1, bf16, HiFi2, SFPI 7.67.0, 35,200 cols (demo maximum). Op: test_lm_head_gather. "
        "Step: 3.172 → 3.001 (mcast) / 3.037 ms (ring). E2E: test_demo_spec_decode, dots = rounds, bar = median "
        "paired Δ. mcast split is bit-exact (acceptance identical); the ring is not, and changes untuned "
        "acceptance 1.59 → 1.62.",
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    save(fig, out, "fig6c_lm_head_split")


# --------------------------------------------------------------------------- Fig 7
def fig7_budget(out):
    # Panel A: §7.0 kernel census, DRAM arm, eager, per-core max, us/step (no argmax op in it)
    head, pinnable, matmul_total, total = 1409.3, 53.1, 2486.5, 2957.6
    segs = [
        ("dense lm_head", head, ORANGE),
        ("pinnable down_proj (1 of 4)", pinnable, AQUA),
        ("other 21 matmuls", matmul_total - head - pinnable, BLUE),
        ("non-matmul ops", total - matmul_total, INK2),
    ]
    # Panel B: e2e ms/iter per lever. argmax: §4.4 rounds; lm_head split: §5.5.7 tuned paired rounds;
    # down_proj pin+ring: §6.3 −113.80 us/trace, PROJECTED onto 101.29 ms/iter (no e2e run exists).
    levers = [
        ("argmax pad removal\n(measured, §4.4)", [-4.73, -4.52, -3.14], AQUA, False),
        ("lm_head L1 slice\n(measured, §5.5.7)", [-0.44, -0.54, +0.77, -0.48, -0.65, -0.63], MCAST, False),
        ("down_proj pin + ring\n(projected, §6.3)", [-0.1138], RING, True),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
    left = 0
    for lbl, v, c in segs:
        a.barh(0, v, left=left, height=0.45, color=c, **BAR)
        left += v
    a.legend(
        handles=[Patch(color=c, label=f"{lbl}: {v:.0f} μs ({100 * v / total:.1f}%)") for lbl, v, c in segs],
        loc="upper left",
        bbox_to_anchor=(0, 1.0),
        ncol=1,
        fontsize=8.5,
    )
    a.set_ylim(-0.45, 1.25)
    a.set_yticks([])
    a.set_xlim(0, total)
    a.set_xlabel("drafter step kernel time, μs (per-core max, eager)")
    a.grid(axis="y", visible=False)
    a.set_title(
        f"The pinnable down_proj is {100 * pinnable / total:.1f}% of the step; the head {100 * head / total:.1f}%"
    )
    a.text(
        0,
        -0.4,
        "drafter = 9.4% of a 101.29 ms iteration post-fix (3 × 3.157 ms; cross-instrument, §4.4)",
        fontsize=8,
        color=INK2,
    )

    for i, (lbl, ds, c, projected) in enumerate(levers):
        m = statistics.median(ds)
        b.barh(i, m, 0.5, color=c, hatch="//" if projected else None, **BAR)
        if len(ds) > 1:
            b.scatter(ds, [i] * len(ds), s=26, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        b.text(
            m / 2 if m < -1 else m - 0.1,
            i - 0.3,
            sg(m),
            ha="center" if m < -1 else "right",
            va="bottom",
            color=INK,
            fontsize=9,
        )
    b.axvline(0, color=AXIS, lw=1)
    b.set_yticks(range(3), [lv[0] for lv in levers], fontsize=8.5)
    b.invert_yaxis()
    b.set_xlim(-5.6, 1.2)
    b.set_xlabel("Δ ms / iteration (median; dots = rounds)")
    b.grid(axis="y", visible=False)
    b.set_title("End-to-end value of each lever")
    caption(
        fig,
        "Left: §7.0 test_profile_eager_step, DRAM arm, pre-argmax-fix, argmax not in its census. "
        "Right: argmax = §4.4 pad vs fast A/B; lm_head = §5.5.7 tuned paired Δ (r3 outlier kept); "
        "down_proj = §6.3 −113.80 μs/trace ÷ 101.29 ms/iter, hatched = projection, below the ~0.5% e2e floor.",
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.62)
    save(fig, out, "fig7_budget_and_levers")


# ------------------------------------------------------------- Fig 2-5 (campaign)
def load_cells():
    rows = [r for r in csv.DictReader(CELLS.open()) if r["warmup_cell"] != "True"]
    by = {}
    for r in rows:
        by.setdefault(r["round"], {})[r["arm"]] = float(r["trace_us"])
    rounds = [v for v in by.values() if len(v) == 6]
    assert len(rounds) == len(by), "a round is missing an arm; contrasts must be within-round"
    return rounds, rows


def contrast(rounds, a, b):
    return med_range([r[a] - r[b] for r in rounds])


def fig2_to_5(out):
    rounds, rows = load_cells()
    sfpi = {r["sfpi"] for r in rows}
    assert len(sfpi) == 1 and len({r["commit"] for r in rows}) == 1 and len({r["harness_sha256"] for r in rows}) == 1
    tag = (
        f"12B-assistant, 1x1, bf16, HiFi2, K=3, n=1 (L0.down_proj), every ring arm rings 3 matmuls/trace; "
        f"test_gather_matched_trace, 400 replays, {len(rounds)} Latin-square rounds, fresh process/cell, "
        f"SFPI {sfpi.pop()}. Whiskers = observed min-max of within-round contrasts; dots = rounds."
    )
    places = [
        ("DRAM\ninterleaved", "dram", "dram_ring"),
        ("L1 interleaved\n(remote SRAM)", "il", "il_ring"),
        ("L1 width-sharded\n(local SRAM)", "mcast", "ring"),
    ]

    # Fig 2: every arm relative to dram
    fig, a = plt.subplots(figsize=(8.5, 4.6))
    w = 0.36
    for i, (lbl, mc, rg) in enumerate(places):
        for dx, arm, c in ((-w / 2 - 0.01, mc, MCAST), (w / 2 + 0.01, rg, RING)):
            ds = [r[arm] - r["dram"] for r in rounds]
            m, lo, hi = med_range(ds)
            a.bar(i + dx, m, w, color=c, **BAR)
            if arm != "dram":
                a.errorbar(i + dx, m, yerr=[[m - lo], [hi - m]], color=INK, lw=1, capsize=3)
                a.scatter([i + dx] * len(ds), ds, s=10, color=INK, alpha=0.5, zorder=3, linewidth=0)
            a.text(
                i + dx,
                m + (4 if m >= 0 else -4),
                "baseline" if arm == "dram" else sg(m, 1),
                ha="center",
                va="bottom" if m >= 0 else "top",
                color=INK,
                fontsize=9,
            )
    a.axhline(0, color=AXIS, lw=1)
    a.set_ylim(-128, 18)
    a.set_xticks(range(3), [p[0] for p in places])
    a.set_ylabel("μs per K=3 trace vs DRAM + mcast (− = faster)")
    base = statistics.median(r["dram"] for r in rounds)
    a.set_title(f"Weight placement × in0 delivery (baseline DRAM + mcast = {base:.1f} μs/trace)")
    a.grid(axis="x", visible=False)
    a.legend(
        handles=[Patch(color=MCAST, label="1D mcast"), Patch(color=RING, label="gather_in0 ring")], loc="lower left"
    )
    caption(fig, tag)
    fig.subplots_adjust(bottom=0.2)
    save(fig, out, "fig2_placement_x_delivery")

    # Fig 3: ring minus its own mcast, per ringed matmul
    fig, a = plt.subplots(figsize=(7, 4))
    for i, (lbl, mc, rg) in enumerate(places):
        ds = [(r[rg] - r[mc]) / 3 for r in rounds]
        m, lo, hi = med_range(ds)
        a.bar(i, m, 0.5, color=RING if m < 0 else INK2, **BAR)
        a.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, lw=1, capsize=3)
        a.text(i + 0.3, m, sg(m), va="center", color=INK)
    a.axhline(0, color=AXIS, lw=1)
    a.set_xticks(range(3), [p[0] for p in places])
    a.set_ylabel("ring − mcast, μs per ringed matmul")
    a.set_title("The ring's sign is set by where the weight lives")
    a.grid(axis="x", visible=False)
    caption(fig, tag)
    fig.subplots_adjust(bottom=0.24)
    save(fig, out, "fig3_ring_gain_per_matmul")

    # Fig 4: waterfall(s) dram -> ... -> ring
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    paths = [
        (
            "mcast first",
            [
                ("SRAM residency\nil − dram", "il", "dram", AQUA),
                ("locality\nmcast − il", "mcast", "il", BLUE),
                ("gather_in0\nring − mcast", "ring", "mcast", ORANGE),
            ],
        ),
        (
            "ring first",
            [
                ("SRAM residency\nil − dram", "il", "dram", AQUA),
                ("gather_in0\nil_ring − il", "il_ring", "il", ORANGE),
                ("locality\nring − il_ring", "ring", "il_ring", BLUE),
            ],
        ),
    ]
    for a, (title, steps) in zip(axs, paths):
        level = 0
        for j, (lbl, x, y, c) in enumerate(steps):
            m = contrast(rounds, x, y)[0]
            a.bar(j, m, 0.6, bottom=level, color=c, **BAR)
            a.text(j, level + m - 3, sg(m, 1), ha="center", va="top", color=INK, fontsize=9)
            level += m
        tot = contrast(rounds, "ring", "dram")[0]
        a.bar(3, tot, 0.6, color=INK2, **BAR)
        a.text(3, tot - 3, sg(tot, 1), ha="center", va="top", color=INK, fontsize=9)
        a.axhline(0, color=AXIS, lw=1)
        a.set_ylim(-128, 4)
        a.set_xticks(range(4), [s[0] for s in steps] + ["total\nring − dram"], fontsize=8.5)
        a.set_title(f"Decomposition of the L1 + ring win ({title})")
        a.grid(axis="x", visible=False)
    axs[0].set_ylabel("μs per K=3 trace (median contrast)")
    caption(fig, tag + " Steps are medians, so they need not sum exactly to the total's median.")
    fig.subplots_adjust(bottom=0.24)
    save(fig, out, "fig4_waterfall")

    # Fig 5: locality under each delivery
    fig, a = plt.subplots(figsize=(6.5, 4))
    for i, (lbl, x, y, c) in enumerate(
        (("under mcast\nmcast − il", "mcast", "il", MCAST), ("under the ring\nring − il_ring", "ring", "il_ring", RING))
    ):
        ds = [r[x] - r[y] for r in rounds]
        m, lo, hi = med_range(ds)
        a.bar(i, m, 0.5, color=c, **BAR)
        a.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, lw=1, capsize=3)
        a.scatter([i] * len(ds), ds, s=10, color=INK, alpha=0.5, zorder=3, linewidth=0)
        a.text(i + 0.3, m, sg(m), va="center", color=INK)
    a.axhline(0, color=AXIS, lw=1)
    a.set_xticks([0, 1], ["under mcast\nmcast − il", "under the ring\nring − il_ring"])
    a.set_ylabel("locality: sharded − interleaved, μs/trace")
    a.set_title("Locality matters far more once the ring is on")
    a.grid(axis="x", visible=False)
    caption(fig, tag)
    fig.subplots_adjust(bottom=0.3)
    save(fig, out, "fig5_locality_x_delivery")

    # medians for the record (§6.9), printed so the script and the text cannot drift
    for x, y in (
        ("il", "dram"),
        ("mcast", "dram"),
        ("mcast", "il"),
        ("ring", "mcast"),
        ("ring", "dram"),
        ("il_ring", "il"),
        ("ring", "il_ring"),
        ("dram_ring", "dram"),
        ("il_ring", "dram"),
    ):
        m, lo, hi = contrast(rounds, x, y)
        print(f"{x:>9} − {y:<9} {m:+8.2f} [{lo:+.2f}, {hi:+.2f}]")


FIGS = {
    "fig1": fig1_argmax,
    "fig6": fig6_ring_k_per_core,
    "fig6b": fig6b_mcast_steps,
    "fig6c": fig6c_lm_head_split,
    "fig7": fig7_budget,
    "campaign": fig2_to_5,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--only", default=",".join(k for k in FIGS if k != "campaign" or CELLS.exists()))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for k in args.only.split(","):
        FIGS[k](args.out)


if __name__ == "__main__":
    main()

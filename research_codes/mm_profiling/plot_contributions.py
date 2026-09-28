#!/usr/bin/env python3
"""Figures for the 12B drafter contributions: argmax pad removal, L1 pinning, gather_in0.

Every number is copied verbatim from
lab-meeting-notes/documents/gemma4-specdec/gemma4-12b/MEASUREMENT_RECORD.md (section cited
beside each literal) or, for the placement x delivery figures, read from
placement6_cells.csv (run_placement_campaign.py). Medians and observed min-max only, no CIs,
per the record's convention. The one exclusion (tuned lm_head round 3, fig6c/fig7) is stated in
the figure's description; the record keeps every cell. Figures are for an external audience: no
caption inside the image and no record-section references in any visible text (save() asserts it);
each figure's description is written to <name>.txt beside it.

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
        "font.size": 12,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.labelsize": 12,
        "axes.titlecolor": INK,
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
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
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
        "legend.fontsize": 11,
        "hatch.color": SURFACE,
        "hatch.linewidth": 1.5,
    }
)
BAR = dict(edgecolor=SURFACE, linewidth=2)  # the 2px surface gap between touching fills
VAL = dict(color=INK, fontsize=12)  # value labels

SETUP = (
    "Setup: Gemma-4 12B (target) with its 12B-assistant drafter (4 layers, 423M parameters), "
    "speculative decoding with 3 draft tokens per iteration, bf16, one Tenstorrent Blackhole P150 chip."
)


def save(fig, out, name, desc):
    """PNG + PDF, and the description as <name>.txt next to them (no caption inside the image)."""
    texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
    assert not any("§" in t for t in texts + [desc]), f"{name}: internal section reference in a public figure"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    (out / f"{name}.txt").write_text(desc.strip() + "\n\n" + SETUP + "\n")
    plt.close(fig)
    print(f"wrote {out / name}.png/.pdf/.txt")


def sg(v, nd=2):
    """Signed number with a typographic minus."""
    return f"{v:+.{nd}f}".replace("-", "−")


def med_range(xs):
    return statistics.median(xs), min(xs), max(xs)


# --------------------------------------------------------------------------- Fig 1
def fig1_argmax(out):
    # record §4.4: drafter step at ctx 512, and the end-to-end A/B (3 counterbalanced rounds)
    parts = [("4 decoder layers", 1.644, 1.644, BLUE), ("lm_head matmul", 1.411, 1.411, ORANGE)]
    parts.append(("argmax (token pick)", 1.586, 0.102, AQUA))
    fast, pad = [101.21, 101.29, 101.32], [105.94, 105.81, 104.46]
    totals = {1: 4.642, 2: 3.157}  # measured totals; the rounded parts sum 1 us off

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[2.1, 1]))
    rows = ["After\n(unpadded)", "Before\n(padded to 32 rows)"]
    for i, idx in enumerate((2, 1)):
        left = 0
        for label, pre, post, c in parts:
            v = post if idx == 2 else pre
            a.barh(i, v, left=left, height=0.55, color=c, **BAR)
            if v > 0.3:
                a.text(left + v / 2, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=12)
            left += v
        a.text(left + 0.06, i, f"{totals[idx]:.2f} ms", va="center", fontweight="semibold", **VAL)
    a.set_yticks([0, 1], rows)
    a.set_xlim(0, 5.5)
    a.set_xlabel("time per draft token (ms)")
    a.grid(axis="y", visible=False)
    a.set_title("Drafter step −32%: the argmax was 34% of it")
    a.legend(
        handles=[Patch(color=c, label=lbl) for lbl, _, _, c in parts],
        loc="upper center",
        bbox_to_anchor=(0.45, -0.2),
        ncol=3,
    )

    for x, ys, c in ((0, pad, INK2), (1, fast, BLUE)):
        m = statistics.median(ys)
        b.scatter([x] * len(ys), ys, s=60, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        b.hlines(m, x - 0.25, x + 0.25, color=INK, linewidth=2.5)
        b.text(x + 0.32, m, f"{m:.1f}", va="center", **VAL)
    b.set_xticks([0, 1], ["Before", "After"])
    b.set_xlim(-0.5, 1.9)
    b.set_ylim(100.5, 106.5)
    b.set_ylabel("ms per decoding iteration")
    b.grid(axis="x", visible=False)
    b.set_title("End to end −4.3%")
    fig.subplots_adjust(bottom=0.26, wspace=0.28)
    save(
        fig,
        out,
        "fig1_argmax_fix",
        """
Removing a padded argmax from the speculative drafter.

Left: time of one drafter step (one draft token), split into the 4 decoder layers, the lm_head
matmul (1024 x 262,144) and the argmax that picks the token. The old argmax padded the single
logits row to 32 rows before reducing it, so every step scanned 32 x 262,144 values (16 MiB) to
find one index. Reducing the unpadded row returns the same index (verified exact) and cuts the
argmax from 1.586 ms to 0.102 ms (15.5x). The layers and the lm_head are unchanged; the step
goes from 4.642 to 3.157 ms (-32.0%).

Right: end-to-end decoding time per speculative iteration, before vs after, 3 alternating
rounds (dots) and their median (bar): 105.81 -> 101.29 ms/iteration (-4.27%, range -3.01 to
-4.46%), i.e. 25.14 -> 26.26 tokens/s per user. Token acceptance is identical (1.66 of 3 drafts),
so the output is unchanged. One "before" round ran at 104.46 ms, about 1.4 ms faster than the
other two; no cause was found (clock and temperature were not logged in this run) and it is
kept.
""",
    )


# --------------------------------------------------------------------------- Fig 6
def fig6_ring_k_per_core(out):
    # isolated-op us, weight pinned L1 width-sharded (so both arms are shard_*). down_proj: record
    # §6.8 (L1-interleaved activation, the tuner's in0_block_w=8); lm_head: §5.5.7, second run.
    rows = [
        ("down_proj\n32 cores, 8 K-tiles each", 40.22, 12.44, 15.07),
        ("lm_head slice\n32 cores, 1 K-tile each", 21.43, 22.00, 26.57),
        ("lm_head slice\n110 cores, 78 of them empty", 41.09, 70.73, 77.51),
    ]
    fig = plt.figure(figsize=(13, 7.4))
    # strip widths proportional to core count, so no empty axis reads as empty cores
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1], width_ratios=[34, 112], hspace=0.62, wspace=0.08)
    a = fig.add_subplot(gs[0, :])
    w = 0.34
    for i, (lbl, mc, ring, ring_conv) in enumerate(rows):
        a.bar(i - w / 2 - 0.01, mc, w, color=MCAST, **BAR)
        a.bar(i + w / 2 + 0.01, ring, w, color=RING, **BAR)
        a.bar(i + w / 2 + 0.01, ring_conv - ring, w, bottom=ring, color=ORANGE_L, **BAR)
        a.text(i - w / 2, mc + 1.5, f"{mc:.1f}", ha="center", **VAL)
        a.text(i + w / 2, ring_conv + 1.5, f"{ring:.1f} + {ring_conv - ring:.1f}", ha="center", **VAL)
        r = ring / mc
        verdict = f"ring {1 / r:.1f}× faster" if r < 0.9 else ("tie" if r < 1.1 else f"ring {r:.1f}× slower")
        a.text(i, max(mc, ring_conv) + 10, verdict, ha="center", fontweight="semibold", color=INK, fontsize=13)
    a.set_xticks(range(3), [r[0] for r in rows])
    a.set_ylim(0, 100)
    a.set_ylabel("μs per matmul")
    a.grid(axis="x", visible=False)
    a.set_title("The gather_in0 ring pays only when every core gets a thick slice of K")
    a.legend(
        handles=[
            Patch(color=MCAST, label="shard_mcast (activation multicast)"),
            Patch(color=RING, label="shard_ring (gather_in0 ring)"),
            Patch(color=ORANGE_L, label="ring's extra layout conversions"),
        ],
        loc="upper left",
    )

    for col, (title, tiles) in enumerate(
        (("down_proj ring (32 cores)", [8] * 32), ("lm_head ring (110 cores)", [1] * 32 + [0] * 78))
    ):
        b = fig.add_subplot(gs[1, col])
        b.bar(
            range(len(tiles)),
            [max(t, 0.12) for t in tiles],
            0.8,
            color=[RING if t else GRID for t in tiles],
            linewidth=0,
        )
        b.set_title(title, fontsize=12, loc="left" if col == 0 else "center")
        b.set_ylim(0, 9)
        b.set_yticks([0, 1, 8])
        b.set_xlim(-1, len(tiles))
        b.set_xlabel("core in the ring")
        b.grid(axis="x", visible=False)
        if col == 0:
            b.set_ylabel("K tiles")
        else:
            b.text(
                71, 3, "78 cores hold nothing\nbut still take part in the ring", ha="center", fontsize=11, color=INK2
            )
            b.set_yticklabels([])
    save(
        fig,
        out,
        "fig6_ring_k_per_core",
        """
When the gather_in0 ring helps, and when it hurts.

Top: one matmul in isolation, weight pinned in L1 width-sharded, two ways of delivering the
activation. shard_mcast multicasts the activation to every core; shard_ring splits the
activation along K across the cores and passes the pieces around a ring (gather_in0). The light
segment is the extra layout conversion the ring needs (resharding the input, converting the
output back); the number on the ring bar is the ring alone.

- down_proj (K = 8192 = 256 tiles over 32 cores, 8 tiles each): ring 12.44 us vs 40.22 us, 3.2x
  faster (15.07 us with conversions).
- lm_head slice on 32 cores (K = 1024 = 32 tiles, 1 each): 22.00 vs 21.43 us, a tie; the
  conversions make it a loss.
- lm_head slice on 110 cores: only 32 tiles for 110 cores, so 78 cores hold no K. They still
  take part in every ring step, and the ring is 1.7x slower (70.73 vs 41.09 us).

Bottom: how many K tiles each core in the ring holds, for the down_proj ring (left) and the
110-core lm_head ring (right). The ring is also not bit-exact against multicast on lm_head
(max |diff| 0.078); multicast is.
""",
    )


# --------------------------------------------------------------------------- Fig 6b
def fig6b_mcast_steps(out):
    # record §6.8: 12B down_proj shape, 32 cores, per_core_N=1, weight L1 width-sharded, isolated-op
    blk = [1, 2, 4, 8, 16, 32]
    dram = [221.87, 119.45, 66.95, 41.19, 28.80, 23.74]
    l1 = [218.57, 116.29, 65.21, 40.22, 28.18, 23.63]
    steps = [256 // b for b in blk]
    fig, a = plt.subplots(figsize=(10, 5.4))
    a.plot(
        steps, dram, color=MCAST, lw=2.5, marker="o", ms=8, mec=SURFACE, mew=2, label="shard_mcast, activation in DRAM"
    )
    a.plot(steps, l1, color=VIOLET, lw=2.5, marker="o", ms=8, mec=SURFACE, mew=2, label="shard_mcast, activation in L1")
    a.scatter(
        [32],
        [12.44],
        s=130,
        color=RING,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=4,
        label="shard_ring (gather_in0), 32 steps",
    )
    a.annotate(
        "at 32 steps: 40.2 μs multicast\nvs 12.4 μs ring",
        xy=(34, 12.44),
        xytext=(52, 30),
        fontsize=12,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=1),
    )
    a.set_xscale("log", base=2)
    a.set_xticks(steps, [str(s) for s in steps])
    a.set_xlabel("K-steps per matmul (fewer steps = wider blocks)")
    a.set_ylabel("μs per matmul")
    a.set_ylim(0, 235)
    a.set_title("Multicast pays a fixed cost per K-step; the ring does the same 32 steps 3× cheaper")
    a.legend(loc="upper left")
    save(
        fig,
        out,
        "fig6b_mcast_steps_vs_ring",
        """
Why the ring wins on down_proj: the multicast matmul's cost is set by its number of K-steps.

The 12B drafter's down_proj (K = 8192, N = 1024) on 32 cores with the weight pinned in L1,
isolated. The multicast kernel walks K in blocks; sweeping the block width from 1 to 32 tiles
takes it from 256 steps to 8. Its time falls almost linearly with the step count, about 0.8 us
per step whatever a step carries (221.9 us at 256 steps, 41.2 at 32, 23.7 at 8), so the cost is
per-step overhead, not bandwidth. Whether the activation starts in DRAM or L1 moves it by at
most 3.3 us.

The gather_in0 ring (orange point) runs the same 32 steps as the default multicast configuration
in 12.44 us instead of 40.22 us (15.07 us including its layout conversions). One run per point.
""",
    )


# --------------------------------------------------------------------------- Fig 6c
def fig6c_lm_head_split(out):
    # record §5.5.7 (HiFi2, current code). Op level (second run) and e2e paired deltas.
    op = [
        ("whole head\nin DRAM", 1412.84, INK2),
        ("13% in L1\nsplit, shard_mcast", 1222.48, MCAST),
        ("13% in L1\nsplit, shard_ring", 1264.39, RING),
    ]
    e2e = {
        # tuned round 3 left out of the picture for now: its base (99.04) is an unexplained low outlier.
        # The record keeps it (its quoted medians, -0.51 / -0.56, include it).
        ("tuned", "split, shard_mcast"): [-0.44, -0.54, -0.48, -0.65, -0.63],
        ("tuned", "split, shard_ring"): [-0.48, -0.64, -0.31, -0.80, -0.66],
        ("default", "split, shard_mcast"): [-0.83, -0.52, -0.50],
        ("default", "split, shard_ring"): [-0.71, -0.30, -0.55],
    }
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw=dict(width_ratios=[1, 1.3]))
    for i, (lbl, v, c) in enumerate(op):
        a.bar(i, v, 0.55, color=c, **BAR)
        a.text(i, v + 12, f"{v:.0f}" + ("" if i == 0 else f"\n({sg(v - op[0][1], 0)})"), ha="center", **VAL)
    a.set_xticks(range(3), [o[0] for o in op])
    a.set_ylim(1000, 1500)
    a.set_ylabel("μs per lm_head matmul (axis from 1000)")
    a.grid(axis="x", visible=False)
    a.set_title("lm_head: pin 13% of it in L1")

    for i, ((drafter, arm), ds) in enumerate(e2e.items()):
        c = MCAST if "mcast" in arm else RING
        b.scatter([i] * len(ds), ds, s=60, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        m = statistics.median(ds)
        b.hlines(m, i - 0.25, i + 0.25, color=INK, lw=2.5)
        b.text(i + 0.3, m, sg(m), va="center", **VAL)
    b.axhline(0, color=AXIS, lw=1)
    b.set_xticks(range(4), [f"{a_.replace('split, ', '')}\n({d})" for d, a_ in e2e])
    b.set_xlabel("drafter matmul configs: tuned or ttnn default")
    b.set_xlim(-0.5, 3.8)
    b.set_ylabel("Δ ms per iteration (lower = faster)")
    b.grid(axis="x", visible=False)
    b.set_title("End to end: about −0.5 ms per iteration")
    fig.subplots_adjust(wspace=0.28)
    save(
        fig,
        out,
        "fig6c_lm_head_split",
        """
Pinning part of the drafter's lm_head in on-chip SRAM.

The drafter's lm_head (1024 x 262,144, 512 MiB in bf16) is too large for L1, and ttnn cannot
place only part of one tensor in L1. So the weight is split along its columns: 35,200 columns
(13.4%, the most the demo's free L1 allows) are pinned in L1 width-sharded, the rest stays in
DRAM, and the two partial logits are joined before the argmax. The result is bit-exact against
the unsplit head.

Left: the lm_head matmul drops from 1412.8 to 1222.5 us (-190 us) with the pinned slice run as
shard_mcast. Running the slice as a gather_in0 ring instead (shard_ring) gives back 42 us of
that, because the slice's K is too thin for 110 cores (see the ring K-per-core figure).

Right: end-to-end change in ms per decoding iteration against the unsplit head, paired within
rounds (dots) with their median (bar), for the drafter with tuned matmul configs and with
ttnn's default configs. shard_mcast: -0.54 ms tuned (5 rounds), -0.62 ms default (3 rounds).
shard_ring is not bit-exact and changed token acceptance in the default configuration. One
tuned round is left out of this figure: its baseline ran unusually fast (99.04 ms vs about
100.4) for no identified reason; with it the tuned medians are -0.51 (shard_mcast) and -0.56
(shard_ring).
""",
    )


# --------------------------------------------------------------------------- Fig 7
def fig7_budget(out):
    # Panel A: record §7.0 kernel census, DRAM arm, eager, per-core max, us/step (no argmax op in it)
    head, pinnable, matmul_total, total = 1409.3, 53.1, 2486.5, 2957.6
    segs = [
        ("lm_head matmul", head, ORANGE),
        ("the down_proj that fits in L1", pinnable, AQUA),
        ("the other 21 matmuls", matmul_total - head - pinnable, BLUE),
        ("everything else", total - matmul_total, INK2),
    ]
    # Panel B: e2e ms/iter per lever. argmax §4.4 rounds; lm_head split §5.5.7 tuned paired rounds
    # (round 3 left out); down_proj pin+ring §6.3 -113.80 us/trace, PROJECTED onto 101.29 ms/iter.
    levers = [
        ("argmax fix\n(measured)", [-4.73, -4.52, -3.14], AQUA, False),
        ("lm_head 13% in L1\n(measured)", [-0.44, -0.54, -0.48, -0.65, -0.63], MCAST, False),
        ("down_proj in L1 + ring\n(estimated)", [-0.1138], RING, True),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    left = 0
    for lbl, v, c in segs:
        a.barh(0, v, left=left, height=0.45, color=c, **BAR)
        left += v
    a.legend(
        handles=[Patch(color=c, label=f"{lbl}: {100 * v / total:.0f}%") for lbl, v, c in segs],
        loc="upper left",
        bbox_to_anchor=(0, 1.02),
        ncol=2,
    )
    a.set_ylim(-0.35, 0.95)
    a.set_yticks([])
    a.set_xlim(0, total)
    a.set_xlabel("drafter step, kernel time (μs)")
    a.grid(axis="y", visible=False)
    a.set_title("Where one drafter step spends its time")

    for i, (lbl, ds, c, projected) in enumerate(levers):
        m = statistics.median(ds)
        b.barh(i, m, 0.5, color=c, hatch="//" if projected else None, **BAR)
        if len(ds) > 1:
            b.scatter(ds, [i] * len(ds), s=36, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        b.text(0.12, i, sg(m) + " ms", ha="left", va="center", **VAL)
    b.axvline(0, color=AXIS, lw=1)
    b.set_yticks(range(3), [lv[0] for lv in levers])
    b.invert_yaxis()
    b.set_xlim(-5.2, 1.6)
    b.set_xlabel("Δ ms per decoding iteration")
    b.grid(axis="y", visible=False)
    b.set_title("What each optimisation saves end to end")
    fig.subplots_adjust(wspace=0.5)
    save(
        fig,
        out,
        "fig7_budget_and_levers",
        """
The size of each lever, and why.

Left: kernel time of one drafter step on the device, before the argmax fix and without the
argmax itself. The lm_head matmul alone is 48% of it; the one down_proj layer that fits in the
free L1 (weights of the other three do not fit alongside the runtime buffers) is under 2%. That
bounds what pinning down_proj can ever buy.

Right: the end-to-end saving of each optimisation, in ms per speculative decoding iteration
(about 101 ms). Bars are medians, dots are individual rounds.
- argmax fix: -4.52 ms (3 rounds, measured end to end).
- lm_head 13% in L1: -0.54 ms (5 rounds, measured end to end; one outlier round left out).
- down_proj in L1 + gather_in0 ring: about -0.11 ms, estimated from its measured saving on the
  drafter (-114 us per 3-step draft) because an effect this small is below the run-to-run noise
  of an end-to-end measurement (hatched).
The drafter is about 9% of an iteration after the argmax fix; the target model's verification
pass is the rest.
""",
    )


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
    method = (
        f"Method: the drafter's 3-step draft is captured as one device trace and replayed 400 times per "
        f"measurement; one arm per fresh process; {len(rounds)} rounds with every arm in every position "
        f"equally often; the device clock held at 1350 MHz throughout. Every ring arm uses the ring on the "
        f"same 3 matmuls (the one down_proj layer, once per draft step). Contrasts are taken within a round; "
        f"bars are medians and whiskers the full min-max over rounds (the spread is 1.4-1.6 us, about 0.015%)."
    )
    places = [
        ("DRAM\ndram_mcast | dram_ring", "dram_mcast", "dram_ring"),
        ("L1, spread over all banks\nil_mcast | il_ring", "il_mcast", "il_ring"),
        ("L1, on the compute cores\nshard_mcast | shard_ring", "shard_mcast", "shard_ring"),
    ]
    glossary = (
        "Arms are named <weight placement>_<activation delivery>: dram = weight in DRAM; il = weight in L1 "
        "interleaved, i.e. spread page by page over all 110 L1 banks, so cores read it from other cores' SRAM; "
        "shard = weight in L1 width-sharded on the cores that compute with it, so no weight read is needed. "
        "mcast = the activation is multicast to all cores; ring = gather_in0, the activation is split along K "
        "and passed around a ring of cores."
    )

    # Fig 2: every arm relative to dram_mcast
    fig, a = plt.subplots(figsize=(11, 5.8))
    w = 0.36
    for i, (lbl, mc, rg) in enumerate(places):
        for dx, arm, c in ((-w / 2 - 0.01, mc, MCAST), (w / 2 + 0.01, rg, RING)):
            ds = [r[arm] - r["dram_mcast"] for r in rounds]
            m, lo, hi = med_range(ds)
            a.bar(i + dx, m, w, color=c, **BAR)
            if arm != "dram_mcast":
                a.errorbar(i + dx, m, yerr=[[m - lo], [hi - m]], color=INK, lw=1.2, capsize=4)
            a.text(
                i + dx,
                (hi + 2 if m >= 0 else lo - 3) if arm != "dram_mcast" else 3,
                "baseline" if arm == "dram_mcast" else sg(m, 0),
                ha="center",
                va="bottom" if m >= 0 else "top",
                **VAL,
            )
    a.axhline(0, color=AXIS, lw=1)
    a.set_ylim(-130, 20)
    a.set_xticks(range(3), [p[0] for p in places])
    a.set_xlabel("where the down_proj weight lives", labelpad=10)
    a.set_ylabel("Δ μs per 3-step draft (lower = faster)")
    a.set_title("Weight placement × activation delivery: L1 + ring is fastest")
    a.grid(axis="x", visible=False)
    a.legend(
        handles=[
            Patch(color=MCAST, label="activation multicast (mcast)"),
            Patch(color=RING, label="gather_in0 ring (ring)"),
        ],
        loc="lower left",
    )
    base = statistics.median(r["dram_mcast"] for r in rounds)
    save(
        fig,
        out,
        "fig2_placement_x_delivery",
        f"""
One down_proj layer of the drafter, six ways: three places for its weight x two ways of feeding
the activation to the matmul. Each bar is the change in the time of a 3-step draft against
dram_mcast (weight in DRAM, activation multicast; {base:.0f} us).

- Moving the weight into L1 helps with either delivery: il_mcast -37 us, shard_mcast -40 us.
- The ring loses when the weight is in DRAM (dram_ring +8 us) and wins when it is in L1
  (il_ring -78 us, shard_ring -113 us).
- shard_ring, the weight on the compute cores plus the ring, is the fastest: -113 us, -1.19% of
  the draft.

{glossary}

{method}
""",
    )

    # Fig 3: ring minus its own mcast, per ringed matmul
    fig, a = plt.subplots(figsize=(9.5, 5.2))
    for i, (lbl, mc, rg) in enumerate(places):
        ds = [(r[rg] - r[mc]) / 3 for r in rounds]
        m, lo, hi = med_range(ds)
        a.bar(i, m, 0.5, color=RING if m < 0 else INK2, **BAR)
        a.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, lw=1.2, capsize=4)
        a.text(
            i, hi + 0.6 if m >= 0 else lo - 0.6, sg(m, 1) + " μs", ha="center", va="bottom" if m >= 0 else "top", **VAL
        )
    a.axhline(0, color=AXIS, lw=1)
    a.set_ylim(-28, 7)
    a.set_xticks(range(3), [p[0] for p in places])
    a.set_ylabel("ring − mcast per matmul (μs)\nabove 0 = ring slower")
    a.set_title("The ring's benefit flips sign with the weight's placement")
    a.grid(axis="x", visible=False)
    save(
        fig,
        out,
        "fig3_ring_gain_per_matmul",
        f"""
What switching the activation delivery from multicast to the gather_in0 ring is worth, per
matmul, for each weight placement: dram_ring - dram_mcast, il_ring - il_mcast and
shard_ring - shard_mcast, each divided by the 3 ringed matmuls in a draft.

With the weight in DRAM the ring costs +2.7 us per matmul. With the weight in L1 it saves 13.4 us
(spread over all banks) or 24.2 us (on the compute cores). The ring blocks K more finely than
multicast does, so it issues more, smaller weight reads; that is cheap from SRAM and expensive
from DRAM.

{glossary}

{method}
""",
    )

    # Fig 4: waterfall(s) dram_mcast -> ... -> shard_ring
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    paths = [
        (
            "A: move to L1, then add the ring",
            [
                ("weight\nto L1\n(il_mcast)", "il_mcast", "dram_mcast", AQUA),
                ("onto the\ncompute cores\n(shard_mcast)", "shard_mcast", "il_mcast", BLUE),
                ("add\nthe ring\n(shard_ring)", "shard_ring", "shard_mcast", ORANGE),
            ],
        ),
        (
            "B: add the ring before localising",
            [
                ("weight\nto L1\n(il_mcast)", "il_mcast", "dram_mcast", AQUA),
                ("add\nthe ring\n(il_ring)", "il_ring", "il_mcast", ORANGE),
                ("onto the\ncompute cores\n(shard_ring)", "shard_ring", "il_ring", BLUE),
            ],
        ),
    ]
    for a, (title, steps) in zip(axs, paths):
        level = 0
        for j, (lbl, x, y, c) in enumerate(steps):
            m = contrast(rounds, x, y)[0]
            a.bar(j, m, 0.62, bottom=level, color=c, **BAR)
            a.text(j, level + m - 3, sg(m, 0), ha="center", va="top", **VAL)
            if j < 2:
                a.hlines(level + m, j + 0.31, j + 0.69, color=MUTED, lw=1, linestyles="dotted")
            level += m
        tot = contrast(rounds, "shard_ring", "dram_mcast")[0]
        a.bar(3, tot, 0.62, color=INK2, **BAR)
        a.text(3, tot - 3, sg(tot, 0), ha="center", va="top", fontweight="semibold", **{**VAL})
        a.axhline(0, color=AXIS, lw=1)
        a.set_ylim(-130, 5)
        a.set_xticks(range(4), [s[0] for s in steps] + ["total\n\n(shard_ring)"])
        a.set_title(title, fontsize=13)
        a.grid(axis="x", visible=False)
    axs[0].set_ylabel("Δ μs per 3-step draft vs dram_mcast")
    fig.suptitle("Where the −113 μs comes from", x=0.06, ha="left", fontsize=15, fontweight="semibold", color=INK)
    fig.subplots_adjust(top=0.82, wspace=0.08)
    save(
        fig,
        out,
        "fig4_waterfall",
        f"""
The full saving of shard_ring over dram_mcast (-113 us per 3-step draft), taken apart in two
orders.

A: moving the weight from DRAM into L1 gives -37 us; putting it on the compute cores themselves
gives only -3 us more under multicast; adding the ring then gives -72 us.
B: from the same L1 start, the ring alone gives -40 us even though the weight is still read from
other cores' SRAM, and localising the weight then gives -35 us.

The two orders disagree on what "locality" is worth (-3 vs -35 us) because locality and the ring
interact: the ring's many small weight reads are what make local SRAM pay. Steps are medians of
within-round contrasts, so they need not add up exactly to the total.

{glossary}

{method}
""",
    )

    # Fig 5: locality under each delivery
    fig, a = plt.subplots(figsize=(8, 5.2))
    for i, (lbl, x, y, c) in enumerate(
        (
            ("with multicast\nshard_mcast − il_mcast", "shard_mcast", "il_mcast", MCAST),
            ("with the ring\nshard_ring − il_ring", "shard_ring", "il_ring", RING),
        )
    ):
        ds = [r[x] - r[y] for r in rounds]
        m, lo, hi = med_range(ds)
        a.bar(i, m, 0.5, color=c, **BAR)
        a.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, lw=1.2, capsize=4)
        a.text(i, lo - 0.8, sg(m, 1) + " μs", ha="center", va="top", **VAL)
    a.axhline(0, color=AXIS, lw=1)
    a.set_ylim(-41, 2)
    a.set_xticks([0, 1], ["with multicast\nshard_mcast − il_mcast", "with the ring\nshard_ring − il_ring"])
    a.set_ylabel("Δ μs per 3-step draft")
    a.set_title("Keeping the weight on the compute cores\nmatters 12× more with the ring")
    a.grid(axis="x", visible=False)
    save(
        fig,
        out,
        "fig5_locality_x_delivery",
        f"""
The value of locality: the weight in L1 on the cores that compute with it (shard) against the
weight in L1 spread over all banks (il), for each activation delivery.

With multicast it is worth only -2.8 us per 3-step draft: once the weight is in SRAM at all,
where in SRAM barely matters. With the gather_in0 ring it is worth -35.0 us, 12x more, because
the ring's finer K blocking issues many more, smaller weight reads, and each remote read then
costs far more than a local one.

{glossary}

{method}
""",
    )

    # medians for the record, printed so the script and the text cannot drift
    for x, y in (
        ("il_mcast", "dram_mcast"),
        ("shard_mcast", "dram_mcast"),
        ("shard_mcast", "il_mcast"),
        ("shard_ring", "shard_mcast"),
        ("shard_ring", "dram_mcast"),
        ("il_ring", "il_mcast"),
        ("shard_ring", "il_ring"),
        ("dram_ring", "dram_mcast"),
        ("il_ring", "dram_mcast"),
    ):
        m, lo, hi = contrast(rounds, x, y)
        print(f"{x:>11} − {y:<11} {m:+8.2f} [{lo:+.2f}, {hi:+.2f}]")


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

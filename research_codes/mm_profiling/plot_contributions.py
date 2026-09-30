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


DRAFT = (
    "A 3-step draft is the drafter proposing 3 tokens; it runs once per decoding iteration, before the "
    "target model verifies them."
)


def tps_panel(ax, groups, title, ylabel, nd=1):
    """Throughput bars from zero: median per arm, per-round dots when there are rounds, % vs the first arm."""
    base = statistics.median(groups[0][1])
    for x, (lbl, vals, c) in enumerate(groups):
        m = statistics.median(vals)
        ax.bar(x, m, 0.55, color=c, **BAR)
        if len(vals) > 1:
            ax.scatter([x] * len(vals), vals, s=22, color=INK, alpha=0.55, zorder=3, linewidth=0)
        pct = 100 * (m / base - 1)
        txt = f"{m:.{nd}f}" if x == 0 else f"{m:.{nd}f}\n{sg(pct, 2 if abs(pct) < 1 else 1)}%"
        ax.text(x, max(vals) * 1.015, txt, ha="center", va="bottom", **VAL)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylim(0, max(max(g[1]) for g in groups) * 1.22)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.set_title(title)


def e2e_tps(ms_per_iter, n_iters, n_tokens=500):
    """tokens/s/user from ms/iter; exact inverse of the demo's own tok/s/u line (n_tokens / steady time)."""
    return [1000.0 * n_tokens / (n_iters * m) for m in ms_per_iter]


# record §4.4 (drafter step at ctx 512; e2e 3 counterbalanced rounds, 188 iterations, acceptance identical)
ARGMAX_STEP = {"pad": 4.642, "fast": 3.157}
ARGMAX_E2E = {"pad": e2e_tps([105.94, 105.81, 104.46], 188), "fast": e2e_tps([101.21, 101.29, 101.32], 188)}
# record §5.5.7, tuned configs (step: identical over 4 rounds; e2e: 191 iterations, round 3 left out)
LMHEAD_STEP = {"base": 3.172, "shard_mcast": 3.001, "shard_ring": 3.037}
LMHEAD_E2E = {
    "base": e2e_tps([100.38, 100.50, 100.29, 100.47, 100.47], 191),
    "shard_mcast": e2e_tps([99.94, 99.96, 99.81, 99.82, 99.84], 191),
    "shard_ring": e2e_tps([99.90, 99.86, 99.98, 99.67, 99.81], 191),
}


def down_proj_tps():
    """Drafter draft-tokens/s per round (6-arm campaign) and e2e tok/s/u per round (demo A/B)."""
    rounds, _ = load_cells()
    step = {a: [3e6 / r[a] for r in rounds] for a in ("dram_mcast", "shard_mcast", "shard_ring")}
    rows = [r for r in csv.DictReader(E2E_CELLS.open()) if r["warmup"] != "True"]
    e2e = {a: [float(r["tok_s_u"]) for r in rows if r["arm"] == a] for a in ("dram_mcast", "shard_mcast", "shard_ring")}
    return step, e2e


def draft_breakdown(ax, rows, parts, title, xmax):
    """Per 3-step draft, stacked by component. rows: (label, [values ms], total ms)."""
    for i, (lbl, vals, tot) in enumerate(rows):
        left = 0
        for v, (_, c) in zip(vals, parts):
            ax.barh(i, v, left=left, height=0.55, color=c, **BAR)
            if v > 0.9:
                ax.text(left + v / 2, i, f"{v:.2f}", ha="center", va="center", color=INK, fontsize=12)
            left += v
        pct = "" if i == len(rows) - 1 else f"  ({sg(100 * (tot / rows[-1][2] - 1), 1)}%)"
        ax.text(left + 0.1, i, f"{tot:.2f} ms{pct}", va="center", fontweight="semibold", **VAL)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, xmax)
    ax.set_xlabel("ms per 3-step draft")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.legend(
        handles=[Patch(color=c, label=lbl) for lbl, c in parts],
        loc="upper center",
        bbox_to_anchor=(0.45, -0.22),
        ncol=len(parts),
    )


# --------------------------------------------------------------------------- Fig 00
def fig00_throughput_summary(out):
    dstep, de2e = down_proj_tps()
    levers = [
        (
            "argmax fix",
            [1000 / ARGMAX_STEP["pad"]],
            [1000 / ARGMAX_STEP["fast"]],
            ARGMAX_E2E["pad"],
            ARGMAX_E2E["fast"],
            AQUA,
        ),
        (
            "lm_head 13% in L1\n(shard_mcast)",
            [1000 / LMHEAD_STEP["base"]],
            [1000 / LMHEAD_STEP["shard_mcast"]],
            LMHEAD_E2E["base"],
            LMHEAD_E2E["shard_mcast"],
            MCAST,
        ),
        (
            "down_proj in L1 + ring\n(shard_ring)",
            dstep["dram_mcast"],
            dstep["shard_ring"],
            de2e["dram_mcast"],
            de2e["shard_ring"],
            RING,
        ),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6))
    w = 0.36
    for ax, (k0, k1, unit, nd) in zip(
        axs, ((1, 2, "draft tokens/s (drafter alone)", 1), (3, 4, "tokens/s per user (end to end)", 2))
    ):
        for i, lv in enumerate(levers):
            b0, b1 = statistics.median(lv[k0]), statistics.median(lv[k1])
            ax.bar(i - w / 2 - 0.01, b0, w, color=GRID, **BAR)
            ax.bar(i + w / 2 + 0.01, b1, w, color=lv[5], **BAR)
            ax.text(i - w / 2, b0 * 1.015, f"{b0:.{nd}f}", ha="center", va="bottom", fontsize=11, color=INK2)
            pct = 100 * (b1 / b0 - 1)
            ax.text(
                i + w / 2,
                b1 * 1.015,
                f"{b1:.{nd}f}\n{sg(pct, 2 if abs(pct) < 1 else 1)}%",
                ha="center",
                va="bottom",
                **VAL,
            )
        ax.set_xticks(range(3), [lv[0] for lv in levers])
        ax.set_ylim(0, max(statistics.median(lv[k1]) for lv in levers) * 1.25)
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
    axs[0].set_title("Drafter throughput")
    axs[1].set_title("End-to-end throughput")
    axs[0].legend(handles=[Patch(color=GRID, label="before (each optimisation's own baseline)")], loc="upper left")
    fig.subplots_adjust(wspace=0.25)
    save(
        fig,
        out,
        "fig00_throughput_summary",
        f"""
Throughput gained by each of the three drafter optimisations, for the drafter alone and end to
end.

Left: draft tokens per second of the drafter by itself (one token per drafter step). Right:
tokens per second per user of the whole speculative decoder (drafter + target model), batch 1.
Each optimisation is shown against its own baseline, because they were measured one at a time
and cannot be stacked: the pinned lm_head part and the pinned down_proj do not fit in L1
together.

- argmax fix: drafter {1000 / ARGMAX_STEP['pad']:.1f} -> {1000 / ARGMAX_STEP['fast']:.1f} draft tokens/s; end to end
  {statistics.median(ARGMAX_E2E['pad']):.2f} -> {statistics.median(ARGMAX_E2E['fast']):.2f} tokens/s/user.
- lm_head 13% in L1 (shard_mcast): drafter {1000 / LMHEAD_STEP['base']:.1f} -> {1000 / LMHEAD_STEP['shard_mcast']:.1f};
  end to end {statistics.median(LMHEAD_E2E['base']):.2f} -> {statistics.median(LMHEAD_E2E['shard_mcast']):.2f}.
- down_proj in L1 + gather_in0 ring (shard_ring): drafter {statistics.median(dstep['dram_mcast']):.1f} ->
  {statistics.median(dstep['shard_ring']):.1f}; end to end {statistics.median(de2e['dram_mcast']):.2f} -> {statistics.median(de2e['shard_ring']):.2f}.

The drafter gains are larger than the end-to-end ones because the drafter is about 9% of an
iteration after the argmax fix; the target model's verification pass is the rest. Values are
medians over rounds. The argmax baseline is the drafter before that fix; the other two baselines
already include it. End-to-end tokens/s/user is computed per round from ms per iteration as
500 tokens / (iterations x ms per iteration), which reproduces the demo's own tokens/s/user
line.
""",
    )


# --------------------------------------------------------------------------- Fig 01
def fig01_argmax_gain(out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.4))
    ep, ef = statistics.median(ARGMAX_E2E["pad"]), statistics.median(ARGMAX_E2E["fast"])
    dg = 100 * (ARGMAX_STEP["pad"] / ARGMAX_STEP["fast"] - 1)
    tps_panel(
        a,
        [("padded", [1000 / ARGMAX_STEP["pad"]], GRID), ("unpadded", [1000 / ARGMAX_STEP["fast"]], AQUA)],
        f"Drafter: {sg(dg, 0)}% draft tokens/s",
        "draft tokens/s (drafter alone)",
    )
    tps_panel(
        b,
        [("padded", ARGMAX_E2E["pad"], GRID), ("unpadded", ARGMAX_E2E["fast"], AQUA)],
        f"End to end: {sg(100 * (ef / ep - 1), 1)}% tokens/s/user",
        "tokens/s per user",
        nd=2,
    )
    fig.subplots_adjust(wspace=0.3)
    save(
        fig,
        out,
        "fig01_argmax_gain",
        f"""
Throughput gained by removing a padded argmax from the speculative drafter.

Left: draft tokens per second of the drafter alone (one token per drafter step): 215.4 -> 316.8
(+47%), from a drafter step of 4.642 -> 3.157 ms.

Right: tokens per second per user of the whole speculative decoder, 3 alternating rounds (dots)
and their median (bar): {ep:.2f} -> {ef:.2f} ({sg(100 * (ef / ep - 1), 1)}%); in time, 105.81 -> 101.29 ms per iteration
(-4.27%). Token acceptance is identical (1.66 of 3 drafts), so the output is unchanged. In one
round the padded baseline ran about 1.4 ms per iteration faster than in the other two (the
highest "padded" dot); no cause was found (clock and temperature were not logged in this run)
and it is kept.

{DRAFT}
""",
    )


# --------------------------------------------------------------------------- Fig 02
def fig02_argmax_breakdown(out):
    # record §4.4 step breakdown (test_fused_draft_k_steps, ctx 512), x3 = per 3-step draft:
    # the instrument times the 3-step trace and reports it per step, so x3 is its measured value.
    parts = [("4 decoder layers", BLUE), ("lm_head matmul", ORANGE), ("argmax (token pick)", AQUA)]
    rows = [
        ("After\n(unpadded)", [4.932, 4.233, 0.306], 9.471),
        ("Before\n(padded to 32 rows)", [4.932, 4.233, 4.758], 13.926),
    ]
    fig, a = plt.subplots(figsize=(12, 4.4))
    draft_breakdown(a, rows, parts, "The argmax was 34% of the draft; now 3%", 16.5)
    fig.subplots_adjust(bottom=0.28)
    save(
        fig,
        out,
        "fig02_argmax_breakdown",
        f"""
Where the argmax fix comes from, inside one 3-step draft.

Time of a 3-step draft split into the drafter's 4 decoder layers, its lm_head matmul
(1024 x 262,144) and the argmax that picks each token. The old argmax padded the single logits
row to 32 rows before reducing it, so every step scanned 32 x 262,144 values (16 MiB) to find one
index. Reducing the unpadded row returns the same index (verified exact) and cuts the argmax
from 4.76 to 0.31 ms per draft (15.5x). The layers and the lm_head are unchanged; the draft goes
from 13.93 to 9.47 ms (-32.0%).

{DRAFT}
""",
    )


# --------------------------------------------------------------------------- Fig 03
def fig03_lm_head_gain(out):
    arms = [
        ("whole lm_head\nin DRAM", "base", GRID),
        ("split,\nshard_mcast", "shard_mcast", MCAST),
        ("split,\nshard_ring", "shard_ring", RING),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.4))
    e = {k: statistics.median(v) for k, v in LMHEAD_E2E.items()}
    ge = {k: 100 * (e[k] / e["base"] - 1) for k in e}
    gd = 100 * (LMHEAD_STEP["base"] / LMHEAD_STEP["shard_mcast"] - 1)
    tps_panel(
        a,
        [(l_, [1000 / LMHEAD_STEP[k]], c) for l_, k, c in arms],
        f"Drafter: {sg(gd, 1)}% draft tokens/s",
        "draft tokens/s (drafter alone)",
    )
    tps_panel(
        b,
        [(l_, LMHEAD_E2E[k], c) for l_, k, c in arms],
        f"End to end: {sg(ge['shard_mcast'], 1)}% tokens/s/user",
        "tokens/s per user",
        nd=2,
    )
    fig.subplots_adjust(wspace=0.3)
    save(
        fig,
        out,
        "fig03_lm_head_gain",
        f"""
Throughput gained by pinning part of the drafter's lm_head in on-chip SRAM.

The drafter's lm_head (1024 x 262,144, 512 MiB in bf16) is too large for L1, and ttnn cannot
place only part of one tensor in L1. So the weight is split along its columns: 35,200 columns
(13.4%, the most the demo's free L1 allows) are pinned in L1 width-sharded, the rest stays in
DRAM, and the two partial logits are joined before the argmax. The result is bit-exact against
the unsplit head.

Left: draft tokens per second of the drafter alone. With the pinned part's activation
multicast (shard_mcast): 315.3 -> 333.2 (+5.7%); run as a gather_in0 ring instead (shard_ring):
329.3 (+4.4%), because the lm_head's K is too thin for the ring (see the thin-K figure).

Right: tokens per second per user end to end, with the drafter's matmul configs tuned, 5 rounds
(dots) and their median (bar): {e['base']:.2f} -> {e['shard_mcast']:.2f} with shard_mcast ({sg(ge['shard_mcast'], 1)}%) and
{e['shard_ring']:.2f} with shard_ring ({sg(ge['shard_ring'], 1)}%);
in time, -0.54 and -0.64 ms per iteration (paired within rounds). With ttnn's default configs
the shard_mcast saving is -0.62 ms per iteration (3 rounds). shard_ring is not bit-exact and
changed token acceptance in the default configuration, so shard_mcast is the one used. One
tuned round is left out: its baseline ran unusually fast (99.04 ms vs about 100.4) for no
identified reason; with it the tuned medians are -0.51 (shard_mcast) and -0.56 (shard_ring) ms.

{DRAFT}
""",
    )


# --------------------------------------------------------------------------- Fig 04
def fig04_lm_head_breakdown(out):
    # record §5.5.7 step level (4 rotated rounds, identical to +-0.001 ms), x3 = per 3-step draft.
    parts = [("4 decoder layers", BLUE_L), ("lm_head matmul", ORANGE), ("token pick (argmax)", AQUA)]
    rows = [
        ("Split, lm_head part\nas shard_ring", [4.974, 3.774, 0.363], 9.111),
        ("Split, lm_head part\nas shard_mcast", [4.974, 3.660, 0.366], 9.003),
        ("Whole lm_head\nin DRAM", [4.974, 4.236, 0.306], 9.516),
    ]
    fig, a = plt.subplots(figsize=(12, 5))
    draft_breakdown(a, rows, parts, "The lm_head matmul gets 0.58 ms faster per draft", 11.8)
    fig.subplots_adjust(bottom=0.26)
    save(
        fig,
        out,
        "fig04_lm_head_breakdown",
        f"""
Where the lm_head gain comes from, inside one 3-step draft.

Time of a 3-step draft split into the drafter's 4 decoder layers, its lm_head matmul and the
token pick (everything from the logits to the token id). With the pinned part of the lm_head run
as shard_mcast, the lm_head goes from 4.24 to 3.66 ms per draft and the draft from 9.52 to
9.00 ms (-5.4%). The token pick grows by 0.06 ms per draft, but not in the argmax itself. The two
partial logits are each converted to row-major and concatenated into one full 262,144-wide row,
and a single argmax runs on it, the same input as without the split. The extra 0.06 ms is that
second conversion and the concat. Running the pinned part as shard_ring instead gives back 0.11 ms
(9.11 ms, -4.3%), because its K is too thin for the ring (next figure). Times were identical
to +-0.003 ms over 4 rounds.

{DRAFT}
""",
    )


# --------------------------------------------------------------------------- Fig 05
def fig05_lm_head_ring_thin_k(out):
    # record §5.5.7 op-level table, second run: the lm_head slice, weight L1 width-sharded.
    rows = [
        ("10,240 columns\non 32 cores", 21.43, 22.00, 26.57),
        ("35,200 columns\non 110 cores", 41.09, 70.73, 77.51),
    ]
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], width_ratios=[34, 112], hspace=0.62, wspace=0.08)
    a = fig.add_subplot(gs[0, :])
    w = 0.3
    for i, (lbl, mc, ring, ring_conv) in enumerate(rows):
        a.bar(i - w / 2 - 0.01, mc, w, color=MCAST, **BAR)
        a.bar(i + w / 2 + 0.01, ring, w, color=RING, **BAR)
        a.bar(i + w / 2 + 0.01, ring_conv - ring, w, bottom=ring, color=ORANGE_L, **BAR)
        a.text(i - w / 2, mc + 1.5, f"{mc:.1f}", ha="center", **VAL)
        a.text(i + w / 2, ring_conv + 1.5, f"{ring:.1f} + {ring_conv - ring:.1f}", ha="center", **VAL)
        r = ring / mc
        verdict = "tie" if r < 1.1 else f"ring {r:.1f}× slower"
        a.text(i, max(mc, ring_conv) + 10, verdict, ha="center", fontweight="semibold", color=INK, fontsize=13)
    a.set_xticks(range(2), [r[0] for r in rows])
    a.set_xlim(-0.6, 1.6)
    a.set_ylim(0, 100)
    a.set_ylabel("μs per matmul (pinned lm_head part)")
    a.grid(axis="x", visible=False)
    a.set_title("On lm_head the ring never pays: K is only 32 tiles")
    a.legend(
        handles=[
            Patch(color=MCAST, label="shard_mcast (activation multicast)"),
            Patch(color=RING, label="shard_ring (gather_in0 ring)"),
            Patch(color=ORANGE_L, label="ring's extra layout conversions"),
        ],
        loc="upper left",
    )

    for col, (title, tiles) in enumerate((("32-core ring", [1] * 32), ("110-core ring", [1] * 32 + [0] * 78))):
        b = fig.add_subplot(gs[1, col])
        b.bar(
            range(len(tiles)),
            [max(t, 0.06) for t in tiles],
            0.8,
            color=[RING if t else GRID for t in tiles],
            linewidth=0,
        )
        b.set_title(title, fontsize=12, loc="left" if col == 0 else "center")
        b.set_ylim(0, 2)
        b.set_yticks([0, 1])
        b.set_xlim(-1, len(tiles))
        b.set_xlabel("core in the ring")
        b.grid(axis="x", visible=False)
        if col == 0:
            b.set_ylabel("K tiles")
        else:
            b.text(
                71, 1.1, "78 cores hold nothing\nbut still take part in the ring", ha="center", fontsize=11, color=INK2
            )
            b.set_yticklabels([])
    save(
        fig,
        out,
        "fig05_lm_head_ring_thin_k",
        """
Why the pinned part of the lm_head is run with multicast, not the gather_in0 ring.

Top: the pinned lm_head part as one matmul in isolation, weight in L1 width-sharded, with the
activation multicast (shard_mcast) or split along K and passed around a ring of cores
(shard_ring). The light segment is the extra layout conversion the ring needs; the number on the
ring bar is the ring alone + the conversion.

The lm_head's K is 1024 = only 32 tiles. The ring splits K across its cores, so:
- on 32 cores (10,240 columns) each core gets 1 tile: the ring ties multicast (22.00 vs
  21.43 us) and loses once its conversions are counted;
- on 110 cores (35,200 columns, the size actually pinned) 78 cores get nothing, yet still take
  part in every ring step: the ring is 1.7x slower (70.73 vs 41.09 us).

Bottom: K tiles held by each core in the two rings. The ring is also not bit-exact against
multicast here (max |diff| 0.078); multicast is. Contrast with the drafter's down_proj, where
K = 256 tiles gives each of 32 cores 8 tiles and the ring is 3.2x faster (last figure).
""",
    )


# --------------------------------------------------------------------------- Fig 11
def fig11_down_proj_ring_steps(out):
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
    a.set_xticks(steps, [str(s_) for s_ in steps])
    a.set_xlabel("K-steps per matmul (fewer steps = wider blocks)")
    a.set_ylabel("μs per down_proj matmul")
    a.set_ylim(0, 235)
    a.set_title("Why the ring wins on down_proj: multicast pays per K-step")
    a.legend(loc="upper left")
    save(
        fig,
        out,
        "fig11_down_proj_ring_steps",
        """
Why the gather_in0 ring wins on down_proj: the multicast matmul's cost is set by its number of
K-steps.

The drafter's down_proj (K = 8192, N = 1024) on 32 cores with the weight pinned in L1
width-sharded, as one matmul in isolation. The multicast kernel walks K in blocks; sweeping the
block width from 1 to 32 tiles takes it from 256 steps to 8. Its time falls almost linearly with
the step count, about 0.8 us per step whatever a step carries (221.9 us at 256 steps, 41.2 at 32,
23.7 at 8), so the cost is per-step overhead, not bandwidth. Whether the activation starts in
DRAM or L1 moves it by at most 3.3 us.

The ring (orange point) splits K = 256 tiles over the 32 cores, 8 tiles each, and runs the same
32 steps as the default multicast configuration in 12.44 us instead of 40.22 us (15.07 us
including its layout conversions). One run per point.
""",
    )


# --------------------------------------------------------------------------- Fig 06
E2E_CELLS = Path(__file__).with_name("down_proj_e2e_cells.csv")


def fig06_down_proj_gain(out):
    # drafter: 6-arm campaign (3-step trace per round); e2e: run_down_proj_e2e.py, 6 rounds, warm-up dropped
    dstep, de2e = down_proj_tps()
    rows = [r for r in csv.DictReader(E2E_CELLS.open()) if r["warmup"] != "True"]
    same_text = len({r["text_sha"] for r in rows}) == 1 and len({r["accept"] for r in rows}) == 1
    assert same_text, "down_proj e2e arms no longer output-identical; the description below would be wrong"
    by = {}
    for r in rows:
        by.setdefault(r["round"], {})[r["arm"]] = float(r["ms_per_iter"])
    d = {a_: [v[a_] - v["dram_mcast"] for v in by.values()] for a_ in ("shard_mcast", "shard_ring")}
    arms = [("dram_mcast", GRID), ("shard_mcast", MCAST), ("shard_ring", RING)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.4))
    gd = 100 * (statistics.median(dstep["shard_ring"]) / statistics.median(dstep["dram_mcast"]) - 1)
    ge = 100 * (statistics.median(de2e["shard_ring"]) / statistics.median(de2e["dram_mcast"]) - 1)
    tps_panel(
        a,
        [(k, dstep[k], c) for k, c in arms],
        f"Drafter: {sg(gd, 1)}% draft tokens/s",
        "draft tokens/s (drafter alone)",
    )
    tps_panel(
        b, [(k, de2e[k], c) for k, c in arms], f"End to end: {sg(ge, 2)}% tokens/s/user", "tokens/s per user", nd=2
    )
    fig.subplots_adjust(wspace=0.3)
    m = {k: statistics.median(v) for k, v in dstep.items()}
    e = {k: statistics.median(v) for k, v in de2e.items()}
    save(
        fig,
        out,
        "fig06_down_proj_gain",
        f"""
Throughput gained by pinning one drafter down_proj weight in on-chip SRAM, alone and with the
gather_in0 ring.

Only one of the drafter's four down_proj weights (8192 x 1024, 16 MiB) fits in the L1 left free
by the runtime buffers, so this is one layer. shard_mcast pins it in L1 width-sharded;
shard_ring additionally delivers the activation with the gather_in0 ring instead of multicast.
The baseline, dram_mcast, keeps it in DRAM.

Left: draft tokens per second of the drafter alone, 12 rounds (dots) and their median (bar):
{m['dram_mcast']:.1f} -> {m['shard_mcast']:.1f} (shard_mcast) -> {m['shard_ring']:.1f} (shard_ring, +{100 * (m['shard_ring'] / m['dram_mcast'] - 1):.1f}%).

Right: tokens per second per user end to end, 6 rounds (dots) and their median (bar):
{e['dram_mcast']:.3f} -> {e['shard_mcast']:.3f} (shard_mcast) -> {e['shard_ring']:.3f} (shard_ring, {sg(ge, 2)}%). In time, paired within
rounds: shard_ring {sg(statistics.median(d['shard_ring']), 3)} ms per iteration, faster in {sum(x < 0 for x in d['shard_ring'])} of 6 rounds;
shard_mcast {sg(statistics.median(d['shard_mcast']), 2)} ms, smaller than the run-to-run spread (the baseline alone varies by
{max(by[r_]['dram_mcast'] for r_ in by) - min(by[r_]['dram_mcast'] for r_ in by):.2f} ms across rounds), so not resolved end to end. Every one of the runs generated the same
text with the same token acceptance (1.62 of 3 drafts): the pin and the ring do not change the
output.

{DRAFT}
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


def fig07_to_10_down_proj(out):
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
        ("L1 interleaved\nil_mcast | il_ring", "il_mcast", "il_ring"),
        ("L1 width-sharded\nshard_mcast | shard_ring", "shard_mcast", "shard_ring"),
    ]
    glossary = (
        "Arms are named <weight placement>_<activation delivery>: dram = weight in DRAM; il = weight in L1 "
        "interleaved, i.e. spread page by page over all 110 L1 banks; shard = weight in L1 width-sharded, i.e. "
        "split into column slices assigned to the matmul's cores. "
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
        "fig07_down_proj_placement_x_delivery",
        f"""
One down_proj layer of the drafter, six ways: three places for its weight x two ways of feeding
the activation to the matmul. Each bar is the change in the time of a 3-step draft against
dram_mcast (weight in DRAM, activation multicast; {base:.0f} us).

- Moving the weight into L1 helps with either delivery: il_mcast -37 us, shard_mcast -40 us.
- The ring loses when the weight is in DRAM (dram_ring +8 us) and wins when it is in L1
  (il_ring -78 us, shard_ring -113 us).
- shard_ring, the width-sharded weight plus the ring, is the fastest: -113 us, -1.19% of the
  draft.

{glossary}

{method}
""",
    )

    # Fig 3: ring minus its own mcast, per ringed matmul
    fig, a = plt.subplots(figsize=(9.5, 5.2))
    for i, (lbl, mc, rg) in enumerate(places):
        ds = [r[rg] - r[mc] for r in rounds]
        m, lo, hi = med_range(ds)
        a.bar(i, m, 0.5, color=RING if m < 0 else INK2, **BAR)
        a.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, lw=1.2, capsize=4)
        a.text(i, hi + 2 if m >= 0 else lo - 2, sg(m, 0) + " μs", ha="center", va="bottom" if m >= 0 else "top", **VAL)
    a.axhline(0, color=AXIS, lw=1)
    a.set_ylim(-84, 21)
    a.set_xticks(range(3), [p[0] for p in places])
    a.set_ylabel("ring − mcast, μs per 3-step draft\nabove 0 = ring slower")
    a.set_title("The ring's benefit flips sign with the weight's placement")
    a.grid(axis="x", visible=False)
    save(
        fig,
        out,
        "fig08_down_proj_ring_by_placement",
        f"""
What switching the activation delivery from multicast to the gather_in0 ring is worth, per
3-step draft, for each weight placement: dram_ring - dram_mcast, il_ring - il_mcast and
shard_ring - shard_mcast. The ring is used on the same 3 matmuls in every case (the one pinned
down_proj layer, once per draft step).

With the weight in DRAM the ring costs +8 us per draft. With the weight in L1 it saves 40 us
(interleaved) or 72 us (width-sharded). The ring blocks K more finely than multicast does, so it
issues more, smaller weight reads; that is cheap from SRAM and expensive from DRAM.

{glossary}

{method}
""",
    )

    # Fig 4: waterfall(s) dram_mcast -> ... -> shard_ring
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    paths = [
        (
            "A: change the layout, then add the ring",
            [
                ("weight\nto L1\n\n(il_mcast)", "il_mcast", "dram_mcast", AQUA),
                ("layout change\ninterleaved\n→ sharded\n(shard_mcast)", "shard_mcast", "il_mcast", BLUE),
                ("add\nthe ring\n\n(shard_ring)", "shard_ring", "shard_mcast", ORANGE),
            ],
        ),
        (
            "B: add the ring, then change the layout",
            [
                ("weight\nto L1\n\n(il_mcast)", "il_mcast", "dram_mcast", AQUA),
                ("add\nthe ring\n\n(il_ring)", "il_ring", "il_mcast", ORANGE),
                ("layout change\ninterleaved\n→ sharded\n(shard_ring)", "shard_ring", "il_ring", BLUE),
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
        a.set_xticks(range(4), [s[0] for s in steps] + ["total\n\n\n(shard_ring)"])
        a.set_title(title, fontsize=13)
        a.grid(axis="x", visible=False)
    axs[0].set_ylabel("Δ μs per 3-step draft vs dram_mcast")
    fig.suptitle("Where the −113 μs comes from", x=0.06, ha="left", fontsize=15, fontweight="semibold", color=INK)
    fig.subplots_adjust(top=0.82, wspace=0.08)
    save(
        fig,
        out,
        "fig09_down_proj_breakdown",
        f"""
The full saving of shard_ring over dram_mcast (-113 us per 3-step draft), taken apart in two
orders.

A: moving the weight from DRAM into L1 (interleaved) gives -37 us; changing its L1 layout from
interleaved to width-sharded gives only -3 us more under multicast; adding the ring then gives
-72 us.
B: from the same L1-interleaved start, the ring alone gives -40 us, and changing the layout to
width-sharded then gives -35 us.

The two orders disagree on what the layout change is worth (-3 vs -35 us) because the layout and
the ring interact: the ring issues many more, smaller weight reads, and those are what the
layout change speeds up. Steps are medians of within-round contrasts, so they need not add up
exactly to the total.

{glossary}

{method}
""",
    )

    # Fig 10: the interleaved -> width-sharded layout change under each delivery
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
    a.set_title("The interleaved → sharded layout change\nis worth 12× more with the ring")
    a.grid(axis="x", visible=False)
    save(
        fig,
        out,
        "fig10_down_proj_layout_change",
        f"""
The value of the L1 layout change: the weight in L1 width-sharded (shard) against the weight in
L1 interleaved over all banks (il), for each activation delivery.

With multicast it is worth only -2.8 us per 3-step draft: once the weight is in SRAM at all, its
L1 layout barely matters. With the gather_in0 ring it is worth -35.0 us, 12x more: the ring's
finer K blocking issues many more, smaller weight reads, and the width-sharded layout serves
them much faster. Which part of that comes from tiles sitting in the reading core's own L1 has
not been isolated.

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
    "fig00": fig00_throughput_summary,
    "fig01": fig01_argmax_gain,
    "fig02": fig02_argmax_breakdown,
    "fig03": fig03_lm_head_gain,
    "fig04": fig04_lm_head_breakdown,
    "fig05": fig05_lm_head_ring_thin_k,
    "fig06": fig06_down_proj_gain,
    "campaign": fig07_to_10_down_proj,
    "fig11": fig11_down_proj_ring_steps,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    have = {"campaign": CELLS.exists(), "fig06": E2E_CELLS.exists()}
    p.add_argument("--only", default=",".join(k for k in FIGS if have.get(k, True)))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for k in args.only.split(","):
        FIGS[k](args.out)


if __name__ == "__main__":
    main()

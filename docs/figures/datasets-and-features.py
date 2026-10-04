"""The two corpora and what a feature is.

WHY THIS FIGURE EXISTS. Every number in this repository is conditioned on two facts about the input
that prose keeps burying. The rows are sparse binary token sets, at 1.19% density, so a literal
asking for a feature to be ABSENT is satisfied without cost on almost every row. And the primary
corpus ships its vocabulary stripped, so no claim about what a feature MEANS can be made there. A
reader who has those two facts in mind reads the rest of the repository correctly; a reader who does
not will mistake the first for a finding about the model and the second for an oversight.

The category bar exists because "interpretable feature" is not one thing here. More than half of the
named vocabulary is Java class names of an application's own screens, which identify an app and
transfer to nothing. The four categories that constitute an indicator of compromise are 23.0% of it,
and that number is the base rate every claim about attributed features is measured against.

ENCODING. Two panels for the two corpora, on identical rows, so the differences line up: scale,
width, base rate, protocol, and whether the names are released. The strip beneath each shows one row
at its measured density. The category bar is sorted by share and coloured by what the category can
support, with each bar also labelled, so nothing rests on colour alone.

  python docs/figures/datasets-and-features.py

Output: docs/figures/datasets-and-features.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE / "datasets-and-features.png"

FIGW, FIGH = 14.0, 9.6
BLUE, ORANGE = "#2a78d6", "#eb6834"
GREEN, GREY = "#3f8f5c", "#9b9b97"
SURFACE, PANEL, INK, MUTED = "#fcfcfb", "#f4f4f1", "#1a1a19", "#6b6b68"
L, FW = 0.040, 0.920


def pts(n):
    return n / 72.0 / FIGH


def panel(ax, x, y, w, h, accent, fill=PANEL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=0, facecolor=fill, zorder=0))
    ax.add_patch(FancyBboxPatch((x, y), 0.0038, h, boxstyle="square,pad=0",
                                linewidth=0, facecolor=accent, zorder=1))


# Drebin's own prefixes on the APIGraph vocabulary, from research/groundtruth-overlap/.
# "indicator" is a capability requested or an API reached for; "identity" names the app itself.
CATEGORIES = [
    ("activity", 53.1, "identity"),
    ("requested permission", 10.2, "indicator"),
    ("intent filter", 10.1, "other"),
    ("restricted API", 8.3, "indicator"),
    ("broadcast receiver", 5.0, "identity"),
    ("service", 4.5, "identity"),
    ("used permission", 2.8, "indicator"),
    ("URL domain", 2.0, "other"),
    ("suspicious API", 1.8, "indicator"),
    ("hardware component", 1.8, "other"),
    ("content provider", 0.5, "identity"),
]
KIND = {"indicator": ORANGE, "identity": BLUE, "other": GREY}

CORPORA = (
    (ORANGE, "LAMDA", "the primary corpus",
     [("applications", "1,008,381"),
      ("malware / benign", "369,906 / 638,475"),
      ("span", "2013-2025, no 2015, 120 months"),
      ("features after variance filtering", "4,561, so 9,122 literals"),
      ("feature density", "1.19%"),
      ("protocol", "train on 2013-14, test per period"),
      ("vocabulary", "STRIPPED: feat_0 ... feat_4560")]),
    (BLUE, "APIGRAPH", "the replication corpus",
     [("applications in the 2012 pool", "30,533"),
      ("malware share", "10.0%"),
      ("span", "2012 pool, tested 2013-2018"),
      ("features after variance filtering", "1,159, so 2,318 literals"),
      ("feature density", "11%"),
      ("protocol", "train on 2012, test per year"),
      ("vocabulary", "RELEASED: real Drebin token names")]),
)


def sparse_strip(ax, x, y, w, h, density, seed, colour):
    """One row of the matrix at its measured density, so the sparsity is seen and not asserted."""
    n = 150
    rng = np.random.default_rng(seed)
    on = rng.random(n) < density
    cw = w / n
    ax.add_patch(Rectangle((x, y), w, h, linewidth=0.6, edgecolor="#d8d8d4",
                           facecolor="#ffffff", zorder=2))
    for i in np.flatnonzero(on):
        ax.add_patch(Rectangle((x + i * cw, y), cw, h, linewidth=0, facecolor=colour, zorder=3))
    return int(on.sum())


def main():
    fig = plt.figure(figsize=(FIGW, FIGH), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(L, 0.970, "Two corpora of sparse binary token sets", fontsize=19, color=INK,
            va="top", weight="semibold")
    ax.text(L, 0.938,
            "Drebin-style bag-of-tokens over an Android application's manifest and disassembly.",
            fontsize=10.5, color=MUTED, va="top")
    ax.text(L, 0.916,
            "Every feature is present or absent, so no booleanisation step stands between the "
            "corpus and the model.",
            fontsize=10.5, color=MUTED, va="top")

    # ---- the two corpora, on identical rows ----------------------------------- #
    aT, aH = 0.890, 0.368
    W = (FW - 0.022) / 2
    for k, (x, (accent, name, sub, rows)) in enumerate(zip((L, L + W + 0.022), CORPORA)):
        panel(ax, x, aT - aH, W, aH, accent)
        ax.text(x + 0.018, aT - pts(19), name, fontsize=12.5, color=accent, va="top",
                weight="semibold")
        ax.text(x + 0.018, aT - pts(37), sub, fontsize=10, color=MUTED, va="top", style="italic")
        for j, (label, value) in enumerate(rows):
            yy = aT - pts(62 + 21 * j)
            ax.text(x + 0.018, yy, label, fontsize=9.6, color=MUTED, va="center")
            bold = label == "vocabulary"
            ax.text(x + W - 0.018, yy, value, fontsize=9.8 if not bold else 10.2,
                    color=INK if not bold else accent, va="center", ha="right",
                    weight="semibold" if bold else "normal")
        # one row of the matrix, at the density stated above it
        sy = aT - aH + pts(22)
        on = sparse_strip(ax, x + 0.018, sy - pts(4), W - 0.036, pts(13),
                          0.0119 if k == 0 else 0.11, 7 if k == 0 else 12, accent)
        ax.text(x + 0.018, sy + pts(18),
                "a 150-feature window of one row: %d present" % on,
                fontsize=8.8, color=MUTED, va="center")

    # ---- the consequence of the stripped vocabulary --------------------------- #
    hY = aT - aH - pts(30)
    ax.plot([L, L + FW], [hY + pts(15), hY + pts(15)], color="#dededa", lw=1, zorder=1)
    ax.text(0.5, hY - pts(4),
            "A feature's name is released for one corpus only, so what a feature means can be "
            "checked there and nowhere else",
            fontsize=11.5, color=INK, ha="center", va="center")

    # ---- what the named vocabulary is made of --------------------------------- #
    cT, cH = hY - pts(34), 0.385
    panel(ax, L, cT - cH, FW, cH, GREY, fill="#f2f3f1")
    ax.text(L + 0.018, cT - pts(18), "THE NAMED VOCABULARY BY DREBIN CATEGORY", fontsize=11.5,
            color=INK, va="top", weight="semibold")
    ax.text(L + 0.018, cT - pts(36),
            "APIGraph, 1,159 tokens, assigned by each token's own prefix, so the grouping involves "
            "no judgement of ours.",
            fontsize=9.8, color=MUTED, va="top", style="italic")

    gx, gy = L + 0.175, cT - cH + pts(40)
    gw, gh = 0.420, cH - pts(92)
    axb = fig.add_axes([gx, gy, gw, gh])
    axb.set_facecolor("#f2f3f1")
    y = np.arange(len(CATEGORIES))[::-1]
    vals = [c[1] for c in CATEGORIES]
    axb.barh(y, vals, height=0.62, color=[KIND[c[2]] for c in CATEGORIES], zorder=3)
    for yy, (label, v, kind) in zip(y, CATEGORIES):
        axb.text(v + 0.8, yy, "%.1f%%  %s" % (v, kind), fontsize=8.4, color=INK, va="center")
    axb.set_yticks(y)
    axb.set_yticklabels([c[0] for c in CATEGORIES], fontsize=8.6, color=INK)
    axb.set_xlim(0, 72)
    axb.set_xticks([0, 20, 40, 60])
    axb.tick_params(labelsize=8.2, colors=MUTED, length=0)
    axb.set_xlabel("share of the 1,159 released token names", fontsize=8.4, color=MUTED)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)
    axb.spines["bottom"].set_color("#d8d8d4")
    axb.grid(axis="x", color="#e6e7e4", lw=0.8, zorder=0)
    axb.set_axisbelow(True)

    tx = L + 0.625
    ax.text(tx, cT - pts(62), "THE BASE RATE BEHIND EVERY CLAIM", fontsize=9.6,
            color=INK, va="top", weight="semibold")
    for j, (lab, val, col, note) in enumerate((
            ("indicator of compromise", "23.0%", ORANGE,
             "a capability requested or an API reached for"),
            ("application identity", "63.1%", BLUE,
             "Java class names of the app's own screens"),
            ("neither", "13.9%", GREY, "intent filters, domains, hardware"))):
        yy = cT - pts(86 + 42 * j)
        ax.text(tx, yy, val, fontsize=13, color=col, va="center", weight="semibold")
        ax.text(tx + 0.058, yy + pts(5), lab, fontsize=10, color=INK, va="center")
        ax.text(tx + 0.058, yy - pts(10), note, fontsize=8.8, color=MUTED, va="center")

    ax.text(tx, cT - cH + pts(34),
            "Over half the vocabulary identifies an application",
            fontsize=9.6, color=INK, va="center")
    ax.text(tx, cT - cH + pts(20),
            "instead of describing what it does, so chance alone",
            fontsize=9.6, color=MUTED, va="center")
    ax.text(tx, cT - cH + pts(6),
            "puts most of any feature set out of a responder's reach.",
            fontsize=9.6, color=MUTED, va="center")

    ax.text(L, 0.013,
            "Sources: research/b0-composition/ for the LAMDA counts, research/b5-apigraph/ for the "
            "replication protocol, research/blacklist-anatomy/ for the density, "
            "research/groundtruth-overlap/ for the category shares.",
            fontsize=9.2, color=MUTED, va="bottom")

    fig.savefig(OUT, dpi=150, metadata={"Software": None})
    print("wrote %s" % OUT.name)


main()

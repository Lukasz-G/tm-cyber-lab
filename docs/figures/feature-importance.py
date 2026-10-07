"""Exact credit for a clause's literals.

WHY THIS FIGURE EXISTS. The closed form is four lines of algebra whose consequence is hard to see in
prose: once a clause's literals are grouped by how they behave at the explained row against the
background, the vote depends on two counts and nothing else, and a Shapley value over 2^M coalitions
collapses to a handful of terms. Showing the grouping makes the collapse obvious, and it makes the
boundary obvious too, since the argument uses the counts and would fail for anything that asks which
literal is unsatisfied.

The last band is here because exactness is a claim that has to be demonstrated. Two brute-force
checks and one re-implementation in a second language sit beside the number the sampled estimator
reaches at the budget the literature uses.

ENCODING. Four groups, left to right, in the order the derivation uses them, each with its size and
its effect on the vote. Blue marks what a coalition can change and grey what it cannot, which is the
whole argument; every group is also labelled, so colour carries nothing on its own.

  python docs/figures/feature-importance.py

Output: docs/figures/feature-importance.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "feature-importance.png"

FIGW, FIGH = 14.0, 8.6
BLUE, ORANGE = "#2a78d6", "#eb6834"
GREY = "#9b9b97"
SURFACE, PANEL, INK, MUTED = "#fcfcfb", "#f4f4f1", "#1a1a19", "#6b6b68"
L, FW = 0.040, 0.920


def pts(n):
    return n / 72.0 / FIGH


def panel(ax, x, y, w, h, accent, fill=PANEL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=0, facecolor=fill, zorder=0))
    ax.add_patch(FancyBboxPatch((x, y), 0.0038, h, boxstyle="square,pad=0",
                                linewidth=0, facecolor=accent, zorder=1))


# The four groups of the derivation, in the order it uses them. "size" is the symbol the derivation
# gives each group; only C and D have one.
GROUPS = (
    ("A", GREY, "satisfied at both",
     ["the row and the background", "agree, so the literal is", "never a miss"],
     "no contribution", "no size symbol"),
    ("B", GREY, "unsatisfied at both",
     ["they agree the other way, so", "the literal is always a miss", "whatever the coalition is"],
     "a constant, folded into $\\beta$", "no size symbol"),
    ("C", BLUE, "satisfied at the row only",
     ["a miss unless the feature is", "in the coalition, so a", "coalition can remove it"],
     "a share of the credit", "$|C| = g$"),
    ("D", BLUE, "satisfied at the background only",
     ["a miss when the feature IS in", "the coalition, so a coalition", "can create it"],
     "a share of the credit", "$|D| = d$"),
)


def main():
    fig = plt.figure(figsize=(FIGW, FIGH), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(L, 0.964, "Exact credit for a clause's literals", fontsize=19, color=INK,
            va="top", weight="semibold")
    ax.text(L, 0.929,
            "A Shapley value asks what each feature contributes, averaged over every order in which "
            "the features might arrive. For a clause the average has a closed form.",
            fontsize=10.5, color=MUTED, va="top")

    # ---- band 1: the partition ------------------------------------------------ #
    aT, aH = 0.900, 0.338
    panel(ax, L, aT - aH, FW, aH, INK, fill="#f1f1ee")
    ax.text(L + 0.018, aT - pts(18), "THE PARTITION", fontsize=11.5, color=INK, va="top",
            weight="semibold")
    ax.text(L + 0.018, aT - pts(36),
            "Group the clause's literals by how each behaves at the explained row $x$ against the "
            "background row $b$.",
            fontsize=9.8, color=MUTED, va="top", style="italic")

    CW = (FW - 0.036 - 3 * 0.014) / 4
    for k, (name, col, head, lines, effect, size) in enumerate(GROUPS):
        x = L + 0.018 + k * (CW + 0.014)
        yb = aT - aH + pts(26)
        ht = aH - pts(84)
        ax.add_patch(FancyBboxPatch((x, yb), CW, ht, boxstyle="round,pad=0,rounding_size=0.005",
                                    linewidth=1.0, facecolor="#ffffff", edgecolor=col, zorder=2))
        ax.text(x + 0.012, yb + ht - pts(16), name, fontsize=14, color=col, va="center",
                weight="semibold", zorder=3)
        ax.text(x + 0.040, yb + ht - pts(16), head, fontsize=9.2, color=INK, va="center", zorder=3)
        for j, line in enumerate(lines):
            ax.text(x + 0.012, yb + ht - pts(38 + 14 * j), line, fontsize=8.6, color=MUTED,
                    va="top", zorder=3)
        ax.text(x + 0.012, yb + pts(26), effect, fontsize=9.4, color=col, va="center",
                weight="semibold", zorder=3)
        ax.text(x + 0.012, yb + pts(11), size, fontsize=9.0, color=MUTED, va="center", zorder=3)

    # ---- the equation the partition yields ------------------------------------ #
    hY = aT - aH - pts(30)
    ax.text(0.5, hY,
            r"$\mathrm{misses}(S) = \beta + (g - |C \cap S|) + |D \cap S|$",
            fontsize=16, color=INK, ha="center", va="center")
    ax.text(0.5, hY - pts(23),
            "Two counts, and never the identity of a literal. That is the whole of what the "
            "derivation uses.",
            fontsize=10.5, color=MUTED, ha="center", va="center")

    # ---- band 2: what follows ------------------------------------------------- #
    bT, bH = hY - pts(44), 0.205
    panel(ax, L, bT - bH, FW, bH, BLUE, fill="#eef4fc")
    ax.text(L + 0.018, bT - pts(18), "THE CONSEQUENCES", fontsize=11.5, color=BLUE, va="top",
            weight="semibold")
    for k, (val, head, lines) in enumerate((
            ("0", "for every feature outside $C \\cup D$",
             ["Its presence changes no count, so its marginal",
              "contribution is zero in every coalition. On LAMDA",
              "that leaves 496 of 4,561 features with any value."]),
            ("2", "distinct values across the rest",
             ["Members of $C$ are interchangeable and so share one",
              "value; members of $D$ share another. Averaging over",
              "orders reduces to a hypergeometric sum."]),
            ("$O((g+d)^2)$", "per clause, no feature-count term",
             ["Cost is set by the size of the clause. One month of",
              "attribution on LAMDA takes 6.7 s single-threaded at",
              "twenty clauses per class."]))):
        x = L + 0.018 + k * ((FW - 0.036) / 3)
        ax.text(x, bT - pts(46), val, fontsize=16, color=INK, va="center", weight="semibold")
        ax.text(x + (0.052 if k < 2 else 0.104), bT - pts(46), head, fontsize=9.4, color=BLUE,
                va="center", weight="semibold")
        for j, line in enumerate(lines):
            ax.text(x, bT - pts(68 + 14.5 * j), line, fontsize=9.0, color=MUTED, va="top")

    # ---- band 3: the check ---------------------------------------------------- #
    cT, cH = bT - bH - 0.020, 0.142
    panel(ax, L, cT - cH, FW, cH, ORANGE, fill="#faeee8")
    ax.text(L + 0.018, cT - pts(18), "THE CHECK AGAINST BRUTE FORCE", fontsize=11.5, color=ORANGE,
            va="top", weight="semibold")
    CHECKS = (
        ("$8.9 \\times 10^{-16}$", "trained models, all $2^{14}$ coalitions",
         "144 combinations of seed, tolerance, ceiling policy, row and class"),
        ("$1.29 \\times 10^{-14}$", "a RIPPER ensemble, all $2^{13}$ coalitions",
         "rules we did not induce, re-implemented in a second language"),
        ("$0.851$ / $0.889$", "the sampled alternative",
         "distance from the exact values, and from a second run of itself"),
    )
    for k, (val, head, note) in enumerate(CHECKS):
        x = L + 0.018 + k * ((FW - 0.036) / 3)
        col = ORANGE if k == 2 else INK
        ax.text(x, cT - pts(46), val, fontsize=14, color=col, va="center", weight="semibold")
        ax.text(x, cT - pts(66), head, fontsize=9.6, color=col, va="top", weight="semibold")
        ax.text(x, cT - pts(81), note, fontsize=8.8, color=MUTED, va="top")

    ax.text(L, 0.030,
            "Polynomial-time computation for this class follows from known results on Shapley "
            "values over decomposable circuits.",
            fontsize=9.2, color=MUTED, va="bottom")
    ax.text(L, 0.012,
            "What the closed form supplies is an explicit expression with no compilation step. "
            "Sources: research/b2-generality/, research/ripper-exact/, research/b2-drift/.",
            fontsize=9.2, color=MUTED, va="bottom")

    fig.savefig(OUT, dpi=150, metadata={"Software": None})
    print("wrote %s" % OUT.name)


main()

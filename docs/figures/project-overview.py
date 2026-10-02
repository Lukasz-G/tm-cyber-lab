"""What the project found, in one panel: exact attribution, and what it does to a published number.

WHY THIS FIGURE EXISTS. The repository's headline is a conditional that is hard to take in from prose:
IF a rule ensemble is additive and its rules read only how many literals are unsatisfied, THEN exact
Shapley values have a closed form, and at realistic width that closed form is cheaper than the sampled
estimate it replaces. The figure states the condition once, puts the two routes side by side so the cost
comparison is visible and not asserted, and then shows what the exact instrument does to the number
the project set out to check.

The lower band is the result. A single reported figure of 0.958 separates into three contributions, and
their ORDER is the finding: the estimator's own variance dominates, monthly refitting comes next, and the
data the figure is usually read as measuring comes last at roughly a third of it.

ENCODING. One colour per route, blue for exact and orange for sampled, used consistently in both the
panels and the chart. The bars are annotated directly, so no legend is needed, and the reported value is
a dashed reference line because every bar is to be read against it.

  python docs/figures/project-overview.py

Output: docs/figures/project-overview.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "project-overview.png"

FIGW, FIGH = 14.0, 10.0
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, PANEL, INK, MUTED = "#fcfcfb", "#f4f4f1", "#1a1a19", "#6b6b68"

L, R, W, FW = 0.040, 0.516, 0.444, 0.920


def pts(n):
    return n / 72.0 / FIGH


def panel(ax, x, y, w, h, accent, fill=PANEL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=0, facecolor=fill, zorder=0))
    ax.add_patch(FancyBboxPatch((x, y), 0.0038, h, boxstyle="square,pad=0",
                                linewidth=0, facecolor=accent, zorder=1))


def main():
    fig = plt.figure(figsize=(FIGW, FIGH), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(L, 0.967, "Exact attribution at less cost than its approximation",
            fontsize=19, color=INK, va="top", weight="semibold")
    ax.text(L, 0.930,
            "LAMDA, 4,561 Drebin features, 2013–2022, flat Fuzzy-Pattern Tsetlin machine at 20 "
            "clauses per class; 88 months, 100 background and 100 explained rows a month.",
            fontsize=10.5, color=MUTED, va="top")

    # ---- band A: the condition ------------------------------------------ #
    aT, aH = 0.900, 0.145
    panel(ax, L, aT - aH, FW, aH, INK, fill="#f1f1ee")
    ax.text(L + 0.018, aT - pts(19), "THE CONDITION", fontsize=11.5, color=INK, va="top",
            weight="semibold")
    ax.text(0.5, aT - pts(54),
            r"$f(x) = \sum_r w_r\, g_r\!\left(\mathit{misses}_r(x)\right)$"
            r"$\qquad\Longrightarrow\qquad$"
            r"exact Shapley in $O\!\left((g+d)^2\right)$ per rule",
            fontsize=15.5, color=INK, ha="center", va="center")
    ax.text(0.5, aT - aH + pts(20),
            "additive over rules, and each rule reads only HOW MANY of its literals are unsatisfied, "
            "never which. Verified against brute force to $8.9\\times10^{-16}$ over 144 configurations.",
            fontsize=10.5, color=MUTED, ha="center", va="center")

    # ---- band B: the two routes ----------------------------------------- #
    bT, bH = aT - aH - 0.019, 0.272
    panel(ax, L, bT - bH, W, bH, ORANGE)
    panel(ax, R, bT - bH, W, bH, BLUE)

    for x, accent, head, sub, eq, body, foot in (
        (L, ORANGE, "SAMPLED", "the instrument in general use",
         "100 coalitions against 4,561 features",
         ["Kernel SHAP fits a weighted linear surrogate to the target's outputs on sampled",
          "coalitions. The library's own default budget is $2M+2048 = 11{,}170$; the published",
          "series uses 100, which is a factor of forty-five fewer coalitions than features.",
          "→  the variance that would tell a real turnover from a failure to find it twice",
          "     is not reported at all"],
         "0.851 from the truth, 0.889 from ITSELF"),
        (R, BLUE, "EXACT", "the instrument available here",
         "no sampling, no feature-count term",
         ["All but $g+d$ features have value exactly zero, the rest collapse to two shared",
          "values, and the cost per rule is set by rule size. The class covers Fuzzy-Pattern",
          "and classical Tsetlin machines, $m$-of-$n$ thresholds and weighted rule ensembles.",
          "→  ordered rule lists and raw disjunctions are provably OUTSIDE it, measured at",
          "     $1.1\\times10^{-1}$ against $2\\times10^{-13}$ for the additive arms"],
         "6.7 s a month, against the estimate it replaces"),
    ):
        ax.text(x + 0.020, bT - pts(19), head, fontsize=12.5, color=accent, va="top",
                weight="semibold")
        ax.text(x + 0.020, bT - pts(37), sub, fontsize=10, color=MUTED, va="top", style="italic")
        ax.text(x + W / 2, bT - pts(68), eq, fontsize=12.5, color=INK, ha="center", va="center")
        yy = bT - pts(86)
        for line in body:
            yy -= pts(14.6)
            ax.text(x + 0.020, yy, line, fontsize=9.6,
                    color=INK if line.startswith("→") else MUTED, va="top")
        ax.text(x + 0.020, bT - bH + pts(16), foot, fontsize=11, color=accent, va="top",
                weight="semibold")

    # ---- the hinge ------------------------------------------------------ #
    hY = bT - bH - pts(26)
    ax.plot([L, L + FW], [hY + pts(14), hY + pts(14)], color="#dededa", lw=1, zorder=1)
    ax.text(0.5, hY - pts(5),
            "with an exact instrument, one reported number separates into three contributions",
            fontsize=11.5, color=INK, ha="center", va="center")
    ax.text(0.5, hY - pts(21),
            "and the order is the result: the estimator first, the experimental protocol second, "
            "the phenomenon last",
            fontsize=10, color=MUTED, ha="center", va="center")

    # ---- band C: the decomposition -------------------------------------- #
    cT, cH = hY - pts(40), 0.332
    panel(ax, L, cT - cH, FW, cH, ORANGE, fill="#faeee8")
    ax.text(L + 0.018, cT - pts(18), "THE COMPOSITION OF THE PUBLISHED 0.958", fontsize=11.5,
            color=ORANGE, va="top", weight="semibold")

    labels = ["the estimator's own\nnoise floor",
              "refitting a fresh model\nevery month",
              "the data moving,\none fixed model"]
    vals = [0.926, 0.661, 0.294]
    cols = [ORANGE, "#b0803a", BLUE]
    gx, gy, gw, gh = L + 0.145, cT - cH + pts(60), 0.210, cH - pts(104)
    axb = fig.add_axes([gx, gy, gw, gh])
    axb.set_facecolor("#faeee8")
    y = np.arange(3)[::-1]
    axb.barh(y, vals, height=0.52, color=cols, zorder=3)
    axb.axvline(0.958, color=INK, ls="--", lw=1.1, zorder=4)
    axb.text(0.958, 2.62, " reported 0.958", fontsize=9, color=INK, va="center")
    for yy, v in zip(y, vals):
        # a long bar's label would collide with the reference line, so it goes inside instead
        if v > 0.80:
            axb.text(v - 0.015, yy, "%.3f" % v, fontsize=10, color="#ffffff", va="center",
                     ha="right", weight="semibold", zorder=5)
        else:
            axb.text(v + 0.014, yy, "%.3f" % v, fontsize=10, color=INK, va="center",
                     weight="semibold")
    axb.set_yticks(y)
    axb.set_yticklabels(labels, fontsize=9, color=INK)
    axb.set_xlim(0, 1.13)
    axb.set_ylim(-0.6, 2.9)
    axb.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axb.tick_params(labelsize=8.5, colors=MUTED, length=0)
    axb.set_xlabel("Jaccard distance, lower is more stable", fontsize=8.5, color=MUTED)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)
    axb.spines["bottom"].set_color("#d8cdc6")
    axb.grid(axis="x", color="#eadcd4", lw=0.8, zorder=0)
    axb.set_axisbelow(True)

    ax.text(L + 0.400, cT - pts(56),
            "Two runs of the explainer against ONE unchanged model, on one",
            fontsize=9.8, color=MUTED, va="top")
    ax.text(L + 0.400, cT - pts(71),
            "unchanged month, disagree by 0.926. That is 97% of the reported",
            fontsize=9.8, color=MUTED, va="top")
    ax.text(L + 0.400, cT - pts(86),
            "signal, so the published figure sits at its estimator's noise floor.",
            fontsize=9.8, color=MUTED, va="top")
    ax.text(L + 0.400, cT - pts(110),
            "→  refitting contributes MORE than the data does, $+0.367$,",
            fontsize=9.8, color=INK, va="top")
    ax.text(L + 0.400, cT - pts(125),
            "     and it is in the released code but not in its paper",
            fontsize=9.8, color=INK, va="top")
    ax.text(L + 0.400, cT - pts(149),
            "Raising the sampling budget tenfold moves the estimator from",
            fontsize=9.8, color=MUTED, va="top")
    ax.text(L + 0.400, cT - pts(164),
            "0.851 to 0.496, so the budget is the cause.",
            fontsize=9.8, color=MUTED, va="top")
    ax.text(L + 0.400, cT - pts(190), "No month in either exact arm reaches 0.958.",
            fontsize=10.5, color=ORANGE, va="top", weight="semibold")
    ax.text(L + 0.400, cT - pts(205),
            "0 of 174 month-pairs: it is not a value we ever attain.",
            fontsize=9.8, color=MUTED, va="top")

    ax.text(L, 0.013,
            "The same variance-not-bias result reproduces on a RIPPER-induced rule ensemble, to three "
            "decimals, so it is a property of sampled attribution at this budget and not of Tsetlin "
            "machines.",
            fontsize=10, color=INK, va="bottom")

    fig.savefig(OUT, dpi=150, metadata={"Software": None})
    print("wrote %s" % OUT.name)


main()

"""Four claims, and the control that retired each.

WHY THIS FIGURE EXISTS. This project's negative results are its most reusable output, and a reader
skimming a repository will not find them: they are scattered across four experiment directories and
they contradict earlier commits. Gathering them into one panel does two things. It states plainly that
the headline claims did not survive, so nobody builds on them. And it shows that each was retired by a
control fixed BEFORE the run, which is the part worth copying.

The bottom band is there so the figure is not merely a list of failures. What the withdrawals left
behind is a methodological finding that constrains other work instead of describing ours, and it is
larger than the claim it replaced.

ENCODING. One row per claim, read left to right as claim, control, verdict. The verdict column carries
the measured number, because a withdrawal without its number is an opinion. Orange marks what was given
up and blue what replaced it; nothing depends on colour alone, since each row is also labelled.

  python docs/figures/claims-and-controls.py

Output: docs/figures/claims-and-controls.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "claims-and-controls.png"

FIGW, FIGH = 14.0, 8.8
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, PANEL, INK, MUTED = "#fcfcfb", "#f4f4f1", "#1a1a19", "#6b6b68"
L, FW = 0.040, 0.920


def pts(n):
    return n / 72.0 / FIGH


def panel(ax, x, y, w, h, accent, fill=PANEL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=0, facecolor=fill, zorder=0))
    ax.add_patch(FancyBboxPatch((x, y), 0.0038, h, boxstyle="square,pad=0",
                                linewidth=0, facecolor=accent, zorder=1))


ROWS = [
    ("the clauses are readable",
     ["a document-frequency count with no model in it,", "ranking the raw training data"],
     "recovers 62 of the model's top-100",
     "$\\chi^2$ recovers 54; chance recovers 4",
     "interp-dataset-control"),
    ("the clauses give an analyst no\nindicator of compromise",
     ["a random control drawn at the data's own", "feature density of 1.19%"],
     "a random included literal is 94.8% negated",
     "so the model-wide 80% is input sparsity",
     "blacklist-anatomy"),
    ("the rule ensemble is more\nrobust to drift",
     ["the same comparison on a second Android corpus,", "with a third arm holding $s$ fixed"],
     "leads on 1 of 5 later years, trails on all 6",
     "and holding $s$ constant moves it under 2 points",
     "b5-apigraph"),
    ("its operating point survives\ndrift where boosters' does not",
     ["match the predicted positive rate, which needs", "only unlabelled test features"],
     "a booster then leads mean $F_1$ on both corpora",
     "Platt and isotonic are monotone, so they change nothing",
     "calibration-check"),
]


def main():
    fig = plt.figure(figsize=(FIGW, FIGH), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(L, 0.968, "Four withdrawn claims, with the control behind each",
            fontsize=19, color=INK, va="top", weight="semibold")
    ax.text(L, 0.932,
            "Every control was fixed before its run. Each claim is withdrawn in the paper with the "
            "measurement that withdrew it, and its directory is named so it can be checked.",
            fontsize=10.5, color=MUTED, va="top")

    # column anchors
    CX1, CX2, CX3 = L + 0.020, L + 0.250, L + 0.588
    hT = 0.886
    ax.text(CX1, hT, "THE CLAIM", fontsize=9.5, color=MUTED, va="top", weight="semibold")
    ax.text(CX2, hT, "THE CONTROL, FIXED IN ADVANCE", fontsize=9.5, color=MUTED, va="top",
            weight="semibold")
    ax.text(CX3, hT, "THE MEASUREMENT", fontsize=9.5, color=MUTED, va="top", weight="semibold")

    rT, rH, rGap = 0.856, 0.125, 0.013
    for i, (claim, control, verdict, detail, where) in enumerate(ROWS):
        top = rT - i * (rH + rGap)
        panel(ax, L, top - rH, FW, rH, ORANGE)
        ax.text(CX1, top - pts(20), claim, fontsize=10.8, color=INK, va="top",
                linespacing=1.45)
        ax.text(CX1, top - rH + pts(16), "WITHDRAWN", fontsize=8.6, color=ORANGE, va="top",
                weight="semibold")
        ax.text(CX1 + 0.072, top - rH + pts(16), "research/%s/" % where, fontsize=8.2,
                color=MUTED, va="top", family="monospace")
        for k, line in enumerate(control):
            ax.text(CX2, top - pts(22 + 16 * k), line, fontsize=9.6, color=MUTED, va="top")
        ax.text(CX3, top - pts(20), verdict, fontsize=11, color=INK, va="top",
                weight="semibold")
        ax.text(CX3, top - pts(41), detail, fontsize=9.5, color=MUTED, va="top")

    # ---- the hinge ------------------------------------------------------ #
    hY = rT - 4 * (rH + rGap) - pts(10)
    ax.plot([L, L + FW], [hY + pts(14), hY + pts(14)], color="#dededa", lw=1, zorder=1)
    ax.text(0.5, hY - pts(4),
            "the last withdrawal left behind something larger than the claim it replaced",
            fontsize=11.5, color=INK, ha="center", va="center")

    # ---- band: what survives -------------------------------------------- #
    sT, sH = hY - pts(26), 0.182
    panel(ax, L, sT - sH, FW, sH, BLUE, fill="#eef4fc")
    ax.text(L + 0.018, sT - pts(18), "THE SURVIVING FINDING",
            fontsize=11.5, color=BLUE, va="top", weight="semibold")
    for k, line in enumerate((
            "An oracle threshold inflates reported $F_1$ by 0.03 in distribution and by 7 to 19 points "
            "on drifted periods, peaking at 33.4,",
            "unequally across model families. Both common defaults are wrong in opposite directions: "
            "the oracle inflates, while a model's own",
            "argmax deflates by 8.6 on average, so the span between two defensible reporting choices "
            "exceeds most published effect sizes here.")):
        ax.text(L + 0.018, sT - pts(44 + 18 * k), line, fontsize=10.5, color=INK, va="top")
    ax.text(L + 0.018, sT - sH + pts(16),
            "The one instance we can verify is our own: every $F_1$ this project reported before the "
            "check was an oracle value.",
            fontsize=10, color=BLUE, va="top", weight="semibold")

    ax.text(L, 0.012,
            "No detection advantage and no interpretability advantage is claimed for the rule ensemble. "
            "The contribution is the attribution result and the threshold finding.",
            fontsize=10, color=INK, va="bottom")

    fig.savefig(OUT, dpi=150, metadata={"Software": None})
    print("wrote %s" % OUT.name)


main()

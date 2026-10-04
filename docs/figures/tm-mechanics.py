"""From a sparse binary row to a decision.

WHY THIS FIGURE EXISTS. A Fuzzy-Pattern Tsetlin machine is usually described as learning
propositional rules, which sets a reader up to expect something a human reads. What it computes is a
vote, and three properties of that vote govern every result in this repository. The vote counts how
many of a clause's literals are unsatisfied and never asks which, which is the property the exact
attribution rests on. The tolerance is small against the clause, so the clauses are precise and long
instead of short and readable. And at 1.19% density a literal demanding a feature's ABSENCE is
satisfied without cost, so clauses fill up with them: the model looks like a blacklist because the
input is sparse, and a random control at the same density is more negated than the model is.

The last row is there because the figure would otherwise read as advocacy. The machine clears the
gate it was given and holds no interpretability advantage, and both belong next to the mechanism.

ENCODING. One left-to-right pipeline, each stage carrying the measured quantity that stage produces.
The clause panel shows real structure, with an unsatisfied literal marked, since the vote is only
legible once a miss is visible. Orange marks a literal asking for absence and blue one asking for
presence, and each is labelled, so nothing depends on colour alone.

  python docs/figures/tm-mechanics.py

Output: docs/figures/tm-mechanics.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE / "tm-mechanics.png"

FIGW, FIGH = 14.0, 7.5
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


def arrow(ax, x0, y0, x1, y1, colour=INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                 linewidth=0.9, color=colour, shrinkA=0, shrinkB=0, zorder=5))


# One clause, as the decoded models actually look: mostly literals demanding absence, with the
# tolerance letting a few go unsatisfied. Signs are from research/blacklist-anatomy/.
CLAUSE = [
    ("SEND_SMS", "present", True),
    ("getDeviceId", "present", True),
    ("CAMERA", "absent", True),
    ("NFC", "absent", True),
    ("BILLING", "absent", False),       # unsatisfied on this row: one miss
    ("...", "absent", True),
]


def main():
    fig = plt.figure(figsize=(FIGW, FIGH), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(L, 0.962, "From a sparse binary row to a decision", fontsize=19, color=INK,
            va="top", weight="semibold")
    ax.text(L, 0.924,
            "Flat Fuzzy-Pattern Tsetlin machine, 20 clauses per class across both polarities. "
            "Measured on LAMDA at the configuration the gate was set at.",
            fontsize=10.5, color=MUTED, va="top")

    # ---- the pipeline, four stages on one row --------------------------------- #
    T, H = 0.888, 0.526
    GAP = 0.016
    W = (FW - 3 * GAP) / 4
    xs = [L + i * (W + GAP) for i in range(4)]
    heads = (
        (GREY, "1  THE ROW", "4,561 features, present or absent"),
        (BLUE, "2  THE LITERALS", "each feature and its negation"),
        (ORANGE, "3  THE CLAUSE", "a conjunction, scored by misses"),
        (INK, "4  THE DECISION", "a signed sum over two banks"),
    )
    for x, (accent, head, sub) in zip(xs, heads):
        panel(ax, x, T - H, W, H, accent)
        ax.text(x + 0.014, T - pts(19), head, fontsize=11.5, color=accent, va="top",
                weight="semibold")
        ax.text(x + 0.014, T - pts(36), sub, fontsize=9.2, color=MUTED, va="top", style="italic")

    for x in xs[:-1]:
        arrow(ax, x + W + 0.002, T - H / 2, x + W + GAP - 0.002, T - H / 2)

    # stage 1: the row itself
    x = xs[0]
    rng = np.random.default_rng(7)
    on = rng.random(260) < 0.0119
    cw = (W - 0.028) / 260
    ax.add_patch(Rectangle((x + 0.014, T - pts(80)), W - 0.028, pts(16), linewidth=0.6,
                           edgecolor="#d8d8d4", facecolor="#ffffff", zorder=2))
    for i in np.flatnonzero(on):
        ax.add_patch(Rectangle((x + 0.014 + i * cw, T - pts(80)), max(cw, 0.0012), pts(16),
                               linewidth=0, facecolor=GREY, zorder=3))
    for j, line in enumerate((
            "A typical application activates a few hundred",
            "of the 4,561, so the overall density is 1.19%.",
            "",
            "Nothing is binned or thresholded first: the",
            "features arrive binary from the corpus, which",
            "removes the largest confound in this area.")):
        ax.text(x + 0.014, T - pts(102 + 15 * j), line, fontsize=9.0, color=MUTED, va="top")
    ax.text(x + 0.014, T - H + pts(18), "1.19% density", fontsize=11, color=GREY, va="top",
            weight="semibold")

    # stage 2: literals
    x = xs[1]
    for j, (lab, col) in enumerate((("$x_i$", BLUE), ("$\\neg x_i$", ORANGE))):
        yy = T - pts(72 + 26 * j)
        ax.add_patch(FancyBboxPatch((x + 0.014, yy - pts(9)), W - 0.028, pts(19),
                                    boxstyle="round,pad=0,rounding_size=0.004", linewidth=0.8,
                                    facecolor="#ffffff", edgecolor=col, zorder=2))
        ax.text(x + 0.020, yy, lab, fontsize=10, color=col, va="center", zorder=3)
        ax.text(x + W - 0.020, yy, ("feature present", "feature absent")[j], fontsize=8.8,
                color=MUTED, va="center", ha="right", zorder=3)
    for j, line in enumerate((
            "4,561 features give 9,122 literals, and a",
            "clause may include either polarity of each.",
            "",
            "At 1.19% density a literal asking for a",
            "feature's ABSENCE is satisfied on almost",
            "every row, so it costs a clause nothing.")):
        ax.text(x + 0.014, T - pts(130 + 15 * j), line, fontsize=9.0, color=MUTED, va="top")
    ax.text(x + 0.014, T - H + pts(18), "9,122 literals", fontsize=11, color=BLUE, va="top",
            weight="semibold")

    # stage 3: the clause
    x = xs[2]
    for j, (name, sign, satisfied) in enumerate(CLAUSE):
        yy = T - pts(66 + 17 * j)
        col = BLUE if sign == "present" else ORANGE
        mark = "∧" if satisfied else "✘"
        ax.text(x + 0.016, yy, mark, fontsize=8.6,
                color=col if satisfied else "#c0392b", va="center", family="DejaVu Sans")
        ax.text(x + 0.030, yy, name, fontsize=8.8, color=INK if satisfied else "#c0392b",
                va="center", family="DejaVu Sans Mono")
        ax.text(x + W - 0.016, yy, sign, fontsize=8.0, color=col, va="center", ha="right")
    ax.text(x + 0.016, T - pts(176), "one literal unsatisfied, so one miss", fontsize=8.6,
            color="#c0392b", va="center")
    ax.text(x + 0.016, T - pts(200), "vote $= \\max(0,\\ LF - \\mathrm{misses})$", fontsize=11,
            color=INK, va="center")
    for j, line in enumerate((
            "The vote reads how MANY literals are",
            "unsatisfied and never which of them.",
            "The next figure builds on that.")):
        ax.text(x + 0.014, T - pts(222 + 15 * j), line, fontsize=9.0, color=MUTED, va="top")
    ax.text(x + 0.014, T - H + pts(18), "$LF = 10$, median 69 literals", fontsize=11,
            color=ORANGE, va="top", weight="semibold")

    # stage 4: the decision
    x = xs[3]
    for j, (lab, col, note) in enumerate((("10 clauses", BLUE, "evidence FOR the class"),
                                          ("10 clauses", ORANGE, "evidence AGAINST it"))):
        yy = T - pts(74 + 30 * j)
        ax.add_patch(FancyBboxPatch((x + 0.014, yy - pts(11)), W - 0.028, pts(23),
                                    boxstyle="round,pad=0,rounding_size=0.004", linewidth=0.8,
                                    facecolor="#ffffff", edgecolor=col, zorder=2))
        ax.text(x + 0.020, yy, lab, fontsize=9.4, color=col, va="center", zorder=3,
                weight="semibold")
        ax.text(x + W - 0.020, yy, note, fontsize=8.4, color=MUTED, va="center", ha="right",
                zorder=3)
    ax.text(x + 0.016, T - pts(146), "$f(x) = \\sum_{+} \\mathrm{vote} - \\sum_{-} "
                                     "\\mathrm{vote}$", fontsize=11, color=INK, va="center")
    for j, line in enumerate((
            "The sum is additive over clauses, which is",
            "the second property the attribution needs.",
            "",
            "A threshold on the sum gives the verdict.",
            "Which threshold is chosen is worth up to",
            "33 points of reported $F_1$ when drifted.")):
        ax.text(x + 0.014, T - pts(168 + 15 * j), line, fontsize=9.0, color=MUTED, va="top")
    ax.text(x + 0.014, T - H + pts(18), "additive over clauses", fontsize=11, color=INK,
            va="top", weight="semibold")

    # ---- what the mechanism costs and what it does not buy -------------------- #
    bT, bH = T - H - 0.038, 0.213
    panel(ax, L, bT - bH, FW, bH, ORANGE, fill="#faeee8")
    ax.text(L + 0.018, bT - pts(18), "THE MECHANISM BEHIND THREE MEASUREMENTS",
            fontsize=11.5, color=ORANGE, va="top", weight="semibold")

    CW = (FW - 0.072) / 3
    for k, (head, val, lines) in enumerate((
            ("the blacklist", "83.4%",
             ["of a trained model's included literals demand a",
              "feature's ABSENCE. A random control drawn at the",
              "data's own density is 94.8% negated, so the",
              "tendency belongs to the input and not the model."]),
            ("the tolerance", "14.49%",
             ["of a clause may go unsatisfied: $LF = 10$ against a",
              "median of 69 included literals. The clauses are",
              "therefore precise and long, so a decoded clause is",
              "not something a human reads."]),
            ("the detection", "95.11",
             ["IID $F_1$ at twenty clauses per class, within two",
              "points of gradient boosting in distribution. We",
              "claim no advantage in detection and none in",
              "interpretability."]))):
        x = L + 0.018 + k * (CW + 0.018)
        ax.text(x, bT - pts(44), val, fontsize=17, color=INK, va="center", weight="semibold")
        ax.text(x + 0.082, bT - pts(44), head, fontsize=10.5, color=ORANGE, va="center",
                weight="semibold")
        for j, line in enumerate(lines):
            ax.text(x, bT - pts(66 + 15 * j), line, fontsize=9.0, color=MUTED, va="top")

    ax.text(L, 0.013,
            "Sources: research/b1-gate/ for the configuration and the detection number, "
            "research/b0-sanity/ for the tolerance ratio, research/blacklist-anatomy/ for the sign "
            "composition and its control.",
            fontsize=9.2, color=MUTED, va="bottom")

    fig.savefig(OUT, dpi=150, metadata={"Software": None})
    print("wrote %s" % OUT.name)


main()

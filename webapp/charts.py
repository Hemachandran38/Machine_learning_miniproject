# =============================================================================
# charts.py — Probability bar chart for the light-theme UI.
# =============================================================================

from __future__ import annotations

import numpy as np


# Light theme palette
_BG       = "#FFFFFF"   # figure background
_AX_BG    = "#F8FAFC"   # axes background
_GRID     = "#E2E8F0"   # grid lines
_TEXT     = "#1E293B"   # axis labels / tick text
_HIGHLIGHT = "#2563EB"  # predicted bar
_MUTED     = "#CBD5E1"  # other bars
_BORDER    = "#E2E8F0"  # spine colour


def make_probability_chart(probabilities: list, predicted_digit: int) -> plt.Figure:
    """
    Vertical bar chart showing per-class probabilities.
    Predicted class is highlighted in blue; others are grey.

    Parameters
    ----------
    probabilities   : list of 10 floats from softmax output
    predicted_digit : int — bar to highlight

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    digits = list(range(10))
    probs  = [p * 100 for p in probabilities]
    colours = [_HIGHLIGHT if d == predicted_digit else _MUTED for d in digits]

    fig, ax = plt.subplots(figsize=(8, 3.4))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_AX_BG)

    bars = ax.bar(digits, probs, color=colours, width=0.62,
                  edgecolor=_BG, linewidth=1.2, zorder=3)

    # Value labels above each visible bar
    for bar, prob in zip(bars, probs):
        if prob >= 0.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{prob:.1f}%",
                ha="center", va="bottom",
                fontsize=7.5, color=_TEXT, fontweight="600"
            )

    # Axes styling
    ax.set_xticks(digits)
    ax.set_xticklabels([str(d) for d in digits], fontsize=11, color=_TEXT, fontweight="600")
    ax.set_ylabel("Probability (%)", color="#64748B", fontsize=9.5)
    ax.set_xlabel("Digit class",     color="#64748B", fontsize=9.5)
    ax.set_ylim(0, max(probs) * 1.30 + 5)
    ax.tick_params(colors="#64748B", length=3)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor(_BORDER)

    # Legend
    predicted_patch = mpatches.Patch(facecolor=_HIGHLIGHT, edgecolor="none",
                                     label=f"Predicted: {predicted_digit}")
    others_patch    = mpatches.Patch(facecolor=_MUTED,     edgecolor="none",
                                     label="Other classes")
    ax.legend(
        handles=[predicted_patch, others_patch],
        loc="upper right", fontsize=8.5,
        facecolor=_BG, edgecolor=_BORDER,
        labelcolor=_TEXT, framealpha=1,
    )

    plt.tight_layout(pad=0.6)
    return fig

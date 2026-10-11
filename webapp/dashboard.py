# =============================================================================
# dashboard.py — Load and present REAL Phase 1 evaluation metrics.
#
# IMPORTANT DESIGN RULE:
#   All numbers shown in the dashboard come from actually running the model
#   against the MNIST test set.  Nothing is hardcoded or fabricated.
#
# HOW IT WORKS:
#   1. We check if a cached metrics file (data/eval_metrics.json) exists.
#   2. If it does, we load it — fast, no model needed.
#   3. If it doesn't, we compute it now (runs model on all 10,000 test images).
#   4. Results are cached to disk so the computation only happens once.
#
# The confusion matrix PNG from Phase 1 (outputs/confusion_matrix.png) is
# reused directly — no recomputation needed.
# =============================================================================

import os
import json
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from config import (EVAL_METRICS_PATH as METRICS_PATH,
                    CONFUSION_MATRIX_PATH as CM_IMG_PATH,
                    TRAINING_CURVES_PATH as CURVES_PATH)

DIGIT_NAMES   = [str(i) for i in range(10)]


# =============================================================================
# Metrics computation (runs once, then cached)
# =============================================================================

def compute_and_cache_metrics(model) -> dict:
    """
    Run the trained model against the full MNIST test set (10,000 images)
    and compute accuracy, loss, per-class precision/recall/F1, and the
    confusion matrix.

    Results are saved to data/eval_metrics.json so this only runs once.

    Parameters
    ----------
    model : loaded tf.keras.Model

    Returns
    -------
    dict — see _build_metrics_dict() for structure
    """
    import tensorflow as tf
    from sklearn.metrics import (
        classification_report, confusion_matrix
    )

    # Load MNIST test set (uses cached download from Phase 1)
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Evaluate loss + accuracy
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # Get predictions
    probs  = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # Per-class metrics via sklearn
    report = classification_report(
        y_test, y_pred,
        target_names=DIGIT_NAMES,
        output_dict=True
    )

    # Confusion matrix (raw counts)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "test_accuracy" : round(float(test_acc),  4),
        "test_loss"     : round(float(test_loss), 4),
        "per_class"     : {
            str(d): {
                "precision": round(report[str(d)]["precision"], 4),
                "recall"   : round(report[str(d)]["recall"],    4),
                "f1"       : round(report[str(d)]["f1-score"],  4),
                "support"  : int(report[str(d)]["support"]),
            }
            for d in range(10)
        },
        "macro_avg": {
            "precision": round(report["macro avg"]["precision"], 4),
            "recall"   : round(report["macro avg"]["recall"],    4),
            "f1"       : round(report["macro avg"]["f1-score"],  4),
        },
        "confusion_matrix": cm,
    }

    # Cache to disk
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


@st.cache_data(show_spinner="Computing model metrics …")
def load_metrics(_model) -> dict:
    """
    Return cached metrics if available; otherwise compute them.
    Uses st.cache_data so the result is cached in memory for the session.

    The leading underscore on _model tells Streamlit not to try to
    hash the model object (it's not hashable).
    """
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return compute_and_cache_metrics(_model)


def load_cached_metrics() -> Optional[dict]:
    """
    Read the cached metrics JSON if it exists, WITHOUT triggering a full
    evaluation of the model.  Used for lightweight values (e.g. the test
    accuracy shown in the hero banner and footer).

    Returns
    -------
    dict | None — metrics dict, or None if no cached file is present yet.
    """
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


# =============================================================================
# Dashboard rendering helpers
# =============================================================================

def render_key_metrics(metrics: dict) -> None:
    """Render the four headline KPI boxes at the top of the dashboard."""
    acc  = metrics["test_accuracy"]  * 100
    loss = metrics["test_loss"]
    mac  = metrics["macro_avg"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy",   f"{acc:.2f}%",           help="Correct predictions on 10,000 MNIST test images")
    c2.metric("Test Loss",       f"{loss:.4f}",           help="Cross-entropy loss on the test set (lower = better)")
    c3.metric("Macro F1-Score",  f"{mac['f1']:.4f}",      help="Average F1-score across all 10 digit classes")
    c4.metric("Macro Precision", f"{mac['precision']:.4f}", help="Average precision across all 10 digit classes")


def render_per_class_table(metrics: dict) -> None:
    """Render a styled per-class precision / recall / F1 table."""
    import pandas as pd

    rows = []
    for d in range(10):
        pc = metrics["per_class"][str(d)]
        rows.append({
            "Digit"    : f"  {d}",
            "Precision": f"{pc['precision']*100:.2f}%",
            "Recall"   : f"{pc['recall']*100:.2f}%",
            "F1-Score" : f"{pc['f1']*100:.2f}%",
            "Test Samples": pc["support"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Digit"       : st.column_config.TextColumn("Digit",        width="small"),
            "Precision"   : st.column_config.TextColumn("Precision",    width="small"),
            "Recall"      : st.column_config.TextColumn("Recall",       width="small"),
            "F1-Score"    : st.column_config.TextColumn("F1-Score",     width="small"),
            "Test Samples": st.column_config.NumberColumn("Test Samples", width="small"),
        }
    )


def render_f1_bar_chart(metrics: dict) -> None:
    """Plot a bar chart of F1-scores per digit class (light theme)."""
    digits = list(range(10))
    f1s    = [metrics["per_class"][str(d)]["f1"] * 100 for d in digits]

    _BG   = "#FFFFFF"
    _AXBG = "#F8FAFC"
    _GRID = "#E2E8F0"
    _TEXT = "#1E293B"

    colours = ["#2563EB" if f >= 99 else "#D97706" if f >= 97 else "#DC2626"
               for f in f1s]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_AXBG)

    bars = ax.bar(digits, f1s, color=colours, width=0.6,
                  edgecolor=_BG, linewidth=1.0, zorder=3)

    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{f1:.1f}%", ha="center", va="bottom",
                fontsize=8, color=_TEXT, fontweight="600")

    ax.set_xticks(digits)
    ax.set_xticklabels([str(d) for d in digits], color=_TEXT, fontsize=11, fontweight="600")
    ax.set_ylabel("F1-Score (%)", color="#64748B", fontsize=10)
    ax.set_xlabel("Digit class",  color="#64748B", fontsize=10)
    ax.set_ylim(min(f1s) - 2, 102)
    ax.tick_params(colors="#64748B", length=3)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor(_GRID)
    ax.set_title("F1-Score per Digit Class", color=_TEXT, fontsize=12,
                 pad=10, fontweight="600")

    patches = [
        mpatches.Patch(facecolor="#2563EB", edgecolor="none", label="≥ 99% (excellent)"),
        mpatches.Patch(facecolor="#D97706", edgecolor="none", label="97–99% (good)"),
        mpatches.Patch(facecolor="#DC2626", edgecolor="none", label="< 97% (needs attention)"),
    ]
    ax.legend(handles=patches, fontsize=8, facecolor=_BG,
              edgecolor=_GRID, labelcolor=_TEXT, framealpha=1)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_confusion_matrix(metrics: dict) -> None:
    """
    Render the confusion matrix.
    Prefers the PNG saved during Phase 1 evaluate.py.
    Falls back to rendering it from the cached JSON data.
    """
    if os.path.exists(CM_IMG_PATH):
        from PIL import Image as PILImage
        img = PILImage.open(CM_IMG_PATH)
        st.image(img, caption="Confusion Matrix — MNIST Test Set (10,000 images)")
    else:
        # Render from JSON data as fallback (light theme)
        cm      = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(range(10), color="#1E293B")
        ax.set_yticklabels(range(10), color="#1E293B")
        ax.set_xlabel("Predicted", color="#1E293B", fontsize=11)
        ax.set_ylabel("True label", color="#1E293B", fontsize=11)
        for i in range(10):
            for j in range(10):
                color = "white" if cm[i, j] > cm.max() / 2 else "#1E293B"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=8, color=color)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def render_training_curves() -> None:
    """Display the training curves PNG saved during Phase 1."""
    if os.path.exists(CURVES_PATH):
        from PIL import Image as PILImage
        img = PILImage.open(CURVES_PATH)
        st.image(img, caption="Training & Validation Accuracy/Loss (Phase 1)")
    else:
        st.info("Training curves not found. Run `python main.py` to generate them.")

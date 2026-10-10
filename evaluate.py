# =============================================================================
# evaluate.py — Evaluate the saved model with full metrics and visualisations.
#
# Run:  python evaluate.py
# (run AFTER train.py has completed successfully)
#
# Produces:
#   outputs/confusion_matrix.png
#   outputs/sample_predictions.png
#   outputs/training_curves.png   (requires history — see note below)
# =============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from config      import MODEL_PATH, OUTPUTS_DIR
from data_loader import load_and_preprocess


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model_safe():
    """Load the saved Keras model with a helpful error message if not found."""
    if not os.path.exists(MODEL_PATH):
        sys.exit(
            f"\n[ERROR] Model not found at:\n  {MODEL_PATH}\n"
            "Please run  python train.py  first.\n"
        )
    print(f"Loading model from {MODEL_PATH} …")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.\n")
    return model


def evaluate_on_test_set(model, x_test, y_test):
    """Print test loss and accuracy."""
    print("=" * 55)
    print("  Test Set Evaluation")
    print("=" * 55)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test accuracy : {acc * 100:.2f} %")
    print(f"  Test loss     : {loss:.4f}")
    print("=" * 55 + "\n")
    return acc, loss


def get_predictions(model, x_test):
    """Return predicted class labels for every test image."""
    # model.predict() returns a probability vector of shape (N, 10).
    # np.argmax picks the index with the highest probability → predicted digit.
    probs  = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    return y_pred, probs


def plot_confusion_matrix(y_test, y_pred):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=list(range(10)))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)

    ax.set_title("Confusion Matrix — Digit Recognition CNN", fontsize=14, pad=15)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",      fontsize=12)

    # Annotate the diagonal (correct predictions) in a distinct colour
    for i in range(10):
        disp.text_[i, i].set_color("white")
        disp.text_[i, i].set_fontweight("bold")

    path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {path}")

    return cm


def print_classification_report(y_test, y_pred):
    """Print precision, recall, and F1-score for every digit class."""
    print("Classification Report")
    print("─" * 55)
    # target_names gives human-readable labels ("Digit 0", "Digit 1", …)
    report = classification_report(
        y_test, y_pred,
        target_names=[f"Digit {i}" for i in range(10)]
    )
    print(report)


def plot_sample_predictions(x_test, y_test, y_pred, probs, n_correct=8, n_wrong=8):
    """
    Show a grid of sample images with their true vs predicted labels.

    Correct predictions are shown with a green border.
    Wrong   predictions are shown with a red border.
    """
    correct_idx = np.where(y_pred == y_test)[0]
    wrong_idx   = np.where(y_pred != y_test)[0]

    # Pick random samples
    rng = np.random.default_rng(42)
    c_samples = rng.choice(correct_idx, size=min(n_correct, len(correct_idx)), replace=False)
    w_samples = rng.choice(wrong_idx,   size=min(n_wrong,   len(wrong_idx)),   replace=False)

    total = n_correct + n_wrong
    fig, axes = plt.subplots(2, max(n_correct, n_wrong), figsize=(18, 5))
    fig.suptitle("Sample Predictions  (Green = Correct  |  Red = Wrong)",
                 fontsize=14, y=1.02)

    def draw_row(ax_row, indices, label_colour):
        for col, idx in enumerate(indices):
            ax = ax_row[col]
            ax.imshow(x_test[idx].squeeze(), cmap="gray")
            confidence = probs[idx][y_pred[idx]] * 100
            ax.set_title(
                f"True: {y_test[idx]}\nPred: {y_pred[idx]} ({confidence:.0f}%)",
                fontsize=8
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(label_colour)
                spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])
        # Hide unused axes in the row
        for col in range(len(indices), ax_row.shape[0]):
            ax_row[col].axis("off")

    draw_row(axes[0], c_samples, "green")
    draw_row(axes[1], w_samples, "red")

    axes[0][0].set_ylabel("Correct", fontsize=11, color="green", labelpad=8)
    axes[1][0].set_ylabel("Wrong",   fontsize=11, color="red",   labelpad=8)

    path = os.path.join(OUTPUTS_DIR, "sample_predictions.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample predictions saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate():
    # 1. Load data
    _, _, x_test, y_test = load_and_preprocess()

    # 2. Load model
    model = load_model_safe()

    # 3. Test set accuracy / loss
    evaluate_on_test_set(model, x_test, y_test)

    # 4. Predictions
    y_pred, probs = get_predictions(model, x_test)

    # 5. Confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # 6. Classification report
    print_classification_report(y_test, y_pred)

    # 7. Sample correct / incorrect images
    plot_sample_predictions(x_test, y_test, y_pred, probs)

    print("\nEvaluation complete.  Check the  outputs/  folder for all charts.")


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        raise

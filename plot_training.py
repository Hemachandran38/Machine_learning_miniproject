# =============================================================================
# plot_training.py — Plot and save training vs validation accuracy/loss curves.
#
# This is called automatically from main.py after training, but you can also
# import and call plot_history() directly if you already have a history object.
#
# Output:  outputs/training_curves.png
# =============================================================================

import os
import matplotlib.pyplot as plt
from config import OUTPUTS_DIR


def plot_history(history):
    """
    Plot four sub-charts:
      1. Training Accuracy vs Validation Accuracy
      2. Training Loss    vs Validation Loss

    A large gap between training and validation curves (over-fitting) would
    indicate the model has memorised the training data without generalising.
    Ideally both curves should be close together and both improving.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        The object returned by model.fit().
    """
    acc      = history.history["accuracy"]
    val_acc  = history.history["val_accuracy"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs   = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — Digit Recognition CNN", fontsize=14)

    # ── Accuracy plot ─────────────────────────────────────────────────────────
    ax1.plot(epochs, acc,     "b-o", markersize=4, label="Training accuracy")
    ax1.plot(epochs, val_acc, "r-o", markersize=4, label="Validation accuracy")
    ax1.set_title("Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0.9, 1.01])          # zoom in to the interesting range
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Annotate the best validation accuracy
    best_val_epoch = val_acc.index(max(val_acc)) + 1
    ax1.axvline(x=best_val_epoch, color="gray", linestyle="--", alpha=0.5,
                label=f"Best val epoch ({best_val_epoch})")
    ax1.annotate(
        f"Best: {max(val_acc)*100:.2f}%",
        xy=(best_val_epoch, max(val_acc)),
        xytext=(best_val_epoch + 0.5, max(val_acc) - 0.01),
        fontsize=8, color="darkred"
    )

    # ── Loss plot ─────────────────────────────────────────────────────────────
    ax2.plot(epochs, loss,     "b-o", markersize=4, label="Training loss")
    ax2.plot(epochs, val_loss, "r-o", markersize=4, label="Validation loss")
    ax2.set_title("Loss over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    path = os.path.join(OUTPUTS_DIR, "training_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved → {path}")

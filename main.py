# =============================================================================
# main.py — Single entry point for the Digit Recognition project.
#
# Run:  python main.py
#
# This script ties everything together:
#   1. Load and preprocess data
#   2. Build the CNN
#   3. Train the CNN
#   4. Plot training curves
#   5. Evaluate with confusion matrix, report, and sample predictions
# =============================================================================

import os
import sys
import tensorflow as tf

# ── Suppress TensorFlow info/warning logs (keeps the console clean) ───────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from config          import EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED, MODEL_PATH
from data_loader     import load_and_preprocess, describe_dataset
from model           import build_cnn, summarise_model
from train           import get_callbacks
from plot_training   import plot_history
from evaluate        import (
    evaluate_on_test_set,
    get_predictions,
    plot_confusion_matrix,
    print_classification_report,
    plot_sample_predictions,
)


def main():
    print("\n" + "=" * 65)
    print("  Handwritten Digit Recognition — Phase 1")
    print("=" * 65 + "\n")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("Step 1/5 — Loading and preprocessing MNIST …")
    x_train, y_train, x_test, y_test = load_and_preprocess()
    describe_dataset(x_train, y_train, x_test, y_test)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("Step 2/5 — Building the CNN model …")
    tf.random.set_seed(RANDOM_SEED)
    model = build_cnn()
    summarise_model(model)

    # ── 3. Training ───────────────────────────────────────────────────────────
    print(f"Step 3/5 — Training (epochs={EPOCHS}, batch_size={BATCH_SIZE}) …\n")
    history = model.fit(
        x_train, y_train,
        epochs           = EPOCHS,
        batch_size       = BATCH_SIZE,
        validation_split = VALIDATION_SPLIT,
        callbacks        = get_callbacks(),
        verbose          = 1
    )

    # ── 4. Training curves ────────────────────────────────────────────────────
    print("\nStep 4/5 — Plotting training curves …")
    plot_history(history)

    # ── 5. Full evaluation ────────────────────────────────────────────────────
    print("\nStep 5/5 — Evaluating on the test set …\n")
    evaluate_on_test_set(model, x_test, y_test)

    y_pred, probs = get_predictions(model, x_test)
    plot_confusion_matrix(y_test, y_pred)
    print_classification_report(y_test, y_pred)
    plot_sample_predictions(x_test, y_test, y_pred, probs)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Phase 1 Complete!")
    print("=" * 65)
    print(f"\n  Trained model : {MODEL_PATH}")
    print(f"  Output charts : outputs/")
    print(
        "\n  To re-evaluate without retraining:\n"
        "    python evaluate.py\n"
        "\n  To retrain from scratch:\n"
        "    python train.py\n"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Run interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise

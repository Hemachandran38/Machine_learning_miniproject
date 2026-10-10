# =============================================================================
# train.py — Train the CNN on MNIST and save the trained model.
#
# Run:  python train.py
# =============================================================================

import os
import tensorflow as tf

from config      import MODEL_PATH, EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED
from data_loader import load_and_preprocess, describe_dataset
from model       import build_cnn, summarise_model


def get_callbacks():
    """
    Return a list of Keras callbacks that improve and monitor training.

    EarlyStopping : stops training automatically when the model stops
                    improving on the validation set (avoids wasting time
                    and prevents over-fitting).
    ModelCheckpoint: saves the best version of the model seen so far —
                    not just the last epoch.
    ReduceLROnPlateau: if the validation loss stops improving for 3 epochs,
                    the learning rate is halved.  This helps the model
                    fine-tune instead of overshooting.
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor   = "val_loss",
        patience  = 5,           # wait 5 epochs before stopping
        restore_best_weights = True,
        verbose   = 1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath       = MODEL_PATH,
        monitor        = "val_accuracy",
        save_best_only = True,
        verbose        = 1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,          # multiply LR by this when triggered
        patience = 3,
        min_lr   = 1e-6,
        verbose  = 1
    )

    return [early_stop, checkpoint, reduce_lr]


def train():
    """Full training pipeline: load → build → train → save."""

    # ── 0. Reproducibility ────────────────────────────────────────────────────
    tf.random.set_seed(RANDOM_SEED)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    x_train, y_train, x_test, y_test = load_and_preprocess()
    describe_dataset(x_train, y_train, x_test, y_test)

    # ── 2. Build model ────────────────────────────────────────────────────────
    model = build_cnn()
    summarise_model(model)

    # ── 3. Train ───────────────────────────────────────────────────────────────
    print(f"Starting training  (epochs={EPOCHS}, batch_size={BATCH_SIZE}) …\n")

    history = model.fit(
        x_train, y_train,
        epochs           = EPOCHS,
        batch_size       = BATCH_SIZE,
        validation_split = VALIDATION_SPLIT,  # reserve 10 % for validation
        callbacks        = get_callbacks(),
        verbose          = 1
    )

    # ── 4. Quick evaluation on the hold-out test set ──────────────────────────
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy : {test_acc * 100:.2f} %")
    print(f"Test loss     : {test_loss:.4f}")

    # ── 5. The best model was already saved by ModelCheckpoint.
    #       We also save once more explicitly to be safe.
    model.save(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    return history, model, x_test, y_test


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        history, model, x_test, y_test = train()
        print("\nTraining complete!  Run  python evaluate.py  to see full metrics.")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise

# =============================================================================
# data_loader.py — Load, preprocess, and describe the MNIST dataset.
#
# MNIST contains 70,000 handwritten digit images:
#   • 60,000 training images  — used to teach the model
#   •  10,000 testing images  — used to measure how well it learned
#
# Each image is 28×28 pixels, grayscale, with a label (0–9).
# =============================================================================

import json
import numpy as np
import tensorflow as tf

from config import (
    IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS,
    NUM_CLASSES, PREPROC_PATH, RANDOM_SEED
)


def load_and_preprocess():
    """
    Download (first run only) and preprocess the MNIST dataset.

    Returns
    -------
    x_train : np.ndarray, shape (60000, 28, 28, 1), float32 in [0, 1]
    y_train : np.ndarray, shape (60000,), int  — class labels 0–9
    x_test  : np.ndarray, shape (10000, 28, 28, 1), float32 in [0, 1]
    y_test  : np.ndarray, shape (10000,), int  — class labels 0–9
    """
    # ── 1. Set random seed for reproducibility ────────────────────────────────
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # ── 2. Download / load MNIST ──────────────────────────────────────────────
    # Keras downloads MNIST automatically on the first call and caches it
    # locally at ~/.keras/datasets/mnist.npz
    print("Loading MNIST dataset …")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print(f"  Training images : {x_train.shape}  |  labels: {y_train.shape}")
    print(f"  Testing  images : {x_test.shape}   |  labels: {y_test.shape}")
    print(f"  Pixel value range BEFORE normalisation: [{x_train.min()}, {x_train.max()}]")

    # ── 3. Normalise pixel values: 0–255 → 0.0–1.0 ───────────────────────────
    # Dividing every pixel by 255.0 scales all values into [0, 1].
    # This helps gradient-based optimisers converge faster and more stably.
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # ── 4. Reshape: add the channel dimension ─────────────────────────────────
    # Original shape: (N, 28, 28)
    # Required shape: (N, 28, 28, 1)  ← the "1" means 1 grayscale channel
    # A colour image would have shape (N, 28, 28, 3) for R, G, B channels.
    x_train = x_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)
    x_test  = x_test.reshape(-1,  IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)

    print(f"  Pixel value range AFTER  normalisation: [{x_train.min():.1f}, {x_train.max():.1f}]")
    print(f"  Reshaped training images: {x_train.shape}")
    print(f"  Reshaped testing  images: {x_test.shape}")

    # ── 5. Labels ─────────────────────────────────────────────────────────────
    # Labels are already integers (0–9).  We keep them as-is because we will
    # use sparse_categorical_crossentropy loss which does NOT need one-hot
    # encoding.  (One-hot: 3 → [0,0,0,1,0,0,0,0,0,0]; we skip that step.)
    print(f"  Unique labels : {np.unique(y_train)}")

    # ── 6. Save preprocessing configuration ──────────────────────────────────
    # This JSON records exactly how we transformed the data.  When we later
    # predict new images, we apply the same transformation so results are
    # consistent.
    preproc_config = {
        "img_height"   : IMG_HEIGHT,
        "img_width"    : IMG_WIDTH,
        "num_channels" : NUM_CHANNELS,
        "num_classes"  : NUM_CLASSES,
        "normalisation": "divide_by_255",
        "pixel_min"    : 0.0,
        "pixel_max"    : 1.0,
        "label_type"   : "integer_0_to_9"
    }

    with open(PREPROC_PATH, "w") as f:
        json.dump(preproc_config, f, indent=4)

    print(f"  Preprocessing config saved → {PREPROC_PATH}")
    print("Dataset loading complete.\n")

    return x_train, y_train, x_test, y_test


def describe_dataset(x_train, y_train, x_test, y_test):
    """Print a summary table of the dataset splits and class distribution."""
    import collections

    print("=" * 55)
    print("  MNIST Dataset Summary")
    print("=" * 55)
    print(f"  Training samples   : {len(x_train):,}")
    print(f"  Testing  samples   : {len(x_test):,}")
    print(f"  Image shape        : {x_train.shape[1:]}")
    print(f"  Pixel value range  : [{x_train.min():.2f}, {x_train.max():.2f}]")
    print()

    train_counts = collections.Counter(y_train.tolist())
    print("  Training samples per digit:")
    for digit in range(10):
        bar = "█" * (train_counts[digit] // 200)
        print(f"    Digit {digit}: {train_counts[digit]:,}  {bar}")
    print("=" * 55 + "\n")

# =============================================================================
# model.py — Define the CNN architecture for digit recognition.
#
# Architecture overview
# ─────────────────────
# Input (28×28×1)
#   → Conv Block 1 : Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout
#   → Conv Block 2 : Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout
#   → Conv Block 3 : Conv2D(64) → BatchNorm → ReLU → Dropout
#   → Flatten
#   → Dense(128) → BatchNorm → ReLU → Dropout
#   → Dense(10)  → Softmax  (output probabilities for digits 0–9)
# =============================================================================

import tensorflow as tf
from tensorflow.keras import layers, models

from config import IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES


def build_cnn():
    """
    Build and compile the CNN model.

    Returns
    -------
    model : tf.keras.Model  (compiled, ready to train)
    """

    # tf.keras.Sequential lets us stack layers one after another, like LEGO bricks.
    model = models.Sequential(name="digit_cnn")

    # ── Input layer ───────────────────────────────────────────────────────────
    # Tells the model what shape to expect.
    # Shape = (height, width, channels) = (28, 28, 1).
    model.add(layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)))

    # =========================================================================
    # CONV BLOCK 1
    # =========================================================================
    # Conv2D(32, (3,3)):
    #   • 32 filters — each filter is a 3×3 "detector" that scans the image
    #     looking for a specific low-level feature (edge, dot, curve …)
    #   • padding="same" keeps the output the same size as the input (28×28)
    #   • activation="relu" — Rectified Linear Unit: replaces negative values
    #     with 0.  Simple but very effective.  Think of it as "only pass
    #     positive signals forward."
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                            name="conv1"))

    # BatchNormalization stabilises training by keeping layer outputs
    # consistently scaled.  It often allows higher learning rates.
    model.add(layers.BatchNormalization(name="bn1"))

    # MaxPooling2D(2,2):
    #   Divides the 28×28 feature map into 2×2 blocks and keeps the
    #   MAXIMUM value from each block → output becomes 14×14.
    #   This makes the network translation-tolerant (a "7" slightly shifted
    #   still looks like a "7" after pooling).
    model.add(layers.MaxPooling2D((2, 2), name="pool1"))

    # Dropout(0.25):
    #   During each training step, randomly set 25 % of neurons to 0.
    #   This prevents over-fitting — the model can't just memorise
    #   the training data; it has to learn robust features.
    model.add(layers.Dropout(0.25, name="drop1"))

    # =========================================================================
    # CONV BLOCK 2
    # =========================================================================
    # 64 filters now — the network learns more complex patterns in deeper layers.
    # After a second MaxPool the spatial size goes from 14×14 → 7×7.
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                            name="conv2"))
    model.add(layers.BatchNormalization(name="bn2"))
    model.add(layers.MaxPooling2D((2, 2), name="pool2"))
    model.add(layers.Dropout(0.25, name="drop2"))

    # =========================================================================
    # CONV BLOCK 3
    # =========================================================================
    # An additional convolution without pooling lets the model extract even
    # richer features while keeping the 7×7 spatial resolution.
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                            name="conv3"))
    model.add(layers.BatchNormalization(name="bn3"))
    model.add(layers.Dropout(0.25, name="drop3"))

    # =========================================================================
    # FLATTEN
    # =========================================================================
    # The output of the last Conv layer is a 3D tensor: (7, 7, 64).
    # Flatten "unrolls" it into a 1D vector of 7×7×64 = 3136 numbers.
    # This is necessary before the fully-connected Dense layers.
    model.add(layers.Flatten(name="flatten"))

    # =========================================================================
    # FULLY CONNECTED (DENSE) LAYERS
    # =========================================================================
    # Dense(128): every one of the 3136 numbers is connected to every one of
    # 128 neurons.  This is where the model combines all the detected features
    # to form a "decision."
    model.add(layers.Dense(128, activation="relu", name="fc1"))
    model.add(layers.BatchNormalization(name="bn4"))
    model.add(layers.Dropout(0.5, name="drop4"))  # higher dropout near the output

    # =========================================================================
    # OUTPUT LAYER
    # =========================================================================
    # Dense(10): one neuron per digit class (0–9).
    # Softmax converts the raw scores into probabilities that sum to 1.0.
    # Example output: [0.01, 0.02, 0.01, 0.90, 0.01, …] → predicted digit: 3
    model.add(layers.Dense(NUM_CLASSES, activation="softmax", name="output"))

    # ── Compile ───────────────────────────────────────────────────────────────
    # optimizer="adam"  — Adam adjusts the learning rate automatically.
    #                     It is the go-to choice for most deep learning tasks.
    # loss="sparse_categorical_crossentropy"
    #                   — Measures how wrong the model's probability
    #                     distribution is compared to the true label.
    #                     "sparse" means labels are integers (not one-hot).
    # metrics=["accuracy"] — We also track the percentage of correct guesses.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def summarise_model(model):
    """Print the layer-by-layer summary including parameter counts."""
    print("\n" + "=" * 65)
    print("  CNN Model Architecture")
    print("=" * 65)
    model.summary()
    total = model.count_params()
    print(f"\n  Total trainable parameters: {total:,}")
    print("=" * 65 + "\n")

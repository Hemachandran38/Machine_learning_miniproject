# =============================================================================
# predictor.py — Load the trained CNN model and run inference.
#
# Key design decision: the model is loaded ONCE using Streamlit's caching
# (@st.cache_resource).  Without caching, the model would reload from disk
# on every button click or canvas update, making the app feel very slow.
#
# @st.cache_resource is the right decorator here because:
#   • The model is a "resource" (large object, expensive to create).
#   • It is shared across all users / sessions.
#   • It never changes while the app is running.
# =============================================================================

import os
import sys
import numpy as np
import streamlit as st
import tensorflow as tf

# ── Suppress TensorFlow info/warning messages in the console ──────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ── Model path (mirrors Phase 1 config.py) ────────────────────────────────────
# We compute the path relative to this file so the app works from any
# working directory.
_WEBAPP_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_WEBAPP_DIR)
MODEL_PATH   = os.path.join(_PROJECT_DIR, "models", "digit_recognizer.keras")

NUM_CLASSES  = 10   # digits 0–9


@st.cache_resource(show_spinner="Loading digit recognition model …")
def load_model() -> tf.keras.Model:
    """
    Load the trained Keras model from disk.

    This function is decorated with @st.cache_resource, meaning Streamlit
    runs it only once per server session and then reuses the cached result.

    Returns
    -------
    model : tf.keras.Model — ready for inference (no recompile needed)

    Raises
    ------
    SystemExit if the .keras file does not exist (gives a clear user message).
    """
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"**Model not found.**\n\n"
            f"Expected path: `{MODEL_PATH}`\n\n"
            "Please run **`python main.py`** (or **`python train.py`**) "
            "from the `digit_recognition/` folder first to train and save the model."
        )
        st.stop()   # Halts Streamlit rendering cleanly (no traceback shown to user)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def predict(model: tf.keras.Model, tensor: np.ndarray) -> dict:
    """
    Run the model on a preprocessed image tensor and return structured results.

    Parameters
    ----------
    model  : loaded Keras model
    tensor : np.ndarray, shape (1, 28, 28, 1), float32 in [0, 1]

    Returns
    -------
    dict with keys:
        "digit"       : int   — predicted digit (0–9)
        "confidence"  : float — probability of the predicted class (0.0–1.0)
        "probabilities": list[float] — probabilities for all 10 classes
    """
    # model.predict() returns a 2D array of shape (1, 10).
    # Each value is the probability for one digit class (they sum to 1.0).
    probs     = model.predict(tensor, verbose=0)[0]   # shape → (10,)
    digit     = int(np.argmax(probs))                 # index of highest probability
    confidence = float(probs[digit])

    return {
        "digit"        : digit,
        "confidence"   : confidence,
        "probabilities": probs.tolist(),
    }

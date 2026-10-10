# =============================================================================
# config.py — Central configuration file for the digit recognition project.
#
# Keeping all settings here means you only need to change one file to
# experiment with different training parameters.
# =============================================================================

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR  = os.path.join(BASE_DIR, "outputs")
DATA_DIR     = os.path.join(BASE_DIR, "data")

MODEL_PATH   = os.path.join(MODELS_DIR, "digit_recognizer.keras")
PREPROC_PATH = os.path.join(DATA_DIR,   "preprocessing_config.json")

# ── Image settings ─────────────────────────────────────────────────────────────
# MNIST images are 28×28 pixels in grayscale (1 channel).
IMG_HEIGHT  = 28
IMG_WIDTH   = 28
NUM_CHANNELS = 1          # 1 = grayscale, 3 = RGB colour
NUM_CLASSES  = 10         # digits 0–9

# ── Training hyperparameters ───────────────────────────────────────────────────
# EPOCHS     — how many times the model sees the full training dataset.
# BATCH_SIZE — how many images are processed together in one step.
#              Smaller = slower but sometimes more accurate.
# VALIDATION_SPLIT — fraction of training data reserved for validation during training.
EPOCHS           = 15
BATCH_SIZE       = 64
VALIDATION_SPLIT = 0.1     # 10 % of training data used for validation

# ── Random seed ───────────────────────────────────────────────────────────────
# Fixing the seed makes results reproducible across runs.
RANDOM_SEED = 42

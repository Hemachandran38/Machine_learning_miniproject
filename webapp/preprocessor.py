# =============================================================================
# preprocessor.py — Convert a raw canvas image into the exact tensor format
#                   that the Phase 1 CNN model expects.
#
# Phase 1 training facts (from data_loader.py + preprocessing_config.json):
#   • Input shape  : (1, 28, 28, 1)
#   • Pixel range  : float32 in [0.0, 1.0]
#   • Polarity     : WHITE digit on BLACK background  (MNIST convention)
#   • Normalisation: divide by 255
#
# The streamlit-drawable-canvas delivers a RGBA NumPy array (H×W×4).
# This module turns that into the correct tensor in six steps.
# =============================================================================

import numpy as np
import streamlit as st
from typing import List, Tuple
from PIL import Image, ImageOps, ImageFilter

from config import IMG_HEIGHT, MNIST_PADDING


# ── Constants (imported from config.py so training and inference always agree) ─
IMG_SIZE = IMG_HEIGHT     # final image is 28×28
PADDING  = MNIST_PADDING  # black border kept around the digit after centering


def is_canvas_empty(image_data: np.ndarray, threshold: int = 10) -> bool:
    """
    Return True if the user has not drawn anything on the canvas yet.

    The canvas starts as a blank black image (all zeros).  We check whether
    any pixel is bright enough to represent a drawn stroke.

    Parameters
    ----------
    image_data : RGBA NumPy array from streamlit-drawable-canvas, shape (H, W, 4)
    threshold  : minimum pixel value (0–255) to count as "drawn"
    """
    if image_data is None:
        return True

    # Look at just the RGB channels; ignore alpha.
    rgb = image_data[:, :, :3]
    return int(rgb.max()) < threshold


def _rgba_to_grayscale_pil(image_data: np.ndarray) -> Image.Image:
    """
    Convert the RGBA NumPy array from the canvas into a grayscale PIL image.

    Steps:
    1. Convert RGBA → RGB (drops transparency).
    2. Convert RGB  → L   (L = luminance = grayscale, single channel).

    PIL's 'L' mode gives each pixel a value from 0 (black) to 255 (white).
    """
    pil_img = Image.fromarray(image_data.astype(np.uint8), mode="RGBA")
    pil_img = pil_img.convert("RGB").convert("L")
    return pil_img


def _crop_to_digit(gray_img: Image.Image, margin: int = 6) -> Image.Image:
    """
    Crop the image down to just the bounding box of the drawn digit,
    then add a small margin so the digit doesn't touch the edges.

    Why? — The canvas is much larger (e.g. 280×280) than 28×28.  If we
    just resize without cropping, the digit shrinks to a tiny dot in the
    corner and the model sees mostly empty space.

    Parameters
    ----------
    gray_img : grayscale PIL Image
    margin   : pixels of padding around the tight bounding box
    """
    # getbbox() returns (left, upper, right, lower) of non-background pixels.
    # It treats 0 (black) as background.  If the image is all black → None.
    bbox = gray_img.getbbox()
    if bbox is None:
        return gray_img   # nothing drawn; return as-is

    left, upper, right, lower = bbox
    h, w = gray_img.size[1], gray_img.size[0]

    # Expand the bounding box by 'margin' pixels, clamped to image boundaries.
    left   = max(0, left   - margin)
    upper  = max(0, upper  - margin)
    right  = min(w, right  + margin)
    lower  = min(h, lower  + margin)

    return gray_img.crop((left, upper, right, lower))


def _resize_preserving_aspect(digit_img: Image.Image,
                               target: int = IMG_SIZE,
                               padding: int = PADDING) -> Image.Image:
    """
    Resize the cropped digit to fit inside a (target - 2*padding) square
    while preserving its aspect ratio (no squishing), then paste it centred
    on a black target×target canvas.

    Example: a tall "1" (height >> width) would become a thin vertical bar
    in the centre of the 28×28 image — just like in MNIST.

    Parameters
    ----------
    digit_img : cropped grayscale PIL Image of the digit
    target    : final canvas side length (28)
    padding   : pixels of black border to keep around the digit
    """
    inner = target - 2 * padding   # usable area inside the padding

    # Compute scaling factor to fit within the inner square.
    w, h       = digit_img.size
    scale      = inner / max(w, h)
    new_w      = max(1, int(w * scale))
    new_h      = max(1, int(h * scale))

    # LANCZOS gives the best quality when shrinking images.
    resized    = digit_img.resize((new_w, new_h), Image.LANCZOS)

    # Create a black canvas and paste the resized digit centred.
    canvas     = Image.new("L", (target, target), color=0)
    paste_x    = (target - new_w) // 2
    paste_y    = (target - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return canvas


def _smooth(img: Image.Image) -> Image.Image:
    """
    Apply a very gentle Gaussian blur (radius=0.5) to smooth jagged
    mouse-drawn strokes.  This makes drawn digits look slightly more
    like the smooth pen strokes in the MNIST training data.
    """
    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


def _gray_to_tensor(gray_img: Image.Image) -> tuple:
    """
    Turn a grayscale PIL image of a single digit into a model-ready tensor.

    This is the shared backbone used by both single-digit and multi-digit
    preprocessing:
        crop to bounding box → resize (aspect preserved, centred)
        → smooth → normalise ÷ 255 → add batch + channel dims

    Returns
    -------
    tensor      : np.ndarray, shape (1, 28, 28, 1), float32 in [0, 1]
    preview_img : PIL Image (28×28 grayscale)
    """
    # ── Crop to digit bounding box ────────────────────────────────────────
    cropped = _crop_to_digit(gray_img, margin=6)

    # ── Resize to 28×28, preserving aspect ratio, centred ─────────────────
    resized = _resize_preserving_aspect(cropped, target=IMG_SIZE, padding=PADDING)

    # ── Smooth ────────────────────────────────────────────────────────────
    smoothed = _smooth(resized)

    # ── Convert to float32 NumPy array and normalise ÷ 255 ────────────────
    arr = np.array(smoothed, dtype=np.float32) / 255.0

    # ── Add batch and channel dimensions: (28,28) → (1,28,28,1) ───────────
    tensor = np.expand_dims(np.expand_dims(arr, -1), 0)

    # Keep an unmodified PIL copy for the "preprocessed preview" widget.
    preview_img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")

    return tensor, preview_img


def preprocess_canvas_image(image_data: np.ndarray) -> tuple:
    """
    Full preprocessing pipeline: canvas RGBA → model-ready tensor.

    Returns
    -------
    tensor      : np.ndarray, shape (1, 28, 28, 1), float32 in [0, 1]
    preview_img : PIL Image (28×28 grayscale) for display in the UI
    is_empty    : bool — True if nothing was drawn
    """
    # ── Guard: empty canvas ───────────────────────────────────────────────
    if is_canvas_empty(image_data):
        return None, None, True

    # ── Step 1: RGBA → grayscale ──────────────────────────────────────────
    gray = _rgba_to_grayscale_pil(image_data)

    # ── Steps 2–6: crop → resize → smooth → normalise → add dims ──────────
    tensor, preview_img = _gray_to_tensor(gray)

    return tensor, preview_img, False


def segment_canvas_image(image_data: np.ndarray,
                         min_area: int = 40) -> tuple:
    """
    Split a canvas containing MULTIPLE digits into individual digit tensors.

    Connected bright regions (components) are found with scipy's label()
    and processed left-to-right, so "1 2 3" becomes three 28×28 tensors in
    the order the digits appear.

    Parameters
    ----------
    image_data : RGBA NumPy array from the canvas, shape (H, W, 4)
    min_area   : ignore components smaller than this many pixels (noise)

    Returns
    -------
    tensors  : list[np.ndarray]  — each (1, 28, 28, 1), float32 in [0, 1]
    previews : list[PIL.Image]   — matching 28×28 previews
    is_empty : bool              — True if no digit-sized component found
    """
    if is_canvas_empty(image_data):
        return [], [], True

    # Find each digit's bounding box, sorted left-to-right.
    boxes = _find_component_boxes(image_data, min_area=min_area)

    if not boxes:
        return [], [], True

    tensors, previews = [], []
    gray = np.array(_rgba_to_grayscale_pil(image_data))
    for (x0, x1, y0, y1) in boxes:
        # Isolate this component on a black background, then reuse the
        # standard single-digit pipeline on the crop.
        sub = gray[y0:y1 + 1, x0:x1 + 1]
        sub_img = Image.fromarray(sub, mode="L")
        tensor, preview = _gray_to_tensor(sub_img)
        tensors.append(tensor)
        previews.append(preview)

    return tensors, previews, False


def _find_component_boxes(image_data: np.ndarray,
                          min_area: int = 40,
                          threshold: int = 50) -> List[tuple]:
    """
    Locate the connected bright components on the canvas and return their
    bounding boxes, sorted left-to-right.

    Returns
    -------
    list of (x0, x1, y0, y1) tuples — one per digit-sized component.
    """
    from scipy import ndimage

    gray     = np.array(_rgba_to_grayscale_pil(image_data))
    labeled, n = ndimage.label(gray > threshold)

    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) < min_area:
            continue
        boxes.append((xs.min(), xs.max(), ys.min(), ys.max()))

    boxes.sort(key=lambda b: b[0])
    return boxes


def get_example_image(digit: int = 3) -> np.ndarray:
    """
    Load one example image from the MNIST test set and convert it to
    the same RGBA format as the canvas so it can be processed by the
    same pipeline.

    Used by the 'Try Example' button in the UI.

    Parameters
    ----------
    digit : which digit class to sample (0–9)

    Returns
    -------
    RGBA NumPy array, shape (280, 280, 4)  — upscaled for visibility
    """
    # The test set is cached so the 11 MB dataset is only loaded once,
    # instead of on every "Try Example" click.
    x_test, y_test = _load_mnist_test()

    # Find all test images of the requested digit.
    indices = np.where(y_test == digit)[0]
    idx     = np.random.choice(indices)
    img_28  = x_test[idx]                # shape (28, 28), values 0–255

    # Upscale 28×28 → 280×280 so it looks reasonable on the canvas.
    pil_img  = Image.fromarray(img_28, mode="L").resize((280, 280), Image.NEAREST)
    pil_rgba = pil_img.convert("RGBA")
    return np.array(pil_rgba)


@st.cache_data(show_spinner=False)
def _load_mnist_test():
    """Load (and cache) the MNIST test set.  Ensures a single download/read."""
    import tensorflow as tf

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_test, y_test

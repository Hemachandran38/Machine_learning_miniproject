# =============================================================================
# test_preprocessor.py — Automated tests for the image preprocessing pipeline.
#
# Run:  pytest tests/ -v          (from digit_recognition/ folder)
#
# These tests verify that whatever a user draws (or whatever MNIST image
# is loaded), the tensor that reaches the model is ALWAYS the correct shape,
# type, range, and polarity.  A wrong tensor silently produces bad predictions
# — that's why we test this specifically.
# =============================================================================

import sys
import os
import numpy as np
import pytest
from PIL import Image

# ── Make sure the project root is on sys.path ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp.preprocessor import (
    is_canvas_empty,
    preprocess_canvas_image,
    _rgba_to_grayscale_pil,
    _crop_to_digit,
    _resize_preserving_aspect,
)


# =============================================================================
# Fixtures — synthetic canvas images
# =============================================================================

def _blank_canvas(h=280, w=280) -> np.ndarray:
    """Return a completely black RGBA array (simulates empty canvas)."""
    return np.zeros((h, w, 4), dtype=np.uint8)


def _canvas_with_stroke(h=280, w=280, brightness=255) -> np.ndarray:
    """
    Return an RGBA array with a white rectangle in the centre.
    Simulates a drawn digit stroke.
    """
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    # Draw a thick white rectangle in the middle
    arr[80:200, 110:170, :3] = brightness   # RGB channels
    arr[80:200, 110:170,  3] = 255          # alpha = fully opaque
    return arr


def _tiny_stroke_canvas(h=280, w=280) -> np.ndarray:
    """Canvas with a very small, faint mark (edge case)."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[140, 140, :3] = 50    # single dim pixel
    arr[140, 140,  3] = 255
    return arr


# =============================================================================
# Test group 1: is_canvas_empty
# =============================================================================

class TestIsCanvasEmpty:

    def test_blank_canvas_is_empty(self):
        """A fully black canvas must be detected as empty."""
        assert is_canvas_empty(_blank_canvas()) is True

    def test_canvas_with_stroke_is_not_empty(self):
        """A canvas with a bright stroke must NOT be detected as empty."""
        assert is_canvas_empty(_canvas_with_stroke()) is False

    def test_none_input_is_empty(self):
        """None input (before canvas renders) must be treated as empty."""
        assert is_canvas_empty(None) is True

    def test_very_dim_pixel_below_threshold(self):
        """A very dim mark (brightness=5) is below the threshold and counts as empty."""
        arr = np.zeros((280, 280, 4), dtype=np.uint8)
        arr[140, 140, :3] = 5
        assert is_canvas_empty(arr, threshold=10) is True

    def test_pixel_at_threshold_is_not_empty(self):
        """A pixel at exactly the threshold is NOT considered empty."""
        arr = np.zeros((280, 280, 4), dtype=np.uint8)
        arr[140, 140, :3] = 10
        assert is_canvas_empty(arr, threshold=10) is False


# =============================================================================
# Test group 2: output tensor properties
# =============================================================================

class TestPreprocessOutputShape:

    def test_tensor_shape(self):
        """Output tensor must be (1, 28, 28, 1) — the exact shape the model expects."""
        tensor, _, _ = preprocess_canvas_image(_canvas_with_stroke())
        assert tensor.shape == (1, 28, 28, 1), \
            f"Expected (1,28,28,1) but got {tensor.shape}"

    def test_tensor_dtype_float32(self):
        """Tensor values must be float32 (not int, not float64)."""
        tensor, _, _ = preprocess_canvas_image(_canvas_with_stroke())
        assert tensor.dtype == np.float32, \
            f"Expected float32 but got {tensor.dtype}"

    def test_pixel_range_0_to_1(self):
        """All pixel values must be in [0.0, 1.0] after normalisation."""
        tensor, _, _ = preprocess_canvas_image(_canvas_with_stroke())
        assert tensor.min() >= 0.0, f"Min pixel {tensor.min()} is below 0"
        assert tensor.max() <= 1.0, f"Max pixel {tensor.max()} is above 1"

    def test_empty_canvas_returns_none_tensor(self):
        """Empty canvas must return (None, None, True) — not crash."""
        tensor, preview, is_empty = preprocess_canvas_image(_blank_canvas())
        assert tensor   is None
        assert preview  is None
        assert is_empty is True

    def test_preview_img_is_28x28(self):
        """The preview PIL image must be exactly 28×28 pixels."""
        _, preview_img, _ = preprocess_canvas_image(_canvas_with_stroke())
        assert preview_img.size == (28, 28), \
            f"Expected (28,28) but got {preview_img.size}"


# =============================================================================
# Test group 3: polarity (white digit on black background)
# =============================================================================

class TestPolarity:

    def test_stroke_is_bright_in_tensor(self):
        """
        A drawn stroke (white pixels on canvas) must appear as BRIGHT values
        in the tensor.  MNIST uses white digits on black — if this is inverted,
        the model will see noise instead of a digit.
        """
        canvas = _canvas_with_stroke(brightness=255)
        tensor, _, _ = preprocess_canvas_image(canvas)
        # The tensor should have pixels close to 1.0 (bright) where we drew
        assert tensor.max() > 0.5, \
            "Stroke pixels are too dark — polarity may be inverted"

    def test_background_is_dark(self):
        """
        Regions without a stroke must be close to 0 (black background).
        """
        canvas = _canvas_with_stroke()
        tensor, _, _ = preprocess_canvas_image(canvas)
        # The very first pixel (top-left corner) has no stroke
        corner_pixel = float(tensor[0, 0, 0, 0])
        assert corner_pixel < 0.1, \
            f"Background pixel {corner_pixel:.3f} is too bright — polarity issue"


# =============================================================================
# Test group 4: internal helpers
# =============================================================================

class TestInternalHelpers:

    def test_rgba_to_grayscale_output_mode(self):
        """_rgba_to_grayscale_pil must return a single-channel 'L' mode image."""
        arr  = _canvas_with_stroke()
        gray = _rgba_to_grayscale_pil(arr)
        assert gray.mode == "L"

    def test_crop_to_digit_shrinks_image(self):
        """After cropping, the image must be smaller than the original."""
        arr     = _canvas_with_stroke()
        gray    = _rgba_to_grayscale_pil(arr)
        cropped = _crop_to_digit(gray)
        orig_area    = gray.size[0] * gray.size[1]
        cropped_area = cropped.size[0] * cropped.size[1]
        assert cropped_area < orig_area, "Crop did not reduce image size"

    def test_resize_preserving_aspect_output_size(self):
        """Output of _resize_preserving_aspect must be exactly 28×28."""
        # Create a non-square crop (tall digit like "1")
        tall_img = Image.new("L", (20, 80), color=0)
        result   = _resize_preserving_aspect(tall_img, target=28, padding=4)
        assert result.size == (28, 28), f"Expected (28,28) but got {result.size}"

    def test_resize_aspect_ratio_preserved(self):
        """
        For a tall narrow image, the resized digit inside the 28×28 canvas
        must be taller than it is wide (aspect ratio preserved, not squished).
        """
        # Create a tall white bar on black background (like the digit "1")
        tall_img = Image.new("L", (10, 60), color=0)
        # Draw a white vertical bar
        pix = tall_img.load()
        for y in range(60):
            for x in range(3, 7):
                pix[x, y] = 255

        result = _resize_preserving_aspect(tall_img, target=28, padding=4)
        arr    = np.array(result)

        # Count bright rows vs bright columns
        bright_rows = np.sum(arr.max(axis=1) > 128)
        bright_cols = np.sum(arr.max(axis=0) > 128)
        assert bright_rows > bright_cols, \
            "Aspect ratio not preserved: tall digit was squished to wide"


# =============================================================================
# Test group 5: all 10 MNIST digit classes
# =============================================================================

class TestAllDigitClasses:
    """
    Load one MNIST example per digit and verify preprocessing succeeds
    for every class.  This catches class-specific issues.
    """

    @pytest.mark.parametrize("digit", list(range(10)))
    def test_mnist_example_preprocesses_correctly(self, digit):
        """
        For digit 0–9: load an MNIST example, convert to canvas format,
        preprocess, and verify the tensor is valid.
        """
        import tensorflow as tf

        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        indices = np.where(y_test == digit)[0]
        idx     = indices[0]   # always use the first example for reproducibility
        img_28  = x_test[idx]  # shape (28, 28), values 0–255

        # Convert to canvas-like RGBA (280×280)
        pil_l    = Image.fromarray(img_28, mode="L").resize((280, 280), Image.NEAREST)
        pil_rgba = pil_l.convert("RGBA")
        canvas   = np.array(pil_rgba)

        tensor, preview, is_empty = preprocess_canvas_image(canvas)

        assert not is_empty,                          f"Digit {digit}: canvas wrongly detected as empty"
        assert tensor is not None,                    f"Digit {digit}: tensor is None"
        assert tensor.shape == (1, 28, 28, 1),        f"Digit {digit}: wrong shape {tensor.shape}"
        assert tensor.dtype == np.float32,            f"Digit {digit}: wrong dtype {tensor.dtype}"
        assert 0.0 <= tensor.min(),                   f"Digit {digit}: pixel below 0"
        assert tensor.max() <= 1.0,                   f"Digit {digit}: pixel above 1"
        assert preview is not None,                   f"Digit {digit}: preview is None"
        assert preview.size == (28, 28),              f"Digit {digit}: preview size {preview.size}"

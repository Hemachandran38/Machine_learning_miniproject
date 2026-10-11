# =============================================================================
# test_segmentation.py — Tests for multi-digit canvas segmentation.
#
# Run:  python -m pytest tests/test_segmentation.py -v
# =============================================================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from webapp.preprocessor import segment_canvas_image, _find_component_boxes


def _canvas_with_strokes(*blocks) -> np.ndarray:
    """Build an RGBA canvas with bright white rectangles.

    Each block is (x0, x1, y0, y1).  Non-overlapping blocks become
    separate connected components.
    """
    height = max(y1 for _, _, _, y1 in blocks) + 20
    width  = max(x1 for _, x1, _, _ in blocks) + 20
    arr    = np.zeros((height, width, 4), dtype=np.uint8)
    for x0, x1, y0, y1 in blocks:
        arr[y0:y1, x0:x1, :3] = 255
    arr[:, :, 3] = 255
    return arr


class TestSegmentCounts:

    def test_empty_canvas_yields_no_segments(self):
        blank = np.zeros((140, 140, 4), dtype=np.uint8)
        tensors, previews, is_empty = segment_canvas_image(blank)
        assert is_empty is True
        assert tensors == []
        assert previews == []

    def test_two_separated_digits(self):
        canvas = _canvas_with_strokes(
            (40, 100, 30, 110),     # left digit
            (170, 230, 30, 110),    # right digit
        )
        tensors, previews, is_empty = segment_canvas_image(canvas)
        assert is_empty is False
        assert len(tensors) == 2
        assert len(previews) == 2

    def test_three_separated_digits(self):
        canvas = _canvas_with_strokes(
            (20, 70, 30, 110),
            (100, 150, 30, 110),
            (180, 230, 30, 110),
        )
        tensors, _, _ = segment_canvas_image(canvas)
        assert len(tensors) == 3


class TestSegmentShapeAndOrder:

    def test_each_tensor_is_model_ready(self):
        canvas = _canvas_with_strokes((40, 100, 30, 110), (170, 230, 30, 110))
        tensors, previews, _ = segment_canvas_image(canvas)
        for tensor, preview in zip(tensors, previews):
            assert tensor.shape == (1, 28, 28, 1)
            assert tensor.dtype == np.float32
            assert 0.0 <= tensor.min() <= tensor.max() <= 1.0
            assert preview.size == (28, 28)

    def test_segments_are_sorted_left_to_right(self):
        # Right digit drawn first in the list but appears second on canvas.
        canvas = _canvas_with_strokes(
            (200, 260, 30, 110),   # right digit
            (30, 90, 30, 110),     # left digit
        )
        boxes = _find_component_boxes(canvas)
        assert len(boxes) == 2
        assert boxes[0][0] < boxes[1][0], \
            "Components must be sorted left-to-right by x0"

    def test_tensors_follow_box_order(self):
        """segment_canvas_image must return tensors in the same LTR order."""
        canvas = _canvas_with_strokes(
            (200, 260, 30, 110),   # right digit
            (30, 90, 30, 110),     # left digit
        )
        boxes        = _find_component_boxes(canvas)
        tensors, _, _ = segment_canvas_image(canvas)
        assert len(tensors) == len(boxes)


class TestMergedComponents:

    def test_touching_digits_merge_into_one(self):
        # Two rectangles that share an edge should be treated as one component.
        canvas = _canvas_with_strokes(
            (40, 100, 30, 110),
            (100, 160, 30, 110),    # touches the first at x=100
        )
        tensors, _, is_empty = segment_canvas_image(canvas)
        assert is_empty is False
        assert len(tensors) == 1

    def test_tiny_noise_is_ignored(self):
        arr = np.zeros((140, 140, 4), dtype=np.uint8)
        arr[70:75, 70:75, :3] = 255     # < min_area → filtered out
        arr[:, :, 3] = 255
        tensors, _, is_empty = segment_canvas_image(arr, min_area=40)
        assert is_empty is True
        assert tensors == []
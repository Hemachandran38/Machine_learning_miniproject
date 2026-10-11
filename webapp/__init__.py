"""
webapp package — Streamlit application for handwritten digit recognition.

Public API re-exported here so callers can do:

    from webapp import preprocess_canvas_image, predict, get_history
"""

from webapp.preprocessor import (
    preprocess_canvas_image,
    segment_canvas_image,
    get_example_image,
    is_canvas_empty,
)
from webapp.predictor import load_model, predict
from webapp.history import (
    init_db,
    save_prediction,
    get_history,
    get_summary_stats,
    clear_history,
    delete_prediction,
    set_feedback,
    export_history_csv,
)

__all__ = [
    "preprocess_canvas_image",
    "segment_canvas_image",
    "get_example_image",
    "is_canvas_empty",
    "load_model",
    "predict",
    "init_db",
    "save_prediction",
    "get_history",
    "get_summary_stats",
    "clear_history",
    "delete_prediction",
    "set_feedback",
    "export_history_csv",
]

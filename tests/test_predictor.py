# =============================================================================
# test_predictor.py — Automated tests for model loading and prediction output.
#
# Run:  pytest tests/ -v          (from digit_recognition/ folder)
#
# These tests verify that the model:
#   1. Loads successfully from disk
#   2. Returns valid output structure for any valid input
#   3. Produces probabilities that sum to 1.0 (softmax guarantee)
#   4. Achieves reasonable accuracy on the MNIST test set
#   5. Handles edge-case inputs without crashing
# =============================================================================

import sys
import os
import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(_PROJECT_DIR, "models", "digit_recognizer.keras")


# =============================================================================
# Fixture: load the model once for all tests in this module
# =============================================================================

@pytest.fixture(scope="module")
def model():
    """
    Load the saved Keras model once and share it across all tests.
    scope="module" means it's loaded once per test file, not once per test.
    """
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        pytest.skip(
            f"Model not found at {MODEL_PATH}. "
            "Run 'python main.py' first to train and save the model."
        )

    return tf.keras.models.load_model(MODEL_PATH)


@pytest.fixture(scope="module")
def mnist_test_data():
    """Load MNIST test set once for all tests."""
    import tensorflow as tf
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)
    return x_test, y_test


def _make_tensor(brightness: float = 0.8) -> np.ndarray:
    """Create a synthetic (1, 28, 28, 1) tensor with a white rectangle."""
    arr = np.zeros((1, 28, 28, 1), dtype=np.float32)
    arr[0, 8:20, 10:18, 0] = brightness
    return arr


# =============================================================================
# Test group 1: model loading
# =============================================================================

class TestModelLoading:

    def test_model_file_exists(self):
        """The .keras file must exist before any test can proceed."""
        assert os.path.exists(MODEL_PATH), \
            f"Model file not found: {MODEL_PATH}\nRun 'python main.py' first."

    def test_model_loads_without_error(self, model):
        """tf.keras.models.load_model must not raise an exception."""
        assert model is not None

    def test_model_has_correct_input_shape(self, model):
        """The model must accept input shape (None, 28, 28, 1)."""
        expected = (None, 28, 28, 1)
        actual   = tuple(model.input_shape)
        assert actual == expected, \
            f"Expected input shape {expected}, got {actual}"

    def test_model_has_correct_output_shape(self, model):
        """The model must produce output shape (None, 10) — 10 digit classes."""
        expected = (None, 10)
        actual   = tuple(model.output_shape)
        assert actual == expected, \
            f"Expected output shape {expected}, got {actual}"


# =============================================================================
# Test group 2: prediction output structure
# =============================================================================

class TestPredictionOutput:

    def test_returns_dict_with_required_keys(self, model):
        """predict() must return a dict with digit, confidence, probabilities."""
        from webapp.predictor import predict
        tensor = _make_tensor()
        result = predict(model, tensor)

        assert isinstance(result, dict)
        assert "digit"         in result
        assert "confidence"    in result
        assert "probabilities" in result

    def test_digit_is_valid_integer(self, model):
        """Predicted digit must be an integer in range [0, 9]."""
        from webapp.predictor import predict
        result = predict(model, _make_tensor())
        assert isinstance(result["digit"], int)
        assert 0 <= result["digit"] <= 9

    def test_confidence_is_valid_float(self, model):
        """Confidence must be a float in [0.0, 1.0]."""
        from webapp.predictor import predict
        result = predict(model, _make_tensor())
        conf = result["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_probabilities_length_is_10(self, model):
        """Probabilities list must have exactly 10 entries (one per digit)."""
        from webapp.predictor import predict
        result = predict(model, _make_tensor())
        assert len(result["probabilities"]) == 10

    def test_probabilities_sum_to_one(self, model):
        """Softmax output must sum to 1.0 (within floating point tolerance)."""
        from webapp.predictor import predict
        result = predict(model, _make_tensor())
        total  = sum(result["probabilities"])
        assert abs(total - 1.0) < 1e-5, \
            f"Probabilities sum to {total:.6f}, expected ~1.0"

    def test_confidence_matches_max_probability(self, model):
        """Confidence must equal the highest value in probabilities."""
        from webapp.predictor import predict
        result = predict(model, _make_tensor())
        assert abs(result["confidence"] - max(result["probabilities"])) < 1e-6

    def test_digit_matches_argmax_of_probabilities(self, model):
        """Predicted digit must be the index of the highest probability."""
        from webapp.predictor import predict
        result      = predict(model, _make_tensor())
        expected_d  = int(np.argmax(result["probabilities"]))
        assert result["digit"] == expected_d


# =============================================================================
# Test group 3: MNIST test set accuracy
# =============================================================================

class TestMNISTAccuracy:

    def test_test_accuracy_above_99_percent(self, model, mnist_test_data):
        """
        The model must achieve ≥ 99% accuracy on the MNIST test set.
        A well-trained CNN on MNIST routinely achieves 99%+.
        If this fails, the model was not trained properly.
        """
        x_test, y_test = mnist_test_data
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        assert acc >= 0.99, \
            f"Test accuracy {acc*100:.2f}% is below the expected 99% threshold. Retrain the model."

    @pytest.mark.parametrize("digit", list(range(10)))
    def test_per_class_accuracy_above_97_percent(self, model, mnist_test_data, digit):
        """
        Each individual digit class must achieve ≥ 97% accuracy.
        This catches cases where the model is good overall but fails
        on a specific digit (e.g., confusing 4 and 9).
        """
        x_test, y_test = mnist_test_data

        # Filter to only this digit's test images
        mask   = y_test == digit
        x_sub  = x_test[mask]
        y_sub  = y_test[mask]

        probs  = model.predict(x_sub, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        acc    = np.mean(y_pred == y_sub)

        assert acc >= 0.97, \
            f"Digit {digit} accuracy is {acc*100:.2f}%, below 97% threshold"


# =============================================================================
# Test group 4: edge cases
# =============================================================================

class TestEdgeCases:

    def test_all_black_tensor(self, model):
        """A tensor of all zeros (blank image) must not crash the model."""
        from webapp.predictor import predict
        tensor = np.zeros((1, 28, 28, 1), dtype=np.float32)
        result = predict(model, tensor)
        # Should return some digit without crashing
        assert 0 <= result["digit"] <= 9

    def test_all_white_tensor(self, model):
        """A tensor of all ones (fully white image) must not crash the model."""
        from webapp.predictor import predict
        tensor = np.ones((1, 28, 28, 1), dtype=np.float32)
        result = predict(model, tensor)
        assert 0 <= result["digit"] <= 9

    def test_prediction_is_deterministic(self, model):
        """The same input must always produce the same prediction."""
        from webapp.predictor import predict
        tensor  = _make_tensor(brightness=0.9)
        result1 = predict(model, tensor)
        result2 = predict(model, tensor)
        assert result1["digit"]      == result2["digit"]
        assert result1["confidence"] == result2["confidence"]

    @pytest.mark.parametrize("digit", list(range(10)))
    def test_correct_prediction_on_clean_mnist_sample(self, model, mnist_test_data, digit):
        """
        For each digit, the first clean MNIST test sample must be
        predicted correctly.  This verifies end-to-end inference.
        """
        x_test, y_test = mnist_test_data
        from webapp.predictor import predict

        idx    = np.where(y_test == digit)[0][0]
        tensor = x_test[idx : idx + 1]   # shape (1, 28, 28, 1)
        result = predict(model, tensor)

        assert result["digit"] == digit, \
            f"Clean MNIST sample for digit {digit} was predicted as {result['digit']} " \
            f"with confidence {result['confidence']*100:.1f}%"

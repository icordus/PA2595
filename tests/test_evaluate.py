"""Unit tests for metric computation in src.evaluate."""

import unittest

import pandas as pd

from src.evaluate import evaluate_model


class DummyModel:
    """Minimal model stub that returns predefined predictions."""

    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, _X):
        return self._predictions


class TestEvaluate(unittest.TestCase):
    """Validate the evaluate_model result dictionary for known inputs."""

    def test_evaluate_model_returns_expected_metrics(self):
        """Computed metrics should match expected values for a fixed prediction set."""
        y_test = pd.Series([0, 1, 1, 0])
        y_pred = [0, 1, 0, 0]
        X_test = pd.DataFrame({"feature": [1, 2, 3, 4]})

        result = evaluate_model(DummyModel(y_pred), X_test, y_test, "dummy")

        self.assertEqual(result["model"], "dummy")
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1_score"], 2 / 3)


if __name__ == "__main__":
    unittest.main()

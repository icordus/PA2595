"""Unit tests for prediction helper behavior in src.predict."""

import unittest
from unittest.mock import patch

from src.predict import predict


class DummyPredictModel:
    """Model stub that returns fixed class and probability outputs."""

    def predict(self, _df):
        return [1]

    def predict_proba(self, _df):
        return [[0.2, 0.8]]


class TestPredict(unittest.TestCase):
    """Validate output schema and values of predict()."""

    @patch("src.predict.load_feature_columns", return_value=["studytime", "absences"])
    @patch("src.predict.load_model", return_value=DummyPredictModel())
    def test_predict_returns_label_and_probability(self, _load_model, _load_columns):
        """predict() should return Pass label and probability from model output."""
        result = predict({"studytime": 2, "absences": 3}, model_name="random_forest")

        self.assertEqual(result["label"], "Pass")
        self.assertAlmostEqual(result["probability"], 0.8)


if __name__ == "__main__":
    unittest.main()

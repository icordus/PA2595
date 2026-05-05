"""Unit tests for FastAPI endpoints in src.api."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app


class TestApi(unittest.TestCase):
    """Validate API health and prediction endpoint behavior."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health_returns_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    @patch("src.api.model_predict", return_value={"label": "Pass", "probability": 0.91})
    @patch("src.api.load_feature_columns", return_value=["studytime", "failures", "absences", "G1", "G2"])
    def test_predict_uses_model_when_artifacts_exist(self, _load_cols, _model_predict):
        payload = {"studytime": 3, "failures": 0, "absences": 2, "G1": 15, "G2": 16}

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["prediction"], "pass")
        self.assertEqual(data["risk"], "low")
        self.assertEqual(data["source"], "model")
        self.assertEqual(data["probability"], 0.91)
        self.assertEqual(data["score"], 15.5)

    @patch("src.api.load_feature_columns", side_effect=FileNotFoundError)
    def test_predict_falls_back_to_heuristic(self, _load_cols):
        payload = {"studytime": 1, "failures": 2, "absences": 25, "G1": 6, "G2": 7}

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["prediction"], "fail")
        self.assertEqual(data["risk"], "high")
        self.assertEqual(data["source"], "heuristic")
        self.assertIsNone(data["probability"])
        self.assertEqual(data["score"], 6.5)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for preprocessing helpers in src.preprocess."""

import unittest

import pandas as pd

from src.preprocess import create_target, build_features_and_target


class TestPreprocess(unittest.TestCase):
    """Validate target creation and leakage-safe feature preparation."""

    def test_create_target_uses_threshold(self):
        """Students with G3 >= threshold should be labeled as Pass (1)."""
        df = pd.DataFrame({"G3": [9, 10, 14]})
        out = create_target(df, threshold=10)
        self.assertEqual(out["target"].tolist(), [0, 1, 1])

    def test_build_features_and_target_removes_g3(self):
        """G3 must be removed from X to avoid data leakage in training."""
        df = pd.DataFrame(
            {
                "G3": [8, 14],
                "studytime": [2, 3],
                "internet": ["yes", "no"],
                "target": [0, 1],
            }
        )
        X, y = build_features_and_target(df)

        self.assertNotIn("G3", X.columns)
        self.assertEqual(y.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()

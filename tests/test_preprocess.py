"""Unit tests for preprocessing helpers in src.preprocess."""

import unittest

import pandas as pd

from src.preprocess import create_target, encode_features


class TestPreprocess(unittest.TestCase):
    """Validate target creation and categorical encoding behavior."""

    def test_create_target_uses_threshold(self):
        """Students with G3 >= threshold should be labeled as Pass (1)."""
        df = pd.DataFrame({"G3": [9, 10, 14]})
        out = create_target(df, threshold=10)
        self.assertEqual(out["target"].tolist(), [0, 1, 1])

    def test_encode_features_encodes_present_columns(self):
        """Only existing categorical columns should be encoded to numeric labels."""
        df = pd.DataFrame({"internet": ["yes", "no", "yes"], "studytime": [1, 2, 3]})
        out = encode_features(df, ["internet", "higher"])

        self.assertIn("internet", out.columns)
        self.assertEqual(sorted(out["internet"].unique().tolist()), [0, 1])
        self.assertEqual(out["studytime"].tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()

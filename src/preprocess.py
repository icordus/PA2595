"""Data loading and split stage for the student performance pipeline."""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

DATASET_FILE = "student-mat.csv"
PASS_THRESHOLD = 10
RANDOM_SEED = 42


def load_data(raw_dir: str, filename: str) -> pd.DataFrame:
    """Load the raw semicolon-separated UCI dataset from disk."""
    filepath = os.path.join(raw_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}.\n"
            "Download it from https://archive.ics.uci.edu/dataset/320/student%2Bperformance "
            "and place it in data/raw/."
        )
    df = pd.read_csv(filepath, sep=";")
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def create_target(df: pd.DataFrame, threshold: int = PASS_THRESHOLD) -> pd.DataFrame:
    """Create binary target from G3 where Pass=1 if G3 >= threshold else 0."""
    if "G3" not in df.columns:
        raise ValueError("Required column 'G3' not found in dataset.")

    out = df.copy()
    out["target"] = (out["G3"] >= threshold).astype(int)
    print(f"Target distribution:\n{out['target'].value_counts().rename({0: 'Fail', 1: 'Pass'})}")
    return out


def build_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X and y while removing G3 from features to avoid data leakage."""
    y = df["target"].copy()
    X = df.drop(columns=["target", "G3"]).copy()
    return X, y


def preprocess(raw_dir: str = RAW_DIR, processed_dir: str = PROCESSED_DIR) -> None:
    """Load raw data, create leakage-safe split, and save train/test CSV files."""
    os.makedirs(processed_dir, exist_ok=True)

    df = load_data(raw_dir, DATASET_FILE)
    df = create_target(df)
    X, y = build_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    print("\nPreprocessing complete.")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")
    print(f"  Features used    : {X.shape[1]}")
    print(f"  Saved to         : {processed_dir}")


if __name__ == "__main__":
    preprocess()

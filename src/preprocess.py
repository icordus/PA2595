"""
preprocess.py
-------------
Loads the UCI Student Performance dataset, cleans it, encodes categorical
features, defines a binary target (Pass / Fail based on final grade G3),
and saves the processed data to data/processed/.

Run:
    python src/preprocess.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Change this to "student-por.csv" if you prefer the Portuguese dataset
DATASET_FILE = "student-mat.csv"

# Students with G3 >= PASS_THRESHOLD are labelled Pass (1), else Fail (0)
PASS_THRESHOLD = 10

# Features selected for the model
SELECTED_FEATURES = [
    "studytime",       # Weekly study time (1–4 scale)
    "absences",        # Number of school absences
    "failures",        # Number of past class failures
    "G1",              # First period grade
    "G2",              # Second period grade
    "Medu",            # Mother's education level
    "Fedu",            # Father's education level
    "traveltime",      # Home to school travel time
    "freetime",        # Free time after school
    "goout",           # Going out with friends
    "Dalc",            # Workday alcohol consumption
    "Walc",            # Weekend alcohol consumption
    "health",          # Current health status
    "internet",        # Internet access at home (yes/no)
    "higher",          # Wants higher education (yes/no)
    "sex",             # Student sex (M/F)
    "address",         # Urban or rural (U/R)
    "famsize",         # Family size (LE3 / GT3)
    "Pstatus",         # Parent cohabitation status (T/A)
    "schoolsup",       # Extra educational support (yes/no)
    "famsup",          # Family educational support (yes/no)
    "paid",            # Extra paid classes (yes/no)
    "activities",      # Extra-curricular activities (yes/no)
    "nursery",         # Attended nursery school (yes/no)
    "romantic",        # In a romantic relationship (yes/no)
]

BINARY_COLS = [
    "internet", "higher", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities", "nursery", "romantic",
]


def load_data(raw_dir: str, filename: str) -> pd.DataFrame:
    """Load the raw UCI CSV file from disk.

    The dataset is expected to be semicolon-separated, as distributed by UCI.
    """
    filepath = os.path.join(raw_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}.\n"
            "Download it from https://archive.ics.uci.edu/dataset/320/student%2Bperformance "
            "and place it in data/raw/."
        )
    # The UCI file uses semicolons as separators
    df = pd.read_csv(filepath, sep=";")
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def create_target(df: pd.DataFrame, threshold: int = PASS_THRESHOLD) -> pd.DataFrame:
    """Create a binary target column named `target` from `G3`.

    `target` is 1 (Pass) when G3 >= threshold, otherwise 0 (Fail).
    """
    df = df.copy()
    df["target"] = (df["G3"] >= threshold).astype(int)
    print(f"Target distribution:\n{df['target'].value_counts().rename({0: 'Fail', 1: 'Pass'})}")
    return df


def encode_features(df: pd.DataFrame, binary_cols: list) -> pd.DataFrame:
    """Label-encode selected categorical columns.

    This function only encodes columns that are present in the DataFrame.
    """
    df = df.copy()
    le = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    return df


def preprocess(raw_dir: str = RAW_DIR, processed_dir: str = PROCESSED_DIR) -> None:
    """Run the full preprocessing pipeline and save train/test CSV files."""
    os.makedirs(processed_dir, exist_ok=True)

    df = load_data(raw_dir, DATASET_FILE)
    df = create_target(df)
    df = encode_features(df, BINARY_COLS)

    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    missing = set(SELECTED_FEATURES) - set(available_features)
    if missing:
        print(f"Warning: features not found in dataset and will be skipped: {missing}")

    X = df[available_features]
    y = df["target"]

    # Handle any remaining missing values
    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    print(f"\nPreprocessing complete.")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")
    print(f"  Features used    : {len(available_features)}")
    print(f"  Saved to         : {processed_dir}")


if __name__ == "__main__":
    preprocess()

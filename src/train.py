"""
train.py
--------
Trains three classifiers (Decision Tree, Random Forest, Logistic Regression)
on the preprocessed data and saves each trained model to models/.

Run:
    python src/train.py

Requires that preprocess.py has been run first.
"""

import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS = {
    "decision_tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    ),
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    ),
}


def load_processed_data(processed_dir: str):
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    return X_train, y_train


def train_all(processed_dir: str = PROCESSED_DIR, models_dir: str = MODELS_DIR) -> None:
    os.makedirs(models_dir, exist_ok=True)

    X_train, y_train = load_processed_data(processed_dir)
    print(f"Training on {len(X_train)} samples with {X_train.shape[1]} features.\n")

    for name, model in MODELS.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        save_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(model, save_path)
        print(f"  Saved to {save_path}")

    # Also save the feature column order so the prototype can align inputs
    feature_path = os.path.join(models_dir, "feature_columns.pkl")
    joblib.dump(list(X_train.columns), feature_path)
    print(f"\nFeature column list saved to {feature_path}")
    print("\nAll models trained successfully.")


if __name__ == "__main__":
    train_all()

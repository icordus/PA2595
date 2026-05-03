"""
evaluate.py
-----------
Loads each saved model, evaluates it on the test set, and prints a full
comparison report including Accuracy, Precision, Recall, F1-score, and
the Confusion Matrix for every model.

Run:
    python src/evaluate.py

Requires that preprocess.py and train.py have been run first.
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODEL_NAMES = ["decision_tree", "random_forest", "logistic_regression"]


def load_test_data(processed_dir: str):
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    return X_test, y_test


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"Model: {model_name.replace('_', ' ').title()}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted Fail  Predicted Pass")
    print(f"  Actual Fail        {cm[0][0]:<15} {cm[0][1]}")
    print(f"  Actual Pass        {cm[1][0]:<15} {cm[1][1]}")
    print(f"\nFull Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
    }


def evaluate_all(processed_dir: str = PROCESSED_DIR, models_dir: str = MODELS_DIR) -> None:
    X_test, y_test = load_test_data(processed_dir)
    results = []

    for name in MODEL_NAMES:
        model_path = os.path.join(models_dir, f"{name}.pkl")
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path} — skipping.")
            continue
        model = joblib.load(model_path)
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)

    if results:
        print(f"\n{'='*50}")
        print("Summary Comparison")
        print(f"{'='*50}")
        summary = pd.DataFrame(results).set_index("model")
        print(summary.to_string())

        best = summary["f1_score"].idxmax()
        print(f"\nBest model by F1-score: {best.replace('_', ' ').title()}")


if __name__ == "__main__":
    evaluate_all()

"""Evaluation stage for the Decision Tree student risk pipeline."""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn import tree

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

PIPELINE_FILE = "decision_tree_pipeline.pkl"


def load_test_data(processed_dir: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the held-out test split generated in preprocess.py."""
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    return X_test, y_test


def evaluate(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute evaluation metrics, with explicit Fail-class recall."""
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    cm = confusion_matrix(y_test, y_pred)
    # Row 0 in cm corresponds to the Fail class (target=0).
    fail_recall = cm[0][0] / cm[0].sum() if cm[0].sum() else 0.0
    metrics["recall_fail"] = fail_recall

    print("\nClassification Report")
    print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))
    return metrics


def save_metrics(metrics: dict, results_dir: str) -> None:
    """Persist metrics in text format for report inclusion."""
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Decision Tree Pipeline Evaluation\n")
        f.write("=" * 40 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Saved metrics to: {metrics_path}")


def save_confusion_matrix(pipeline, X_test: pd.DataFrame, y_test: pd.Series, results_dir: str) -> None:
    """Render and save confusion matrix heatmap."""
    os.makedirs(results_dir, exist_ok=True)
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    out_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved confusion matrix plot to: {out_path}")


def save_decision_tree_plot(pipeline, X_test: pd.DataFrame, results_dir: str) -> None:
    """Render and save a readable decision tree visualization."""
    os.makedirs(results_dir, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out(X_test.columns)

    plt.figure(figsize=(24, 10))
    tree.plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=["Fail", "Pass"],
        filled=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title("Decision Tree (first 3 levels)")
    out_path = os.path.join(results_dir, "decision_tree.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved decision tree plot to: {out_path}")


def evaluate_pipeline(
    processed_dir: str = PROCESSED_DIR,
    models_dir: str = MODELS_DIR,
    results_dir: str = RESULTS_DIR,
) -> None:
    """Load saved pipeline, evaluate, and persist report artifacts."""
    pipeline_path = os.path.join(models_dir, PIPELINE_FILE)
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline artifact not found: {pipeline_path}")

    pipeline = joblib.load(pipeline_path)
    X_test, y_test = load_test_data(processed_dir)

    metrics = evaluate(pipeline, X_test, y_test)
    print("\nMetrics Summary")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    save_metrics(metrics, results_dir)
    save_confusion_matrix(pipeline, X_test, y_test, results_dir)
    save_decision_tree_plot(pipeline, X_test, results_dir)


if __name__ == "__main__":
    evaluate_pipeline()

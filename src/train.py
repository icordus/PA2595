"""Training script, this is the main script for building and saving the machine learning model
What it does:
1. Loads the dataset
2. Splits the dataset into input features and target labels
3. Splits the data into training and test sets
4. Builds a preprocessing and Decision Tree pipeline
5. Trains the model
6. Evaluates the model
7. Saves the trained model, metrics, confusion matrix and tree plot
"""
# This import allows modern Python type annotation behavior
from __future__ import annotations
# argparse is used to read command-line arguments, such as the dataset path
import argparse
# joblib is used to save the trained machine learning pipeline to disk
import joblib
# train_test_split is used to split the dataset into training and testing parts
from sklearn.model_selection import train_test_split
# DecisionTreeClassifier is the machine learning model used in this project
from sklearn.tree import DecisionTreeClassifier

# These settings and output paths are imported from the central configuration file
from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    DECISION_TREE_PARAMS,
    MODEL_PATH,
    METRICS_PATH,
    CONFUSION_MATRIX_PATH,
    TREE_PLOT_PATH,
)
# This function loads the UCI Student Performance dataset from a CSV file
from src.data_loader import load_student_data
# These functions prepare the input features, target variable and full sklearn pipeline.
from src.preprocessing import split_features_and_target, build_pipeline
# These functions evaluate the model and save the reportable output files.
from src.evaluate import (
    evaluate_classifier,
    save_metrics,
    save_confusion_matrix,
    save_tree_plot,
)

def train_model(csv_path: str):
    """Train the Decision Tree pipeline and save all outputs"""
    # 1. Load dataset
    df = load_student_data(csv_path)
    # 2. Create X and y
    X, y = split_features_and_target(df)
    # 3. Train and test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    # 4. Create the Decision Tree model
    decision_tree = DecisionTreeClassifier(**DECISION_TREE_PARAMS)
    # 5. Build full pipeline: preprocessing and model
    pipeline = build_pipeline(X_train, decision_tree)
    # 6. Train the pipeline
    pipeline.fit(X_train, y_train)
    # 7. Evaluate the model
    metrics = evaluate_classifier(pipeline, X_train, X_test, y_train, y_test)
    # 8. Save the trained pipeline
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # This saves the full trained pipeline to the models folder
    joblib.dump(pipeline, MODEL_PATH)

    # 9. Save reportable outputs
    save_metrics(metrics, METRICS_PATH)
    # This saves the confusion matrix as an image file
    save_confusion_matrix(pipeline, X_test, y_test, CONFUSION_MATRIX_PATH)
    # This saves a visual plot of the trained Decision Tree as an image file
    save_tree_plot(pipeline, TREE_PLOT_PATH)
    # 10. Print summary to terminal
    # These messages confirm that training finished and show where the outputs were saved
    print("Training completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    print(f"Decision tree plot saved to: {TREE_PLOT_PATH}")
    print()
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Recall for Fail class: {metrics['recall_fail_class']:.4f}")
    return pipeline, metrics

def parse_args():
    # This creates a command-line parser for the training script.
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree model for student performance prediction."
    )

    # This adds the required --data argument.
    parser.add_argument(
        "--data",
        required=True,
        help="Path to student-mat.csv or student-por.csv",
    )
    return parser.parse_args()


# This block runs only when the file is executed directly
if __name__ == "__main__":
    args = parse_args()
    train_model(args.data)
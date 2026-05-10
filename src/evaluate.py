"""Model evaluation and result export,
This file evaluates the trained model and saves useful result files
This file calculates general performance metrics and also focuses on
precision, recall and F1-score for the Fail class.
"""

# This import allows modern Python type annotation behavior
from __future__ import annotations

# pathlib is used to work with file paths in a clean and cross-platform way
from pathlib import Path

# matplotlib is used to create and save plots as image files
import matplotlib.pyplot as plt

# These sklearn metrics are used to evaluate the classification model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# plot_tree is used to create a visual representation of the trained Decision Tree
from sklearn.tree import plot_tree


def evaluate(model, X_test, y_test) -> dict:
    """Evaluate predictions on test data using metrics expected by unit tests."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "recall_fail": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
    }

def evaluate_classifier(model, X_train, X_test, y_train, y_test) -> dict:
    """Evaluate the trained model  
    model:The trained sklearn pipeline
    X_train, X_test: Training and test input data
    y_train, y_test: Training and test target labels
    """

    # This makes predictions on the training data
    # These predictions are used to calculate training accuracy
    train_pred = model.predict(X_train)

    # This makes predictions on the unseen test data
    # These predictions are more important because they show how the model behaves on new data
    test_pred = model.predict(X_test)

    # This dictionary stores all evaluation results in one place
    metrics = {
        # Training accuracy shows how well the model performs on the data it learned from
        "train_accuracy": accuracy_score(y_train, train_pred),

        # Test accuracy shows how well the model performs on unseen data
        "test_accuracy": accuracy_score(y_test, test_pred),

        # When the model predicts Fail, how often is it actually correct?
        # pos_label=0 means that class 0 is treated as the positive class here
        # class 0 means Fail
        # zero_division=0 prevents errors if the metric cannot be calculated
        "precision_fail_class": precision_score(y_test, test_pred, pos_label=0, zero_division=0),

     
        # Of all students who actually failed, how many did the model correctly identify?
        "recall_fail_class": recall_score(y_test, test_pred, pos_label=0, zero_division=0),

        # F1-score combines precision and recall into one balanced metric for the Fail class
        "f1_fail_class": f1_score(y_test, test_pred, pos_label=0, zero_division=0),

        # The classification report gives precision, recall, F1-score and support for both classes
        # target_names makes the output easier to read by showing Fail and Pass instead of 0 and 1
        "classification_report": classification_report(
            y_test,
            test_pred,
            target_names=["Fail", "Pass"],
            zero_division=0,
        ),
    }

    return metrics


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Save metrics to a text file.
    """

    # This creates the output folder if it does not already exist
    # parents=True also creates any missing parent folders
    # exist_ok=True avoids an error if the folder already exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # This opens the output text file in write mode using UTF-8 encoding.
    with output_path.open("w", encoding="utf-8") as f:
        # These lines write a simple title at the top of the metrics file.
        f.write("Student Performance Decision Tree Metrics\n")
        f.write("========================================\n\n")
        # This loop writes all numeric metrics except the classification report
        for key, value in metrics.items():
            # The classification report is text, not a single numeric value
            # Therefore, it is handled separately below
            if key != "classification_report":
                # The metric value is written with four decimal places
                f.write(f"{key}: {value:.4f}\n")

        # This writes the full classification report after the numeric metrics
        f.write("\nClassification report:\n")
        f.write(metrics["classification_report"])

def save_confusion_matrix(model, X_test, y_test, output_path: Path) -> None:
    """Save the confusion matrix plot, the confusion matrix shows what the model predicted correctly and incorrectly.
       Rows = actual classes.
       Columns = predicted classes.
    """
    # This creates the output folder if it does not already exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # This predicts the classes for the test dataset
    y_pred = model.predict(X_test)

    # This calculates the confusion matrix
    # labels=[0, 1] keeps the order fixed as Fail first and Pass second
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # This prepares a readable confusion matrix display with class names
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail", "Pass"],
    )

    # This draws the confusion matrix
    # values_format="d" means the values are displayed as integers
    display.plot(values_format="d")
    plt.title("Confusion Matrix")
    # This adjusts the layout so that labels and titles fit better
    plt.tight_layout()

    # This saves the confusion matrix as an image file
    plt.savefig(output_path, dpi=200)
    # This closes the plot to free memory and avoid overlapping plots later.
    plt.close()


def save_tree_plot(pipeline, output_path: Path) -> None:
    """Save a visualization of the trained Decision Tree
    """
    # This creates the output folder if it does not already exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The pipeline contains both preprocessing and the trained model
    # Here we extract the preprocessing step.
    preprocessor = pipeline.named_steps["preprocessor"]
    # Here we extract the Decision Tree model from the pipeline
    model = pipeline.named_steps["model"]

    # After one-hot encoding, feature names change
    # For example, a categorical column such as internet may become feature names
    # like categorical__internet_yes and categorical__internet_no.
    try:
        # This gets the final feature names after preprocessing
        feature_names = preprocessor.get_feature_names_out()

    # If feature names cannot be extracted, the plot is still created without them
    except Exception:
        feature_names = None
    # This sets the size of the Decision Tree plot
    # A large figure is used because tree diagrams can become wide
    plt.figure(figsize=(24, 12))
    # This draws the trained Decision Tree
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Fail", "Pass"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Decision Tree Classifier")

    # This adjusts the layout so the plot fits better in the saved image
    plt.tight_layout()
    # This saves the Decision Tree plot as an image file
    plt.savefig(output_path, dpi=200)
    # This closes the plot to free memory and avoid interfering with other plots
    plt.close()
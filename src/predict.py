# This import allows modern Python type annotation behavior
from __future__ import annotations

# argparse is used to read command-line arguments, such as the dataset path
import argparse

# joblib is used to load the saved machine learning pipeline from the models folder
import joblib

# MODEL_PATH contains the location of the saved trained model pipeline
from src.config import MODEL_PATH

# load_student_data loads the UCI Student Performance CSV file
from src.data_loader import load_student_data

# split_features_and_target separates the input features from the target column
from src.preprocessing import split_features_and_target

def load_model():
    """Load the saved pipeline.The saved object contains:
    - preprocessing
    - trained Decision Tree
    """
    # This checks whether the trained model file exists in the expected location
    # If the model has not been trained yet, prediction cannot continue
    if not MODEL_PATH.exists():
        # A clear error is raised so the user knows that training must be done first
        raise FileNotFoundError(
            f"Model was not found at {MODEL_PATH}. "
            "Train the model first with: python -m src.train --data data/student-mat.csv"
        )
    # This loads and returns the saved pipeline.
    return joblib.load(MODEL_PATH)

def predict_single_student(student_row):
    """Predict Pass or Fail for one student row.
    """
    # This loads the saved trained model pipeline
    model = load_model()

    # This predicts the class for the selected student row.
    prediction = model.predict(student_row)[0]

    # This returns the prediction probabilities for both classes
    # The order is [Fail, Pass] because class 0 is Fail and class 1 is Pass
    probability = model.predict_proba(student_row)[0]

    # This converts the numeric prediction into a readable label.
    label = "Pass" if prediction == 1 else "Fail"

    # This returns both the readable prediction and the probability values
    return label, probability


def demo_prediction(csv_path: str):
    """Run a prediction on the first row of the dataset."""

    # This loads the dataset from the CSV path provided by the user
    df = load_student_data(csv_path)

    # This uses the same preprocessing logic as training
    # It creates the target variable and removes G3 from the input features.
    X, _ = split_features_and_target(df)

    # This selects the first student row as a simple demonstration example
    example = X.iloc[[0]]

    # This runs the prediction for the selected example student
    label, probability = predict_single_student(example)

    # These print statements show the selected input and prediction results in the terminal
    print("Example input:")
    print(example.T)
    print()
    print(f"Prediction: {label}")
    print(f"Probability [Fail, Pass]: {probability}")


def parse_args():
    # This creates a command-line argument parser for the prediction script
    parser = argparse.ArgumentParser(description="Run a demo prediction.")

    # This adds the required --data argument.
    parser.add_argument(
        "--data",
        required=True,
        help="Path to student-mat.csv or student-por.csv",
    )
    # This reads the arguments from the command line and returns them
    return parser.parse_args()

# This block runs only when the file is executed directly as a script.
if __name__ == "__main__":
    # This reads the dataset path from the command-line arguments
    args = parse_args()

    # This runs the demo prediction using the provided dataset path
    demo_prediction(args.data)
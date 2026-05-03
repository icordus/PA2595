"""Prediction helper for the saved Decision Tree pipeline artifact."""

import os
import joblib
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PIPELINE_FILE = "decision_tree_pipeline.pkl"
FEATURE_COLUMNS_FILE = "feature_columns.pkl"


def load_model(model_name: str = "decision_tree_pipeline"):
    """Load a trained pipeline artifact from the models directory."""
    model_path = os.path.join(MODELS_DIR, PIPELINE_FILE if model_name == "decision_tree_pipeline" else f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {model_path}. Run train.py first."
        )
    return joblib.load(model_path)


def load_feature_columns() -> list:
    """Load the feature column order used during training."""
    path = os.path.join(MODELS_DIR, FEATURE_COLUMNS_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError("feature_columns.pkl not found. Run train.py first.")
    return joblib.load(path)


def predict(features: dict, model_name: str = "decision_tree_pipeline") -> dict:
    """
    Parameters
    ----------
    features : dict
        Keys are feature names, values are the input values.
    model_name : str
        Default: 'decision_tree_pipeline'

    Returns
    -------
    dict with keys:
        'label'       : 'Pass' or 'Fail'
        'probability' : float, probability of Pass
    """
    model = load_model(model_name)
    columns = load_feature_columns()

    df = pd.DataFrame([features], columns=columns).fillna(0)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]  # probability of class 1 (Pass)

    return {
        "label": "Pass" if prediction == 1 else "Fail",
        "probability": float(probability),
    }


if __name__ == "__main__":
    # Quick sanity check with a sample student profile
    sample = {
        "studytime": 2,
        "absences": 4,
        "failures": 0,
        "G1": 12,
        "G2": 13,
        "Medu": 3,
        "Fedu": 2,
        "traveltime": 1,
        "freetime": 3,
        "goout": 2,
        "Dalc": 1,
        "Walc": 2,
        "health": 4,
        "internet": 1,
        "higher": 1,
        "sex": 1,
        "address": 1,
        "famsize": 0,
        "Pstatus": 1,
        "schoolsup": 0,
        "famsup": 1,
        "paid": 0,
        "activities": 1,
        "nursery": 1,
        "romantic": 0,
    }

    result = predict(sample)
    print(f"Prediction : {result['label']}")
    print(f"Pass probability : {result['probability']:.2%}")

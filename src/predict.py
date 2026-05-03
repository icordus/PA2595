"""
predict.py
----------
Loads the best saved model and makes a single prediction given a dictionary
of feature values. Used internally by the Streamlit prototype.

Can also be run directly for a quick sanity check:
    python src/predict.py
"""

import os
import joblib
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_model(model_name: str = "random_forest"):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {model_path}. Run train.py first."
        )
    return joblib.load(model_path)


def load_feature_columns() -> list:
    path = os.path.join(MODELS_DIR, "feature_columns.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("feature_columns.pkl not found. Run train.py first.")
    return joblib.load(path)


def predict(features: dict, model_name: str = "random_forest") -> dict:
    """
    Parameters
    ----------
    features : dict
        Keys are feature names, values are the input values.
    model_name : str
        One of: 'decision_tree', 'random_forest', 'logistic_regression'

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

    result = predict(sample, model_name="random_forest")
    print(f"Prediction : {result['label']}")
    print(f"Pass probability : {result['probability']:.2%}")

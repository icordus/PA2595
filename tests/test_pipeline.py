# pandas is used here to create a small fake dataset for testing
import pandas as pd
# DecisionTreeClassifier is used as a simple model inside the pipeline test
from sklearn.tree import DecisionTreeClassifier
# These project functions are tested in this file.
from src.preprocessing import split_features_and_target, build_pipeline
def make_fake_dataset():
    """Create a tiny fake dataset for testing
    """
    # This returns a small pandas DataFrame that imitates the structure of the real dataset
    # It contains both categorical and numerical columns so that preprocessing can be tested
    return pd.DataFrame(
        {
            # These are categorical features similar to the real UCI dataset
            "school": ["GP", "MS", "GP", "MS"],
            "sex": ["F", "M", "F", "M"],
            # These are numerical input features.
            "age": [17, 18, 16, 19],
            "studytime": [2, 1, 3, 1],
            "failures": [0, 2, 0, 3],
            "absences": [4, 10, 2, 15],
            "G1": [12, 8, 15, 6],
            "G2": [13, 7, 15, 5],
            # G3 is the final grade.
            # It is used to create the Pass or Fail target, but it must be removed from X.
            "G3": [14, 8, 16, 4],
        }
    )

def test_split_features_and_target_removes_g3():
    # This creates the small fake dataset used by the test
    df = make_fake_dataset()
    # This applies the same feature - target split used in the real pipeline
    X, y = split_features_and_target(df)
    # This checks that G3 was removed from the input features.
    assert "G3" not in X.columns
    # This checks that the number of input rows matches the number of target labels
    assert len(X) == len(y)
    # This checks that the target contains only valid binary labels: 0 and 1.
    assert set(y.unique()).issubset({0, 1})


def test_pipeline_can_train_and_predict():
    """Check that preprocessing and Decision Tree can train and predict
    """

    # This creates the small fake dataset used by the test
    df = make_fake_dataset()
    # This splits the fake dataset into input features and target labels
    X, y = split_features_and_target(df)
    # This creates a small Decision Tree model for testing
    model = DecisionTreeClassifier(max_depth=2, random_state=42)
    # This builds the full pipeline with preprocessing and the Decision Tree model
    pipeline = build_pipeline(X, model)
    # This trains the pipeline on the fake dataset
    pipeline.fit(X, y)
    # This uses the trained pipeline to make predictions on the same fake dataset
    predictions = pipeline.predict(X)
    # This checks that the pipeline returns one prediction for each input row
    assert len(predictions) == len(X)
    # This checks that all predictions are valid binary labels
    # The model should only return 0 for Fail or 1 for Pass
    assert set(predictions).issubset({0, 1})
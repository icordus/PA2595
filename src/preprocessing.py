"""Preprocessing functions, This file prepares the raw dataset for machine learning.
Main steps:
1. Create the binary Pass or Fail target from G3
2. Remove G3 from the input features to avoid data leakage
3. Detect categorical and numerical columns
4. Build a preprocessing transformer for the model pipeline
"""

# This import allows modern Python type annotation behavior
from __future__ import annotations
# Tuple is used in the function type hints to show that a function returns two values
from typing import Tuple
# pandas is used for working with tabular data such as DataFrames and Series
import pandas as pd
# ColumnTransformer allows different preprocessing steps to be applied to different column types
from sklearn.compose import ColumnTransformer
# OneHotEncoder converts categorical text values into numerical columns
from sklearn.preprocessing import OneHotEncoder
# Pipeline connects preprocessing and the machine learning model into one object
from sklearn.pipeline import Pipeline
# These configuration values are imported from config.py so they are defined in one central place
from src.config import TARGET_COLUMN, PASS_THRESHOLD, DROP_COLUMNS

def create_target(df: pd.DataFrame) -> pd.Series:
    # This checks that the expected target column exists in the dataset
    if TARGET_COLUMN not in df.columns:
        # A clear error is raised if the wrong dataset is loaded
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    # This creates the binary target variable.
    # If the final grade is greater than or equal to the threshold, the value becomes 1
    # If the final grade is below the threshold, the value becomes 0
    # G3 = 12 -> 1 (Pass)
    # G3 = 8  -> 0 (Fail)
    y = (df[TARGET_COLUMN] >= PASS_THRESHOLD).astype(int)

    # This returns the target values as a pandas Series
    return y

def split_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # This creates the target variable before removing G3 from the input features
    y = create_target(df)
    # This keeps only the columns from DROP_COLUMNS that actually exist in the dataset
    columns_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    # This removes G3 from the input features.
    X = df.drop(columns=columns_to_drop)
    # This returns both the model input features and the target variable.
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # This finds all categorical columns
    # In this dataset, categorical columns usually have the object data type
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    # This finds all numerical columns
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()
    # This creates a preprocessing transformer
    # It applies OneHotEncoder to categorical features and passes numeric features unchanged
    preprocessor = ColumnTransformer(
        transformers=[
            # This transformer converts categorical values into numerical one-hot encoded columns
            # handle_unknown="ignore" prevents errors if a new category appears during prediction
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            # Decision Trees do not require feature scaling
            # Therefore, numeric columns can pass through unchanged
            ("numeric", "passthrough", numeric_features),
        ]
    )
    return preprocessor


def build_pipeline(X: pd.DataFrame, model) -> Pipeline:
    # This builds the preprocessing step based on the input feature columns
    preprocessor = build_preprocessor(X)
    # This creates one sklearn Pipeline containing two steps:
    # 1. preprocessor: prepares the data
    # 2. model: trains or predicts using the prepared data
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline
"""Data loading utilities
This file is responsible for loading the original CSV dataset safely
The project uses the UCI Student Performance dataset. These CSV files use
semicolon separators instead of normal comma separators, so the file must be
loaded carefully. This file also performs simple validation checks before the dataset is returned
to the rest of the machine learning pipeline.
"""
# pandas is used for reading the CSV file and storing the dataset as a DataFrame
import pandas as pd

def load_student_data(csv_path: str) -> pd.DataFrame:
    """Load the UCI Student Performance dataset.The UCI Student Performance CSV files use semicolon separators, not commas.
    If the default comma separator is used, pandas will read the whole row as
    one single column, which would break the preprocessing and training steps
    """
    # This line reads the CSV dataset from the provided file path
    # sep=";" is required because the UCI dataset separates values with semicolons
    df = pd.read_csv(csv_path, sep=";")

    # This check confirms that the dataset contains the G3 column
    # G3 is necessary because the project uses it as the original final grade
    # before converting it into the binary Pass οor Fail target
    if "G3" not in df.columns:
        # If G3 is missing, the wrong file was probably loaded
        # The program stops here with a clear error message instead of failing later
        raise ValueError(
            "Expected column 'G3' was not found. "
            "Check that you loaded the correct UCI Student Performance file"
        )
    # This check confirms that the dataset contains at least one row
    # An empty dataset cannot be used for training, testing or prediction
    if df.empty:
        # The program stops with a clear error message if the dataset has no records
        raise ValueError("The dataset is empty.")

    # If all validation checks pass the loaded dataset is returned
    return df
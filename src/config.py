"""Central configuration file for the project
Instead of repeating paths, target names, thresholds and model settings in many files,
we keep them here. This makes the project easier to maintain and easier to explain in
the report. If we want to change the pass or fail threshold or the Decision Tree depth later, we change
it here only once.
"""
# pathlib is used to work with file and folder paths in a clean and cross-platform way.
from pathlib import Path

# ---------------------------------------------------------------------------
# Project folders
# ---------------------------------------------------------------------------
# __file__ is the path of this file, which is src/config.py.
# resolve() converts it into an absolute path
# parents[1] moves two levels up:
# src/config.py -> src/ -> project root/
# This gives us the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# This folder is used to store the trained machine learning model
# The model is saved here after running the training script
MODEL_DIR = PROJECT_ROOT / "models"

# This folder is used to store evaluation results
# For example, metrics and plots are saved in this folder
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------
# This is the file path where the trained pipeline will be saved
# The project saves the full pipeline, not only the Decision Tree model
# This is important because the pipeline includes both preprocessing and prediction
MODEL_PATH = MODEL_DIR / "student_performance_decision_tree.joblib"

# This text file stores the evaluation metrics after training
# It can contain results such as accuracy, precision, recall, F1-score and classification report
METRICS_PATH = RESULTS_DIR / "metrics.txt"

# This image file stores the confusion matrix plot
# The confusion matrix helps us see correct and incorrect predictions for Pass and Fail
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"

# This image file stores the Decision Tree visualization
# It helps explain how the trained tree makes decisions
TREE_PLOT_PATH = RESULTS_DIR / "decision_tree.png"


# ---------------------------------------------------------------------------
# Dataset and target settings
# ---------------------------------------------------------------------------
# In the UCI Student Performance dataset, G3 is the final grade column
# The project uses this column to create the binary Pass/Fail target
TARGET_COLUMN = "G3"

# This threshold converts the final grade into a binary class
# If G3 is greater than or equal to 10, the student is classified as Pass
# If G3 is less than 10, the student is classified as Fail
PASS_THRESHOLD = 10

# G3 must be removed from the input features after the target is created.
# If G3 remained inside the input features, the model would already see the final answer.
# That would cause data leakage and make the evaluation unrealistic.
DROP_COLUMNS = ["G3"]

# ---------------------------------------------------------------------------
# Reproducibility settings
# ---------------------------------------------------------------------------
# This random seed makes the results more reproducible.
# It is used in the train or test split and in the Decision Tree model.
# With the same seed, another run should produce the same or very similar results.
RANDOM_STATE = 42

# This means that 20% of the dataset is used for testing.
# The remaining 80% is used for training the model.
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Decision Tree settings
# ---------------------------------------------------------------------------
# These are the main parameters of the Decision Tree Classifier.
# Keeping them here makes the model configuration easy to find and change.
DECISION_TREE_PARAMS = {
    # The Gini criterion is used to measure how well a split separates the classes.
    "criterion": "gini",

    # max_depth limits how deep the tree can grow.
    # This helps prevent the model from becoming too complex and overfitting the training data.
    "max_depth": 4,

    # min_samples_leaf requires each final leaf node to contain at least 5 samples.
    # This also helps reduce overfitting and makes the tree more stable.
    "min_samples_leaf": 5,

    # The random state makes the model training more reproducible.
    "random_state": RANDOM_STATE,
}

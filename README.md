# Student Performance Risk Prediction
## Project Overview
This project is a small Machine Learning Engineering prototype for predicting student academic performance risk. The goal is to classify a student as either Pass or Fail based on information from the UCI Student Performance dataset. The project uses a supervised binary classification approach. The original dataset contains the final grade in the column G3. In this project, G3 is converted into a binary target variable:
G3 >= 10  -> Pass
G3 < 10   -> Fail
After the target is created, G3 is removed from the input features. This is an important design decision because keeping G3 inside the training data would create data leakage. In that case, the model would already have access to the final answer and the evaluation results would not be meaningful.
The model used in this project is a Decision Tree Classifier. This model was selected because it is simple, suitable for tabular data and easier to explain than many more complex machine learning models. 
------------------------------------------------------------------------------------------------------------------------------------------------------
## Project Structure
The project is organised into separate folders and files so that each part of the machine learning pipeline has a clear responsibility
StudentPerformanceDecision/
│
├── data/
├── models/
├── results/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   └── test_pipeline.py
│
├── app.py
├── pytest.ini
└── README.md
------------------------------------------------------------------------------------------------------------------------------------------------------
## Folders and Files Explanation
### Folder data
The data folder is used to store the dataset files used by the project.
The project expects the UCI Student Performance dataset to be placed inside this folder. For example, the dataset file can be stored as: data/student-mat.csv or data/student-por.csv. The dataset is not included in this project folder. It must be downloaded separately and placed inside the data folder before training the model.This folder is separated from the source code because data should not be mixed with Python implementation files. Keeping the dataset in a dedicated folder makes the project cleaner and easier to run.
------------------------------------------------------------------------------------------------------------------------------------------------------
### Folder models
The models folder is used to store the trained machine learning model.After the training script is executed, the project saves the trained pipeline in this folder. The expected saved model file is: models/student_performance_decision_tree.joblib. The saved object is not only the Decision Tree model. It is the full machine learning pipeline, including both preprocessing and the trained classifier. This is important because the same preprocessing steps used during training must also be used during prediction. If only the model was saved without preprocessing, prediction could fail or produce inconsistent results. Saving the full pipeline makes the project more reproducible and safer to use.
------------------------------------------------------------------------------------------------------------------------------------------------------
### Folder results
The results folder is used to store evaluation outputs generated after training.The training process can produce files such as:
results/metrics.txt, results/confusion_matrix.png and results/decision_tree.png
- The metrics.txt file contains numerical evaluation results such as accuracy, precision, recall, F1-score and the classification report
- The confusion_matrix.png file shows how many students were correctly or incorrectly classified as Pass or Fail
- The decision_tree.png file visualises the trained Decision Tree. 
This folder is useful because it keeps the model outputs separate from the source code. 
------------------------------------------------------------------------------------------------------------------------------------------------------
### Folder src
The src folder contains the main Python source code of the machine learning pipeline. The code is divided into several files instead of being written in one large script. This makes the project easier to understand, test, debug, and maintain. Each file has a specific responsibility, such as loading data, preparing features, training the model, evaluating results or running predictions. This structure also supports good Machine Learning Engineering practice because the pipeline is more modular and reproducible. 
------------------------------------------------------------------------------------------------------------------------------------------------------
###  File__init__.py in src folder
The __init__.py file marks the src folder as a Python package.
This allows the project files to import functions from the src folder using imports such as:
- from src.preprocessing import split_features_and_target
The file does not need to contain much code. Its purpose is mainly structural. Without it, Python package imports may not work correctly in some environments. 
------------------------------------------------------------------------------------------------------------------------------------------------------
### File config.py in src folder
The config.py file contains the central configuration settings of the project
This file defines important values such as:
- the project root folder
- the model output path
- the results output paths
- the target column name
- the Pass or Fail threshold
- the train or test split size
- the random seed
- the Decision Tree parameters
For example, the project uses G3 as the original final grade column and converts it into the binary target using this rule:
G3 >= 10  -> Pass
G3 < 10   -> Fail
The file also defines that G3 must be removed from the input features. This prevents data leakage.
Using a separate configuration file is useful because important settings are not repeated across many files. If the threshold, model path, or Decision Tree parameters need to change later, they can be changed in one place. 
------------------------------------------------------------------------------------------------------------------------------------------------------
### File data_loader.py in src folder
The data_loader.py file is responsible for loading the dataset.The UCI Student Performance dataset uses semicolon-separated CSV files. For this reason, the loader reads the file using: 
- pd.read_csv(csv_path, sep=";")
This is important because using the normal comma separator would load the data incorrectly.
The file also performs basic validation. It checks that the dataset contains the G3 column, because G3 is required to create the Pass or Fail target. It also checks that the dataset is not empty. This file exists so that data loading is handled in one place instead of being repeated in different scripts.
------------------------------------------------------------------------------------------------------------------------------------------------------
### File preprocessing.py in src folder
The preprocessing.py file prepares the raw dataset for machine learning.
Its main responsibilities are:
- creating the binary target variable from G3
- splitting the dataset into input features X and target labels y
- removing G3 from the input features
- detecting categorical and numerical columns
- applying one-hot encoding to categorical variables
- building the preprocessing and model pipeline
The target is created as follows: 1 = Pass or 0 = Fail
Categorical columns, such as school, sex, internet or guardian, cannot be used directly by scikit-learn models. They must first be converted into numerical form. This is done with OneHotEncoder
Numerical columns, such as age, studytime, failures and absences are passed through without scaling. This is acceptable because Decision Tree models do not require feature scaling in the same way that some other models do. This file is one of the most important parts of the project because it controls how raw student data becomes usable input for the machine learning model 
------------------------------------------------------------------------------------------------------------------------------------------------------
### File train.py in src folder
The train.py file is the main training script of the project.
It runs the full training workflow from start to finish. 
The script:
1. Loads the dataset.
2. Creates the Pass or Fail target.
3. Splits the data into features and target labels.
4. Splits the dataset into training and test sets.
5. Builds the preprocessing and Decision Tree pipeline.
6. Trains the model.
7. Evaluates the model.
8. Saves the trained pipeline.
9. Saves the metrics and plots.
The training script uses an 80 - 20 train - test split. This means that 80% of the data is used for training and 20% is kept for testing. The split also uses stratification. This keeps the Pass or Fail distribution approximately similar in both the training and test sets. This is useful because the dataset may not contain an equal number of Pass and Fail examples. The script can be run from the project root with:
- python -m src.train --data data/student-mat.csv
After successful training, the trained model is saved in the models folder and the evaluation outputs are saved in the results folder.
------------------------------------------------------------------------------------------------------------------------------------------------------
### File evaluate.py in src Folder
 The evaluate.py file contains the evaluation logic for the trained model
This file calculates several metrics, including:
- training accuracy
- test accuracy
- precision for the Fail class
- recall for the Fail class
- F1-score for the Fail class
- classification report
Accuracy alone is not enough for this project. The Fail class is especially important because the purpose of the prototype is to identify students who may be at academic risk. For example, recall for the Fail class shows how many actual failing students were correctly identified by the model. This is important because missing an at-risk student may be more serious than incorrectly flagging a student for review.
The file also saves visual outputs which are results/confusion_matrix.png and results/decision_tree.png 
------------------------------------------------------------------------------------------------------------------------------------------------------
### File predict.py in src Folder
The predict.py file demonstrates how to load the saved model and make a prediction. It loads the trained pipeline from the models folder and uses it to predict whether a selected student is classified as Pass or Fail. The script can be run from the project root with:
- python -m src.predict --data data/student-mat.csv
For simplicity, the script uses the first row of the dataset as an example prediction. This is useful as a command-line demonstration that the saved model can be loaded and used after training. The prediction output also includes probabilities for both classes: Probability [Fail, Pass]
This file is useful because it shows that the project does not stop at training. It also demonstrates how the trained model can be reused for prediction.
------------------------------------------------------------------------------------------------------------------------------------------------------
### Folder tests
The tests folder contains software tests for the project. The tests do not prove that the model is highly accurate. They prove that important parts of the software pipeline work as expected.
------------------------------------------------------------------------------------------------------------------------------------------------------
###  File test_pipeline.py in tests folder
The test_pipeline.py file contains unit tests for the preprocessing and pipeline logic. The tests use a small fake dataset instead of the real CSV file. This makes the tests fast, simple and independent from external data files. The tests check that:
- G3 is removed from the input features
- the target variable contains only binary values
- the pipeline can train successfully
- the pipeline can return valid predictions
One important test protects against data leakage. It checks that G3 is not included in the input features after preprocessing. This is important because G3 is the final grade and should not be used as an input feature when predicting Pass or Fail. The tests can be run with: pytest
------------------------------------------------------------------------------------------------------------------------------------------------------
### File app.py
The app.py file contains the Streamlit prototype interface. The Streamlit app provides a simple user interface where the user can:
- enter the dataset path
- load the dataset
- select an existing student row
- run a Pass or Fail prediction
- view the predicted class
- view the probability of Fail and Pass
The app can be started with: streamlit run app.py
The app requires a trained model before it can run correctly. If the model file does not exist, the app shows an error message and asks the user to train the model first. The prototype also includes an ethical warning. It clearly states that the prediction must not be used as the only basis for real student decisions. This is important because educational decisions should involve human judgement and context, not only an automated prediction.
------------------------------------------------------------------------------------------------------------------------------------------------------
### File pytest.ini
The pytest.ini file contains configuration for pytest. It tells pytest where the test files are located and ensures that the project root is included in the Python path. The file includes:
ini
[pytest]
pythonpath = .
testpaths = tests
This makes it easier to run tests without import errors. It also keeps the testing configuration explicit and clear.
------------------------------------------------------------------------------------------------------------------------------------------------------
### File README.md
The README.md file explains the project to another person who opens the folder.
It should describe:
- what the project does
- how the project is structured
- what each folder and file is used for
- how to run training
- how to run prediction
- how to run the Streamlit prototype
- how to run tests
- what the limitations of the project are
A clear README is important because the examiner or another developer should be able to understand the project without reading every Python file first.
------------------------------------------------------------------------------------------------------------------------------------------------------
## How to Run the Project
### 1. Place the dataset in the data folder
Place the CSV file inside the data folder.
Example: data/student-mat.csv
------------------------------------------------------------------------------------------------------------------------------------------------------
### 2. Install the required Python packages
The project uses packages such as:
text
pandas
scikit-learn
matplotlib
joblib
streamlit
pytest
If a requirements.txt file is added the installation can be done with: pip install -r requirements.txt
Otherwise, the packages can be installed manually.
------------------------------------------------------------------------------------------------------------------------------------------------------
### 3. Train the model
Run the training script from the project root: python -m src.train --data data/student-mat.csv
This command trains the Decision Tree pipeline and saves the model and evaluation outputs.
------------------------------------------------------------------------------------------------------------------------------------------------------
### 4. Run a command-line prediction
After training, run: python -m src.predict --data data/student-mat.csv
This loads the saved model and performs a simple example prediction.
------------------------------------------------------------------------------------------------------------------------------------------------------
### 5. Run the Streamlit prototype
After training, run: streamlit run app.py
This starts the prototype interface in the browser.
------------------------------------------------------------------------------------------------------------------------------------------------------
### 6. Run the tests
Run: pytest
This checks that the basic pipeline works correctly.
------------------------------------------------------------------------------------------------------------------------------------------------------





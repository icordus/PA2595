"""docstring 
app.py is the Streamlit prototype of the project. It does not train the model.
It uses an already trained model from the models/ folder.
The basic flow is:
1. It opens a simple web interface.
2. It asks for the dataset path.
3. It checks whether a trained model exists.
4. It loads the dataset.
5. It removes the target variable and keeps only the input features.
6. It loads the saved model pipeline.
7. It allows the user to select an existing student row.
8. It makes a Pass or Fail prediction.
9. It displays the probabilities for Fail amd Pass.
Run from project root: streamlit run app.py
"""

#This import makes Python handle type annotations in a more modern and flexible way
from __future__ import annotations

#joblib is used to load the trained machine learning pipeline from the models folder 
import joblib

#streamlit is used to create the simple web interface for the prototype 
import streamlit as st

#MODEL_PATH stores the file path where the trained model pipeline should be saved
from src.config import MODEL_PATH

#load_student_data reads the student dataset from the selected CSV file path
from src.data_loader import load_student_data

#split_features_and_target prepares the input features and removes the target column from the model input 
from src.preprocessing import split_features_and_target

#This sets the basic Streamlit page settings, including the browser page title and layout
st.set_page_config(
    page_title="Student Performance Risk Prediction",
    layout="centered",
)

#This displays the main title at the top of the web application 
st.title("Student Performance Risk Prediction")

#This text explains the purpose and limitation of the prototype to the user. 
st.write(
    "This prototype predicts whether a student is likely to pass or fail. "
    "It is a decision-support prototype not an automatic educational decision system."
)

#This warning makes the ethical limitation clear inside the prototype interface. 
st.warning(
    "Important: This prediction must not be used as the only basis for real student decisions."
)

#This input field allows the user to provide the dataset path. The default value points to the expected dataset location. 
data_path = st.text_input("Dataset path", "data/student-mat.csv")

#The app needs a trained model before it can make predictions. This condition checks whether the model file exists. 
if not MODEL_PATH.exists():
    #If the trained model is missing, the app shows an error message with the training mand. 
    st.error(
        "Trained model not found. Train it first by using: "
        "python -m src.train --data data/student-mat.csv"
    )
    #st.stop() stops the Streamlit app here, because prediction is impossible without a trained model. 
    st.stop()

#This block tries to load the dataset and prepare the input features used for prediction. 
try:
    #This loads the student dataset from the path entered by the user. 
    df = load_student_data(data_path)
    #This separates the input features from the target variable. The target is ignored here because the app only needs input features for prediction 
    X, _ = split_features_and_target(df)

#If loading or preprocessing fails, the app catches the error and shows it to the user instead of crashing. 
except Exception as ex:
    #This displays the dataset loading or preprocessing error inside the Streamlit interface. 
    st.error(f"Could not load dataset: {ex}")

    #The app stops because it cannot continue without valid data. 
    st.stop()

#This loads the saved preprocessing and Decision Tree pipeline from the model file. 
model = joblib.load(MODEL_PATH)

#This section title tells the user that the next control is used to select a student record. 
st.subheader("Select an existing student row for demo prediction")

#The app selects an existing student row instead of asking the user to manually enter all feature values. 
row_index = st.number_input(
    "Student row index",

    #The minimum valid row index is 0 because pandas rows are zero-based. 
    min_value=0,

    #The maximum valid row index is the last available row in the feature dataset. 
    max_value=len(X) - 1,

    #The default selected row is the first student record. 
    value=0,

    #The input increases or decreases by one row at a time. 
    step=1,
)

#This selects one student row as a DataFrame, because the trained pipeline expects a table-like input. 
student = X.iloc[[row_index]]

#This label is shown before displaying the selected student input values. 
st.write("Selected student input:")

#This displays the selected student row in a table inside the Streamlit app. 
st.dataframe(student)

#This button starts the prediction only when the user clicks it. 
if st.button("Predict"):
    #This uses the trained model pipeline to predict the class for the selected student. 
    prediction = model.predict(student)[0]

    #This returns the prediction probabilities for Fail and Pass. 
    probability = model.predict_proba(student)[0]

    #The model uses 1 for Pass and 0 for Fail, so this converts the numeric output into a readable label. 
    label = "Pass" if prediction == 1 else "Fail"

    #This converts the predicted label into a simple academic risk explanation. 
    risk_text = "Low academic risk" if label == "Pass" else "High academic risk"

    #This section shows the final prediction output to the user. 
    st.subheader("Prediction result")

    #This displays the predicted class as Pass or Fail. 
    st.write(f"Predicted class: **{label}**")

    #This displays the simple risk interpretation of the prediction. 
    st.write(f"Risk indication: **{risk_text}**")

    #This displays the model probability for the Fail class. 
    st.write(f"Probability of Fail: **{probability[0]:.2f}")

    #This displays the model probability for the Pass class. 
    st.write(f"Probability of Pass: **{probability[1]:.2f}")
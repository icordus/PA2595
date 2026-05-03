"""
app.py — Streamlit Prediction Prototype
----------------------------------------
A simple web interface where the user enters student attributes
and the system predicts whether the student is expected to Pass or Fail.

Run:
    streamlit run prototype/app.py

Requires that preprocess.py and train.py have been run first.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from src.predict import predict, load_feature_columns

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Student Academic Performance Predictor")
st.markdown(
    "Enter the student's details below and the system will predict whether "
    "the student is likely to **Pass** or **Fail**."
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — model selection
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox(
    "Select model",
    options=["random_forest", "decision_tree", "logistic_regression"],
    format_func=lambda x: x.replace("_", " ").title(),
)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Student Profile")

col1, col2 = st.columns(2)

with col1:
    studytime = st.slider("Weekly study time (1=<2h … 4=>10h)", 1, 4, 2)
    absences = st.number_input("Number of absences", min_value=0, max_value=93, value=4)
    failures = st.slider("Past class failures (0–4)", 0, 4, 0)
    G1 = st.slider("First period grade (0–20)", 0, 20, 12)
    G2 = st.slider("Second period grade (0–20)", 0, 20, 13)
    Medu = st.slider("Mother's education (0=none … 4=higher)", 0, 4, 2)
    Fedu = st.slider("Father's education (0=none … 4=higher)", 0, 4, 2)
    traveltime = st.slider("Travel time to school (1=<15min … 4=>1h)", 1, 4, 1)
    freetime = st.slider("Free time after school (1=very low … 5=very high)", 1, 5, 3)
    goout = st.slider("Going out with friends (1=very low … 5=very high)", 1, 5, 3)
    Dalc = st.slider("Workday alcohol consumption (1=very low … 5=very high)", 1, 5, 1)
    Walc = st.slider("Weekend alcohol consumption (1=very low … 5=very high)", 1, 5, 2)
    health = st.slider("Current health status (1=very bad … 5=very good)", 1, 5, 4)

with col2:
    internet = st.selectbox("Internet access at home", ["Yes", "No"])
    higher = st.selectbox("Wants to pursue higher education", ["Yes", "No"])
    sex = st.selectbox("Sex", ["Female", "Male"])
    address = st.selectbox("Address type", ["Urban", "Rural"])
    famsize = st.selectbox("Family size", ["≤ 3 members (LE3)", "> 3 members (GT3)"])
    Pstatus = st.selectbox("Parents living together", ["Yes (T)", "No (A)"])
    schoolsup = st.selectbox("Extra educational school support", ["Yes", "No"])
    famsup = st.selectbox("Family educational support", ["Yes", "No"])
    paid = st.selectbox("Extra paid classes", ["Yes", "No"])
    activities = st.selectbox("Extra-curricular activities", ["Yes", "No"])
    nursery = st.selectbox("Attended nursery school", ["Yes", "No"])
    romantic = st.selectbox("In a romantic relationship", ["Yes", "No"])

# ---------------------------------------------------------------------------
# Encode inputs to match training encoding
# ---------------------------------------------------------------------------
def yes_no(val: str) -> int:
    return 1 if val == "Yes" else 0

features = {
    "studytime": studytime,
    "absences": absences,
    "failures": failures,
    "G1": G1,
    "G2": G2,
    "Medu": Medu,
    "Fedu": Fedu,
    "traveltime": traveltime,
    "freetime": freetime,
    "goout": goout,
    "Dalc": Dalc,
    "Walc": Walc,
    "health": health,
    "internet": yes_no(internet),
    "higher": yes_no(higher),
    "sex": 1 if sex == "Male" else 0,
    "address": 1 if address == "Urban" else 0,
    "famsize": 0 if "LE3" in famsize else 1,
    "Pstatus": 1 if "T" in Pstatus else 0,
    "schoolsup": yes_no(schoolsup),
    "famsup": yes_no(famsup),
    "paid": yes_no(paid),
    "activities": yes_no(activities),
    "nursery": yes_no(nursery),
    "romantic": yes_no(romantic),
}

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    try:
        result = predict(features, model_name=model_name)
        label = result["label"]
        prob = result["probability"]

        if label == "Pass":
            st.success(f"### Prediction: **{label}** ✅")
            st.progress(prob, text=f"Pass probability: {prob:.1%}")
        else:
            st.error(f"### Prediction: **{label}** ❌")
            st.progress(1 - prob, text=f"Fail probability: {1 - prob:.1%}")

        st.caption(
            f"Model used: **{model_name.replace('_', ' ').title()}** — "
            f"Pass probability: {prob:.2%}"
        )
    except FileNotFoundError as e:
        st.error(
            f"Model not found. Please run `python src/preprocess.py` and "
            f"`python src/train.py` first.\n\nDetails: {e}"
        )

st.divider()
st.caption(
    "PA2595 Machine Learning Engineering — Student Academic Performance Predictor. "
    "Dataset: UCI Student Performance (https://archive.ics.uci.edu/dataset/320/student%2Bperformance)"
)

"""Streamlit decision-support prototype for student risk prediction."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st
from src.predict import predict

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "student-mat.csv")

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
    "This tool predicts whether a student is likely to **Pass** or **Fail** "
    "based on structured academic and demographic data."
)
st.warning(
    "Decision-support only: this prototype should support, not replace, human academic judgment."
)
st.divider()

if not os.path.exists(RAW_DATA_PATH):
    st.error("Dataset not found at data/raw/student-mat.csv. Add the file and refresh.")
    st.stop()

df = pd.read_csv(RAW_DATA_PATH, sep=";")
features_df = df.drop(columns=["G3"], errors="ignore")

st.sidebar.header("Data Source")
st.sidebar.caption(f"Loaded {len(features_df)} student records")

selected_index = st.sidebar.slider(
    "Select student row",
    min_value=0,
    max_value=max(len(features_df) - 1, 0),
    value=0,
)

selected_record = features_df.iloc[selected_index].to_dict()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Selected Student Record")
st.dataframe(pd.DataFrame([selected_record]), use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    try:
        result = predict(selected_record)
        label = result["label"]
        prob = result["probability"]

        if label == "Pass":
            st.success(f"### Prediction: **{label}** ✅")
            st.progress(prob, text=f"Pass probability: {prob:.1%}")
        else:
            st.error(f"### Prediction: **{label}** ❌")
            st.progress(1 - prob, text=f"Fail probability: {1 - prob:.1%}")

        st.caption(f"Model used: **Decision Tree Pipeline** — Pass probability: {prob:.2%}")
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

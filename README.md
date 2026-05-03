# PA2595 — Student Academic Performance Prediction

**Course:** PA2595 Machine Learning Engineering  
**Project deadline:** 29 May 2026  
**Next meeting:** 3 May 2026 at 12:00 PM (Sweden time)

---

## Project Overview

This project predicts student academic performance (Pass / Fail) using supervised machine learning on structured educational and demographic data. Three models are compared:

- **Decision Tree**
- **Random Forest**
- **Logistic Regression**

Models are evaluated using Accuracy, Precision, Recall, F1-score, and Confusion Matrix. A simple Streamlit prototype allows users to enter student attributes and receive a real-time prediction.

---

## Dataset

**UCI Student Performance Dataset**  
649 records — demographic, social, and school-related features + grades.  
URL: https://archive.ics.uci.edu/dataset/320/student%2Bperformance

Place the downloaded `student-mat.csv` (or `student-por.csv`) file in `data/raw/`.

---

## Project Structure

```
PA2595/
├── data/
│   ├── raw/                  # Original downloaded dataset
│   └── processed/            # Cleaned and encoded data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── preprocess.py         # Data loading, cleaning, encoding, splitting
│   ├── train.py              # Model training and saving
│   ├── evaluate.py           # Metrics and confusion matrix
│   └── predict.py            # Load saved model and predict
├── models/                   # Saved trained model files (.pkl)
├── prototype/
│   └── app.py                # Streamlit prediction prototype
├── report/                   # Report, timesheet, contribution statement
├── tests/
│   ├── test_preprocess.py
│   ├── test_evaluate.py
│   └── test_predict.py
├── ROADMAP.md                # Project plan and milestones
├── start.sh                  # One-command pipeline start (bash)
├── start.ps1                 # One-command pipeline start (PowerShell)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Download the dataset

Download `student-mat.csv` from https://archive.ics.uci.edu/dataset/320/student%2Bperformance  
and place it in `data/raw/`.

---

## Running the Pipeline

### One-command start (recommended)

Linux/macOS/Git Bash:

```bash
bash start.sh
```

Windows PowerShell:

```powershell
.\start.ps1
```

This command runs all steps in order:

1. Installs dependencies (`python -m pip install -r requirements.txt`)
2. Runs preprocessing
3. Trains all models
4. Evaluates models
5. Starts the Streamlit prototype

If `student-mat.csv` or `student-por.csv` is missing from `data/raw/`, the script stops and shows what to download.

In PowerShell, if script execution is blocked on your machine, run this once in that terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

### Manual run (step by step)

### Preprocess data

```bash
python src/preprocess.py
```

### Train and evaluate models

```bash
python src/train.py
python src/evaluate.py
```

### Run the Streamlit prototype

```bash
python -m streamlit run prototype/app.py
```

### Run unit tests

```bash
python -m unittest discover -s tests -v
```

If you use the project virtual environment directly:

```bash
venv/Scripts/python -m unittest discover -s tests -v
```

Run a single test file:

```bash
venv/Scripts/python -m unittest tests/test_predict.py -v
```

Run a single test case:

```bash
venv/Scripts/python -m unittest tests.test_predict.TestPredict.test_predict_returns_label_and_probability -v
```

---

## Deliverables

- [ ] Project report (IEEE format)
- [ ] Timesheet diary (each member)
- [ ] Statement of individual contribution (each member)
- [ ] Source code with run instructions
- [ ] Trained model files

---

## Resources

| Resource | URL |
|---|---|
| UCI Student Performance | https://archive.ics.uci.edu/dataset/320/student%2Bperformance |
| pandas docs | https://pandas.pydata.org/docs/getting_started/index.html |
| scikit-learn docs | https://scikit-learn.org/stable/getting_started.html |
| Streamlit docs | https://docs.streamlit.io/get-started |
| Streamlit Community Cloud | https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started |
| IEEE Template Selector | https://template-selector.ieee.org/ |


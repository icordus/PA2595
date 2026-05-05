# PA2595 — Student Academic Performance Prediction

**Course:** PA2595 Machine Learning Engineering  
**Project deadline:** 29 May 2026  
**Next meeting:** 3 May 2026 at 12:00 PM (Sweden time)

---

## Project Overview

This project predicts student academic performance risk (Pass / Fail) using a **reproducible Decision Tree pipeline** on structured educational and demographic data.

The pipeline includes:

- Data loading and validation
- Target creation from final grade (`G3 >= 10 => Pass`, otherwise `Fail`)
- Leakage prevention by removing `G3` from model inputs
- One-hot encoding for categorical features (inside sklearn pipeline)
- Decision Tree training and evaluation
- Result artifact storage (metrics and plots)
- Streamlit decision-support prototype

The model is evaluated using Accuracy, Precision, Recall, F1-score, and Confusion Matrix, with explicit attention to **Fail-class recall**.

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
├── src/
│   ├── preprocess.py         # Load data, create target, split train/test
│   ├── train.py              # One-hot + Decision Tree pipeline training
│   ├── evaluate.py           # Metrics + confusion matrix + tree plot
│   ├── predict.py            # Load saved pipeline and predict
│   └── api.py                # FastAPI REST API (GET /health, POST /predict)
├── models/                   # Saved pipeline artifact (.pkl)
├── results/                  # metrics.txt, confusion_matrix.png, decision_tree.png
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

If you want to install the published package version from TestPyPI instead of using the local source code, use:

```bash
python -m pip install -r requirements-package.txt
```

`requirements-package.txt` installs the published `pa2595-student-risk` package from TestPyPI and uses the normal PyPI index for third-party dependencies such as pandas and scikit-learn.

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
3. Trains Decision Tree pipeline
4. Evaluates pipeline and saves result artifacts
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

Generated artifacts:

- `models/decision_tree_pipeline.pkl`
- `models/feature_columns.pkl`
- `results/metrics.txt`
- `results/confusion_matrix.png`
- `results/decision_tree.png`

### Run the Streamlit prototype

```bash
python -m streamlit run prototype/app.py
```

### Run the REST API

```bash
uvicorn src.api:app --reload --port 8000
```

Or via the installed entry point:

```bash
pa2595-api
```

The API will be available at `http://localhost:8000`.  
Interactive documentation (Swagger UI) is at `http://localhost:8000/docs`.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Check if the API is running |
| POST | `/predict` | Predict pass/fail risk for a student |

#### Example: student who passes

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"studytime": 3, "failures": 0, "absences": 2, "G1": 15, "G2": 16}'
```

```json
{"prediction": "pass", "risk": "low", "score": 15.5, "probability": 0.95, "source": "model"}
```

#### Example: student who fails

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"studytime": 1, "failures": 2, "absences": 25, "G1": 6, "G2": 7}'
```

```json
{"prediction": "fail", "risk": "high", "score": 6.5, "probability": 0.12, "source": "model"}
```

> **Note:** If no trained model is found, the API falls back to a grade-based heuristic and returns `"source": "heuristic"`. Run the pipeline first to generate the model artifacts.

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


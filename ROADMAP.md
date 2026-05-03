# Project Roadmap — PA2595 Student Academic Performance Prediction

**Project deadline: 29 May 2026**  
**Current date: 3 May 2026**  
**Time remaining: ~26 days**

---

## Milestones Overview

| # | Milestone | Target Date | Status |
|---|-----------|-------------|--------|
| 1 | Project setup & dataset download | 3 May 2026 | ✅ Done |
| 2 | Data exploration & preprocessing | 7 May 2026 | 🔲 Todo |
| 3 | Model training & initial evaluation | 12 May 2026 | 🔲 Todo |
| 4 | Model comparison & best model selection | 16 May 2026 | 🔲 Todo |
| 5 | Streamlit prototype development | 19 May 2026 | 🔲 Todo |
| 6 | Report writing (all sections) | 24 May 2026 | 🔲 Todo |
| 7 | Final review, timesheets, contribution statements | 27 May 2026 | 🔲 Todo |
| 8 | Submission | 29 May 2026 | 🔲 Todo |

---

## Detailed Task Breakdown

### Phase 1 — Setup (by 3 May)
- [x] Define project idea and write proposal
- [x] Set up GitHub repository structure
- [x] Scaffold Python source code (preprocess, train, evaluate, prototype)
- [x] Create requirements.txt
- [ ] Download UCI Student Performance dataset → place in `data/raw/`
- [ ] Install dependencies: `pip install -r requirements.txt`

---

### Phase 2 — Data Exploration & Preprocessing (by 7 May)
- [ ] Run `notebooks/01_data_exploration.ipynb` to understand the dataset
  - Check distributions of features
  - Check for missing values
  - Visualise Pass/Fail split
- [ ] Run `src/preprocess.py` to produce cleaned train/test splits
  - Confirm output files in `data/processed/`
- [ ] Review and adjust `SELECTED_FEATURES` list if needed
- [ ] Document decisions in the report (Data Collection & Preprocessing section)

**Responsible:** All members (split by notebook section)

---

### Phase 3 — Model Training (by 12 May)
- [ ] Run `src/train.py` to train all three models
  - Decision Tree
  - Random Forest
  - Logistic Regression
- [ ] Verify that `.pkl` files are saved in `models/`
- [ ] Tune basic hyperparameters if accuracy is low (e.g. `max_depth` for DT)
- [ ] Document training approach and hyperparameter choices

**Responsible:** Assign one model per member

---

### Phase 4 — Evaluation & Model Comparison (by 16 May)
- [ ] Run `src/evaluate.py` on the test set
- [ ] Record results:
  - Accuracy, Precision, Recall, F1-score per model
  - Confusion matrices
- [ ] Create a comparison table for the report
- [ ] Select the best model and justify the choice
- [ ] Fill in `notebooks/04_evaluation.ipynb` with plots (optional but good for report)

**Responsible:** All members

---

### Phase 5 — Prototype Development (by 19 May)
- [ ] Run `streamlit run prototype/app.py` and test the UI
- [ ] Test with several student profiles (edge cases: high absences, low grades)
- [ ] Fix any encoding issues between the form and model input
- [ ] (Optional) Deploy to Streamlit Community Cloud for easy presentation

**Responsible:** One member (frontend/prototype role)

---

### Phase 6 — Report Writing (by 24 May)
Report must include (IEEE format):
- [ ] **Title page:** Authors, study group number, project title
- [ ] **Abstract** (100–150 words)
- [ ] **Introduction:** Problem statement, motivation, objectives
- [ ] **Project Description:** Dataset, features, target variable
- [ ] **Requirements:** Functional and non-functional
- [ ] **Design & Architecture:** System diagram, component overview
- [ ] **Prototype Implementation:** How the code works, key choices
- [ ] **Testing & Evaluation:** Results table, confusion matrices, discussion
- [ ] **Deployment & Maintenance Plan:** Hypothetical plan (e.g. Streamlit Cloud)
- [ ] **Conclusions:** Summary, limitations, future work
- [ ] **References:** UCI dataset, scikit-learn, pandas, Streamlit

**Template:** https://template-selector.ieee.org/

**Responsible:** Split sections among members

---

### Phase 7 — Final Review (by 27 May)
- [ ] Proofread the full report
- [ ] Verify code runs end-to-end from a fresh install (`requirements.txt`)
- [ ] Each member completes their **Timesheet Diary**
- [ ] Each member writes their **Statement of Individual Contribution**
- [ ] Final group review of all deliverables

---

### Phase 8 — Submission (29 May)
- [ ] Submit:
  - [ ] Project report (PDF, IEEE format)
  - [ ] Timesheet diary (one per member)
  - [ ] Statement of individual contribution (one per member)
  - [ ] Source code with README and run instructions
  - [ ] (Optional) Link to live Streamlit prototype

---

## Deliverable Checklist

| Deliverable | Required | Responsible |
|---|---|---|
| Project report (IEEE PDF) | ✅ | All |
| Timesheet diary | ✅ | Each member individually |
| Statement of contribution | ✅ | Each member individually |
| Source code + run instructions | ✅ | All |
| Trained model files | ✅ | Included in repo or provided separately |
| Live Streamlit prototype | Optional | One member |

---

## Recommended Role Split (adapt as needed)

| Role | Tasks |
|---|---|
| Data Lead | Download dataset, run preprocessing, write Data section of report |
| ML Engineer | Train models, tune hyperparameters, run evaluation, write Methodology |
| Prototype Dev | Build and test Streamlit app, optional deployment |
| Report Lead | Write Introduction, Abstract, Conclusions; coordinate final report |

> All members must complete their own Timesheet Diary and Statement of Contribution.

---

## Notes

- Pass threshold for target variable: **G3 ≥ 10** (adjustable in `src/preprocess.py`)
- Models to compare: **Decision Tree**, **Random Forest**, **Logistic Regression**
- Primary evaluation metric: **F1-score** (balances precision and recall)
- Dataset: UCI Student Performance (`student-mat.csv`, Mathematics course)

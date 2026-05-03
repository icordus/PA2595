#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Checking dataset..."
if [[ ! -f "data/raw/student-mat.csv" && ! -f "data/raw/student-por.csv" ]]; then
  echo "Dataset not found in data/raw/."
  echo "Download student-mat.csv (or student-por.csv) from:"
  echo "https://archive.ics.uci.edu/dataset/320/student%2Bperformance"
  exit 1
fi

echo "[2/5] Installing dependencies..."
python -m pip install -r requirements.txt

echo "[3/5] Preprocessing data..."
python src/preprocess.py

echo "[4/5] Training Decision Tree pipeline..."
python src/train.py

echo "[5/5] Evaluating pipeline and starting Streamlit..."
python src/evaluate.py
python -m streamlit run prototype/app.py

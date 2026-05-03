$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "[1/5] Checking dataset..."
$hasMat = Test-Path "data/raw/student-mat.csv"
$hasPor = Test-Path "data/raw/student-por.csv"
if (-not ($hasMat -or $hasPor)) {
    Write-Host "Dataset not found in data/raw/."
    Write-Host "Download student-mat.csv (or student-por.csv) from:"
    Write-Host "https://archive.ics.uci.edu/dataset/320/student%2Bperformance"
    exit 1
}

Write-Host "[2/5] Installing dependencies..."
python -m pip install -r requirements.txt

Write-Host "[3/5] Preprocessing data..."
python src/preprocess.py

Write-Host "[4/5] Training models..."
python src/train.py

Write-Host "[5/5] Evaluating models and starting Streamlit..."
python src/evaluate.py
python -m streamlit run prototype/app.py

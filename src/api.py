"""FastAPI REST API for student risk prediction."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict as model_predict, load_feature_columns

app = FastAPI(
    title="PA2595 Student Risk API",
    description="Decision Tree pipeline for student academic risk prediction.",
    version="0.1.0",
)


class StudentInput(BaseModel):
    studytime: int = Field(..., ge=1, description="Weekly study time (hours)")
    failures: int = Field(..., ge=0, le=4, description="Number of past class failures")
    absences: int = Field(..., ge=0, description="Number of school absences")
    G1: int = Field(..., ge=0, le=20, description="First period grade (0–20)")
    G2: int = Field(..., ge=0, le=20, description="Second period grade (0–20)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: StudentInput):
    """
    Predict whether a student will pass or fail.

    Uses the trained Decision Tree pipeline when available.
    Falls back to a grade-based heuristic if no model is found.
    """
    features = {
        "studytime": data.studytime,
        "failures": data.failures,
        "absences": data.absences,
        "G1": data.G1,
        "G2": data.G2,
    }

    try:
        # Fill remaining feature columns with 0 (neutral defaults)
        all_columns = load_feature_columns()
        full_features = {col: 0 for col in all_columns}
        full_features.update(features)

        result = model_predict(full_features)
        prediction = result["label"].lower()
        probability = result["probability"]

        if prediction == "fail" or probability < 0.4:
            risk = "high"
        elif probability < 0.65:
            risk = "medium"
        else:
            risk = "low"

        return {
            "prediction": prediction,
            "risk": risk,
            "score": round((data.G1 + data.G2) / 2, 2),
            "probability": round(probability, 4),
            "source": "model",
        }

    except FileNotFoundError:
        # Fallback heuristic when model artifacts are not yet generated
        score = (data.G1 + data.G2) / 2

        if data.failures > 1 or data.absences > 20 or score < 10:
            prediction = "fail"
            risk = "high"
        elif score < 13:
            prediction = "pass"
            risk = "medium"
        else:
            prediction = "pass"
            risk = "low"

        return {
            "prediction": prediction,
            "risk": risk,
            "score": round(score, 2),
            "probability": None,
            "source": "heuristic",
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run():
    """Entry point for `pa2595-api` CLI command."""
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()

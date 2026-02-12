from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API (Pipeline)")

pipeline = joblib.load("artifacts/churn_pipeline.joblib")

# Load columns used by the preprocessor
cols = joblib.load("artifacts/columns.joblib")
NUM_COLS = cols["num_cols"]
CAT_COLS = cols["cat_cols"]
ALL_COLS = NUM_COLS + CAT_COLS


class CustomerInput(BaseModel):
    data: dict  # raw inputs: {"gender":"Female", "tenure":1, ...}


@app.get("/")
def home():
    return {"message": "Churn Pipeline API is running", "artifact": "artifacts/churn_pipeline.joblib"}


@app.post("/predict")
def predict(payload: CustomerInput):
    row = {c: payload.data.get(c, None) for c in ALL_COLS}
    df = pd.DataFrame([row], columns=ALL_COLS)

    pred = int(pipeline.predict(df)[0])
    proba = None
    if hasattr(pipeline, "predict_proba"):
        proba = float(pipeline.predict_proba(df)[0][1])

    return {
        "prediction": pred,
        "churn_label": "Yes" if pred == 1 else "No",
        "churn_probability": proba
    }

import random

import mlflow
import pandas as pd
from env import (
    MLFLOW_URI,
    MODEL_NAME,
    WINEDATA,
)
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from train import train_model

app = FastAPI(title="Wine Quality Prediction API")

# Probability for routing to current model (default: 0.8)
p = 0.8


def load_latest_model():
    mlflow.set_tracking_uri(MLFLOW_URI)
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")


# At startup, both current and next are the same
current_model = load_latest_model()
next_model = current_model


@app.post("/predict")
def predict(data: WINEDATA):
    try:
        df = pd.DataFrame([data.model_dump()])
        if random.random() < p:
            preds = current_model.predict(df)
            model_used = "current"
        else:
            preds = next_model.predict(df)
            model_used = "next"
        return {"prediction": preds.tolist(), "model_used": model_used}
    except MlflowException as e:
        raise HTTPException(status_code=404, detail=f"MLflow error: {str(e)}")


@app.post("/update-model")
def update_model():
    global next_model
    model_info, _ = train_model()
    next_model = load_latest_model()
    return {
        "status": "success",
        "message": "Next model retrained and updated successfully",
        "run_id": model_info.run_id,
    }


@app.post("/accept-next-model")
def accept_next_model():
    global current_model, next_model
    current_model = next_model
    return {"status": "success", "message": "Next model accepted as current model."}

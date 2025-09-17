import mlflow
import pandas as pd
from env import (
    MODEL_NAME,
    WINEDATA,
)
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException

from train import train_model

app = FastAPI(title="Wine Quality Prediction API")


def load_latest_model():
    try:
        return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")
    except MlflowException as e:
        train_model()
        return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")


loaded_model = load_latest_model()


@app.post("/predict")
def predict(data: WINEDATA):
    try:
        df = pd.DataFrame([data.model_dump()])
        preds = loaded_model.predict(df)
        return {"prediction": preds.tolist()}
    except MlflowException as e:
        raise HTTPException(status_code=404, detail=f"MLflow error: {str(e)}")


@app.post("/update-model")
def update_model():
    global loaded_model
    model_info, _ = train_model()
    loaded_model = load_latest_model()
    return {
        "status": "success",
        "message": "Model retrained and updated successfully",
        "run_id": model_info.run_id,
    }

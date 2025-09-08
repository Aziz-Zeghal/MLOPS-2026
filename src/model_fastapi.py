from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib

model = joblib.load("regression.joblib")

app = FastAPI()

@app.get("/predict")
def predict(size: float, bedrooms: int, garden: bool):
    features = [[size, bedrooms, garden]]
    prediction = model.predict(features)[0]
    return JSONResponse(content={"y_pred": prediction})


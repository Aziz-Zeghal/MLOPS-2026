import requests
import json

fastAPI = "http://127.0.0.1:5000"

# 1. Predict
with open("example.json") as f:
    sample = json.load(f)

    print(
        "Prediction before update:",
        requests.post(f"{fastAPI}/predict", json=sample).json(),
    )

    # 2. Update model
    print("Update response:", requests.post(f"{fastAPI}/update-model").json())

    # 3. Predict again
    print(
        "Prediction after update:",
        requests.post(f"{fastAPI}/predict", json=sample).json(),
    )

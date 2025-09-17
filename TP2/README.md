# Wine Quality Prediction Project

This project demonstrates how to train, track, and deploy a wine quality prediction model using MLflow and FastAPI.

## 1. Set Up Conda Environment


This project requires the `MLOPS` conda environment for reproducibility.

Activate the environment:

```bash
conda activate MLOPS
```

## 2. Run MLflow Tracking Server

Start the MLflow tracking server to log and view experiments:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Or use the UI shortcut:

```bash
mlflow ui
```

Access the MLflow UI at: [http://127.0.0.1:8080](http://127.0.0.1:8080)

## 3. Train and Serve the Model with FastAPI

Start the FastAPI server to train the model and serve predictions:

```bash
fastapi run model_api.py --port 5000
```

- The model will be trained and logged to MLflow automatically.
- The API will be available at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 4. Make Predictions and Update Model

Use the provided `test.py` script to:
- Send a prediction request using `example.json` as input
- Retrain/update the model
- Send another prediction request

```bash
python test.py
```

## 5. Project Structure
- `model_api.py`: FastAPI app and ML model logic
- `train.py`: Model training and logging to MLflow
- `env.py`: Environment variables and input schema
- `example.json`: Example input for prediction
- `test.py`: Script to test prediction and model update endpoints
- `data/wine_quality_merged.csv`: Dataset
- `requirements.txt`: Python dependencies (for dockerization)
- `conda_env.yaml`: Conda environment dump

## References
- MLflow docs: https://mlflow.org/docs/latest/ml/tracking/quickstart/
- FastAPI docs: https://fastapi.tiangolo.com/

---

For more details, see the notebook `demo.ipynb` for step-by-step data analysis and model training.

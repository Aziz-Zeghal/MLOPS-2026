"""Environment variable file for the project."""

from pydantic import BaseModel


DATA_PATH = "data/wine_quality_merged.csv"
MLFLOW_URI = "http://127.0.0.1:8080"
API_URI = "http://127.0.0.1:8000"
EXPERIMENT_NAME = "Logistic_Regression_Experiment"
MODEL_NAME = "logistic-regression-model"
DATASET_NAME = "wine-quality-dataset"


# Define the input data structure
class WINEDATA(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    quality: float
    sulphates: float
    alcohol: float

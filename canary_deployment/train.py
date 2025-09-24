import mlflow
import pandas as pd
from env import (
    DATA_PATH,
    DATASET_NAME,
    EXPERIMENT_NAME,
    MLFLOW_URI,
    MODEL_NAME,
)
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_model():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # To stop inferring schema
    for col in df.select_dtypes(include=["int"]).columns:
        df[col] = df[col].astype("float")

    # Make every column lowercase and underscore separated
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=["type"], axis=1), df["type"], test_size=0.2, random_state=42
    )

    hyperparams = {
        "C": 1.0,
        "max_iter": 50000,
    }
    # Train model
    model = LogisticRegression(**hyperparams)
    model.fit(x_train, y_train)

    # Evaluate model
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri=MLFLOW_URI)

    # Create the experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params(hyperparams)

        # Log metric
        mlflow.log_metric("accuracy", float(acc))
        print(f"Run finished. Accuracy: {acc}")

        signature = infer_signature(x_train, model.predict(x_train))

        model_info = mlflow.sklearn.log_model(  # type: ignore
            sk_model=model,
            name=DATASET_NAME,
            signature=signature,
            input_example=x_train.iloc[:5],
            registered_model_name=MODEL_NAME,
        )
        print(f"Model logged to MLflow with run_id: {run.info.run_id}")
        return model_info, model


if __name__ == "__main__":
    train_model()

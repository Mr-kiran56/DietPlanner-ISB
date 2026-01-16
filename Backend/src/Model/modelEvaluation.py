import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature
from pathlib import Path

# =========================================================
# PROJECT ROOT
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# =========================================================
# LOGGING CONFIGURATION
# =========================================================
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# =========================================================
# UTILS
# =========================================================
def load_params(params_path: Path) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_data(file_path: Path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=["label", "Unnamed: 0"], errors="ignore")
    y = df["label"]
    return X, y


def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return report, cm


def log_confusion_matrix(cm, name: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    path = f"confusion_matrix_{name}.png"
    plt.savefig(path)
    plt.close()
    mlflow.log_artifact(path)


def save_run_info(run_id: str, model_path: str, file_path: str):
    with open(file_path, "w") as f:
        json.dump(
            {"run_id": run_id, "model_path": model_path},
            f,
            indent=4
        )

# =========================================================
# MAIN
# =========================================================
def main():
    mlflow.set_tracking_uri(
        "http://ec2-13-236-200-53.ap-southeast-2.compute.amazonaws.com:5000/"
    )
    mlflow.set_experiment("dvc-pipeline-runs")

    with mlflow.start_run() as run:
        try:
            PARAMS_PATH = PROJECT_ROOT / "params.yaml"
            TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test_processed.csv"
            MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_medical_model.pkl"

            # Load params (log only model params)
            params = load_params(PARAMS_PATH)
            for k, v in params["model_building"].items():
                mlflow.log_param(k, v)

            # Load model + test data
            model = load_model(MODEL_PATH)
            X_test, y_test = load_data(TEST_PATH)

            # Evaluation
            report, cm = evaluate_model(model, X_test, y_test)

            # Log metrics
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics["precision"],
                        f"test_{label}_recall": metrics["recall"],
                        f"test_{label}_f1-score": metrics["f1-score"],
                    })

            # Log confusion matrix
            log_confusion_matrix(cm, "test")

            # Infer model signature (CORRECT WAY)
            input_example = X_test.iloc[:5]
            signature = infer_signature(
                input_example,
                model.predict(input_example)
            )

            mlflow.sklearn.log_model(
                model,
                artifact_path="xgboost_model",
                signature=signature,
                input_example=input_example
            )

            save_run_info(
                run.info.run_id,
                "xgboost_model",
                "experiment_info.json"
            )

            mlflow.set_tag("model_type", "XGBoostClassifier")
            mlflow.set_tag("task", "Medical Disease Prediction")
            mlflow.set_tag("dataset", "Patient Health Records")

            logger.info("Model evaluation completed successfully")

        except Exception as e:
            logger.error("Model evaluation failed: %s", e)
            raise


if __name__ == "__main__":
    main()

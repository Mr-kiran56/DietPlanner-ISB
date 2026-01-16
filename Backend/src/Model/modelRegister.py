# register_model.py

import json
import mlflow
import logging
from mlflow.tracking import MlflowClient

# =========================================================
# MLFLOW CONFIG
# =========================================================
mlflow.set_tracking_uri(
    "http://ec2-13-236-200-53.ap-southeast-2.compute.amazonaws.com:5000/"
)

# =========================================================
# LOGGING CONFIGURATION
# =========================================================
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_registration_errors.log")
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
def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load model info: %s", e)
        raise


def register_model(model_name: str, model_info: dict):
    try:
        client = MlflowClient()

        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        # Preferred (future-proof) way
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=model_version.version
        )

        logger.info(
            "Model '%s' version %s registered and set to STAGING",
            model_name,
            model_version.version
        )

    except Exception as e:
        logger.error("Model registration failed: %s", e)
        raise

# =========================================================
# MAIN
# =========================================================
def main():
    try:
        model_info = load_model_info("experiment_info.json")

        model_name = "medical_xgboost_classifier"

        register_model(model_name, model_info)

    except Exception as e:
        logger.error("Registration pipeline failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

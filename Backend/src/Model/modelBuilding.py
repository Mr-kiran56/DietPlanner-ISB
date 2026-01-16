import logging
import numpy as np
import pandas as pd
import pickle
import yaml
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from pathlib import Path

# =========================================================
# PROJECT ROOT
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# =========================================================
# LOGGING CONFIGURATION
# =========================================================
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_building_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def load_params(params_path: Path) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except Exception as e:
        logger.error("Failed to load params: %s", e)
        raise


def load_data(file_path: Path):
    try:
        df = pd.read_csv(file_path)
        X = df.drop(columns=["label", "Unnamed: 0"], errors="ignore")
        y = df["label"]
        logger.info("Loaded data from %s | Shape: %s", file_path, X.shape)
        return X, y
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        raise


def calculate_sample_weights(y: np.ndarray) -> np.ndarray:
    try:
        classes, counts = np.unique(y, return_counts=True)
        N, C = len(y), len(classes)
        class_weights = {c: N / (C * cnt) for c, cnt in zip(classes, counts)}
        sample_weights = np.array([class_weights[i] for i in y])
        logger.info("Sample weights calculated for %d classes", C)
        return sample_weights
    except Exception as e:
        logger.error("Failed to calculate sample weights: %s", e)
        raise


def train_xgboost(
    X_train,
    y_train,
    num_classes,
    sample_weights,
    params: dict
):
    try:
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            random_state=42,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
        )

        model.fit(X_train, y_train, sample_weight=sample_weights)
        logger.info("XGBoost training completed")
        return model
    except Exception as e:
        logger.error("Model training failed: %s", e)
        raise


def save_model(model, path: Path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved at %s", path)
    except Exception as e:
        logger.error("Failed to save model: %s", e)
        raise


def predict_patient(model, input_row: pd.Series):
    input_df = pd.DataFrame([input_row])
    probs = model.predict_proba(input_df)[0]
    pred = np.argmax(probs)
    return pred, probs[pred]

# =========================================================
# MAIN PIPELINE
# =========================================================
def main():
    try:
        PARAMS_PATH = PROJECT_ROOT / "params.yaml"

        TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train_processed.csv"
        TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test_processed.csv"

        MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_medical_model.pkl"

        # Load params
        params = load_params(PARAMS_PATH)
        model_params = params["model_building"]

        # Load train & test (already split)
        X_train, y_train = load_data(TRAIN_PATH)
        X_test, y_test = load_data(TEST_PATH)

        # Sample weights ONLY from training labels
        sample_weights = calculate_sample_weights(y_train)
        num_classes = len(np.unique(y_train))

        # Train model
        model = train_xgboost(
            X_train=X_train,
            y_train=y_train,
            num_classes=num_classes,
            sample_weights=sample_weights,
            params=model_params
        )

        # Evaluate on test set
        y_pred = model.predict(X_test)
        print("\n===== Classification Report (TEST SET) =====\n")
        print(classification_report(y_test, y_pred))

        # Save model
        save_model(model, MODEL_PATH)

        # Example prediction
        pred, conf = predict_patient(model, X_test.iloc[0])
        print(f"\nExample prediction → Class: {pred}, Confidence: {conf:.4f}")

    except Exception as e:
        logger.error("Model building pipeline failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

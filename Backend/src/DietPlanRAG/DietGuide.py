import pickle
import numpy as np
from fastapi import FastAPI, HTTPException, Query
import uvicorn
from pathlib import Path
import traceback
import logging

# ---------- PROJECT ROOT ----------
# B:/PersonalDietPlan-Infosys/DietPlanner
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------- LOGGING CONFIGURATION ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- LOCAL IMPORTS ----------
from DataExtraction.patientData import DataExtraction
from DataPreparation.dataIgnestion import preprocess_data
from DataPreparation.dataPreprocess import Preprocess_data
from IntentLLM.BertModel import split_sentences, predict_intent
from .LLM import generate_diet
from dotenv import load_dotenv
load_dotenv()
# ---------- APP ----------
app = FastAPI(title="Diet Planner ML API")

# ---------- LOAD MODEL ----------
def load_model(model_path: Path):
    """Load the ML model from disk."""
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

MODEL_PATH = PROJECT_ROOT / "models/xgboost_medical_model.pkl"
model = load_model(MODEL_PATH)

# ---------- API ----------
@app.get("/ML/Predict")
def ML_prediction(
    file_path: str = Query(
        default="Backend/src/DataPreparation/narrative_report_1.txt",
        description="Relative path to patient report file"
    )
):
    
    try:
        # ---------- RESOLVE FILE PATH ----------
        full_path = (PROJECT_ROOT / file_path).resolve()

        logger.info("======== DEBUG ========")
        logger.info(f"PROJECT_ROOT : {PROJECT_ROOT}")
        logger.info(f"INPUT PATH   : {file_path}")
        logger.info(f"FULL PATH    : {full_path}")
        logger.info(f"EXISTS       : {full_path.exists()}")
        logger.info("=======================")

        if not full_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"File not found: {full_path}"
            )

        # ---------- DATA EXTRACTION ----------
        logger.info("Starting data extraction...")
        dataDF, text = DataExtraction(str(full_path))
        dataDF = dataDF.replace({None: np.nan}).infer_objects(copy=False)

        # ---------- PREPROCESS ----------
        logger.info("Preprocessing data...")
        dataDF = preprocess_data(dataDF)
        final_data = Preprocess_data(dataDF)

        # ADD THIS DEBUG CODE:
        logger.info(f"final_data TYPE: {type(final_data)}")
        logger.info(f"final_data: {final_data}")

        # ---------- ML PREDICTION ----------
        logger.info("Running ML prediction...")

        # ADD MORE DEBUG:
        logger.info(f"Trying to access iloc[0]...")
        first_row = final_data.iloc[0]
        logger.info(f"first_row TYPE: {type(first_row)}")

        X = first_row.tolist()
        logger.info(f"X TYPE: {type(X)}")
        logger.info(f"X VALUE: {X}")

        input_data = np.array(X).reshape(1, -1)
        logger.info(f"input_data shape: {input_data.shape}")

        probs = model.predict_proba(input_data)

        pred_class = model.classes_[probs.argmax()]
        confidence = round(float(probs.max()), 4)

        logger.info(f"Prediction: {pred_class} with confidence {confidence}")

        # ---------- INTENT EXTRACTION ----------
        logger.info("Extracting intents...")
        sentences = split_sentences(text)
        detected_intents = []

        for s in sentences:
            intent, conf = predict_intent(s)
            if conf >= 0.7:
                detected_intents.append({
                    "sentence": s,
                    "intent": intent,
                    "confidence": round(conf, 4)
                })

        logger.info(f"Detected {len(detected_intents)} intents")

        # ---------- USER PAYLOAD ----------
        user_payload = {
            "patient_profile": dataDF.iloc[0].to_dict(),
            "ml_prediction": {
                "predicted_disease": str(pred_class),
                "confidence": confidence
            },
            "detected_intents": detected_intents,
            "user_preferences": {
                "food_type": "veg",
                "budget": "medium",
                "days": 7
            }
        }

        # ---------- LLM GENERATION ----------
        logger.info("Generating diet plan with LLM...")
        diet_response = generate_diet(
            context=text,
            payload=user_payload,
            days=user_payload["user_preferences"]["days"]
        )

        logger.info("Diet plan generated successfully")

        return {
            "ml_prediction": user_payload["ml_prediction"],
            "diet_plan": diet_response
        }

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------- HEALTH CHECK ----------
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Diet Planner ML API is running",
        "model_loaded": MODEL_PATH.exists()
    }


# ---------- RUN ----------
if __name__ == "__main__":
    uvicorn.run(
        "DietPlanRAG.DietGuide:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
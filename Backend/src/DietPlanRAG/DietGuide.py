import pickle
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import traceback
import logging
import shutil
from typing import Optional, List, Dict, Any

# ---------- PROJECT ROOT ----------
# This file should be at: DietPlanner/Backend/src/DietPlanRAG/DietGuide.py
# So we go up to get to DietPlanner root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ---------- UPLOAD DIRECTORY ----------
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------- LOGGING CONFIGURATION ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- ADD BACKEND TO PATH ----------
import sys
BACKEND_SRC = PROJECT_ROOT / "Backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

# ---------- LOCAL IMPORTS ----------
try:
    from DataExtraction.patientData import DataExtraction
    from DataPreparation.dataIgnestion import preprocess_data
    from DataPreparation.dataPreprocess import Preprocess_data
    from IntentLLM.BertModel import split_sentences, predict_intent
    from DietPlanRAG.LLM import generate_diet
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error(f"Make sure Backend/src is in PYTHONPATH")
    logger.error(f"Current sys.path: {sys.path}")
    raise

from dotenv import load_dotenv
load_dotenv()

# ---------- APP ----------
app = FastAPI(
    title="Diet Planner ML API",
    description="API for personalized diet planning based on medical reports",
    version="1.0.0"
)

# ---------- CORS MIDDLEWARE ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:5173", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- PYDANTIC MODELS ----------
class PredictionRequest(BaseModel):
    file_path: str
    food_type: Optional[str] = "veg"
    budget: Optional[str] = "medium"
    days: Optional[int] = 7
    

class MLPrediction(BaseModel):
    predicted_disease: str
    confidence: float

class IntentDetection(BaseModel):
    sentence: str
    intent: str
    confidence: float

class PredictionResponse(BaseModel):
    ml_prediction: MLPrediction
    diet_plan: Any

class UploadResponse(BaseModel):
    message: str
    file_path: str
    filename: str

# ---------- DISEASE LABELS ----------
labels = [
    "Healthy",
    "Pre-Hypertension",
    "Hypertension Stage 1",
    "Hypertension Stage 2",
    "Prediabetes",
    "Diabetes",
    "Obesity",
    "Cardiovascular Risk",
    "Anemia Mild",
    "Anemia Severe",
    "Metabolic Syndrome"
]

# ---------- LOAD MODEL ----------
def load_model(model_path: Path):
    """Load the ML model from disk."""
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

MODEL_PATH = PROJECT_ROOT / "models/xgboost_medical_model.pkl"
model = load_model(MODEL_PATH)

# ---------- FILE UPLOAD ENDPOINT ----------
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a medical report file (PDF, TXT, DOC, DOCX, or image).
    Returns the relative file path for use in prediction endpoint.
    """
    try:
        # Validate file type
        allowed_extensions = ['.txt', '.pdf', '.doc', '.docx', '.jpg', '.jpeg', '.png']
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Create unique filename to avoid conflicts
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Save file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Return relative path from project root
        relative_path = file_path.relative_to(PROJECT_ROOT)
        
        logger.info(f"File uploaded successfully: {relative_path}")
        
        return UploadResponse(
            message="File uploaded successfully",
            file_path=str(relative_path),
            filename=file.filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )

# ---------- GET PREDICTION (With Query Parameters) ----------
@app.get("/ML/Predict")
async def ml_prediction_get(
    file_path: str,
    food_type: str = "veg",
    budget: str = "medium",
    days: int = 7
):
    """
    Generate diet plan based on uploaded medical report.
    Accepts user preferences via query parameters.
    
    Parameters:
    - file_path: Path to uploaded file
    - food_type: "veg" or "nonveg" (default: "veg")
    - budget: "low", "medium", or "high" (default: "medium")
    - days: Number of days for diet plan (1-30, default: 7)
    """
    try:
        # Validate parameters
        if food_type not in ["veg", "nonveg"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid food_type: {food_type}. Must be 'veg' or 'nonveg'"
            )
        
        if budget not in ["low", "medium", "high"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid budget: {budget}. Must be 'low', 'medium', or 'high'"
            )
        
        if not (1 <= days <= 30):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid days: {days}. Must be between 1 and 30"
            )
        
        # ---------- RESOLVE FILE PATH ----------
        full_path = (PROJECT_ROOT / file_path).resolve()

        logger.info("======== DEBUG ========")
        logger.info(f"PROJECT_ROOT : {PROJECT_ROOT}")
        logger.info(f"INPUT PATH   : {file_path}")
        logger.info(f"FULL PATH    : {full_path}")
        logger.info(f"EXISTS       : {full_path.exists()}")
        logger.info(f"PREFERENCES  : food_type={food_type}, budget={budget}, days={days}")
        logger.info("=======================")

        if not full_path.exists():
            raise HTTPException(
                status_code=404,
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

        logger.info(f"final_data TYPE: {type(final_data)}")
        logger.info(f"final_data shape: {final_data.shape}")

        # ---------- ML PREDICTION ----------
        logger.info("Running ML prediction...")
        
        first_row = final_data.iloc[0]
        X = first_row.tolist()
        input_data = np.array(X).reshape(1, -1)

        logger.info(f"input_data shape: {input_data.shape}")

        probs = model.predict_proba(input_data)

        pred_index = int(np.argmax(probs))
        predicted_disease = labels[pred_index]
        confidence = round(float(probs[0][pred_index]), 4)

        logger.info(f"Prediction: {predicted_disease} (confidence: {confidence})")

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

        # ---------- USER PAYLOAD WITH PREFERENCES ----------
        user_payload = {
            "patient_profile": dataDF.iloc[0].to_dict(),
            "ml_prediction": {
                "predicted_disease": predicted_disease,
                "confidence": confidence
            },
            "detected_intents": detected_intents,
            "user_preferences": {
                "food_type": food_type,
                "budget": budget,
                "days": days
            }
        }

        # ---------- LLM GENERATION ----------
        logger.info(f"Generating {days}-day diet plan with preferences: {food_type}, {budget}...")
        diet_response = generate_diet(
            context=text,
            payload=user_payload,
            days=days
        )

        logger.info("Diet plan generated successfully")

        return PredictionResponse(
            ml_prediction=MLPrediction(
                predicted_disease=predicted_disease,
                confidence=confidence
            ),
            diet_plan=diet_response
        )

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
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ---------- ML PREDICTION ENDPOINT (POST) ----------
@app.post("/ML/Predict", response_model=PredictionResponse)
async def ml_prediction(request: PredictionRequest):
    """
    POST endpoint for diet plan generation.
    Redirects to GET endpoint with extracted parameters.
    """
    return await ml_prediction_get(
        file_path=request.file_path,
        food_type=request.food_type,
        budget=request.budget,
        days=request.days
    )

# ---------- HEALTH CHECK ----------
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Diet Planner ML API is running",
        "model_loaded": MODEL_PATH.exists(),
        "upload_dir": str(UPLOAD_DIR),
        "endpoints": {
            "upload": "POST /upload",
            "predict": "GET /ML/Predict?file_path=<path>&food_type=<veg|nonveg>&budget=<low|medium|high>&days=<1-30>",
            "health": "GET /"
        }
    }

# ---------- GET LABELS ----------
@app.get("/labels")
def get_labels():
    """Get list of possible disease predictions."""
    return {
        "labels": labels,
        "count": len(labels)
    }

# ---------- RUN ----------
if __name__ == "__main__":
    uvicorn.run(
        "DietPlanRAG.DietGuide:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
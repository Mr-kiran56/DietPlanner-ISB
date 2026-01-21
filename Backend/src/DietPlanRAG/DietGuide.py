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
import os
import aiofiles
import datetime

# ---------- PROJECT ROOT ----------
# FIXED: Use absolute path in Docker container
PROJECT_ROOT = Path("/app")  # Changed from relative to absolute

# ---------- UPLOAD DIRECTORY ----------
UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# ---------- LOGGING CONFIGURATION ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- ADD BACKEND TO PATH ----------
import sys
# FIXED: Correct path for Docker container
sys.path.insert(0, "/app/Backend/src")

# ---------- LOCAL IMPORTS ----------
try:
    from DataExtraction.patientData import DataExtraction
    from DataPreparation.dataIgnestion import preprocess_data
    from DataPreparation.dataPreprocess import Preprocess_data
    from IntentLLM.BertModel import split_sentences, predict_intent
    from DietPlanRAG.LLM import generate_diet
    logger.info("✅ All imports successful!")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    logger.error("Checking directory structure...")
    try:
        logger.error(f"Does /app exist? {Path('/app').exists()}")
        logger.error(f"Does /app/Backend exist? {Path('/app/Backend').exists()}")
        logger.error(f"Does /app/Backend/src exist? {Path('/app/Backend/src').exists()}")
        logger.error(f"Contents of /app: {list(Path('/app').iterdir())}")
        logger.error(f"Contents of /app/Backend: {list(Path('/app/Backend').iterdir())}")
    except:
        pass
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
    allow_origins=["*"],
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
    file_size: int

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

# FIXED: Use absolute path for model
MODEL_PATH = Path("/app/Backend/models/xgboost_medical_model.pkl")
try:
    logger.info(f"Attempting to load model from: {MODEL_PATH}")
    logger.info(f"Model path exists: {MODEL_PATH.exists()}")
    if MODEL_PATH.exists():
        model = load_model(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        logger.warning(f"⚠️ Model file not found at {MODEL_PATH}")
        logger.warning(f"Looking for model in alternative locations...")
        # Try alternative paths
        alt_paths = [
            Path("/app/models/xgboost_medical_model.pkl"),
            Path("/app/Backend/src/models/xgboost_medical_model.pkl"),
            Path("models/xgboost_medical_model.pkl")
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                MODEL_PATH = alt_path
                model = load_model(MODEL_PATH)
                logger.info(f"✅ Model loaded from alternative path: {MODEL_PATH}")
                break
        else:
            logger.error("❌ Model not found in any location")
            model = None
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    traceback.print_exc()
    model = None

# ---------- FILE UPLOAD ENDPOINT ----------
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a medical report file."""
    try:
        allowed_extensions = ['.txt', '.pdf', '.doc', '.docx', '.jpg', '.jpeg', '.png']
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported."
            )
        
        import uuid
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = f"{timestamp}_{unique_id}_{Path(file.filename).name}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as buffer:
            content = await file.read()
            await buffer.write(content)
        
        file_size = os.path.getsize(file_path)
        relative_path = f"uploads/{safe_filename}"
        
        logger.info(f"✅ File uploaded: {relative_path} ({file_size} bytes)")
        
        return UploadResponse(
            message="File uploaded successfully",
            file_path=relative_path,
            filename=file.filename,
            file_size=file_size
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

# ---------- GET PREDICTION ----------
@app.get("/ML/Predict")
async def ml_prediction_get(
    file_path: str,
    food_type: str = "veg",
    budget: str = "medium",
    days: int = 7
):
    """Generate diet plan based on uploaded medical report."""
    try:
        # Validate parameters
        if food_type not in ["veg", "nonveg"]:
            raise HTTPException(400, "food_type must be 'veg' or 'nonveg'")
        
        if budget not in ["low", "medium", "high"]:
            raise HTTPException(400, "budget must be 'low', 'medium', or 'high'")
        
        if not (1 <= days <= 30):
            raise HTTPException(400, "days must be between 1 and 30")
        
        # Resolve file path
        if file_path.startswith("uploads/"):
            full_path = UPLOAD_DIR / Path(file_path).name
        else:
            full_path = UPLOAD_DIR / Path(file_path).name
        
        logger.info(f"Looking for file: {full_path}")
        logger.info(f"File exists: {full_path.exists()}")
        
        if not full_path.exists():
            raise HTTPException(404, f"File not found: {file_path}")
        
        # Data extraction
        logger.info("Starting data extraction...")
        dataDF, text = DataExtraction(str(full_path))
        
        # Preprocess
        logger.info("Preprocessing data...")
        dataDF = preprocess_data(dataDF)
        final_data = Preprocess_data(dataDF)
        
        # ML prediction
        logger.info("Running ML prediction...")
        if model is None:
            raise HTTPException(500, "ML model not loaded")
        
        first_row = final_data.iloc[0]
        X = first_row.tolist()
        input_data = np.array(X).reshape(1, -1)
        
        probs = model.predict_proba(input_data)
        pred_index = int(np.argmax(probs))
        predicted_disease = labels[pred_index]
        confidence = round(float(probs[0][pred_index]), 4)
        
        logger.info(f"Prediction: {predicted_disease} (confidence: {confidence})")
        
        # Intent extraction
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
        
        # User payload
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
        
        # LLM generation
        logger.info(f"Generating {days}-day diet plan...")
        diet_response = generate_diet(
            context=text,
            payload=user_payload,
            days=days
        )
        
        logger.info("✅ Diet plan generated successfully")
        
        return {
            "ml_prediction": {
                "predicted_disease": predicted_disease,
                "confidence": confidence
            },
            "diet_plan": diet_response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(500, f"Internal server error: {str(e)}")

# ---------- POST PREDICTION ----------
@app.post("/ML/Predict")
async def ml_prediction(request: PredictionRequest):
    """POST endpoint for diet plan generation."""
    return await ml_prediction_get(
        file_path=request.file_path,
        food_type=request.food_type,
        budget=request.budget,
        days=request.days
    )

# ---------- SIMPLE HEALTH CHECK ----------
@app.get("/health")
async def health_check_simple():
    """Simple health check for Docker."""
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

# ---------- HEALTH CHECK ----------
@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Diet Planner ML API is running",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists() if hasattr(MODEL_PATH, 'exists') else False,
        "upload_dir": str(UPLOAD_DIR),
        "python_path": sys.path,
        "endpoints": {
            "upload": "POST /upload",
            "predict": "GET /ML/Predict?file_path=<path>&food_type=<type>&budget=<range>&days=<num>",
            "health": "GET /health",
            "root": "GET /"
        }
    }

# ---------- GET LABELS ----------
@app.get("/labels")
async def get_labels():
    """Get list of possible disease predictions."""
    return {"labels": labels, "count": len(labels)}

# ---------- LIST UPLOADED FILES ----------
@app.get("/uploads")
async def list_uploads():
    """List all uploaded files."""
    files = []
    for f in UPLOAD_DIR.glob("*"):
        if f.is_file():
            files.append({
                "name": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime
            })
    return {"files": files}

if __name__ == "__main__":
    uvicorn.run(
        "DietGuide:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
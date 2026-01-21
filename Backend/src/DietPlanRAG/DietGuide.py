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

# ---------- PROJECT ROOT ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ---------- UPLOAD DIRECTORY ----------
# Use absolute path in container
UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

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
    allow_origins=["*"],  # Allow all origins in production, or specify frontend URL
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

MODEL_PATH = PROJECT_ROOT / "models/xgboost_medical_model.pkl"
try:
    model = load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

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
        import uuid
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = f"{timestamp}_{unique_id}_{Path(file.filename).name}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Save file asynchronously
        async with aiofiles.open(file_path, 'wb') as buffer:
            content = await file.read()
            await buffer.write(content)
        
        file_size = os.path.getsize(file_path)
        
        # Return relative path (for API calls)
        relative_path = f"uploads/{safe_filename}"
        
        logger.info(f"File uploaded successfully: {relative_path} ({file_size} bytes)")
        
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
    """
    Generate diet plan based on uploaded medical report.
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
        
        # Resolve file path
        if file_path.startswith("uploads/"):
            full_path = UPLOAD_DIR / Path(file_path).name
        else:
            full_path = Path(file_path)
        
        if not full_path.exists():
            # Try absolute path
            full_path = UPLOAD_DIR / Path(file_path).name
            if not full_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found: {file_path}. Searched in: {UPLOAD_DIR}"
                )
        
        logger.info("======== FILE DEBUG ========")
        logger.info(f"Input file_path: {file_path}")
        logger.info(f"Resolved path: {full_path}")
        logger.info(f"Exists: {full_path.exists()}")
        logger.info(f"File size: {full_path.stat().st_size if full_path.exists() else 'N/A'}")
        logger.info("==========================")
        
        # Data extraction
        logger.info("Starting data extraction...")
        dataDF, text = DataExtraction(str(full_path))
        dataDF = dataDF.replace({None: np.nan}).infer_objects(copy=False)
        
        # Preprocess
        logger.info("Preprocessing data...")
        dataDF = preprocess_data(dataDF)
        final_data = Preprocess_data(dataDF)
        
        # ML prediction
        logger.info("Running ML prediction...")
        if model is None:
            raise HTTPException(status_code=500, detail="ML model not loaded")
        
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
        
        logger.info(f"Detected {len(detected_intents)} intents")
        
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
        
        logger.info("Diet plan generated successfully")
        
        return {
            "ml_prediction": {
                "predicted_disease": predicted_disease,
                "confidence": confidence
            },
            "diet_plan": diet_response
        }
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ---------- POST PREDICTION ----------
@app.post("/ML/Predict")
async def ml_prediction(request: PredictionRequest):
    """
    POST endpoint for diet plan generation.
    """
    return await ml_prediction_get(
        file_path=request.file_path,
        food_type=request.food_type,
        budget=request.budget,
        days=request.days
    )

# ---------- HEALTH CHECK ----------
@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Diet Planner ML API is running",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "upload_dir": str(UPLOAD_DIR),
        "upload_dir_exists": UPLOAD_DIR.exists(),
        "upload_dir_files": len(list(UPLOAD_DIR.glob("*"))) if UPLOAD_DIR.exists() else 0,
        "endpoints": {
            "upload": "POST /upload",
            "predict_get": "GET /ML/Predict?file_path=<path>&food_type=<veg|nonveg>&budget=<low|medium|high>&days=<1-30>",
            "predict_post": "POST /ML/Predict",
            "health": "GET /",
            "labels": "GET /labels"
        }
    }

# ---------- GET LABELS ----------
@app.get("/labels")
async def get_labels():
    """Get list of possible disease predictions."""
    return {
        "labels": labels,
        "count": len(labels)
    }

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
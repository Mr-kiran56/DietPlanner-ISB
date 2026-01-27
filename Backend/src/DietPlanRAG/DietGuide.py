"""
Diet Planner ML API
A FastAPI application for personalized diet planning based on medical reports.
"""

import os
import sys
import time
import pickle
import logging
import traceback
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import aiofiles
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv



# ========== CONFIGURATION ==========
class Config:
    """Application configuration."""
    PROJECT_ROOT = Path("/app")
    UPLOAD_DIR = Path("/app/uploads")
    MODEL_PATH = Path("/app/Backend/models/xgboost_medical_model.pkl")
    BACKEND_SRC_PATH = "/app/Backend/src"
    
    ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.doc', '.docx', '.jpg', '.jpeg', '.png'}
    MAX_FILE_SIZE_MB = 10
    
    DISEASE_LABELS = [
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
    
    FOOD_TYPES = {"veg", "nonveg"}
    BUDGET_TYPES = {"low", "medium", "high"}
    MIN_DAYS = 1
    MAX_DAYS = 30


# ========== LOGGING SETUP ==========
def setup_logging():
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ========== STARTUP TRACKER ==========
class StartupTracker:
    """Track and log application startup progress."""
    
    def __init__(self):
        self.start_time = time.time()
    
    def log(self, message: str, emoji: str = "📋"):
        """Log startup message with elapsed time."""
        elapsed = time.time() - self.start_time
        logger.info(f"{emoji} [{elapsed:6.2f}s] {message}")


startup_tracker = StartupTracker()
startup_tracker.log("DietPlanner backend initializing...", "🚀")


# ========== DIRECTORY SETUP ==========
def setup_directories():
    """Create necessary directories."""
    Config.UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
    startup_tracker.log(f"Upload directory ready: {Config.UPLOAD_DIR}", "📁")


setup_directories()


# ========== IMPORT DEPENDENCIES ==========
def setup_imports():
    """Add backend source to Python path and import modules."""
    sys.path.insert(0, Config.BACKEND_SRC_PATH)
    
    try:
        global DataExtraction, preprocess_data, Preprocess_data
        global split_sentences, predict_intent, generate_diet
        
        from DataExtraction.patientData import DataExtraction
        from DataPreparation.dataIgnestion import preprocess_data
        from DataPreparation.dataPreprocess import Preprocess_data
        from IntentLLM.BertModel import split_sentences, predict_intent
        from DietPlanRAG.LLM import generate_diet
        
        startup_tracker.log("All imports successful", "✅")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error(f"sys.path: {sys.path}")
        _log_directory_structure()
        raise


def _log_directory_structure():
    """Log directory structure for debugging."""
    paths_to_check = [
        Path('/app'),
        Path('/app/Backend'),
        Path('/app/Backend/src')
    ]
    
    for path in paths_to_check:
        if path.exists():
            logger.error(f"Contents of {path}: {list(path.iterdir())}")
        else:
            logger.error(f"{path} does not exist")


setup_imports()
load_dotenv()


# ========== PYDANTIC MODELS ==========
class PredictionRequest(BaseModel):
    """Request model for diet plan generation."""
    file_path: str = Field(..., description="Path to uploaded medical report")
    food_type: str = Field(default="veg", description="Dietary preference")
    budget: str = Field(default="medium", description="Budget constraint")
    days: int = Field(default=7, ge=1, le=30, description="Number of days for diet plan")
    
    @field_validator('food_type')
    @classmethod
    def validate_food_type(cls, v):
        if v not in Config.FOOD_TYPES:
            raise ValueError(f"food_type must be one of {Config.FOOD_TYPES}")
        return v
    
    @field_validator('budget')
    @classmethod
    def validate_budget(cls, v):
        if v not in Config.BUDGET_TYPES:
            raise ValueError(f"budget must be one of {Config.BUDGET_TYPES}")
        return v


class MLPrediction(BaseModel):
    """ML model prediction result."""
    predicted_disease: str
    confidence: float = Field(ge=0.0, le=1.0)


class IntentDetection(BaseModel):
    """Detected intent from text."""
    sentence: str
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Complete prediction response."""
    ml_prediction: MLPrediction
    diet_plan: Any


class UploadResponse(BaseModel):
    """File upload response."""
    message: str
    file_path: str
    filename: str
    file_size: int


# ========== MODEL LOADING ==========
class ModelManager:
    """Manage ML model loading and predictions."""
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self._load_model()
    
    def _load_model(self):
        """Load the ML model from disk."""
        startup_tracker.log("Loading ML model...", "📦")
        
        paths_to_try = [
            Config.MODEL_PATH,
            Path("/app/models/xgboost_medical_model.pkl"),
            Path("/app/Backend/src/models/xgboost_medical_model.pkl"),
            Path("models/xgboost_medical_model.pkl")
        ]
        
        for path in paths_to_try:
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        self.model = pickle.load(f)
                    self.model_path = path
                    startup_tracker.log(f"Model loaded from {path}", "✅")
                    return
                except Exception as e:
                    logger.error(f"Failed to load model from {path}: {e}")
        
        logger.warning("Model not found in any location - running without ML predictions")
        startup_tracker.log("Model not loaded", "⚠️")
    
    def predict(self, input_data: np.ndarray) -> tuple[str, float]:
        """Make prediction using loaded model."""
        if self.model is None:
            raise RuntimeError("ML model not loaded")
        
        probs = self.model.predict_proba(input_data)
        pred_index = int(np.argmax(probs))
        predicted_disease = Config.DISEASE_LABELS[pred_index]
        confidence = round(float(probs[0][pred_index]), 4)
        
        return predicted_disease, confidence
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


model_manager = ModelManager()


# ========== FASTAPI APP ==========
app = FastAPI(
    title="Diet Planner ML API",
    description="API for personalized diet planning based on medical reports",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== UTILITY FUNCTIONS ==========
def generate_unique_filename(original_filename: str) -> str:
    """Generate unique filename with timestamp and UUID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    safe_name = Path(original_filename).name
    return f"{timestamp}_{unique_id}_{safe_name}"


def validate_file_extension(filename: str) -> bool:
    """Check if file extension is allowed."""
    return Path(filename).suffix.lower() in Config.ALLOWED_EXTENSIONS


def resolve_file_path(file_path: str) -> Path:
    """Resolve relative file path to absolute path."""
    if file_path.startswith("uploads/"):
        return Config.UPLOAD_DIR / Path(file_path).name
    return Config.UPLOAD_DIR / Path(file_path).name


# ========== API ENDPOINTS ==========
@app.on_event("startup")
async def startup_event():
    """Log startup completion."""
    total_time = time.time() - startup_tracker.start_time
    startup_tracker.log(f"FastAPI ready! Total startup: {total_time:.2f}s", "✅")
    startup_tracker.log("Health endpoint: http://0.0.0.0:8000/health", "🌐")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "status": "healthy",
        "message": "Diet Planner ML API",
        "version": "1.0.0",
        "model_loaded": model_manager.is_loaded,
        "endpoints": {
            "upload": "POST /upload",
            "predict_get": "GET /ML/Predict",
            "predict_post": "POST /ML/Predict",
            "health": "GET /health",
            "labels": "GET /labels",
            "uploads": "GET /uploads"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "loaded": model_manager.is_loaded,
            "path": str(model_manager.model_path) if model_manager.model_path else None
        },
        "directories": {
            "upload": str(Config.UPLOAD_DIR),
            "exists": Config.UPLOAD_DIR.exists()
        }
    }


@app.post("/upload", response_model=UploadResponse, tags=["Files"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a medical report file.
    
    Supported formats: txt, pdf, doc, docx, jpg, jpeg, png
    """
    try:
        if not validate_file_extension(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not supported. Allowed: {Config.ALLOWED_EXTENSIONS}"
            )
        
        safe_filename = generate_unique_filename(file.filename)
        file_path = Config.UPLOAD_DIR / safe_filename
        
        # Save file asynchronously
        async with aiofiles.open(file_path, 'wb') as buffer:
            content = await file.read()
            await buffer.write(content)
        
        file_size = os.path.getsize(file_path)
        relative_path = f"uploads/{safe_filename}"
        
        logger.info(f"File uploaded: {relative_path} ({file_size} bytes)")
        
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )


# @app.get("/health")
# async def health_check():
#     return {
#         "status": "ok",
#         "service": "dietplanner-backend"
#     }


@app.get("/ML/Predict", tags=["Prediction"])
async def predict_get(
    file_path: str,
    food_type: str = "veg",
    budget: str = "medium",
    days: int = 7
):
    """Generate diet plan based on uploaded medical report (GET method)."""
    request = PredictionRequest(
        file_path=file_path,
        food_type=food_type,
        budget=budget,
        days=days
    )
    return await _generate_diet_plan(request)


@app.post("/ML/Predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_post(request: PredictionRequest):
    """Generate diet plan based on uploaded medical report (POST method)."""
    return await _generate_diet_plan(request)


async def _generate_diet_plan(request: PredictionRequest) -> Dict[str, Any]:
    """
    Core logic for diet plan generation.
    
    Process flow:
    1. Validate and locate file
    2. Extract data from medical report
    3. Preprocess data
    4. Run ML prediction
    5. Extract intents from text
    6. Generate personalized diet plan
    """
    try:
        # Validate model is loaded
        if not model_manager.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model not loaded"
            )
        
        # Resolve and validate file path
        full_path = resolve_file_path(request.file_path)
        logger.info(f"Processing file: {full_path}")
        
        if not full_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {request.file_path}"
            )
        
        # Extract data from medical report
        logger.info("Extracting data from medical report...")
        dataDF, text = DataExtraction(str(full_path))
        
        # Preprocess data
        logger.info("Preprocessing data..")
        dataDF = preprocess_data(dataDF)
        final_data = Preprocess_data(dataDF)
        import pandas as pd
        final_data = final_data.apply(pd.to_numeric, errors="coerce")
        final_data = final_data.fillna(0.0)

        
        # Prepare input for ML model
        first_row = final_data.iloc[0].tolist()
        input_data = np.array(first_row).reshape(1, -1)
        
        # Run ML prediction
        logger.info("Running ML prediction...")
        predicted_disease, confidence = model_manager.predict(input_data)
        logger.info(f"Prediction: {predicted_disease} (confidence: {confidence})")
        
        # Extract intents from text
        logger.info("Extracting intents...")
        sentences = split_sentences(text)
        detected_intents = []
        
        for sentence in sentences:
            intent, conf = predict_intent(sentence)
            
            if intent != "unknown" and conf >= 0.7:
                detected_intents.append({
                    "sentence": sentence,
                    "intent": intent,
                    "confidence": round(conf, 4)
                })
        
        # Prepare user payload for LLM
        user_payload = {
            "patient_profile": dataDF.iloc[0].to_dict(),
            "ml_prediction": {
                "predicted_disease": predicted_disease,
                "confidence": confidence
            },
            "detected_intents": detected_intents,
            "user_preferences": {
                "food_type": request.food_type,
                "budget": request.budget,
                "days": request.days
            }
        }
        
        # Generate diet plan using LLMs
        logger.info(f"Generating {request.days}-day diet plan...")
        diet_response = generate_diet(user_payload)
        
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
    except Exception as e:
        logger.error(f"Unexpected error in diet plan generation: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/labels", tags=["Reference"])
async def get_labels():
    """Get list of possible disease predictions."""
    return {
        "labels": Config.DISEASE_LABELS,
        "count": len(Config.DISEASE_LABELS)
    }


@app.get("/uploads", tags=["Files"])
async def list_uploads():
    """List all uploaded files."""
    files = []
    for file in Config.UPLOAD_DIR.glob("*"):
        if file.is_file():
            files.append({
                "name": file.name,
                "size": file.stat().st_size,
                "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
            })
    
    return {
        "files": files,
        "count": len(files),
        "directory": str(Config.UPLOAD_DIR)
    }


# ========== APPLICATION ENTRY POINT ==========
if __name__ == "__main__":
    uvicorn.run(
        "DietGuide:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
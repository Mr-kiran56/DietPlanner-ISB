import os
import re
import torch
import joblib
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import logging

# ============================================================
# ENV CONFIG (SAFE FOR DOCKER / CI)
# ============================================================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_MEMORY_MAPPING"] = "1"

logger = logging.getLogger(__name__)

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "INTENT_MODEL_PATH",
        "/app/Backend/src/IntentLLM/IntentLLM-F/export_results/checkpoint-500"
    )
)

LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

# ============================================================
# GLOBAL MODEL OBJECTS (LAZY LOADED)
# ============================================================
tokenizer = None
model = None
label_encoder = None

NUM_ENCODER_CLASSES = None
NUM_MODEL_CLASSES = None

# ============================================================
# LAZY MODEL LOADER (CRITICAL FIX)
# ============================================================
def load_intent_model():
    """
    Load tokenizer, model, and label encoder lazily.
    This prevents Docker healthcheck failures.
    """
    global tokenizer, model, label_encoder
    global NUM_ENCODER_CLASSES, NUM_MODEL_CLASSES

    if tokenizer is not None and model is not None:
        return

    logger.info("[IntentLLM] Loading intent classification model...")

    if not MODEL_PATH.exists():
        raise RuntimeError(f"[IntentLLM] Model path not found: {MODEL_PATH}")

    if not LABEL_ENCODER_PATH.exists():
        raise RuntimeError(f"[IntentLLM] Label encoder not found: {LABEL_ENCODER_PATH}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        use_fast=True
    )

    # ---- Model ----
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    model.to(device)
    model.eval()

    # ---- Label Encoder ----
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    if isinstance(label_encoder.classes_, list):
        label_encoder.classes_ = np.array(label_encoder.classes_)

    NUM_ENCODER_CLASSES = len(label_encoder.classes_)
    NUM_MODEL_CLASSES = model.config.num_labels

    logger.info("[IntentLLM] Model loaded successfully")
    logger.info(f"[IntentLLM] Model classes   : {NUM_MODEL_CLASSES}")
    logger.info(f"[IntentLLM] Encoder classes : {NUM_ENCODER_CLASSES}")
    logger.info(f"[IntentLLM] Encoder labels  : {label_encoder.classes_}")

# ============================================================
# SENTENCE SPLITTER
# ============================================================
def split_sentences(text: str):
    """
    Split text into meaningful sentences.
    """
    if not text or not isinstance(text, str):
        return []

    return [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', text)
        if len(s.strip()) > 5
    ]

# ============================================================
# INTENT PREDICTION (SAFE)
# ============================================================
def predict_intent(sentence: str):
    """
    Predict intent for a single sentence.
    Safe for Docker, CI/CD, and CPU servers.
    """
    try:
        if not sentence or len(sentence.strip()) < 3:
            return "unknown", 0.0

        # Load model only when neededd
        load_intent_model()

        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)
        pred_id = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_id].item())

        # Safety check
        if pred_id >= NUM_ENCODER_CLASSES:
            return "unknown", round(confidence, 4)

        intent = label_encoder.inverse_transform([pred_id])[0]
        return intent, round(confidence, 4)

    except Exception as e:
        logger.error(f"[IntentLLM] Prediction failed: {e}")
        return "unknown", 0.0

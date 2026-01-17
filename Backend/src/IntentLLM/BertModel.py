import torch
import joblib
import numpy as np  # ← ADD THIS IMPORT
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import torch.nn.functional as F

# ---------- 1. DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. ABSOLUTE PATHS ----------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "IntentLLM-F" / "export" / "results" / "checkpoint-500"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

# DEBUG
print("MODEL_PATH:", MODEL_PATH)
print("Exists:", MODEL_PATH.exists())

# ---------- 3. LOAD TOKENIZER ----------
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

# ---------- 4. LOAD MODEL ----------
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.to(device)
model.eval()

# ---------- 5. LOAD LABEL ENCODER ----------
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# FIX: Convert classes_ to numpy array if it's a list
if isinstance(label_encoder.classes_, list):
    label_encoder.classes_ = np.array(label_encoder.classes_)

# ---------- 6. PREDICTION FUNCTIONS ----------

def split_sentences(text: str):
    return [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', text)
        if len(s.strip()) > 5
    ]


def predict_intent(sentence: str):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=1)

    pred_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_id].item()

    intent = label_encoder.inverse_transform([pred_id])[0]

    return intent, round(confidence, 4)
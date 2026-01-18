import os
import torch
import joblib
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import torch.nn.functional as F

# -------------------------------------------------
# 🔴 CRITICAL WINDOWS MEMORY FIX
# -------------------------------------------------
# Disable HuggingFace memory-mapped loading
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_MEMORY_MAPPING"] = "1"

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "IntentLLM-F" / "export" / "results" / "checkpoint-500"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

print("[IntentLLM] MODEL_PATH:", MODEL_PATH)
print("[IntentLLM] Exists:", MODEL_PATH.exists())

# -------------------------------------------------
# LOAD TOKENIZER (lightweight)
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    use_fast=True
)

# -------------------------------------------------
# LOAD MODEL (MEMORY SAFE)
# -------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.float32,   # ⬅ avoid fp16 mmap issues
    low_cpu_mem_usage=True       # ⬅ VERY IMPORTANT
)

model.to(device)
model.eval()

# -------------------------------------------------
# LOAD LABEL ENCODER
# -------------------------------------------------
label_encoder = joblib.load(LABEL_ENCODER_PATH)

if isinstance(label_encoder.classes_, list):
    label_encoder.classes_ = np.array(label_encoder.classes_)

NUM_ENCODER_CLASSES = len(label_encoder.classes_)
NUM_MODEL_CLASSES = model.config.num_labels

print("[IntentLLM] Model classes :", NUM_MODEL_CLASSES)
print("[IntentLLM] Encoder classes:", NUM_ENCODER_CLASSES)
print("[IntentLLM] Encoder labels :", label_encoder.classes_)

# -------------------------------------------------
# SENTENCE SPLITTER
# -------------------------------------------------
def split_sentences(text: str):
    if not text:
        return []

    return [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', text)
        if len(s.strip()) > 5
    ]

# -------------------------------------------------
# INTENT PREDICTION (SAFE)
# -------------------------------------------------
def predict_intent(sentence: str):
    if not sentence or len(sentence.strip()) < 3:
        return "unknown", 0.0

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

    # 🛡️ ABSOLUTE SAFETY (NO CRASH)
    if pred_id >= NUM_ENCODER_CLASSES:
        return "unknown", round(confidence, 4)

    intent = label_encoder.inverse_transform([pred_id])[0]
    return intent, round(confidence, 4)

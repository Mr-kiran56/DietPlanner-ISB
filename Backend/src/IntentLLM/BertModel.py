# import torch
# import joblib
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # ---------- 1. DEVICE ----------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ---------- 2. ABSOLUTE PATHS (THIS IS THE FIX) ----------
# BASE_DIR = Path(__file__).resolve().parent

# MODEL_PATH = BASE_DIR / "IntentLLM-F" / "export" / "results" / "checkpoint-500"
# LABEL_ENCODER_PATH = BASE_DIR / "IntentLLM-F" / "export" / "results" / "label_encoder.pkl"

# # DEBUG (optional but useful)
# print("MODEL_PATH:", MODEL_PATH)
# print("Exists:", MODEL_PATH.exists())

# # ---------- 3. LOAD TOKENIZER ----------
# tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

# # ---------- 4. LOAD MODEL ----------
# model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
# model.to(device)
# model.eval()

# # ---------- 5. LOAD LABEL ENCODER ----------
# label_encoder = joblib.load(LABEL_ENCODER_PATH)

# # ---------- 6. PREDICTION FUNCTION ----------
# def predict_intent(sentence: str):
#     inputs = tokenizer(
#         sentence,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=128
#     ).to(device)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     pred_id = torch.argmax(outputs.logits, dim=1).item()
#     return label_encoder.inverse_transform([pred_id])[0]

# # ---------- 7. TEST ----------
# test_sentence = "History of coronary artery disease since 9"
# print("Prediction:", predict_intent(test_sentence))



import torch
import joblib
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- 1. DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. ABSOLUTE PATHS (THIS IS THE FIX) ----------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "IntentLLM-F" / "export" / "results" / "checkpoint-500"
LABEL_ENCODER_PATH =BASE_DIR / "label_encoder.pkl"

# DEBUG (optional but useful)
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

# ---------- 6. PREDICTION FUNCTION ----------

def sentences_splitter(text):
    text = "This is the first sentence. Here is the second one. Python is easy to learn."
    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
    return sentences

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

    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([pred_id])[0]



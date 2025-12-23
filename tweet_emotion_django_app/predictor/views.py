# predictor/views.py
from pathlib import Path
import json

from django.http import JsonResponse
from django.shortcuts import render

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# ----------------------------
# Paths (MODEL_DIR must exist)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # tweet_emotion_django_app/
MODEL_DIR = BASE_DIR / "export_emotion_model"      # <- folder next to manage.py

if not MODEL_DIR.exists():
    raise RuntimeError(f"MODEL_DIR not found: {MODEL_DIR}")

# ----------------------------
# Load tokenizer + model ONCE
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR.as_posix(), local_files_only=True)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR.as_posix(), local_files_only=True)

# If you saved a label map during training, load it. Otherwise keep a default.
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
if LABEL_MAP_PATH.exists():
    label_map = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    # label_map is likely {"0":"sadness", ...} -> convert keys to int
    label_map = {int(k): v for k, v in label_map.items()}
else:
    # Default (change if your dataset differs)
    label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}


def home(request):
    return render(request, "predictor/home.html")
# predictor/views.py

def info(request):
    return render(request, "predictor/info.html")


def predict(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    text = request.POST.get("text", "").strip()
    if not text:
        return JsonResponse({"error": "Empty text"}, status=400)

    inputs = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=96
    )

    outputs = model(inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=1).numpy()[0].tolist()

    pred_id = int(tf.argmax(logits, axis=1).numpy()[0])
    pred_label = label_map.get(pred_id, str(pred_id))

    return JsonResponse({
        "text": text,
        "pred_id": pred_id,
        "pred_label": pred_label,
        "probs": probs
    })

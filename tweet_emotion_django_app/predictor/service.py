from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


def probabilities(logits):
    values = np.asarray(logits, dtype=float).reshape(-1)
    if len(values) != 6 or not np.isfinite(values).all():
        raise RuntimeError("Expected six finite emotion logits")
    exponentials = np.exp(values - values.max())
    return (exponentials / exponentials.sum()).tolist()


class EmotionService:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.labels = {
            int(key): value
            for key, value in json.loads(
                (model_dir / "label_map.json").read_text(encoding="utf-8")
            ).items()
        }
        if set(self.labels) != set(range(6)):
            raise RuntimeError("Label map must contain exactly six classes")
        self.version = "unloaded"

    @property
    def artifact_present(self):
        return (self.model_dir / "tf_model.h5").exists()

    def load(self):
        weight_path = self.model_dir / "tf_model.h5"
        if not weight_path.exists():
            raise RuntimeError("Trained TensorFlow weights are missing")
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        self.version = "distilbert-" + hashlib.sha256(weight_path.read_bytes()).hexdigest()[:12]

    def predict(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("text must contain 1 to 4000 characters")
        if self.model is None:
            self.load()
        started = time.perf_counter()
        inputs = self.tokenizer(
            text.strip(), return_tensors="tf", truncation=True, padding=True, max_length=96
        )
        outputs = self.model(inputs, training=False)
        logits = outputs.logits.numpy() if hasattr(outputs.logits, "numpy") else outputs.logits
        probs = probabilities(logits)
        predicted = int(np.argmax(probs))
        return {
            "text": text.strip(),
            "pred_id": predicted,
            "pred_label": self.labels[predicted],
            "probs": probs,
            "model_version": self.version,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }

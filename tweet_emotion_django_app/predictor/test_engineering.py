from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.test import Client, SimpleTestCase

from .service import EmotionService, probabilities


class EmotionTests(SimpleTestCase):
    def test_probability_schema_and_stable_softmax(self):
        probs = probabilities([1000, 1001, 1002, 1003, 1004, 1005])
        self.assertAlmostEqual(sum(probs), 1)
        with self.assertRaises(RuntimeError):
            probabilities([1, 2])

    def test_fake_model_contract_and_real_label_map(self):
        service = EmotionService(Path(__file__).resolve().parents[1] / "export_emotion_model")
        service.model = lambda inputs, training: SimpleNamespace(logits=[[0, 4, 0, 0, 0, 0]])
        service.tokenizer = lambda text, **kwargs: {}
        result = service.predict("happy")
        self.assertEqual(result["pred_label"], "joy")
        self.assertEqual(len(result["probs"]), 6)

    def test_missing_weights_and_invalid_input(self):
        client = Client()
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/predict/").status_code, 405)
        self.assertEqual(client.post("/predict/", {"text": ""}).status_code, 400)
        self.assertEqual(client.post("/predict/", {"text": "hello"}).status_code, 503)
        self.assertFalse(client.get("/health/").json()["trained_weights_present"])

    def test_csrf_and_success_response(self):
        self.assertEqual(
            Client(enforce_csrf_checks=True).post("/predict/", {"text": "hello"}).status_code, 403
        )
        with patch(
            "predictor.views.service.predict",
            return_value={"pred_label": "joy", "probs": [0, 1, 0, 0, 0, 0]},
        ):
            self.assertEqual(
                Client().post("/predict/", {"text": "hello"}).json()["pred_label"], "joy"
            )

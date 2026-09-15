import logging
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .service import EmotionService

logger = logging.getLogger(__name__)
service = EmotionService(Path(__file__).resolve().parents[1] / "export_emotion_model")


def home(request):
    return render(request, "predictor/home.html")


def info(request):
    return render(request, "predictor/info.html")


def health(_request):
    return JsonResponse(
        {
            "status": "ok",
            "trained_weights_present": service.artifact_present,
            "model_loaded": service.model is not None,
            "model_version": service.version,
        }
    )


@require_POST
def predict(request):
    try:
        result = service.predict(request.POST.get("text", ""))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception:
        logger.exception("Emotion inference failed")
        return JsonResponse({"error": "The trained emotion model is unavailable."}, status=503)
    return JsonResponse(result)

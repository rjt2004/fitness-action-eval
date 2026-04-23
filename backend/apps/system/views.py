from __future__ import annotations

from django.conf import settings
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from config.api_response import api_success
from fitness_action_eval.model_options import POSE_MODEL_OPTIONS


def health_check(request):
    data = {
        "service": "fitness-action-eval-backend",
        "status": "UP",
        "debug": settings.DEBUG,
        "timezone": settings.TIME_ZONE,
        "database_engine": settings.DATABASES["default"]["ENGINE"],
    }
    return api_success(data=data, message="health check passed")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def pose_model_options(request):
    options = [
        {
            "value": key,
            "label": meta["label"],
            "description": meta["description"],
        }
        for key, meta in POSE_MODEL_OPTIONS.items()
    ]
    return api_success(data=options, message="获取姿态模型选项成功")

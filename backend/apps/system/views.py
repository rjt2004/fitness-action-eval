from __future__ import annotations

from django.conf import settings

from config.api_response import api_success


def health_check(request):
    data = {
        "service": "fitness-action-eval-backend",
        "status": "UP",
        "debug": settings.DEBUG,
        "timezone": settings.TIME_ZONE,
        "database_engine": settings.DATABASES["default"]["ENGINE"],
    }
    return api_success(data=data, message="health check passed")

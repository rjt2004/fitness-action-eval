from __future__ import annotations

from typing import Any

from django.http import JsonResponse


def api_success(data: Any = None, message: str = "ok", status_code: int = 200) -> JsonResponse:
    payload = {
        "code": 0,
        "message": message,
        "data": data,
    }
    return JsonResponse(payload, status=status_code)


def api_error(message: str = "error", code: int = 1, status_code: int = 400, data: Any = None) -> JsonResponse:
    payload = {
        "code": code,
        "message": message,
        "data": data,
    }
    return JsonResponse(payload, status=status_code)

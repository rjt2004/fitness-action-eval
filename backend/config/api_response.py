from __future__ import annotations

from typing import Any

from django.http import JsonResponse


def api_success(data: Any = None, message: str = "ok", status_code: int = 200) -> JsonResponse:
    """统一返回成功响应，便于前后端保持固定的数据结构。"""

    payload = {
        "code": 0,
        "message": message,
        "data": data,
    }
    return JsonResponse(payload, status=status_code)


def api_error(message: str = "error", code: int = 1, status_code: int = 400, data: Any = None) -> JsonResponse:
    """统一返回失败响应，错误信息和额外数据都放在 data 中。"""

    payload = {
        "code": code,
        "message": message,
        "data": data,
    }
    return JsonResponse(payload, status=status_code)

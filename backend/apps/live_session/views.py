from __future__ import annotations

from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from apps.accounts.models import User
from apps.template_manager.models import TemplateVideo
from config.api_response import api_error, api_success

from .models import LiveSession
from .serializers import LiveSessionDetailSerializer, LiveSessionListSerializer, LiveSessionStartSerializer
from .services import (
    delete_live_session,
    generate_session_no,
    get_live_session_preview_frame,
    repair_stale_live_session,
    session_registry_status,
    start_live_session,
    stop_live_session,
)


def _visible_sessions(user: User):
    """管理员看全部，普通用户只看自己的实时会话。"""

    queryset = LiveSession.objects.select_related("user", "template", "template__category", "template__created_by")
    if user.role == User.Role.ADMIN:
        return queryset.all()
    return queryset.filter(user=user)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def live_session_list_view(request):
    """实时会话列表。"""

    sessions = [repair_stale_live_session(session) for session in _visible_sessions(request.user)]
    return api_success(data=LiveSessionListSerializer(sessions, many=True).data, message="获取实时会话列表成功")


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def live_session_start_view(request):
    """创建并启动实时跟练会话。"""

    serializer = LiveSessionStartSerializer(data=request.data)
    if not serializer.is_valid():
        return api_error(message="启动实时会话失败", data=serializer.errors, status_code=400)

    validated = serializer.validated_data
    template = TemplateVideo.objects.get(id=validated["template_id"])
    session = LiveSession.objects.create(
        session_no=generate_session_no(),
        user=request.user,
        template=template,
        session_name=validated.get("session_name") or f"{template.template_name}-实时跟练",
        camera_source=str(validated.get("camera_source", "0")),
        camera_width=validated.get("camera_width"),
        camera_height=validated.get("camera_height"),
        camera_mirror=validated.get("camera_mirror", True),
        capture_error_frames=validated.get("capture_error_frames", False),
        frame_stride=validated.get("frame_stride", 1),
        smooth_window=validated.get("smooth_window", 1),
        pose_model=validated.get("pose_model", "lite"),
        score_scale=validated.get("score_scale", "8.00"),
        hint_threshold=validated.get("hint_threshold", "0.200"),
        hint_min_interval=validated.get("hint_min_interval", 60),
        max_hints=validated.get("max_hints", 360),
        ref_search_window=validated.get("ref_search_window", 60),
    )
    start_live_session(session)
    session.refresh_from_db()
    payload = LiveSessionDetailSerializer(session).data
    payload["runtime"] = session_registry_status(session.id)
    return api_success(data=payload, message="实时会话已启动", status_code=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def live_session_detail_view(request, session_id: int):
    """实时会话详情。"""

    session = get_object_or_404(_visible_sessions(request.user), id=session_id)
    session = repair_stale_live_session(session)
    payload = LiveSessionDetailSerializer(session).data
    payload["runtime"] = session_registry_status(session.id)
    return api_success(data=payload, message="获取实时会话详情成功")


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def live_session_stop_view(request, session_id: int):
    """向后台线程发送停止信号。"""

    session = get_object_or_404(_visible_sessions(request.user), id=session_id)
    if session.status not in {LiveSession.Status.PENDING, LiveSession.Status.RUNNING, LiveSession.Status.PAUSED}:
        return api_error(message="当前会话不可停止。", status_code=400)
    signaled = stop_live_session(session)
    payload = LiveSessionDetailSerializer(session).data
    payload["runtime"] = session_registry_status(session.id)
    payload["stop_signal_sent"] = signaled
    return api_success(data=payload, message="已发送停止信号")


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def live_session_delete_view(request, session_id: int):
    """删除已结束的实时会话及其文件。"""

    session = get_object_or_404(_visible_sessions(request.user), id=session_id)
    session = repair_stale_live_session(session)
    try:
        delete_live_session(session)
    except ValueError as exc:
        return api_error(message=str(exc), status_code=400)
    return api_success(data={"session_id": session_id}, message="实时会话删除成功")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def live_session_preview_frame_view(request, session_id: int):
    """保留 HTTP 预览帧接口，作为 WebSocket 的兼容回退。"""

    get_object_or_404(_visible_sessions(request.user), id=session_id)
    frame_bytes = get_live_session_preview_frame(session_id)
    if not frame_bytes:
        return HttpResponse(status=204)
    return HttpResponse(frame_bytes, content_type="image/jpeg")

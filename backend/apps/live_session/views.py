from __future__ import annotations

from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from config.api_response import api_error, api_success

from apps.accounts.models import User
from apps.template_manager.models import TemplateVideo

from .models import LiveSession
from .serializers import LiveSessionDetailSerializer, LiveSessionListSerializer, LiveSessionStartSerializer
from .services import generate_session_no, session_registry_status, start_live_session, stop_live_session


def _visible_sessions(user: User):
    queryset = LiveSession.objects.select_related("user", "template", "template__category", "template__created_by")
    if user.role == User.Role.ADMIN:
        return queryset.all()
    return queryset.filter(user=user)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def live_session_list_view(request):
    sessions = _visible_sessions(request.user)
    return api_success(data=LiveSessionListSerializer(sessions, many=True).data, message="获取实时会话列表成功")


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def live_session_start_view(request):
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
        preview=validated.get("preview", False),
        frame_stride=validated.get("frame_stride", 4),
        smooth_window=validated.get("smooth_window", 7),
        score_scale=validated.get("score_scale", "8.00"),
        hint_threshold=validated.get("hint_threshold", "0.180"),
        ref_search_window=validated.get("ref_search_window", 90),
        max_frames=validated.get("max_frames"),
    )
    start_live_session(session)
    session.refresh_from_db()
    payload = LiveSessionDetailSerializer(session).data
    payload["runtime"] = session_registry_status(session.id)
    return api_success(data=payload, message="实时会话已启动", status_code=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def live_session_detail_view(request, session_id: int):
    session = get_object_or_404(_visible_sessions(request.user), id=session_id)
    payload = LiveSessionDetailSerializer(session).data
    payload["runtime"] = session_registry_status(session.id)
    return api_success(data=payload, message="获取实时会话详情成功")


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def live_session_stop_view(request, session_id: int):
    session = get_object_or_404(_visible_sessions(request.user), id=session_id)
    if session.status not in {LiveSession.Status.PENDING, LiveSession.Status.RUNNING}:
        return api_error(message="当前会话不在运行状态，无法停止。", status_code=400)
    signaled = stop_live_session(session)
    payload = LiveSessionDetailSerializer(session).data
    payload["runtime"] = session_registry_status(session.id)
    payload["stop_signal_sent"] = signaled
    return api_success(data=payload, message="已发送停止信号")

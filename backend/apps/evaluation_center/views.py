from __future__ import annotations

from pathlib import Path

from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from apps.accounts.models import User
from apps.template_manager.models import FileAsset, TemplateVideo
from config.api_response import api_error, api_success

from .models import EvaluationHint, EvaluationPhaseResult, EvaluationTask
from .serializers import (
    EvaluationHintSerializer,
    EvaluationPhaseResultSerializer,
    EvaluationTaskCreateSerializer,
    EvaluationTaskDetailSerializer,
    EvaluationTaskListSerializer,
)
from .services import (
    delete_evaluation_task,
    generate_task_no,
    load_result_payload,
    repair_stale_task,
    start_evaluation_task,
    task_registry_status,
)


def _visible_tasks(user: User):
    """管理员可查看全部任务，普通用户仅查看自己的任务。"""

    queryset = EvaluationTask.objects.select_related("user", "template", "template__category", "template__created_by")
    if user.role == User.Role.ADMIN:
        return queryset.all()
    return queryset.filter(user=user)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def evaluation_task_list_view(request):
    """离线评估任务列表。"""

    tasks = [repair_stale_task(task) for task in _visible_tasks(request.user)]
    payload = []
    for task in tasks:
        row = EvaluationTaskListSerializer(task).data
        row["runtime"] = task_registry_status(task.id)
        payload.append(row)
    return api_success(data=payload, message="获取评估任务列表成功")


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def evaluation_task_create_view(request):
    """创建新的离线评估任务并放入后台线程。"""

    serializer = EvaluationTaskCreateSerializer(data=request.data)
    if not serializer.is_valid():
        return api_error(message="创建评估任务失败", data=serializer.errors, status_code=400)

    validated = serializer.validated_data
    template = TemplateVideo.objects.get(id=validated["template_id"])

    task = EvaluationTask.objects.create(
        task_no=generate_task_no(),
        user=request.user,
        template=template,
        task_name=validated.get("task_name") or f"{template.template_name}-评估任务",
        query_video=validated["query_video"],
        score_scale=validated.get("score_scale", "8.00"),
        hint_threshold=validated.get("hint_threshold", "0.180"),
        hint_min_interval=validated.get("hint_min_interval", 8),
        max_hints=validated.get("max_hints", 40),
        export_video=validated.get("export_video", False),
        frame_stride=validated.get("frame_stride", 1),
        smooth_window=validated.get("smooth_window", 5),
        pose_model=validated.get("pose_model", "follow_template"),
        progress_percent=0,
        progress_text="任务已创建，等待开始处理",
    )

    FileAsset.objects.update_or_create(
        biz_type="evaluation_input",
        biz_id=task.id,
        defaults={
            "file_name": Path(task.query_video.name).name,
            "file_path": task.query_video.path,
            "file_type": Path(task.query_video.name).suffix.lower().lstrip("."),
            "file_size": task.query_video.size or 0,
        },
    )

    start_evaluation_task(task)
    task.refresh_from_db()
    payload = EvaluationTaskDetailSerializer(task).data
    payload["runtime"] = task_registry_status(task.id)
    payload["result_payload"] = {}
    return api_success(data=payload, message="评估任务已启动", status_code=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def evaluation_task_detail_view(request, task_id: int):
    """离线评估任务详情。"""

    task = get_object_or_404(_visible_tasks(request.user), id=task_id)
    task = repair_stale_task(task)
    payload = EvaluationTaskDetailSerializer(task).data
    payload["runtime"] = task_registry_status(task.id)
    payload["result_payload"] = load_result_payload(task)
    return api_success(data=payload, message="获取评估任务详情成功")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def evaluation_task_phases_view(request, task_id: int):
    """离线评估阶段结果。"""

    task = get_object_or_404(_visible_tasks(request.user), id=task_id)
    task = repair_stale_task(task)
    rows = EvaluationPhaseResult.objects.filter(task=task)
    return api_success(data=EvaluationPhaseResultSerializer(rows, many=True).data, message="获取评估阶段结果成功")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def evaluation_task_hints_view(request, task_id: int):
    """离线评估纠错提示。"""

    task = get_object_or_404(_visible_tasks(request.user), id=task_id)
    task = repair_stale_task(task)
    rows = EvaluationHint.objects.filter(task=task)
    return api_success(data=EvaluationHintSerializer(rows, many=True).data, message="获取评估提示成功")


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def evaluation_task_delete_view(request, task_id: int):
    """删除离线评估任务及其结果文件。"""

    task = get_object_or_404(_visible_tasks(request.user), id=task_id)
    try:
        delete_evaluation_task(task)
    except ValueError as exc:
        return api_error(message=str(exc), status_code=400)
    return api_success(data={"task_id": task_id}, message="评估任务删除成功")

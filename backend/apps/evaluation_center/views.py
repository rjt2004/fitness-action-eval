from __future__ import annotations

from pathlib import Path

from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from config.api_response import api_error, api_success

from apps.accounts.models import User
from apps.template_manager.models import FileAsset, TemplateVideo

from .models import EvaluationHint, EvaluationPhaseResult, EvaluationTask
from .serializers import (
    EvaluationHintSerializer,
    EvaluationPhaseResultSerializer,
    EvaluationTaskCreateSerializer,
    EvaluationTaskDetailSerializer,
    EvaluationTaskListSerializer,
)
from .services import generate_task_no, load_result_payload, persist_result_details, run_evaluation_task


def _visible_tasks(user: User):
    queryset = EvaluationTask.objects.select_related("user", "template", "template__category", "template__created_by")
    if user.role == User.Role.ADMIN:
        return queryset.all()
    return queryset.filter(user=user)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def evaluation_task_list_view(request):
    tasks = _visible_tasks(request.user)
    return api_success(data=EvaluationTaskListSerializer(tasks, many=True).data, message="获取评估任务列表成功")


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def evaluation_task_create_view(request):
    serializer = EvaluationTaskCreateSerializer(data=request.data)
    if not serializer.is_valid():
        return api_error(message="创建评估任务失败", data=serializer.errors, status_code=400)

    validated = serializer.validated_data
    template = TemplateVideo.objects.get(id=validated["template_id"])

    task = EvaluationTask.objects.create(
        task_no=generate_task_no(),
        user=request.user,
        template=template,
        task_name=validated.get("task_name") or f"{template.template_name}-评分任务",
        query_video=validated["query_video"],
        score_scale=validated.get("score_scale", "8.00"),
        hint_threshold=validated.get("hint_threshold", "0.180"),
        hint_min_interval=validated.get("hint_min_interval", 8),
        max_hints=validated.get("max_hints", 40),
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

    try:
        run_evaluation_task(task)
        persist_result_details(task)
    except Exception as exc:
        task.refresh_from_db()
        return api_error(
            message="评估任务执行失败",
            data={
                "task_id": task.id,
                "task_no": task.task_no,
                "detail": str(exc),
            },
            status_code=500,
        )

    task.refresh_from_db()
    return api_success(data=EvaluationTaskDetailSerializer(task).data, message="评估任务执行成功", status_code=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def evaluation_task_detail_view(request, task_id: int):
    task = get_object_or_404(_visible_tasks(request.user), id=task_id)
    payload = EvaluationTaskDetailSerializer(task).data
    payload["result_payload"] = load_result_payload(task)
    return api_success(data=payload, message="获取评估任务详情成功")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def evaluation_task_phases_view(request, task_id: int):
    task = get_object_or_404(_visible_tasks(request.user), id=task_id)
    rows = EvaluationPhaseResult.objects.filter(task=task)
    return api_success(data=EvaluationPhaseResultSerializer(rows, many=True).data, message="获取评估阶段结果成功")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def evaluation_task_hints_view(request, task_id: int):
    task = get_object_or_404(_visible_tasks(request.user), id=task_id)
    rows = EvaluationHint.objects.filter(task=task)
    return api_success(data=EvaluationHintSerializer(rows, many=True).data, message="获取评估提示成功")

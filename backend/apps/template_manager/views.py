from __future__ import annotations

from django.db import transaction
from django.db.models import ProtectedError
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from apps.accounts.models import User
from apps.accounts.permissions import IsAdminRole
from config.api_response import api_error, api_success

from .models import ActionCategory, TemplateVideo
from .serializers import (
    ActionCategorySerializer,
    TemplateUploadSerializer,
    TemplateVideoDetailSerializer,
    TemplateVideoListSerializer,
)
from .services import (
    delete_template_bundle,
    ensure_default_baduanjin_category,
    register_source_asset,
    start_template_build,
)


def _visible_templates(user: User):
    queryset = TemplateVideo.objects.select_related("category", "created_by").order_by("-id")
    if user.role == User.Role.ADMIN:
        return queryset
    return queryset.filter(status=TemplateVideo.Status.READY)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def category_list_view(request):
    ensure_default_baduanjin_category()
    categories = ActionCategory.objects.filter(is_active=True).order_by("id")
    return api_success(data=ActionCategorySerializer(categories, many=True).data, message="获取动作类别成功")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def template_list_view(request):
    templates = _visible_templates(request.user)
    return api_success(data=TemplateVideoListSerializer(templates, many=True).data, message="获取模板列表成功")


@api_view(["POST"])
@permission_classes([IsAuthenticated, IsAdminRole])
def template_upload_view(request):
    serializer = TemplateUploadSerializer(data=request.data)
    if not serializer.is_valid():
        return api_error(message="模板上传失败", data=serializer.errors, status_code=400)

    validated = serializer.validated_data
    category = ensure_default_baduanjin_category()

    with transaction.atomic():
        template = TemplateVideo.objects.create(
            category=category,
            template_name=validated["template_name"],
            version=validated.get("version", "v1"),
            source_video=validated["source_video"],
            frame_stride=validated.get("frame_stride", 4),
            smooth_window=validated.get("smooth_window", 7),
            pose_model=validated.get("pose_model", "heavy"),
            created_by=request.user,
        )
        register_source_asset(template)

    template = start_template_build(template)
    return api_success(data=TemplateVideoDetailSerializer(template).data, message="模板已上传，正在生成", status_code=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def template_detail_view(request, template_id: int):
    template = get_object_or_404(_visible_templates(request.user), id=template_id)
    return api_success(data=TemplateVideoDetailSerializer(template).data, message="获取模板详情成功")


@api_view(["DELETE"])
@permission_classes([IsAuthenticated, IsAdminRole])
def template_delete_view(request, template_id: int):
    template = get_object_or_404(TemplateVideo, id=template_id)
    if template.status == TemplateVideo.Status.BUILDING:
        return api_error(message="模板正在生成中，请生成结束后再删除。", status_code=400)
    template_name = template.template_name
    try:
        delete_template_bundle(template)
    except ProtectedError:
        return api_error(
            message="模板已被评估任务或实时会话使用，暂时不能删除。",
            status_code=400,
        )
    return api_success(data={"template_id": template_id, "template_name": template_name}, message="模板删除成功")

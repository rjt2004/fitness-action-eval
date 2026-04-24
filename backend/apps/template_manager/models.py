from __future__ import annotations

from django.conf import settings
from django.db import models

from fitness_action_eval.model_options import DEFAULT_TEMPLATE_MODEL_KEY, get_pose_model_choices


class ActionCategory(models.Model):
    """动作分类，目前系统主要使用八段锦分类。"""

    code = models.CharField(max_length=50, unique=True, verbose_name="动作编码")
    name = models.CharField(max_length=100, verbose_name="动作名称")
    description = models.TextField(blank=True, default="", verbose_name="说明")
    is_active = models.BooleanField(default=True, verbose_name="是否启用")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        db_table = "action_category"
        verbose_name = "动作分类"
        verbose_name_plural = "动作分类"

    def __str__(self) -> str:
        return f"{self.name}({self.code})"


class TemplateVideo(models.Model):
    """动作模板表，保存标准视频及其姿态模板参数。"""

    class Status(models.TextChoices):
        DRAFT = "draft", "草稿"
        BUILDING = "building", "生成中"
        READY = "ready", "已生成"
        FAILED = "failed", "生成失败"

    category = models.ForeignKey(ActionCategory, on_delete=models.PROTECT, related_name="templates", verbose_name="动作分类")
    template_name = models.CharField(max_length=100, verbose_name="模板名称")
    version = models.CharField(max_length=30, default="v1", verbose_name="版本")
    source_video = models.FileField(upload_to="template_center/source/", verbose_name="标准视频")
    template_file_path = models.CharField(max_length=255, blank=True, default="", verbose_name="模板文件路径")
    cover_image_path = models.CharField(max_length=255, blank=True, default="", verbose_name="封面图路径")
    frame_stride = models.PositiveIntegerField(default=4, verbose_name="抽帧步长")
    smooth_window = models.PositiveIntegerField(default=7, verbose_name="平滑窗口")
    pose_model = models.CharField(
        max_length=20,
        choices=get_pose_model_choices(),
        default=DEFAULT_TEMPLATE_MODEL_KEY,
        verbose_name="姿态模型",
    )
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT, verbose_name="状态")
    progress_percent = models.PositiveIntegerField(default=0, verbose_name="进度百分比")
    progress_text = models.CharField(max_length=100, blank=True, default="", verbose_name="进度说明")
    build_error = models.TextField(blank=True, default="", verbose_name="生成错误")
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="created_templates",
        verbose_name="创建人",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        db_table = "template_video"
        verbose_name = "动作模板"
        verbose_name_plural = "动作模板"
        ordering = ["-id"]

    def __str__(self) -> str:
        return self.template_name


class FileAsset(models.Model):
    """统一登记模板、评估、会话产生的文件资源。"""

    class BizType(models.TextChoices):
        TEMPLATE_SOURCE = "template_source", "模板源视频"
        TEMPLATE_FILE = "template_file", "姿态模板文件"
        EVALUATION_INPUT = "evaluation_input", "评估输入视频"
        EVALUATION_JSON = "evaluation_json", "评估结果 JSON"
        EVALUATION_PLOT = "evaluation_plot", "评估结果图"
        EVALUATION_VIDEO = "evaluation_video", "评估结果视频"

    biz_type = models.CharField(max_length=30, choices=BizType.choices, verbose_name="业务类型")
    biz_id = models.BigIntegerField(verbose_name="业务主键")
    file_name = models.CharField(max_length=255, verbose_name="文件名")
    file_path = models.CharField(max_length=255, verbose_name="文件路径")
    file_type = models.CharField(max_length=30, blank=True, default="", verbose_name="文件类型")
    file_size = models.BigIntegerField(default=0, verbose_name="文件大小")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        db_table = "file_asset"
        verbose_name = "文件资源"
        verbose_name_plural = "文件资源"
        ordering = ["-id"]

    def __str__(self) -> str:
        return self.file_name

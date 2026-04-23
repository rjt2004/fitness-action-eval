from __future__ import annotations

from django.conf import settings
from django.db import models

from apps.template_manager.models import TemplateVideo
from fitness_action_eval.model_options import get_pose_model_choices


class LiveSession(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "待启动"
        RUNNING = "running", "运行中"
        PAUSED = "paused", "已暂停"
        SUCCESS = "success", "已完成"
        FAILED = "failed", "失败"
        STOPPED = "stopped", "已停止"

    session_no = models.CharField(max_length=50, unique=True, verbose_name="会话编号")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="live_sessions",
        verbose_name="所属用户",
    )
    template = models.ForeignKey(
        TemplateVideo,
        on_delete=models.PROTECT,
        related_name="live_sessions",
        verbose_name="模板",
    )
    session_name = models.CharField(max_length=100, blank=True, default="", verbose_name="会话名称")
    camera_source = models.CharField(max_length=255, default="0", verbose_name="摄像头来源")
    camera_width = models.PositiveIntegerField(null=True, blank=True, verbose_name="摄像头宽度")
    camera_height = models.PositiveIntegerField(null=True, blank=True, verbose_name="摄像头高度")
    camera_mirror = models.BooleanField(default=True, verbose_name="是否镜像")
    preview = models.BooleanField(default=False, verbose_name="是否本地预览")
    export_video = models.BooleanField(default=False, verbose_name="是否导出视频")
    capture_error_frames = models.BooleanField(default=True, verbose_name="是否保存错误动作帧")
    error_frame_count = models.PositiveIntegerField(default=0, verbose_name="错误动作帧数量")
    frame_stride = models.PositiveIntegerField(default=1, verbose_name="抽帧步长")
    smooth_window = models.PositiveIntegerField(default=1, verbose_name="平滑窗口")
    pose_model = models.CharField(max_length=20, choices=get_pose_model_choices(include_follow_template=True), default="lite", verbose_name="姿态模型")
    score_scale = models.DecimalField(max_digits=6, decimal_places=2, default=8.00, verbose_name="评分尺度")
    hint_threshold = models.DecimalField(max_digits=6, decimal_places=3, default=0.200, verbose_name="提示阈值")
    hint_min_interval = models.PositiveIntegerField(default=60, verbose_name="提示最小间隔")
    max_hints = models.PositiveIntegerField(default=360, verbose_name="提示上限")
    ref_search_window = models.PositiveIntegerField(default=60, verbose_name="模板搜索窗口")
    max_frames = models.PositiveIntegerField(null=True, blank=True, verbose_name="最大处理帧数")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING, verbose_name="状态")
    summary_json_path = models.CharField(max_length=255, blank=True, default="", verbose_name="摘要 JSON 路径")
    output_video_path = models.CharField(max_length=255, blank=True, default="", verbose_name="输出视频路径")
    avg_score = models.DecimalField(max_digits=6, decimal_places=2, default=0, verbose_name="平均得分")
    matched_frames = models.PositiveIntegerField(default=0, verbose_name="匹配帧数")
    final_phase_name = models.CharField(max_length=100, blank=True, default="", verbose_name="最终动作阶段")
    final_phase_cue = models.CharField(max_length=255, blank=True, default="", verbose_name="最终动作要领")
    final_part = models.CharField(max_length=50, blank=True, default="", verbose_name="最终关注部位")
    error_message = models.TextField(blank=True, default="", verbose_name="错误信息")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    started_at = models.DateTimeField(null=True, blank=True, verbose_name="开始时间")
    ended_at = models.DateTimeField(null=True, blank=True, verbose_name="结束时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        db_table = "live_session"
        verbose_name = "实时跟练会话"
        verbose_name_plural = "实时跟练会话"
        ordering = ["-id"]

    def __str__(self) -> str:
        return self.session_no

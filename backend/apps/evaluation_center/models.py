from __future__ import annotations

from django.conf import settings
from django.db import models

from apps.template_manager.models import TemplateVideo


class EvaluationTask(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "待处理"
        RUNNING = "running", "处理中"
        SUCCESS = "success", "成功"
        FAILED = "failed", "失败"

    task_no = models.CharField(max_length=50, unique=True, verbose_name="任务编号")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="evaluation_tasks",
        verbose_name="提交用户",
    )
    template = models.ForeignKey(
        TemplateVideo,
        on_delete=models.PROTECT,
        related_name="evaluation_tasks",
        verbose_name="模板",
    )
    task_name = models.CharField(max_length=100, blank=True, default="", verbose_name="任务名称")
    query_video = models.FileField(upload_to="evaluation_center/query/", verbose_name="待评估视频")
    result_json_path = models.CharField(max_length=255, blank=True, default="", verbose_name="结果 JSON 路径")
    result_plot_path = models.CharField(max_length=255, blank=True, default="", verbose_name="结果图路径")
    result_video_path = models.CharField(max_length=255, blank=True, default="", verbose_name="结果视频路径")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING, verbose_name="状态")
    score = models.DecimalField(max_digits=6, decimal_places=2, default=0, verbose_name="得分")
    normalized_distance = models.DecimalField(max_digits=10, decimal_places=4, default=0, verbose_name="归一化距离")
    hint_count = models.PositiveIntegerField(default=0, verbose_name="提示数量")
    max_hints = models.PositiveIntegerField(default=40, verbose_name="最大提示数量")
    hint_threshold = models.DecimalField(max_digits=6, decimal_places=3, default=0.180, verbose_name="提示阈值")
    hint_min_interval = models.PositiveIntegerField(default=8, verbose_name="提示最小间隔")
    score_scale = models.DecimalField(max_digits=6, decimal_places=2, default=8.00, verbose_name="评分尺度")
    error_message = models.TextField(blank=True, default="", verbose_name="错误信息")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    started_at = models.DateTimeField(null=True, blank=True, verbose_name="开始时间")
    finished_at = models.DateTimeField(null=True, blank=True, verbose_name="结束时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        db_table = "evaluation_task"
        verbose_name = "评估任务"
        verbose_name_plural = "评估任务"
        ordering = ["-id"]

    def __str__(self) -> str:
        return self.task_no


class EvaluationPhaseResult(models.Model):
    task = models.ForeignKey(
        EvaluationTask,
        on_delete=models.CASCADE,
        related_name="phase_results",
        verbose_name="所属任务",
    )
    phase_type = models.CharField(max_length=20, verbose_name="阶段类型")
    phase_id = models.IntegerField(verbose_name="阶段编号")
    phase_name = models.CharField(max_length=100, verbose_name="阶段名称")
    cue = models.CharField(max_length=255, blank=True, default="", verbose_name="动作要领")
    start_seq_idx = models.IntegerField(default=0, verbose_name="开始序列下标")
    end_seq_idx = models.IntegerField(default=0, verbose_name="结束序列下标")
    start_time_s = models.DecimalField(max_digits=10, decimal_places=3, default=0, verbose_name="开始时间")
    end_time_s = models.DecimalField(max_digits=10, decimal_places=3, default=0, verbose_name="结束时间")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        db_table = "evaluation_phase_result"
        verbose_name = "评估阶段结果"
        verbose_name_plural = "评估阶段结果"
        ordering = ["id"]

    def __str__(self) -> str:
        return f"{self.task.task_no}-{self.phase_type}-{self.phase_name}"


class EvaluationHint(models.Model):
    task = models.ForeignKey(
        EvaluationTask,
        on_delete=models.CASCADE,
        related_name="hints",
        verbose_name="所属任务",
    )
    phase_id = models.IntegerField(null=True, blank=True, verbose_name="阶段编号")
    phase_name = models.CharField(max_length=100, blank=True, default="", verbose_name="阶段名称")
    cue = models.CharField(max_length=255, blank=True, default="", verbose_name="动作要领")
    query_phase_id = models.IntegerField(null=True, blank=True, verbose_name="测试阶段编号")
    ref_index = models.IntegerField(default=0, verbose_name="参考序列下标")
    query_index = models.IntegerField(default=0, verbose_name="测试序列下标")
    query_frame = models.IntegerField(default=0, verbose_name="测试帧号")
    query_time_s = models.DecimalField(max_digits=10, decimal_places=3, default=0, verbose_name="测试时间")
    ref_time_s = models.DecimalField(max_digits=10, decimal_places=3, default=0, verbose_name="参考时间")
    part = models.CharField(max_length=50, blank=True, default="", verbose_name="身体部位")
    part_error = models.DecimalField(max_digits=10, decimal_places=4, default=0, verbose_name="部位误差")
    point_error = models.DecimalField(max_digits=10, decimal_places=4, default=0, verbose_name="关键点误差")
    angle_error = models.DecimalField(max_digits=10, decimal_places=4, default=0, verbose_name="角度误差")
    message = models.CharField(max_length=255, verbose_name="提示内容")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        db_table = "evaluation_hint"
        verbose_name = "评估提示"
        verbose_name_plural = "评估提示"
        ordering = ["id"]

    def __str__(self) -> str:
        return f"{self.task.task_no}-{self.part}-{self.query_frame}"

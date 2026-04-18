from __future__ import annotations

from rest_framework import serializers

from apps.template_manager.models import FileAsset, TemplateVideo
from apps.template_manager.serializers import TemplateVideoListSerializer

from .models import EvaluationHint, EvaluationPhaseResult, EvaluationTask


class EvaluationTaskCreateSerializer(serializers.Serializer):
    template_id = serializers.IntegerField()
    task_name = serializers.CharField(max_length=100, required=False, allow_blank=True, default="")
    query_video = serializers.FileField()
    score_scale = serializers.DecimalField(max_digits=6, decimal_places=2, required=False, default="8.00")
    hint_threshold = serializers.DecimalField(max_digits=6, decimal_places=3, required=False, default="0.180")
    hint_min_interval = serializers.IntegerField(required=False, min_value=1, default=8)
    max_hints = serializers.IntegerField(required=False, min_value=1, default=40)

    def validate_template_id(self, value):
        try:
            template = TemplateVideo.objects.get(id=value)
        except TemplateVideo.DoesNotExist as exc:
            raise serializers.ValidationError("模板不存在。") from exc
        if template.status != TemplateVideo.Status.READY or not template.template_file_path:
            raise serializers.ValidationError("模板尚未生成完成，无法用于评估。")
        return value


class EvaluationTaskListSerializer(serializers.ModelSerializer):
    template = TemplateVideoListSerializer(read_only=True)
    username = serializers.CharField(source="user.username", read_only=True)
    query_video_url = serializers.SerializerMethodField()

    class Meta:
        model = EvaluationTask
        fields = (
            "id",
            "task_no",
            "task_name",
            "status",
            "score",
            "normalized_distance",
            "hint_count",
            "created_at",
            "started_at",
            "finished_at",
            "username",
            "template",
            "query_video_url",
        )

    def get_query_video_url(self, obj: EvaluationTask) -> str:
        return obj.query_video.url if obj.query_video else ""


class EvaluationTaskDetailSerializer(EvaluationTaskListSerializer):
    result_json_path = serializers.CharField()
    result_plot_path = serializers.CharField()
    result_video_path = serializers.CharField()
    error_message = serializers.CharField()
    file_assets = serializers.SerializerMethodField()

    class Meta(EvaluationTaskListSerializer.Meta):
        fields = EvaluationTaskListSerializer.Meta.fields + (
            "result_json_path",
            "result_plot_path",
            "result_video_path",
            "error_message",
            "max_hints",
            "hint_threshold",
            "hint_min_interval",
            "score_scale",
            "file_assets",
        )

    def get_file_assets(self, obj: EvaluationTask):
        assets = FileAsset.objects.filter(biz_id=obj.id, biz_type__startswith="evaluation_")
        return [
          {
              "id": item.id,
              "biz_type": item.biz_type,
              "file_name": item.file_name,
              "file_path": item.file_path,
              "file_type": item.file_type,
              "file_size": item.file_size,
              "created_at": item.created_at,
          }
          for item in assets
        ]


class EvaluationPhaseResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = EvaluationPhaseResult
        fields = (
            "id",
            "phase_type",
            "phase_id",
            "phase_name",
            "cue",
            "start_seq_idx",
            "end_seq_idx",
            "start_time_s",
            "end_time_s",
        )


class EvaluationHintSerializer(serializers.ModelSerializer):
    class Meta:
        model = EvaluationHint
        fields = (
            "id",
            "phase_id",
            "phase_name",
            "cue",
            "query_phase_id",
            "ref_index",
            "query_index",
            "query_frame",
            "query_time_s",
            "ref_time_s",
            "part",
            "part_error",
            "point_error",
            "angle_error",
            "message",
        )

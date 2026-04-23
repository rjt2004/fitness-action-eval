from __future__ import annotations

import json
from pathlib import Path

from rest_framework import serializers

from apps.template_manager.models import TemplateVideo
from apps.template_manager.serializers import TemplateVideoListSerializer
from fitness_action_eval.model_options import (
    DEFAULT_RUNTIME_MODEL_KEY,
    FOLLOW_TEMPLATE_MODEL_KEY,
    get_pose_model_label,
    normalize_pose_model_key,
)

from .models import LiveSession


class LiveSessionStartSerializer(serializers.Serializer):
    template_id = serializers.IntegerField()
    session_name = serializers.CharField(max_length=100, required=False, allow_blank=True, default="")
    camera_source = serializers.CharField(required=False, default="0")
    camera_width = serializers.IntegerField(required=False, allow_null=True, min_value=1, default=480)
    camera_height = serializers.IntegerField(required=False, allow_null=True, min_value=1, default=270)
    camera_mirror = serializers.BooleanField(required=False, default=True)
    preview = serializers.BooleanField(required=False, default=False)
    export_video = serializers.BooleanField(required=False, default=False)
    capture_error_frames = serializers.BooleanField(required=False, default=False)
    frame_stride = serializers.IntegerField(required=False, min_value=1, default=1)
    smooth_window = serializers.IntegerField(required=False, min_value=1, default=1)
    pose_model = serializers.CharField(required=False, default="lite")
    score_scale = serializers.DecimalField(max_digits=6, decimal_places=2, required=False, default="8.00")
    hint_threshold = serializers.DecimalField(max_digits=6, decimal_places=3, required=False, default="0.200")
    hint_min_interval = serializers.IntegerField(required=False, min_value=1, default=60)
    max_hints = serializers.IntegerField(required=False, min_value=1, default=360)
    ref_search_window = serializers.IntegerField(required=False, min_value=10, default=60)

    def validate_template_id(self, value):
        try:
            template = TemplateVideo.objects.get(id=value)
        except TemplateVideo.DoesNotExist as exc:
            raise serializers.ValidationError("模板不存在。") from exc
        if template.status != TemplateVideo.Status.READY or not template.template_file_path:
            raise serializers.ValidationError("模板尚未生成完成，无法启动实时跟练。")
        return value

    def validate_pose_model(self, value: str) -> str:
        return normalize_pose_model_key(
            value,
            default=DEFAULT_RUNTIME_MODEL_KEY,
            include_follow_template=True,
        )


class LiveSessionListSerializer(serializers.ModelSerializer):
    template = TemplateVideoListSerializer(read_only=True)
    username = serializers.CharField(source="user.username", read_only=True)
    hint_count = serializers.SerializerMethodField()
    pose_model_label = serializers.SerializerMethodField()

    class Meta:
        model = LiveSession
        fields = (
            "id",
            "session_no",
            "session_name",
            "status",
            "avg_score",
            "hint_count",
            "final_phase_name",
            "created_at",
            "started_at",
            "ended_at",
            "camera_source",
            "pose_model",
            "pose_model_label",
            "username",
            "template",
        )

    def get_hint_count(self, obj: LiveSession) -> int:
        payload = self._load_summary_payload(obj)
        try:
            return int(payload.get("hint_count", 0))
        except (TypeError, ValueError):
            return 0

    def get_pose_model_label(self, obj: LiveSession) -> str:
        if obj.pose_model == FOLLOW_TEMPLATE_MODEL_KEY:
            return f"跟随模板（{get_pose_model_label(obj.template.pose_model)}）"
        return get_pose_model_label(obj.pose_model)

    @staticmethod
    def _load_summary_payload(obj: LiveSession) -> dict:
        if not obj.summary_json_path:
            return {}
        path = Path(obj.summary_json_path)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


class LiveSessionDetailSerializer(LiveSessionListSerializer):
    summary_json_path = serializers.CharField()
    output_video_path = serializers.CharField()
    final_phase_cue = serializers.CharField()
    final_part = serializers.CharField()
    error_message = serializers.CharField()
    summary_payload = serializers.SerializerMethodField()
    runtime_payload = serializers.SerializerMethodField()

    class Meta(LiveSessionListSerializer.Meta):
        fields = LiveSessionListSerializer.Meta.fields + (
            "camera_width",
            "camera_height",
            "camera_mirror",
            "preview",
            "export_video",
            "capture_error_frames",
            "error_frame_count",
            "frame_stride",
            "smooth_window",
            "pose_model",
            "pose_model_label",
            "score_scale",
            "hint_threshold",
            "hint_min_interval",
            "max_hints",
            "ref_search_window",
            "summary_json_path",
            "output_video_path",
            "final_phase_cue",
            "final_part",
            "error_message",
            "summary_payload",
            "runtime_payload",
        )

    def get_summary_payload(self, obj: LiveSession):
        return self._load_summary_payload(obj)

    def get_runtime_payload(self, obj: LiveSession):
        from .services import get_live_session_runtime_payload

        return get_live_session_runtime_payload(obj.id)

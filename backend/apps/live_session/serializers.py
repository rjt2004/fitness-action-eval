from __future__ import annotations

from pathlib import Path

from rest_framework import serializers

from apps.template_manager.serializers import TemplateVideoListSerializer

from .models import LiveSession


class LiveSessionStartSerializer(serializers.Serializer):
    template_id = serializers.IntegerField()
    session_name = serializers.CharField(max_length=100, required=False, allow_blank=True, default="")
    camera_source = serializers.CharField(required=False, default="0")
    camera_width = serializers.IntegerField(required=False, allow_null=True, min_value=1)
    camera_height = serializers.IntegerField(required=False, allow_null=True, min_value=1)
    camera_mirror = serializers.BooleanField(required=False, default=True)
    preview = serializers.BooleanField(required=False, default=False)
    frame_stride = serializers.IntegerField(required=False, min_value=1, default=4)
    smooth_window = serializers.IntegerField(required=False, min_value=1, default=7)
    score_scale = serializers.DecimalField(max_digits=6, decimal_places=2, required=False, default="8.00")
    hint_threshold = serializers.DecimalField(max_digits=6, decimal_places=3, required=False, default="0.180")
    ref_search_window = serializers.IntegerField(required=False, min_value=10, default=90)
    max_frames = serializers.IntegerField(required=False, allow_null=True, min_value=1)

    def validate_template_id(self, value):
        from apps.template_manager.models import TemplateVideo

        try:
            template = TemplateVideo.objects.get(id=value)
        except TemplateVideo.DoesNotExist as exc:
            raise serializers.ValidationError("模板不存在。") from exc
        if template.status != TemplateVideo.Status.READY or not template.template_file_path:
            raise serializers.ValidationError("模板尚未生成完成，无法启动实时跟练。")
        return value


class LiveSessionListSerializer(serializers.ModelSerializer):
    template = TemplateVideoListSerializer(read_only=True)
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = LiveSession
        fields = (
            "id",
            "session_no",
            "session_name",
            "status",
            "avg_score",
            "matched_frames",
            "final_phase_name",
            "created_at",
            "started_at",
            "ended_at",
            "camera_source",
            "username",
            "template",
        )


class LiveSessionDetailSerializer(LiveSessionListSerializer):
    summary_json_path = serializers.CharField()
    output_video_path = serializers.CharField()
    final_phase_cue = serializers.CharField()
    final_part = serializers.CharField()
    error_message = serializers.CharField()
    summary_payload = serializers.SerializerMethodField()

    class Meta(LiveSessionListSerializer.Meta):
        fields = LiveSessionListSerializer.Meta.fields + (
            "camera_width",
            "camera_height",
            "camera_mirror",
            "preview",
            "frame_stride",
            "smooth_window",
            "score_scale",
            "hint_threshold",
            "ref_search_window",
            "max_frames",
            "summary_json_path",
            "output_video_path",
            "final_phase_cue",
            "final_part",
            "error_message",
            "summary_payload",
        )

    def get_summary_payload(self, obj: LiveSession):
        if not obj.summary_json_path:
            return {}
        path = Path(obj.summary_json_path)
        if not path.exists():
            return {}
        import json

        return json.loads(path.read_text(encoding="utf-8"))

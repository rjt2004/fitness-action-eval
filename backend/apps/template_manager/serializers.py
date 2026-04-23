from __future__ import annotations

from rest_framework import serializers

from fitness_action_eval.baduanjin import BADUANJIN_PHASES
from fitness_action_eval.model_options import DEFAULT_TEMPLATE_MODEL_KEY, POSE_MODEL_OPTIONS, get_pose_model_label, normalize_pose_model_key

from .models import ActionCategory, FileAsset, TemplateVideo


class ActionCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = ActionCategory
        fields = ("id", "code", "name", "description", "is_active")


class FileAssetSerializer(serializers.ModelSerializer):
    class Meta:
        model = FileAsset
        fields = ("id", "biz_type", "biz_id", "file_name", "file_path", "file_type", "file_size", "created_at")


class TemplateVideoListSerializer(serializers.ModelSerializer):
    category = ActionCategorySerializer(read_only=True)
    created_by_name = serializers.CharField(source="created_by.username", read_only=True)
    pose_model_label = serializers.SerializerMethodField()

    class Meta:
        model = TemplateVideo
        fields = (
            "id",
            "template_name",
            "version",
            "status",
            "progress_percent",
            "progress_text",
            "frame_stride",
            "smooth_window",
            "pose_model",
            "pose_model_label",
            "template_file_path",
            "cover_image_path",
            "build_error",
            "created_at",
            "updated_at",
            "category",
            "created_by_name",
        )

    def get_pose_model_label(self, obj: TemplateVideo) -> str:
        return get_pose_model_label(obj.pose_model)


class TemplateVideoDetailSerializer(TemplateVideoListSerializer):
    source_video_path = serializers.SerializerMethodField()
    file_assets = serializers.SerializerMethodField()
    phase_guides = serializers.SerializerMethodField()

    class Meta(TemplateVideoListSerializer.Meta):
        fields = TemplateVideoListSerializer.Meta.fields + ("source_video_path", "file_assets", "phase_guides")

    def get_source_video_path(self, obj: TemplateVideo) -> str:
        return obj.source_video.url if obj.source_video else ""

    def get_file_assets(self, obj: TemplateVideo):
        assets = FileAsset.objects.filter(
            biz_id=obj.id,
            biz_type__in=[FileAsset.BizType.TEMPLATE_SOURCE, FileAsset.BizType.TEMPLATE_FILE],
        )
        return FileAssetSerializer(assets, many=True).data

    def get_phase_guides(self, obj: TemplateVideo):
        return [
            {
                "phase_id": phase.phase_id,
                "key": phase.key,
                "phase_name": phase.display_name,
                "cue": phase.cue,
            }
            for phase in BADUANJIN_PHASES
        ]


class TemplateUploadSerializer(serializers.Serializer):
    category_id = serializers.IntegerField(required=False)
    template_name = serializers.CharField(max_length=100)
    version = serializers.CharField(max_length=30, required=False, default="v1")
    source_video = serializers.FileField()
    frame_stride = serializers.IntegerField(required=False, default=4, min_value=1)
    smooth_window = serializers.IntegerField(required=False, default=7, min_value=1)
    pose_model = serializers.ChoiceField(choices=tuple(POSE_MODEL_OPTIONS.keys()), required=False, default=DEFAULT_TEMPLATE_MODEL_KEY)

    def validate_category_id(self, value):
        if not ActionCategory.objects.filter(id=value, is_active=True).exists():
            raise serializers.ValidationError("动作类别不存在或已禁用。")
        return value

    def validate_pose_model(self, value: str) -> str:
        return normalize_pose_model_key(value, default=DEFAULT_TEMPLATE_MODEL_KEY)

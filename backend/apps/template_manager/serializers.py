from __future__ import annotations

from rest_framework import serializers

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

    class Meta:
        model = TemplateVideo
        fields = (
            "id",
            "template_name",
            "version",
            "status",
            "frame_stride",
            "smooth_window",
            "template_file_path",
            "cover_image_path",
            "build_error",
            "created_at",
            "updated_at",
            "category",
            "created_by_name",
        )


class TemplateVideoDetailSerializer(TemplateVideoListSerializer):
    source_video_path = serializers.SerializerMethodField()
    file_assets = serializers.SerializerMethodField()

    class Meta(TemplateVideoListSerializer.Meta):
        fields = TemplateVideoListSerializer.Meta.fields + ("source_video_path", "file_assets")

    def get_source_video_path(self, obj: TemplateVideo) -> str:
        return obj.source_video.url if obj.source_video else ""

    def get_file_assets(self, obj: TemplateVideo):
        assets = FileAsset.objects.filter(
            biz_id=obj.id,
            biz_type__in=[FileAsset.BizType.TEMPLATE_SOURCE, FileAsset.BizType.TEMPLATE_FILE],
        )
        return FileAssetSerializer(assets, many=True).data


class TemplateUploadSerializer(serializers.Serializer):
    category_id = serializers.IntegerField(required=False)
    template_name = serializers.CharField(max_length=100)
    version = serializers.CharField(max_length=30, required=False, default="v1")
    source_video = serializers.FileField()
    frame_stride = serializers.IntegerField(required=False, default=4, min_value=1)
    smooth_window = serializers.IntegerField(required=False, default=7, min_value=1)

    def validate_category_id(self, value):
        if not ActionCategory.objects.filter(id=value, is_active=True).exists():
            raise serializers.ValidationError("动作类别不存在或已禁用。")
        return value


class TemplateBuildSerializer(serializers.Serializer):
    frame_stride = serializers.IntegerField(required=False, min_value=1)
    smooth_window = serializers.IntegerField(required=False, min_value=1)

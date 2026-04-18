from django.contrib import admin

from .models import ActionCategory, FileAsset, TemplateVideo


@admin.register(ActionCategory)
class ActionCategoryAdmin(admin.ModelAdmin):
    list_display = ("id", "code", "name", "is_active", "created_at")
    search_fields = ("code", "name")
    list_filter = ("is_active",)


@admin.register(TemplateVideo)
class TemplateVideoAdmin(admin.ModelAdmin):
    list_display = ("id", "template_name", "category", "version", "status", "frame_stride", "created_by", "created_at")
    search_fields = ("template_name", "version")
    list_filter = ("status", "category")


@admin.register(FileAsset)
class FileAssetAdmin(admin.ModelAdmin):
    list_display = ("id", "biz_type", "biz_id", "file_name", "file_size", "created_at")
    search_fields = ("file_name", "file_path")
    list_filter = ("biz_type",)

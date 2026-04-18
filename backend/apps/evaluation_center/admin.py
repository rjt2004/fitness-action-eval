from django.contrib import admin

from .models import EvaluationHint, EvaluationPhaseResult, EvaluationTask


@admin.register(EvaluationTask)
class EvaluationTaskAdmin(admin.ModelAdmin):
    list_display = ("id", "task_no", "task_name", "user", "template", "status", "score", "hint_count", "created_at")
    search_fields = ("task_no", "task_name", "user__username")
    list_filter = ("status", "template__category")
    readonly_fields = ("task_no", "score", "normalized_distance", "hint_count", "created_at", "started_at", "finished_at")


@admin.register(EvaluationPhaseResult)
class EvaluationPhaseResultAdmin(admin.ModelAdmin):
    list_display = ("id", "task", "phase_type", "phase_id", "phase_name", "start_time_s", "end_time_s")
    search_fields = ("task__task_no", "phase_name")
    list_filter = ("phase_type",)


@admin.register(EvaluationHint)
class EvaluationHintAdmin(admin.ModelAdmin):
    list_display = ("id", "task", "phase_name", "part", "query_frame", "query_time_s", "part_error")
    search_fields = ("task__task_no", "phase_name", "part", "message")
    list_filter = ("phase_name", "part")

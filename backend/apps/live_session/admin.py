from django.contrib import admin

from .models import LiveSession


@admin.register(LiveSession)
class LiveSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "session_no", "session_name", "user", "template", "status", "avg_score", "matched_frames", "created_at")
    search_fields = ("session_no", "session_name", "user__username")
    list_filter = ("status", "template__category")
    readonly_fields = (
        "session_no",
        "avg_score",
        "matched_frames",
        "final_phase_name",
        "final_phase_cue",
        "final_part",
        "summary_json_path",
        "output_video_path",
        "created_at",
        "started_at",
        "ended_at",
    )

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from django.conf import settings

from config.video_utils import transcode_to_browser_mp4
from fitness_action_eval.pipeline import run_dtw_scoring_from_template

from apps.template_manager.models import FileAsset

from .models import EvaluationHint, EvaluationPhaseResult, EvaluationTask


def generate_task_no() -> str:
    return f"EVAL{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid4().hex[:6].upper()}"


def register_evaluation_asset(task: EvaluationTask, biz_type: str, file_path: str) -> None:
    path = Path(file_path)
    FileAsset.objects.update_or_create(
        biz_type=biz_type,
        biz_id=task.id,
        defaults={
            "file_name": path.name,
            "file_path": str(path),
            "file_type": path.suffix.lower().lstrip("."),
            "file_size": path.stat().st_size if path.exists() else 0,
        },
    )


def run_evaluation_task(task: EvaluationTask) -> EvaluationTask:
    result_dir = Path(settings.MEDIA_ROOT) / "evaluation_center" / "results" / task.task_no
    result_dir.mkdir(parents=True, exist_ok=True)

    out_json = result_dir / "result.json"
    out_plot = result_dir / "plot.png"
    out_video_raw = result_dir / "overlay_raw.mp4"
    out_video_web = result_dir / "overlay.mp4"

    task.status = EvaluationTask.Status.RUNNING
    task.started_at = datetime.now()
    task.error_message = ""
    task.save(update_fields=["status", "started_at", "error_message", "updated_at"])

    try:
        summary = run_dtw_scoring_from_template(
            template_path=task.template.template_file_path,
            query_video=task.query_video.path,
            out_json=str(out_json),
            out_plot=str(out_plot),
            out_video=str(out_video_raw),
            preview=False,
            score_scale=float(task.score_scale),
            hint_threshold=float(task.hint_threshold),
            hint_min_interval=int(task.hint_min_interval),
            max_hints=int(task.max_hints),
        )
        transcode_to_browser_mp4(str(out_video_raw), str(out_video_web))
        task.result_json_path = str(out_json)
        task.result_plot_path = str(out_plot)
        task.result_video_path = str(out_video_web)
        task.score = round(float(summary["score"]), 2)
        task.normalized_distance = round(float(summary["norm_dist"]), 4)
        task.hint_count = int(summary["hint_count"])
        task.status = EvaluationTask.Status.SUCCESS
        task.finished_at = datetime.now()
        task.save(
            update_fields=[
                "result_json_path",
                "result_plot_path",
                "result_video_path",
                "score",
                "normalized_distance",
                "hint_count",
                "status",
                "finished_at",
                "updated_at",
            ]
        )

        register_evaluation_asset(task, "evaluation_input", task.query_video.path)
        register_evaluation_asset(task, "evaluation_json", str(out_json))
        register_evaluation_asset(task, "evaluation_plot", str(out_plot))
        register_evaluation_asset(task, "evaluation_video", str(out_video_web))
    except Exception as exc:
        task.status = EvaluationTask.Status.FAILED
        task.error_message = str(exc)
        task.finished_at = datetime.now()
        task.save(update_fields=["status", "error_message", "finished_at", "updated_at"])
        raise
    return task


def load_result_payload(task: EvaluationTask) -> dict:
    if not task.result_json_path:
        return {}
    path = Path(task.result_json_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def persist_result_details(task: EvaluationTask) -> None:
    payload = load_result_payload(task)
    if not payload:
        return

    EvaluationPhaseResult.objects.filter(task=task).delete()
    EvaluationHint.objects.filter(task=task).delete()

    phase_rows = []
    for phase_type, rows in (
        ("reference", payload.get("reference_phases", [])),
        ("query", payload.get("query_phases", [])),
    ):
        for row in rows:
            phase_rows.append(
                EvaluationPhaseResult(
                    task=task,
                    phase_type=phase_type,
                    phase_id=int(row.get("phase_id", 0)),
                    phase_name=str(row.get("phase_name", "")),
                    cue=str(row.get("cue", "")),
                    start_seq_idx=int(row.get("start_seq_idx", 0)),
                    end_seq_idx=int(row.get("end_seq_idx", 0)),
                    start_time_s=float(row.get("start_time_s", 0.0)),
                    end_time_s=float(row.get("end_time_s", 0.0)),
                )
            )
    if phase_rows:
        EvaluationPhaseResult.objects.bulk_create(phase_rows)

    hint_rows = []
    for row in payload.get("hints", []):
        hint_rows.append(
            EvaluationHint(
                task=task,
                phase_id=row.get("phase_id"),
                phase_name=str(row.get("phase_name", "")),
                cue=str(row.get("cue", "")),
                query_phase_id=row.get("query_phase_id"),
                ref_index=int(row.get("ref_index", 0)),
                query_index=int(row.get("query_index", 0)),
                query_frame=int(row.get("query_frame", 0)),
                query_time_s=float(row.get("query_time_s", 0.0)),
                ref_time_s=float(row.get("ref_time_s", 0.0)),
                part=str(row.get("part", "")),
                part_error=float(row.get("part_error", 0.0)),
                point_error=float(row.get("point_error", 0.0)),
                angle_error=float(row.get("angle_error", 0.0)),
                message=str(row.get("message", "")),
            )
        )
    if hint_rows:
        EvaluationHint.objects.bulk_create(hint_rows)

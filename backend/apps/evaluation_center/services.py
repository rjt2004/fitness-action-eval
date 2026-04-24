from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
from django.conf import settings
from django.db import close_old_connections

from apps.template_manager.models import FileAsset
from config.video_utils import transcode_to_browser_mp4
from fitness_action_eval.model_options import FOLLOW_TEMPLATE_MODEL_KEY, resolve_pose_model_path
from fitness_action_eval.pipeline import run_dtw_scoring_from_template

from .models import EvaluationHint, EvaluationPhaseResult, EvaluationTask


_TASK_REGISTRY: dict[int, dict[str, object]] = {}
_TASK_LOCK = threading.Lock()


def generate_task_no() -> str:
    """生成离线评估任务编号。"""

    return f"EVAL{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid4().hex[:6].upper()}"


def task_registry_status(task_id: int) -> dict[str, object]:
    """查询后台线程是否仍在运行。"""

    with _TASK_LOCK:
        entry = _TASK_REGISTRY.get(task_id)
    if not entry:
        return {"active": False}

    thread = entry["thread"]
    assert isinstance(thread, threading.Thread)
    return {"active": thread.is_alive()}


def _result_paths(task: EvaluationTask) -> tuple[Path, Path, Path, Path]:
    """为每个任务固定结果目录，便于后续清理和回显。"""

    result_dir = Path(settings.MEDIA_ROOT) / "evaluation_center" / "results" / task.task_no
    result_dir.mkdir(parents=True, exist_ok=True)
    return (
        result_dir / "result.json",
        result_dir / "plot.png",
        result_dir / "overlay_raw.mp4",
        result_dir / "overlay.mp4",
    )


def _update_progress(task_id: int, percent: int, text: str) -> None:
    """后台线程推进进度时同步更新数据库。"""

    EvaluationTask.objects.filter(id=task_id).update(
        progress_percent=max(0, min(100, int(percent))),
        progress_text=str(text),
        updated_at=datetime.now(),
    )


def _prepare_result_video(out_video_raw: Path, out_video_web: Path) -> Path:
    """尽量把 OpenCV 输出的视频转成浏览器更易播放的 MP4。"""

    if not out_video_raw.exists():
        return out_video_web
    try:
        transcode_to_browser_mp4(str(out_video_raw), str(out_video_web))
        return out_video_web if out_video_web.exists() else out_video_raw
    except Exception:
        cap = cv2.VideoCapture(str(out_video_raw))
        is_readable = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
        cap.release()
        return out_video_raw if is_readable else out_video_web


def register_evaluation_asset(task: EvaluationTask, biz_type: str, file_path: str) -> None:
    """把输入视频、结果图、结果 JSON 等登记到文件资产表。"""

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


def load_result_payload(task: EvaluationTask) -> dict:
    """从结果 JSON 中读取评估摘要。"""

    if not task.result_json_path:
        return {}
    path = Path(task.result_json_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def persist_result_details(task: EvaluationTask) -> None:
    """把 JSON 中的阶段结果和提示结果拆到数据库中，便于详情页查询。"""

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


def _finalize_success(task: EvaluationTask, out_json: Path, out_plot: Path, out_video_web: Path) -> None:
    """任务成功后统一回填数据库和资产表。"""

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    task.result_json_path = str(out_json)
    task.result_plot_path = str(out_plot)
    task.result_video_path = str(out_video_web) if out_video_web.exists() else ""
    task.score = round(float(payload.get("score_0_100", 0.0)), 2)
    task.normalized_distance = round(float(payload.get("normalized_dtw_distance", 0.0)), 4)
    task.hint_count = int(payload.get("hint_count", 0))
    task.progress_percent = 100
    task.progress_text = "评估完成"
    task.status = EvaluationTask.Status.SUCCESS
    task.finished_at = datetime.now()
    task.error_message = ""
    task.save(
        update_fields=[
            "result_json_path",
            "result_plot_path",
            "result_video_path",
            "score",
            "normalized_distance",
            "hint_count",
            "progress_percent",
            "progress_text",
            "status",
            "finished_at",
            "error_message",
            "updated_at",
        ]
    )
    register_evaluation_asset(task, "evaluation_input", task.query_video.path)
    register_evaluation_asset(task, "evaluation_json", str(out_json))
    register_evaluation_asset(task, "evaluation_plot", str(out_plot))
    if out_video_web.exists():
        register_evaluation_asset(task, "evaluation_video", str(out_video_web))
    persist_result_details(task)


def repair_stale_task(task: EvaluationTask) -> EvaluationTask:
    """修复因服务重启或线程中断导致的悬挂任务状态。"""

    if task.status not in {EvaluationTask.Status.PENDING, EvaluationTask.Status.RUNNING}:
        return task
    if task_registry_status(task.id)["active"]:
        return task

    out_json, out_plot, out_video_raw, out_video_web = _result_paths(task)
    result_json = Path(task.result_json_path) if task.result_json_path else out_json
    result_plot = Path(task.result_plot_path) if task.result_plot_path else out_plot
    result_video = Path(task.result_video_path) if task.result_video_path else out_video_web

    if result_json.exists():
        if out_video_raw.exists() and not result_video.exists():
            result_video = _prepare_result_video(out_video_raw, out_video_web)
        task.refresh_from_db()
        _finalize_success(task, result_json, result_plot, result_video)
        return task

    if task.started_at and (datetime.now() - task.started_at).total_seconds() > 300:
        task.status = EvaluationTask.Status.FAILED
        task.progress_text = "任务已中断，请重新提交"
        task.error_message = task.error_message or "任务未完成且没有找到结果文件。"
        task.finished_at = datetime.now()
        task.save(update_fields=["status", "progress_text", "error_message", "finished_at", "updated_at"])
    return task


def _run_evaluation_task_worker(task_id: int) -> None:
    """后台线程：执行离线评估主流程。"""

    close_old_connections()
    try:
        task = EvaluationTask.objects.select_related("template", "user").get(id=task_id)
        out_json, out_plot, out_video_raw, out_video_web = _result_paths(task)

        task.status = EvaluationTask.Status.RUNNING
        task.started_at = datetime.now()
        task.error_message = ""
        task.progress_percent = 5
        task.progress_text = "开始评估"
        task.save(
            update_fields=[
                "status",
                "started_at",
                "error_message",
                "progress_percent",
                "progress_text",
                "updated_at",
            ]
        )

        run_dtw_scoring_from_template(
            template_path=task.template.template_file_path,
            query_video=task.query_video.path,
            out_json=str(out_json),
            out_plot=str(out_plot),
            out_video=str(out_video_raw) if task.export_video else None,
            preview=False,
            score_scale=float(task.score_scale),
            hint_threshold=float(task.hint_threshold),
            hint_min_interval=int(task.hint_min_interval),
            max_hints=int(task.max_hints),
            query_frame_stride=int(task.frame_stride),
            query_smooth_window=int(task.smooth_window),
            query_task_model=None if task.pose_model == FOLLOW_TEMPLATE_MODEL_KEY else resolve_pose_model_path(task.pose_model),
            progress_callback=lambda percent, text: _update_progress(task_id, percent, text),
        )
        if task.export_video:
            _update_progress(task_id, 96, "正在转码结果视频")
            final_video = _prepare_result_video(out_video_raw, out_video_web)
        else:
            _update_progress(task_id, 96, "正在整理评估结果")
            final_video = out_video_web
        task.refresh_from_db()
        _finalize_success(task, out_json, out_plot, final_video)
    except Exception as exc:
        task = EvaluationTask.objects.get(id=task_id)
        task.status = EvaluationTask.Status.FAILED
        task.error_message = str(exc)
        task.progress_text = "评估失败"
        task.finished_at = datetime.now()
        task.save(update_fields=["status", "error_message", "progress_text", "finished_at", "updated_at"])
    finally:
        close_old_connections()
        with _TASK_LOCK:
            _TASK_REGISTRY.pop(task_id, None)


def start_evaluation_task(task: EvaluationTask) -> EvaluationTask:
    """创建后台线程执行离线评估。"""

    thread = threading.Thread(
        target=_run_evaluation_task_worker,
        args=(task.id,),
        name=f"evaluation-task-{task.id}",
        daemon=True,
    )
    with _TASK_LOCK:
        _TASK_REGISTRY[task.id] = {"thread": thread}
    thread.start()
    return task


def _safe_remove_path(path_str: str) -> None:
    """仅允许删除 MEDIA_ROOT 下的文件，避免误删其他目录。"""

    if not path_str:
        return
    path = Path(path_str)
    if not path.exists():
        return
    media_root = Path(settings.MEDIA_ROOT).resolve()
    try:
        resolved = path.resolve()
    except OSError:
        return
    if media_root not in resolved.parents and resolved != media_root:
        return
    if path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def delete_evaluation_task(task: EvaluationTask) -> None:
    """删除已完成的离线评估任务及其关联文件。"""

    if task.status in {EvaluationTask.Status.PENDING, EvaluationTask.Status.RUNNING}:
        raise ValueError("运行中的评估任务请等待完成后再删除。")

    query_video_path = task.query_video.path if task.query_video else ""
    result_dir = Path(settings.MEDIA_ROOT) / "evaluation_center" / "results" / task.task_no

    FileAsset.objects.filter(biz_id=task.id, biz_type__startswith="evaluation_").delete()
    task.delete()

    _safe_remove_path(query_video_path)
    _safe_remove_path(str(result_dir))

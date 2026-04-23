from __future__ import annotations

import json
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
from django.conf import settings
from django.db import close_old_connections

from config.video_utils import transcode_to_browser_mp4
from fitness_action_eval.model_options import FOLLOW_TEMPLATE_MODEL_KEY, resolve_pose_model_path
from fitness_action_eval.pipeline import run_camera_coach

from .models import LiveSession


_SESSION_REGISTRY: dict[int, dict[str, object]] = {}
_REGISTRY_LOCK = threading.Lock()


def generate_session_no() -> str:
    return f"LIVE{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid4().hex[:6].upper()}"


def _session_result_paths(session: LiveSession) -> tuple[str, str]:
    result_dir = Path(settings.MEDIA_ROOT) / "live_session" / "results" / session.session_no
    result_dir.mkdir(parents=True, exist_ok=True)
    return str(result_dir / "summary.json"), str(result_dir / "overlay_raw.mp4")


def _update_session_summary(session: LiveSession, summary: dict, status: str) -> None:
    session.summary_json_path = str(summary.get("summary_json_path", session.summary_json_path))
    session.output_video_path = str(summary.get("output_video", session.output_video_path))
    session.avg_score = round(float(summary.get("avg_score_0_100", 0.0)), 2)
    session.matched_frames = int(summary.get("matched_frames", 0))
    session.final_phase_name = str(summary.get("final_phase_name", ""))
    session.final_phase_cue = str(summary.get("final_phase_cue", ""))
    session.final_part = str(summary.get("final_part", ""))
    session.error_frame_count = int(len(summary.get("error_frames", [])))
    session.status = status
    session.ended_at = datetime.now()
    session.save(
        update_fields=[
            "summary_json_path",
            "output_video_path",
            "avg_score",
            "matched_frames",
            "final_phase_name",
            "final_phase_cue",
            "final_part",
            "error_frame_count",
            "status",
            "ended_at",
            "updated_at",
        ]
    )


def _update_preview_frame(session_id: int, frame) -> None:
    with _REGISTRY_LOCK:
        entry = _SESSION_REGISTRY.get(session_id)
        if entry is None:
            return
        last_frame_at = float(entry.get("last_frame_at", 0.0))
        now = time.perf_counter()
        if now - last_frame_at < 0.12:
            return

    preview_frame = frame
    height, width = preview_frame.shape[:2]
    target_width = 720
    if width > target_width:
        scale = target_width / float(width)
        target_height = max(1, int(round(height * scale)))
        preview_frame = cv2.resize(preview_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(".jpg", preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    if not ok:
        return
    with _REGISTRY_LOCK:
        entry = _SESSION_REGISTRY.get(session_id)
        if entry is None:
            return
        entry["latest_frame"] = encoded.tobytes()
        entry["last_frame_at"] = time.perf_counter()


def _update_runtime_state(session_id: int, state: dict) -> None:
    with _REGISTRY_LOCK:
        entry = _SESSION_REGISTRY.get(session_id)
        if entry is None:
            return
        entry["latest_state"] = dict(state)


def _run_live_session_worker(session_id: int, stop_event: threading.Event, pause_event: threading.Event) -> None:
    close_old_connections()
    try:
        session = LiveSession.objects.select_related("template", "user").get(id=session_id)
        summary_path, video_path = _session_result_paths(session)
        error_frames_dir = str(Path(summary_path).with_name("error_frames")) if session.capture_error_frames else None
        session.status = LiveSession.Status.RUNNING
        session.started_at = datetime.now()
        session.error_message = ""
        session.summary_json_path = summary_path
        session.output_video_path = ""
        session.save(
            update_fields=[
                "status",
                "started_at",
                "error_message",
                "summary_json_path",
                "output_video_path",
                "updated_at",
            ]
        )

        summary = run_camera_coach(
            template_path=session.template.template_file_path,
            ref_video=session.template.source_video.path,
            camera_source=session.camera_source,
            task_model=settings.FITNESS_ACTION_EVAL["MODEL_PATH"],
            num_poses=1,
            smooth_window=session.smooth_window,
            score_scale=float(session.score_scale),
            hint_threshold=float(session.hint_threshold),
            hint_min_interval=int(session.hint_min_interval),
            max_hints=int(session.max_hints),
            ref_search_window=session.ref_search_window,
            frame_stride=session.frame_stride,
            camera_task_model=None if session.pose_model == FOLLOW_TEMPLATE_MODEL_KEY else resolve_pose_model_path(session.pose_model),
            camera_width=session.camera_width,
            camera_height=session.camera_height,
            camera_mirror=session.camera_mirror,
            out_json=summary_path,
            out_video=video_path if session.export_video else None,
            out_error_frames_dir=error_frames_dir,
            preview=session.preview,
            max_frames=session.max_frames,
            stop_checker=stop_event.is_set,
            pause_checker=pause_event.is_set,
            frame_callback=lambda frame: _update_preview_frame(session_id, frame),
            state_callback=lambda state: _update_runtime_state(session_id, state),
        )
        output_video = ""
        raw_output_video = str(summary.get("output_video") or "")
        if session.export_video and raw_output_video and Path(raw_output_video).exists():
            web_video_path = str(Path(raw_output_video).with_name("overlay.mp4"))
            transcode_to_browser_mp4(raw_output_video, web_video_path)
            output_video = web_video_path if Path(web_video_path).exists() else raw_output_video
        summary["summary_json_path"] = summary_path
        summary["output_video"] = output_video
        final_status = LiveSession.Status.STOPPED if stop_event.is_set() else LiveSession.Status.SUCCESS
        session.refresh_from_db()
        _update_session_summary(session, summary, final_status)
    except Exception as exc:
        session = LiveSession.objects.get(id=session_id)
        session.status = LiveSession.Status.FAILED
        session.error_message = str(exc)
        session.ended_at = datetime.now()
        session.save(update_fields=["status", "error_message", "ended_at", "updated_at"])
    finally:
        close_old_connections()
        with _REGISTRY_LOCK:
            _SESSION_REGISTRY.pop(session_id, None)


def start_live_session(session: LiveSession) -> LiveSession:
    stop_event = threading.Event()
    pause_event = threading.Event()
    thread = threading.Thread(
        target=_run_live_session_worker,
        args=(session.id, stop_event, pause_event),
        name=f"live-session-{session.id}",
        daemon=True,
    )
    with _REGISTRY_LOCK:
        _SESSION_REGISTRY[session.id] = {
            "thread": thread,
            "stop_event": stop_event,
            "pause_event": pause_event,
            "latest_frame": b"",
            "last_frame_at": 0.0,
            "latest_state": {},
        }
    thread.start()
    return session


def stop_live_session(session: LiveSession) -> bool:
    with _REGISTRY_LOCK:
        entry = _SESSION_REGISTRY.get(session.id)
    if not entry:
        return False
    stop_event = entry["stop_event"]
    assert isinstance(stop_event, threading.Event)
    stop_event.set()
    return True


def pause_live_session(session: LiveSession) -> bool:
    with _REGISTRY_LOCK:
        entry = _SESSION_REGISTRY.get(session.id)
    if not entry:
        return False
    pause_event = entry["pause_event"]
    assert isinstance(pause_event, threading.Event)
    pause_event.set()
    LiveSession.objects.filter(id=session.id).update(status=LiveSession.Status.PAUSED, updated_at=datetime.now())
    return True


def resume_live_session(session: LiveSession) -> bool:
    with _REGISTRY_LOCK:
        entry = _SESSION_REGISTRY.get(session.id)
    if not entry:
        return False
    pause_event = entry["pause_event"]
    assert isinstance(pause_event, threading.Event)
    pause_event.clear()
    LiveSession.objects.filter(id=session.id).update(status=LiveSession.Status.RUNNING, updated_at=datetime.now())
    return True


def session_registry_status(session_id: int) -> dict[str, object]:
    with _REGISTRY_LOCK:
        entry = _SESSION_REGISTRY.get(session_id)
    if not entry:
        return {"active": False, "stop_requested": False, "pause_requested": False}
    thread = entry["thread"]
    stop_event = entry["stop_event"]
    pause_event = entry["pause_event"]
    assert isinstance(thread, threading.Thread)
    assert isinstance(stop_event, threading.Event)
    assert isinstance(pause_event, threading.Event)
    return {
        "active": thread.is_alive(),
        "stop_requested": stop_event.is_set(),
        "pause_requested": pause_event.is_set(),
    }


def repair_stale_live_session(session: LiveSession) -> LiveSession:
    if session.status not in {LiveSession.Status.PENDING, LiveSession.Status.RUNNING, LiveSession.Status.PAUSED}:
        return session

    runtime = session_registry_status(session.id)
    if runtime.get("active"):
        return session

    summary_path = Path(session.summary_json_path) if session.summary_json_path else None
    if summary_path and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        session.refresh_from_db()
        _update_session_summary(session, summary, LiveSession.Status.SUCCESS)
        return session

    if session.started_at and (datetime.now() - session.started_at).total_seconds() > 60:
        session.status = LiveSession.Status.FAILED
        session.error_message = session.error_message or "实时会话已中断，请重新启动。"
        session.ended_at = datetime.now()
        session.save(update_fields=["status", "error_message", "ended_at", "updated_at"])
    return session


def get_live_session_preview_frame(session_id: int) -> bytes:
    with _REGISTRY_LOCK:
        entry = _SESSION_REGISTRY.get(session_id)
        if not entry:
            return b""
        frame = entry.get("latest_frame", b"")
    return frame if isinstance(frame, bytes) else b""


def get_live_session_runtime_payload(session_id: int) -> dict:
    with _REGISTRY_LOCK:
        entry = _SESSION_REGISTRY.get(session_id)
        if not entry:
            return {}
        state = entry.get("latest_state", {})
    return dict(state) if isinstance(state, dict) else {}


def _safe_remove_path(path_str: str) -> None:
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


def delete_live_session(session: LiveSession) -> None:
    if session.status in {LiveSession.Status.PENDING, LiveSession.Status.RUNNING, LiveSession.Status.PAUSED}:
        raise ValueError("运行中的实时会话请先停止后再删除。")

    result_dir = Path(settings.MEDIA_ROOT) / "live_session" / "results" / session.session_no
    session.delete()
    _safe_remove_path(str(result_dir))

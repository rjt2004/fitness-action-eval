from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from django.conf import settings
from django.db import close_old_connections

from config.video_utils import transcode_to_browser_mp4
from fitness_action_eval.pipeline import run_camera_coach

from apps.accounts.models import User

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
            "status",
            "ended_at",
            "updated_at",
        ]
    )


def _run_live_session_worker(session_id: int, stop_event: threading.Event) -> None:
    close_old_connections()
    try:
        session = LiveSession.objects.select_related("template", "user").get(id=session_id)
        summary_path, video_path = _session_result_paths(session)
        session.status = LiveSession.Status.RUNNING
        session.started_at = datetime.now()
        session.error_message = ""
        session.summary_json_path = summary_path
        session.output_video_path = video_path
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
            ref_search_window=session.ref_search_window,
            frame_stride=session.frame_stride,
            camera_width=session.camera_width,
            camera_height=session.camera_height,
            camera_mirror=session.camera_mirror,
            out_json=summary_path,
            out_video=video_path,
            preview=session.preview,
            max_frames=session.max_frames,
            stop_checker=stop_event.is_set,
        )
        web_video_path = str(Path(video_path).with_name("overlay.mp4"))
        transcode_to_browser_mp4(video_path, web_video_path)
        summary["summary_json_path"] = summary_path
        summary["output_video"] = web_video_path
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
    thread = threading.Thread(
        target=_run_live_session_worker,
        args=(session.id, stop_event),
        name=f"live-session-{session.id}",
        daemon=True,
    )
    with _REGISTRY_LOCK:
        _SESSION_REGISTRY[session.id] = {"thread": thread, "stop_event": stop_event}
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


def session_registry_status(session_id: int) -> dict[str, object]:
    with _REGISTRY_LOCK:
        entry = _SESSION_REGISTRY.get(session_id)
    if not entry:
        return {"active": False, "stop_requested": False}
    thread = entry["thread"]
    stop_event = entry["stop_event"]
    assert isinstance(thread, threading.Thread)
    assert isinstance(stop_event, threading.Event)
    return {"active": thread.is_alive(), "stop_requested": stop_event.is_set()}

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fitness_action_eval.baduanjin import (
    BADUANJIN_MANUAL_BOUNDARIES_S,
    BADUANJIN_STANDARD_DURATION_S,
    get_phase_definition,
    get_substage_by_key,
)
from fitness_action_eval.pipeline import _ensure_baduanjin_features, load_pose_template
from fitness_action_eval.pose import extract_pose_sequence
from fitness_action_eval.visualization import draw_pose_skeleton, draw_text_block, draw_utf8_text


def _format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minute = int(seconds // 60)
    second = int(seconds % 60)
    return f"{minute:02d}:{second:02d}"


def _load_reference_data(video_path: Path, template_path: Path | None, model_path: Path, frame_stride: int, smooth_window: int) -> dict:
    if template_path and template_path.exists():
        data = load_pose_template(str(template_path))
        data["reference_video"] = str(video_path)
        return data
    data = extract_pose_sequence(
        video_path=str(video_path),
        task_model=str(model_path),
        num_poses=1,
        smooth_window=max(1, smooth_window),
        frame_stride=max(1, frame_stride),
        preview=False,
    )
    data["reference_video"] = str(video_path)
    data["task_model"] = str(model_path)
    data["num_poses"] = 1
    data["smooth_window"] = max(1, smooth_window)
    return _ensure_baduanjin_features(data)


def _resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0 or frame.shape[1] <= target_width:
        return frame
    scale = target_width / float(frame.shape[1])
    target_height = int(round(frame.shape[0] * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _stage_for_time(ref_data: dict, time_s: float) -> dict[str, object]:
    ref_times = np.asarray(ref_data["time_s"], dtype=np.float32)
    video_duration = float(ref_times[-1]) if ref_times.size else float(BADUANJIN_STANDARD_DURATION_S)
    scale = video_duration / float(BADUANJIN_STANDARD_DURATION_S) if video_duration > 0 else 1.0
    active_start = float(BADUANJIN_MANUAL_BOUNDARIES_S[0][0]) * scale
    if float(time_s) < active_start:
        return {
            "phase_id": -1,
            "phase_name": "片头准备",
            "substage_name": "未进入预备势",
            "cue": "从 00:20 开始进入预备势，本段不参与八段锦动作划分。",
            "phase_start_s": 0.0,
            "phase_end_s": active_start,
            "substage_start_s": 0.0,
            "substage_end_s": active_start,
            "source": "manual_start",
            "progress": float(np.clip(float(time_s) / max(1e-6, active_start), 0.0, 1.0)),
            "seq_idx": 0,
        }
    idx = int(np.searchsorted(ref_times, float(time_s), side="left"))
    idx = int(np.clip(idx, 0, len(ref_times) - 1))
    phase_id = int(ref_data["phase_ids"][idx])
    phase = get_phase_definition(phase_id)
    substage_key = str(ref_data["substage_keys"][idx])
    substage = get_substage_by_key(phase_id, substage_key)
    phase_rows = [row for row in ref_data.get("phase_rows", []) if int(row.get("phase_id", -1)) == phase_id]
    substage_rows = [
        row
        for row in ref_data.get("substage_rows", [])
        if int(row.get("phase_id", -1)) == phase_id
        and str(row.get("substage_key", "")) == substage_key
        and float(row.get("start_time_s", -1)) <= float(time_s) <= float(row.get("end_time_s", 1e9))
    ]
    phase_row = phase_rows[0] if phase_rows else {}
    substage_row = substage_rows[0] if substage_rows else {}
    phase_start = float(phase_row.get("start_time_s", ref_times[idx]))
    phase_end = float(phase_row.get("end_time_s", ref_times[idx]))
    progress = (float(time_s) - phase_start) / max(1e-6, phase_end - phase_start)
    return {
        "phase_id": phase_id,
        "phase_name": phase.display_name,
        "substage_name": substage.name if substage else substage_key,
        "cue": substage.cue if substage else phase.cue,
        "phase_start_s": phase_start,
        "phase_end_s": phase_end,
        "substage_start_s": float(substage_row.get("start_time_s", ref_times[idx])),
        "substage_end_s": float(substage_row.get("end_time_s", ref_times[idx])),
        "source": substage_row.get("source", "state_machine"),
        "progress": float(np.clip(progress, 0.0, 1.0)),
        "seq_idx": idx,
    }


def _overlay_lines(stage: dict[str, object], time_s: float) -> list[str]:
    phase_id = int(stage["phase_id"])
    phase_title = "片头准备" if phase_id < 0 else f"{phase_id + 1}. {stage['phase_name']}"
    return [
        f"当前式：{phase_title}",
        f"状态机小阶段：{stage['substage_name']}",
        f"当前时间：{_format_time(time_s)} / 本式：{_format_time(stage['phase_start_s'])}-{_format_time(stage['phase_end_s'])}",
        f"小阶段时间：{_format_time(stage['substage_start_s'])}-{_format_time(stage['substage_end_s'])}",
        f"动作要点：{stage['cue']}",
    ]


def render_state_preview(
    video_path: Path,
    output_video: Path,
    output_json: Path,
    template_path: Path | None,
    model_path: Path,
    width: int,
    render_stride: int,
    frame_stride: int,
    smooth_window: int,
) -> Path:
    ref_data = _load_reference_data(video_path, template_path, model_path, frame_stride, smooth_window)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(ref_data.get("substage_rows", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width > 0 and source_width > width:
        scale = width / float(source_width)
        output_size = (width, int(round(source_height * scale)))
    else:
        output_size = (source_width, source_height)

    render_stride = max(1, int(render_stride))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / render_stride,
        output_size,
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {output_video}")

    raw_points = np.asarray(ref_data.get("raw_points", []), dtype=np.float32)
    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % render_stride != 0:
            frame_idx += 1
            continue
        frame = _resize_frame(frame, width)
        time_s = frame_idx / fps
        stage = _stage_for_time(ref_data, time_s)
        seq_idx = int(stage.get("seq_idx", -1))
        if 0 <= seq_idx < raw_points.shape[0]:
            draw_pose_skeleton(frame, raw_points[seq_idx])
        draw_text_block(frame, _overlay_lines(stage, time_s), x=24, y=24, bg_color=(12, 30, 34), fg_color=(246, 250, 238))

        bar_x = 24
        bar_y = frame.shape[0] - 54
        bar_w = min(frame.shape[1] - 48, 760)
        bar_h = 18
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 56, 60), -1)
        fill_w = int(bar_w * float(stage["progress"]))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (60, 190, 160), -1)
        draw_utf8_text(frame, "本式内部进度", (bar_x, bar_y - 30), 20, (246, 250, 238), shadow_color=(0, 0, 0))

        writer.write(frame)
        written += 1
        if written % max(1, int(round((fps / render_stride) * 10))) == 0:
            print(f"rendered {_format_time(time_s)}")
        frame_idx += 1

    writer.release()
    cap.release()
    return output_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Baduanjin state-machine substage preview.")
    parser.add_argument("--input", default="media/template_center/source/baduan.mp4")
    parser.add_argument("--template", default="media/template_center/generated/template_9_v1.npz")
    parser.add_argument("--model", default="pose_landmarker_full.task")
    parser.add_argument("--output", default="output/baduanjin_state_machine_stage_preview.mp4")
    parser.add_argument("--json", default="output/baduanjin_state_machine_substages.json")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()

    output = render_state_preview(
        video_path=Path(args.input),
        output_video=Path(args.output),
        output_json=Path(args.json),
        template_path=Path(args.template) if args.template else None,
        model_path=Path(args.model),
        width=args.width,
        render_stride=args.render_stride,
        frame_stride=args.frame_stride,
        smooth_window=args.smooth_window,
    )
    print(f"saved: {output.resolve()}")


if __name__ == "__main__":
    main()

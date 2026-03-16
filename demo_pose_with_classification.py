import argparse
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision

from mmfit_classifier_utils import fill_missing, normalize_coords, window_features


# BlazePose 33 keypoints topology.
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]

COUNT_RULES = {
    "squats": {"low": 95.0, "high": 155.0, "low_phase": "down", "high_phase": "up"},
    "pushups": {"low": 90.0, "high": 155.0, "low_phase": "down", "high_phase": "up"},
    "lunges": {"low": 95.0, "high": 160.0, "low_phase": "down", "high_phase": "up"},
    # Tuned for current situp capture where torso-hip-knee angle mostly falls in ~30..130.
    "situps": {"low": 72.0, "high": 118.0, "low_phase": "up", "high_phase": "down"},
}


@dataclass
class RepCounterFSM:
    counts: dict = field(default_factory=dict)
    states: dict = field(default_factory=dict)
    rep_start: dict = field(default_factory=dict)
    segments: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for action in COUNT_RULES:
            self.counts[action] = 0
            self.states[action] = "unknown"
            self.rep_start[action] = None
            self.segments[action] = []

    def update(self, action: str, metric: float, frame_idx: int) -> None:
        if action not in COUNT_RULES or not np.isfinite(metric):
            return
        low = COUNT_RULES[action]["low"]
        high = COUNT_RULES[action]["high"]
        state = self.states[action]

        if state == "unknown":
            if metric >= high:
                self.states[action] = "high"
            elif metric <= low:
                self.states[action] = "low"
            return

        if state == "high" and metric <= low:
            self.states[action] = "low"
            self.rep_start[action] = frame_idx
            return

        if state == "low" and metric >= high:
            self.states[action] = "high"
            self.counts[action] += 1
            start_idx = self.rep_start[action] if self.rep_start[action] is not None else frame_idx
            self.segments[action].append([int(start_idx), int(frame_idx)])
            self.rep_start[action] = None

    def current_phase(self, action: str) -> str:
        if action not in COUNT_RULES:
            return "n/a"
        state = self.states[action]
        if state == "low":
            return COUNT_RULES[action]["low_phase"]
        if state == "high":
            return COUNT_RULES[action]["high_phase"]
        return "ready"


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_video_writer(
    output_path: str, width: int, height: int, fps: float
) -> cv2.VideoWriter:
    ensure_parent_dir(output_path)
    ext = os.path.splitext(output_path.lower())[1]
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened() and ext == ".mkv":
        fallback_fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fallback_fourcc, fps, (width, height))
    return writer


def draw_pose_landmarker(frame_bgr, pose_landmarks) -> None:
    if not pose_landmarks:
        return
    h, w = frame_bgr.shape[:2]
    for landmarks in pose_landmarks:
        points = []
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)
        for a, b in POSE_CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(frame_bgr, points[a], points[b], (0, 200, 255), 2)


def pose_bbox(landmarks, width: int, height: int) -> Tuple[int, int, int, int, float, float, float]:
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    cx = float((x1 + x2) / 2.0)
    cy = float((y1 + y2) / 2.0)
    area = float(max(1, (x2 - x1) * (y2 - y1)))
    return x1, y1, x2, y2, cx, cy, area


def select_target_pose(
    pose_landmarks, width: int, height: int, prev_center: Optional[Tuple[float, float]]
):
    if not pose_landmarks:
        return None, prev_center, None

    frame_cx = width / 2.0
    frame_cy = height / 2.0
    diag = (width**2 + height**2) ** 0.5 + 1e-6
    best_idx = -1
    best_score = -1e9
    best_info = None

    for i, landmarks in enumerate(pose_landmarks):
        x1, y1, x2, y2, cx, cy, area = pose_bbox(landmarks, width, height)
        center_dist = ((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5 / diag
        if prev_center is None:
            track_dist = 0.0
        else:
            track_dist = ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5 / diag
        area_norm = area / float(width * height + 1e-6)
        score = (2.0 * area_norm) - (0.7 * center_dist) - (0.9 * track_dist)
        if score > best_score:
            best_score = score
            best_idx = i
            best_info = (x1, y1, x2, y2, cx, cy)

    if best_idx < 0:
        return None, prev_center, None
    selected = pose_landmarks[best_idx]
    next_center = (best_info[4], best_info[5])
    return selected, next_center, best_info


def mp33_to_openpose18_xy(landmarks33: List, width: int, height: int) -> np.ndarray:
    # Convert MediaPipe 33 joints to an OpenPose-like 18-joint layout.
    pts = np.full((18, 2), np.nan, dtype=np.float32)

    def xy(idx: int) -> np.ndarray:
        lm = landmarks33[idx]
        return np.array([lm.x * width, lm.y * height], dtype=np.float32)

    # 0 nose
    pts[0] = xy(0)
    # 1 neck (derived)
    pts[1] = (xy(11) + xy(12)) / 2.0
    # right arm
    pts[2] = xy(12)
    pts[3] = xy(14)
    pts[4] = xy(16)
    # left arm
    pts[5] = xy(11)
    pts[6] = xy(13)
    pts[7] = xy(15)
    # right leg
    pts[8] = xy(24)
    pts[9] = xy(26)
    pts[10] = xy(28)
    # left leg
    pts[11] = xy(23)
    pts[12] = xy(25)
    pts[13] = xy(27)
    # eyes / ears
    pts[14] = xy(5)   # right eye
    pts[15] = xy(2)   # left eye
    pts[16] = xy(8)   # right ear
    pts[17] = xy(7)   # left ear
    return pts


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return float("nan")
    cos_v = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_v)))


def lm_xy(landmarks33: List, idx: int) -> np.ndarray:
    lm = landmarks33[idx]
    return np.array([lm.x, lm.y], dtype=np.float32)


def action_metric(action: str, landmarks33: List) -> float:
    if action == "squats":
        left = angle_deg(lm_xy(landmarks33, 23), lm_xy(landmarks33, 25), lm_xy(landmarks33, 27))
        right = angle_deg(lm_xy(landmarks33, 24), lm_xy(landmarks33, 26), lm_xy(landmarks33, 28))
        return float(np.nanmean([left, right]))
    if action == "pushups":
        left = angle_deg(lm_xy(landmarks33, 11), lm_xy(landmarks33, 13), lm_xy(landmarks33, 15))
        right = angle_deg(lm_xy(landmarks33, 12), lm_xy(landmarks33, 14), lm_xy(landmarks33, 16))
        return float(np.nanmean([left, right]))
    if action == "lunges":
        left = angle_deg(lm_xy(landmarks33, 23), lm_xy(landmarks33, 25), lm_xy(landmarks33, 27))
        right = angle_deg(lm_xy(landmarks33, 24), lm_xy(landmarks33, 26), lm_xy(landmarks33, 28))
        return float(np.nanmin([left, right]))
    if action == "situps":
        left = angle_deg(lm_xy(landmarks33, 11), lm_xy(landmarks33, 23), lm_xy(landmarks33, 25))
        right = angle_deg(lm_xy(landmarks33, 12), lm_xy(landmarks33, 24), lm_xy(landmarks33, 26))
        return float(np.nanmean([left, right]))
    return float("nan")


def classify_from_window(
    clf,
    window_coords: np.ndarray,
    class_names: List[str],
) -> Tuple[str, np.ndarray]:
    coords = fill_missing(window_coords)
    coords = normalize_coords(coords)
    feats = window_features(coords, window_size=coords.shape[0], stride=coords.shape[0])
    pred_id = int(clf.predict(feats)[0])
    prob = clf.predict_proba(feats)[0]
    return class_names[pred_id], prob


def draw_prediction_overlay(
    frame_bgr: np.ndarray,
    pred_label: str,
    prob: Optional[np.ndarray],
    class_names: List[str],
    target_box=None,
    fps_value: Optional[float] = None,
    rep_count: Optional[int] = None,
    rep_phase: Optional[str] = None,
) -> None:
    def draw_text(
        text: str,
        org: Tuple[int, int],
        color: Tuple[int, int, int],
        scale: float = 0.65,
        thickness: int = 2,
    ) -> None:
        # Dark outline keeps text readable over bright backgrounds.
        cv2.putText(
            frame_bgr,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    if target_box is not None:
        x1, y1, x2, y2, _, _ = target_box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 120, 255), 2)
        draw_text("TARGET", (x1, max(24, y1 - 10)), (0, 180, 255), scale=0.7, thickness=2)

    # Build lines first so panel can auto-size and avoid overlaps.
    lines: List[Tuple[str, Tuple[int, int, int], float, int]] = []
    if pred_label:
        lines.append((f"Pred: {pred_label}", (40, 255, 40), 0.9, 2))
    if prob is not None:
        top = np.argsort(-prob)[:3]
        for i in top:
            txt = f"{class_names[int(i)]}: {prob[int(i)]:.2f}"
            lines.append((txt, (240, 240, 240), 0.66, 2))
    if fps_value is not None:
        lines.append((f"FPS: {fps_value:.1f}", (0, 220, 255), 0.72, 2))
    if rep_count is not None:
        lines.append((f"Count: {rep_count}", (80, 255, 255), 0.72, 2))
    if rep_phase:
        lines.append((f"Phase: {rep_phase}", (255, 210, 80), 0.72, 2))

    if lines:
        x, y = 16, 16
        line_h = 28
        panel_w = 360
        panel_h = 16 + len(lines) * line_h + 8
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)
        text_y = y + 26
        for text, color, scale, thickness in lines:
            draw_text(text, (x + 12, text_y), color, scale=scale, thickness=thickness)
            text_y += line_h


def run_video_mode(
    input_path: str,
    output_path: Optional[str],
    display: bool,
    task_model: str,
    clf,
    class_names: List[str],
    window_size: int,
    classify_every: int,
    num_poses: int,
    count_conf: float,
    count_log: Optional[str],
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Cannot read first frame from video: {input_path}")

    height, width = first_frame.shape[:2]
    writer = None
    if output_path:
        writer = get_video_writer(output_path, width, height, fps)
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video writer for: {output_path}")

    if not os.path.exists(task_model):
        cap.release()
        raise FileNotFoundError(f"Task model not found: {task_model}")

    base_options = mp_tasks_python.BaseOptions(model_asset_path=task_model)
    options = mp_tasks_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_tasks_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=num_poses,
    )

    history = deque(maxlen=window_size)
    pred_label = ""
    pred_prob: Optional[np.ndarray] = None
    prev_center = None
    prev_tick = time.perf_counter()
    fps_ema = 0.0
    rep_counter = RepCounterFSM()

    with mp_tasks_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame = first_frame
        frame_idx = 0
        while True:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((frame_idx * 1000.0) / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            target_landmarks, prev_center, target_box = select_target_pose(
                result.pose_landmarks, width=width, height=height, prev_center=prev_center
            )
            if target_landmarks is not None:
                draw_pose_landmarker(frame, [target_landmarks])
                pose18 = mp33_to_openpose18_xy(target_landmarks, width=width, height=height)
                history.append(pose18)
            else:
                target_box = None

            if len(history) >= window_size and frame_idx % classify_every == 0:
                win = np.asarray(history, dtype=np.float32)
                pred_label, pred_prob = classify_from_window(clf, win, class_names)

            rep_count = None
            rep_phase = None
            if target_landmarks is not None and pred_label in COUNT_RULES:
                conf_ok = True
                if pred_prob is not None:
                    conf_ok = float(np.max(pred_prob)) >= count_conf
                if conf_ok:
                    metric = action_metric(pred_label, target_landmarks)
                    rep_counter.update(pred_label, metric, frame_idx)
                rep_count = rep_counter.counts.get(pred_label, 0)
                rep_phase = rep_counter.current_phase(pred_label)

            now_tick = time.perf_counter()
            dt = max(1e-6, now_tick - prev_tick)
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst_fps)
            prev_tick = now_tick
            draw_prediction_overlay(
                frame,
                pred_label,
                pred_prob,
                class_names,
                target_box=target_box,
                fps_value=fps_ema,
                rep_count=rep_count,
                rep_phase=rep_phase,
            )

            if writer is not None:
                writer.write(frame)

            if display:
                cv2.imshow("Pose + Action Classification Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            ok, frame = cap.read()
            if not ok:
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[OK] Saved output video to: {output_path}")
    if display:
        cv2.destroyAllWindows()

    print("[COUNT] Repetition summary:", rep_counter.counts)
    if count_log:
        ensure_parent_dir(count_log)
        with open(count_log, "w", encoding="utf-8") as f:
            json.dump(
                {"counts": rep_counter.counts, "segments": rep_counter.segments},
                f,
                ensure_ascii=True,
                indent=2,
            )
        print(f"[OK] Saved counting log to: {count_log}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pose estimation + 4-action classification demo.")
    p.add_argument("--input", required=True, help="Input video path.")
    p.add_argument("--output", default=None, help="Output video path.")
    p.add_argument("--no_display", action="store_true", help="Disable live preview window.")
    p.add_argument("--task_model", default="pose_landmarker.task", help="Path to pose_landmarker.task.")
    p.add_argument(
        "--model_path",
        default="artifacts/mmfit_action_cls/rf_mmfit_4actions.joblib",
        help="Path to trained classifier model.",
    )
    p.add_argument(
        "--meta_path",
        default="artifacts/mmfit_action_cls/meta.json",
        help="Path to classifier metadata.",
    )
    p.add_argument(
        "--classify_every",
        type=int,
        default=8,
        help="Run classifier every N frames after warm-up window.",
    )
    p.add_argument(
        "--num_poses",
        type=int,
        default=4,
        help="Maximum number of persons to detect before target selection.",
    )
    p.add_argument(
        "--count_conf",
        type=float,
        default=0.55,
        help="Minimum classification confidence required to update repetition counter.",
    )
    p.add_argument(
        "--count_log",
        default=None,
        help="Optional output path for counting summary JSON.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Classifier model not found: {args.model_path}")
    if not os.path.exists(args.meta_path):
        raise FileNotFoundError(f"Classifier meta not found: {args.meta_path}")

    clf = joblib.load(args.model_path)
    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    class_names = meta["actions"]
    window_size = int(meta["window_size"])

    run_video_mode(
        input_path=args.input,
        output_path=args.output,
        display=not args.no_display,
        task_model=args.task_model,
        clf=clf,
        class_names=class_names,
        window_size=window_size,
        classify_every=max(1, args.classify_every),
        num_poses=max(1, args.num_poses),
        count_conf=max(0.0, min(1.0, args.count_conf)),
        count_log=args.count_log,
    )


if __name__ == "__main__":
    main()

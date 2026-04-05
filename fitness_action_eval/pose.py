from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision

from fitness_action_eval.visualization import close_preview_windows, draw_pose_skeleton, draw_text_block, preview_frame


def moving_average_matrix(x: np.ndarray, k: int) -> np.ndarray:
    # 对时间序列特征做滑动平均，减小关键点抖动带来的噪声。
    if x.ndim != 2:
        raise ValueError("moving_average_matrix expects shape (T, D).")
    if k <= 1 or x.shape[0] < k:
        return x
    if k % 2 == 0:
        k += 1
    pad = k // 2
    kernel = np.ones((k,), dtype=np.float32) / float(k)
    out = np.empty_like(x, dtype=np.float32)
    for d in range(x.shape[1]):
        xp = np.pad(x[:, d], (pad, pad), mode="edge")
        out[:, d] = np.convolve(xp, kernel, mode="valid")
    return out


def normalize_matrix(x: np.ndarray) -> np.ndarray:
    # 对每一维特征做标准化，使不同关键点维度具有可比性。
    if x.ndim != 2:
        raise ValueError("normalize_matrix expects shape (T, D).")
    mu = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-6
    return (x - mu) / std


def pose_bbox(
    landmarks, width: int, height: int
) -> Tuple[int, int, int, int, float, float, float]:
    # 根据当前姿态关键点计算包围框、中心点和面积，用于多人场景下的人体目标筛选。
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    cx = float((x1 + x2) / 2.0)
    cy = float((y1 + y2) / 2.0)
    area = float(max(1.0, (x2 - x1) * (y2 - y1)))
    return x1, y1, x2, y2, cx, cy, area


def select_target_pose(
    pose_landmarks,
    width: int,
    height: int,
    prev_center: Optional[Tuple[float, float]],
):
    # 在检测到多人时，优先选择面积较大、靠近画面中心且与上一帧位置连续的人体。
    if not pose_landmarks:
        return None, prev_center

    frame_cx = width / 2.0
    frame_cy = height / 2.0
    diag = (width**2 + height**2) ** 0.5 + 1e-6
    best_idx = -1
    best_score = -1e9
    best_center = prev_center

    for i, landmarks in enumerate(pose_landmarks):
        _, _, _, _, cx, cy, area = pose_bbox(landmarks, width, height)
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
            best_center = (cx, cy)

    if best_idx < 0:
        return None, prev_center
    return pose_landmarks[best_idx], best_center


def normalize_pose_points(points: np.ndarray) -> Optional[np.ndarray]:
    # 以髋部中心为原点、躯干长度为尺度做归一化，降低人物位置与身高差异的影响。
    if points.shape != (33, 2):
        return None
    hip_center = (points[23] + points[24]) / 2.0
    shoulder_center = (points[11] + points[12]) / 2.0
    scale = float(np.linalg.norm(shoulder_center - hip_center))
    if scale < 1e-6:
        return None
    norm = (points - hip_center[None, :]) / scale
    return norm.astype(np.float32)


def extract_pose_sequence(
    video_path: str,
    task_model: str,
    num_poses: int,
    smooth_window: int,
    preview: bool = False,
    preview_title: str = "Pose Preview",
) -> Dict[str, Any]:
    # 从视频中提取逐帧姿态序列，并生成后续 DTW 所需的标准化特征。
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Cannot read first frame from: {video_path}")

    height, width = first.shape[:2]
    options = mp_tasks_vision.PoseLandmarkerOptions(
        base_options=mp_tasks_python.BaseOptions(model_asset_path=task_model),
        running_mode=mp_tasks_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=max(1, num_poses),
    )

    points_seq: List[np.ndarray] = []
    raw_points_seq: List[np.ndarray] = []
    frame_indices: List[int] = []
    time_s: List[float] = []
    prev_center = None
    frame_idx = 0

    with mp_tasks_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame = first
        while True:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx * 1000.0) / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            target, prev_center = select_target_pose(
                result.pose_landmarks, width=width, height=height, prev_center=prev_center
            )

            preview_frame_img = frame.copy()
            if target is not None and len(target) >= 33:
                # 同时保留原始坐标与归一化坐标，分别用于渲染和评分。
                pts = np.array([[lm.x, lm.y] for lm in target[:33]], dtype=np.float32)
                if np.all(np.isfinite(pts)):
                    norm = normalize_pose_points(pts)
                    if norm is not None and np.all(np.isfinite(norm)):
                        points_seq.append(norm)
                        raw_points_seq.append(pts.copy())
                        frame_indices.append(frame_idx)
                        time_s.append(float(frame_idx / fps))
                        if preview:
                            draw_pose_skeleton(preview_frame_img, pts)
            if preview:
                lines = [
                    f"Frame: {frame_idx}",
                    f"Valid Poses: {len(points_seq)}",
                    f"Video: {video_path}",
                ]
                draw_text_block(preview_frame_img, lines, x=20, y=18)
                should_continue = preview_frame(preview_title, preview_frame_img)
                if not should_continue:
                    preview = False
                    close_preview_windows()

            frame_idx += 1
            ok, frame = cap.read()
            if not ok:
                break

    cap.release()
    if preview:
        close_preview_windows()

    if len(points_seq) < 10:
        raise RuntimeError(f"Too few valid pose points ({len(points_seq)}) from {video_path}.")

    points = np.asarray(points_seq, dtype=np.float32)
    flat = points.reshape(points.shape[0], -1)
    flat_smooth = moving_average_matrix(flat, max(1, smooth_window))
    points_smooth = flat_smooth.reshape((-1, 33, 2))
    features = normalize_matrix(flat_smooth)

    return {
        "features": features,
        "points": points_smooth,
        "raw_points": np.asarray(raw_points_seq, dtype=np.float32),
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "time_s": np.asarray(time_s, dtype=np.float32),
        "fps": float(fps),
    }

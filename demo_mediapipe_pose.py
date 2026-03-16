import argparse
import os
import time
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision


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
        # Fallback for some OpenCV/FFmpeg builds that do not support mp4v in mkv.
        fallback_fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fallback_fourcc, fps, (width, height))
    return writer


def infer_output_size(frame) -> Tuple[int, int]:
    height, width = frame.shape[:2]
    return width, height


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

        # Higher score => better target.
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


def draw_pose_landmarker(
    frame_bgr,
    pose_landmarks,
    target_landmarks=None,
    target_box=None,
    show_bbox: bool = False,
) -> None:
    if not pose_landmarks:
        return

    h, w = frame_bgr.shape[:2]
    for landmarks in pose_landmarks:
        points = []
        is_target = target_landmarks is not None and landmarks is target_landmarks
        pt_color = (0, 255, 0) if not is_target else (0, 255, 255)
        line_color = (0, 200, 255) if not is_target else (0, 120, 255)
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame_bgr, (x, y), 3, pt_color, -1)

        for a, b in POSE_CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(frame_bgr, points[a], points[b], line_color, 2)

    if show_bbox and target_box is not None:
        x1, y1, x2, y2, _, _ = target_box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 120, 255), 2)
        cv2.putText(
            frame_bgr,
            "TARGET",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 120, 255),
            2,
            cv2.LINE_AA,
        )


def run_image_mode(
    input_path: str,
    output_path: str,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    task_model: str,
    num_poses: int,
) -> None:
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    if not os.path.exists(task_model):
        raise FileNotFoundError(f"Task model not found: {task_model}")

    base_options = mp_tasks_python.BaseOptions(model_asset_path=task_model)
    options = mp_tasks_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_tasks_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_tracking_confidence,
        min_tracking_confidence=min_tracking_confidence,
        num_poses=num_poses,
    )
    with mp_tasks_vision.PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(input_path)
        result = landmarker.detect(mp_image)
    h, w = image.shape[:2]
    target_landmarks, _, target_box = select_target_pose(
        result.pose_landmarks, width=w, height=h, prev_center=None
    )
    if target_landmarks is not None:
        draw_pose_landmarker(
            image,
            [target_landmarks],
            target_landmarks=target_landmarks,
            target_box=target_box,
            show_bbox=True,
        )
    else:
        draw_pose_landmarker(image, result.pose_landmarks)

    ensure_parent_dir(output_path)
    cv2.imwrite(output_path, image)
    print(f"[OK] Saved pose visualization image to: {output_path}")


def run_video_mode(
    input_path: str,
    output_path: Optional[str],
    min_detection_confidence: float,
    min_tracking_confidence: float,
    display: bool,
    task_model: str,
    num_poses: int,
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

    width, height = infer_output_size(first_frame)
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
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_tracking_confidence,
        min_tracking_confidence=min_tracking_confidence,
        num_poses=num_poses,
    )
    prev_center = None
    prev_tick = time.perf_counter()
    fps_ema = 0.0
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
                draw_pose_landmarker(
                    frame,
                    [target_landmarks],
                    target_landmarks=target_landmarks,
                    target_box=target_box,
                    show_bbox=True,
                )
            else:
                draw_pose_landmarker(frame, result.pose_landmarks)

            now_tick = time.perf_counter()
            dt = max(1e-6, now_tick - prev_tick)
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst_fps)
            prev_tick = now_tick
            cv2.putText(
                frame,
                f"FPS: {fps_ema:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)

            if display:
                cv2.imshow("MediaPipe Pose Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            ok, frame = cap.read()
            if not ok:
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[OK] Saved pose visualization video to: {output_path}")

    if display:
        cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MediaPipe pose estimation demo for image/video visualization."
    )
    parser.add_argument(
        "--mode",
        choices=["image", "video"],
        required=True,
        help="Run mode: image or video.",
    )
    parser.add_argument("--input", required=True, help="Input image/video path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Required for image mode; optional for video mode.",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="Minimum pose detection confidence threshold.",
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
        help="Minimum pose tracking confidence threshold.",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Disable real-time preview window in video mode.",
    )
    parser.add_argument(
        "--task_model",
        default="pose_landmarker.task",
        help="Path to pose_landmarker.task.",
    )
    parser.add_argument(
        "--num_poses",
        type=int,
        default=4,
        help="Maximum number of persons to detect.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.mode == "image":
        if not args.output:
            raise ValueError("--output is required in image mode.")
        run_image_mode(
            input_path=args.input,
            output_path=args.output,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            task_model=args.task_model,
            num_poses=max(1, args.num_poses),
        )
        return

    run_video_mode(
        input_path=args.input,
        output_path=args.output,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        display=not args.no_display,
        task_model=args.task_model,
        num_poses=max(1, args.num_poses),
    )


if __name__ == "__main__":
    main()

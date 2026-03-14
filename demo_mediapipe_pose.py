import argparse
import os
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


def run_image_mode(
    input_path: str,
    output_path: str,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    task_model: str,
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
        num_poses=1,
    )
    with mp_tasks_vision.PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(input_path)
        result = landmarker.detect(mp_image)
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
        num_poses=1,
    )
    with mp_tasks_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame = first_frame
        frame_idx = 0
        while True:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((frame_idx * 1000.0) / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            draw_pose_landmarker(frame, result.pose_landmarks)

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
        "--display",
        action="store_true",
        help="Show real-time preview window in video mode (press q to quit).",
    )
    parser.add_argument(
        "--task_model",
        default="pose_landmarker.task",
        help="Path to pose_landmarker.task.",
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
        )
        return

    run_video_mode(
        input_path=args.input,
        output_path=args.output,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        display=args.display,
        task_model=args.task_model,
    )


if __name__ == "__main__":
    main()

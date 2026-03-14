import argparse
import json
import os
from collections import deque
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


def first_person_33(result) -> Optional[List]:
    if not result.pose_landmarks:
        return None
    if len(result.pose_landmarks) == 0:
        return None
    return result.pose_landmarks[0]


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
    frame_bgr: np.ndarray, pred_label: str, prob: Optional[np.ndarray], class_names: List[str]
) -> None:
    if pred_label:
        cv2.putText(
            frame_bgr,
            f"Pred: {pred_label}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (40, 255, 40),
            2,
            cv2.LINE_AA,
        )
    if prob is not None:
        top = np.argsort(-prob)[:3]
        y = 70
        for i in top:
            txt = f"{class_names[int(i)]}: {prob[int(i)]:.2f}"
            cv2.putText(
                frame_bgr,
                txt,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28


def run_video_mode(
    input_path: str,
    output_path: Optional[str],
    display: bool,
    task_model: str,
    clf,
    class_names: List[str],
    window_size: int,
    classify_every: int,
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
        num_poses=1,
    )

    history = deque(maxlen=window_size)
    pred_label = ""
    pred_prob: Optional[np.ndarray] = None

    with mp_tasks_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame = first_frame
        frame_idx = 0
        while True:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((frame_idx * 1000.0) / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_pose_landmarker(frame, result.pose_landmarks)
            person = first_person_33(result)
            if person is not None:
                pose18 = mp33_to_openpose18_xy(person, width=width, height=height)
                history.append(pose18)

            if len(history) >= window_size and frame_idx % classify_every == 0:
                win = np.asarray(history, dtype=np.float32)
                pred_label, pred_prob = classify_from_window(clf, win, class_names)

            draw_prediction_overlay(frame, pred_label, pred_prob, class_names)

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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pose estimation + 4-action classification demo.")
    p.add_argument("--input", required=True, help="Input video path.")
    p.add_argument("--output", default=None, help="Output video path.")
    p.add_argument("--display", action="store_true", help="Show live preview window.")
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
        display=args.display,
        task_model=args.task_model,
        clf=clf,
        class_names=class_names,
        window_size=window_size,
        classify_every=max(1, args.classify_every),
    )


if __name__ == "__main__":
    main()

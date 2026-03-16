import argparse
import os
import time
from typing import Dict, List

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision


def parse_model_items(items: List[str]) -> List[Dict[str, str]]:
    parsed = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
            parsed.append({"name": name.strip(), "path": path.strip()})
        else:
            name = os.path.splitext(os.path.basename(item))[0]
            parsed.append({"name": name, "path": item})
    return parsed


def evaluate_model(video_path: str, model_path: str, num_poses: int) -> Dict[str, float]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
    options = mp_tasks_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_tasks_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=num_poses,
    )

    total_frames = 0
    detected_frames = 0
    infer_time_sum = 0.0

    start = time.perf_counter()
    with mp_tasks_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total_frames += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((total_frames - 1) * 1000.0 / fps)

            t0 = time.perf_counter()
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            infer_time_sum += time.perf_counter() - t0

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                detected_frames += 1

    elapsed = time.perf_counter() - start
    cap.release()

    dropped = total_frames - detected_frames
    return {
        "frames": float(total_frames),
        "elapsed_s": elapsed,
        "fps": (total_frames / elapsed) if elapsed > 0 else 0.0,
        "avg_infer_ms": (infer_time_sum / total_frames * 1000.0) if total_frames > 0 else 0.0,
        "dropped": float(dropped),
        "drop_rate": (dropped / total_frames) if total_frames > 0 else 0.0,
    }


def print_table(results: List[Dict[str, float]]) -> None:
    print("\nModel Comparison")
    print("| model | fps | elapsed_s | avg_infer_ms | dropped_frames | drop_rate |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['name']} | {r['fps']:.2f} | {r['elapsed_s']:.2f} | "
            f"{r['avg_infer_ms']:.2f} | {int(r['dropped'])} | {r['drop_rate']*100:.2f}% |"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark 3 MediaPipe pose task models.")
    p.add_argument("--input", required=True, help="Input video path.")
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Three models, format: name=path or path. Example: lite=pose_lite.task full=pose_full.task heavy=pose_heavy.task",
    )
    p.add_argument("--num_poses", type=int, default=4, help="Maximum number of persons to detect.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    models = parse_model_items(args.models)
    if len(models) != 3:
        raise ValueError("Please provide exactly 3 models in --models.")

    results = []
    for m in models:
        print(f"[RUN] {m['name']} -> {m['path']}")
        metrics = evaluate_model(args.input, m["path"], num_poses=max(1, args.num_poses))
        metrics["name"] = m["name"]
        results.append(metrics)

    print_table(results)


if __name__ == "__main__":
    main()

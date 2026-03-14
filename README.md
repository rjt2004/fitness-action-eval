# Fitness Action Eval - MediaPipe Pose Demo

This repository now includes a minimal MediaPipe pose visualization demo for:
- single image input
- video input

The demo draws pose landmarks and skeleton connections on frames and saves outputs.

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Run on an image

```bash
python demo_mediapipe_pose.py --mode image --input data\test.jpg --output outputs\pose_test.jpg
```

## 3) Run on a video

```bash
python demo_mediapipe_pose.py --mode video --input data\workout.mp4 --output outputs\pose_workout.mp4
```

Preview while processing (press `q` to quit preview window):

```bash
python demo_mediapipe_pose.py --mode video --input data\workout.mp4 --output outputs\pose_workout.mp4 --display
```

## 4) Optional parameters

- `--min_detection_confidence`: default `0.5`
- `--min_tracking_confidence`: default `0.5`
- `--task_model`: PoseLandmarker model path, default `pose_landmarker.task`

## Notes

- For image mode, `--output` is required.
- For video mode, `--output` is optional; if omitted, only real-time preview is shown when `--display` is enabled.
- This demo uses PoseLandmarker (MediaPipe Tasks API) only.
- For best codec compatibility, prefer `.mp4` output first (e.g. `outputs\result.mp4`).

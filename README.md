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

## MM-Fit 4-Action Classification (squats/pushups/lunges/situps)

Train classifier:

```bash
python train_mmfit_classifier.py --dataset_root mm-fit --output_dir artifacts\mmfit_action_cls
```

Run demo inference on a labeled MM-Fit segment:

```bash
python demo_mmfit_classifier.py --dataset_root mm-fit --worker w00 --segment_index 0 --model_path artifacts\mmfit_action_cls\rf_mmfit_4actions.joblib --meta_path artifacts\mmfit_action_cls\meta.json
```

## Pose Estimation + Classification Demo (New)

This demo keeps MediaPipe PoseLandmarker visualization and adds 4-action classification overlay.

```bash
python demo_pose_with_classification.py --input push_up_test.mp4 --output push_up_cls_result.mp4 --display --task_model pose_landmarker.task --model_path artifacts\mmfit_action_cls\rf_mmfit_4actions.joblib --meta_path artifacts\mmfit_action_cls\meta.json
```

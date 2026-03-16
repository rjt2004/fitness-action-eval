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

Preview is shown by default while processing (press `q` to quit preview window):

```bash
python demo_mediapipe_pose.py --mode video --input data\workout.mp4 --output outputs\pose_workout.mp4
```

## 4) Optional parameters

- `--min_detection_confidence`: default `0.5`
- `--min_tracking_confidence`: default `0.5`
- `--task_model`: PoseLandmarker model path, default `pose_landmarker.task`
- `--num_poses`: default `4` (detect multiple people, auto-lock one target)
- `--no_display`: disable real-time preview window

## Notes

- For image mode, `--output` is required.
- For video mode, `--output` is optional; real-time preview is enabled by default.
- Main pose demo now draws the auto-selected target person by default.
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

## Pose Estimation + Classification Demo

This demo keeps MediaPipe PoseLandmarker visualization and adds 4-action classification overlay.

```bash
python demo_pose_with_classification.py --input push_up_test.mp4 --output push_up_cls_result.mp4 --task_model pose_landmarker.task --model_path artifacts\mmfit_action_cls\rf_mmfit_4actions.joblib --meta_path artifacts\mmfit_action_cls\meta.json
```

This demo now includes repetition counting (threshold + state machine) for:
- `squats`
- `pushups`
- `lunges`
- `situps`

You can also export counting summary:

```bash
python demo_pose_with_classification.py --input input\pushup_test.mp4 --output output\pushup_cls_count.mp4 --task_model pose_landmarker_full.task --model_path artifacts\mmfit_action_cls\rf_mmfit_4actions.joblib --meta_path artifacts\mmfit_action_cls\meta.json --count_log output\pushup_count.json
```

## One-Click Model Benchmark (FPS/Time/Drop)

Compare three MediaPipe task models on the same video:

```bash
python benchmark_pose_models.py --input push_up_test.mp4 --models lite=pose_landmarker_lite.task full=pose_landmarker_full.task heavy=pose_landmarker_heavy.task
```

Output table includes:
- `fps`: end-to-end processing FPS
- `elapsed_s`: total processing time
- `avg_infer_ms`: average model inference time per frame
- `dropped_frames` / `drop_rate`: frames without detected pose

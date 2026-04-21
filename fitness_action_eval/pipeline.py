from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, Union

import cv2
import numpy as np

from fitness_action_eval.baduanjin import (
    apply_phase_feature_weights,
    build_phase_ids,
    get_phase_definition,
    phase_metadata_rows,
    weight_single_feature,
)
from fitness_action_eval.dtw import distance_to_score, dtw_distance_multidim
from fitness_action_eval.feedback import build_feedback, build_live_feedback
from fitness_action_eval.pose import (
    build_current_feature,
    build_pose_feature_bundle,
    create_pose_landmarker,
    detect_pose_in_frame,
    extract_pose_sequence,
)
from fitness_action_eval.visualization import (
    close_preview_windows,
    compose_compare_frame,
    compose_live_query_frame,
    draw_pose_skeleton,
    draw_text_block,
    get_aligned_reference_frame,
    preview_frame,
    render_feedback_video,
    save_phase_plots,
    save_plot,
)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _ensure_baduanjin_features(data: Dict[str, Any]) -> Dict[str, Any]:
    if "angle_features" not in data or "combined_features_raw" not in data or "feature_mean" not in data or "feature_std" not in data:
        feature_bundle = build_pose_feature_bundle(data["points"])
        data["angle_features"] = feature_bundle["angle_features"]
        data["combined_features_raw"] = feature_bundle["combined_features_raw"]
        data["feature_mean"] = feature_bundle["feature_mean"]
        data["feature_std"] = feature_bundle["feature_std"]
        data["base_features"] = feature_bundle["base_features"]
    else:
        data["base_features"] = data.get("base_features", data["features"])

    data["phase_ids"] = build_phase_ids(
        int(data["points"].shape[0]),
        time_s=data.get("time_s"),
    ).astype(np.int32)

    data["phase_rows"] = phase_metadata_rows(data["phase_ids"], data["time_s"])
    data["features"] = apply_phase_feature_weights(data["base_features"], data["phase_ids"])
    return data


def _template_payload(ref_video: str, task_model: str, num_poses: int, smooth_window: int, ref_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {
        "reference_video": np.asarray(ref_video),
        "task_model": np.asarray(task_model),
        "num_poses": np.asarray(int(num_poses), dtype=np.int32),
        "smooth_window": np.asarray(int(smooth_window), dtype=np.int32),
        "frame_stride": np.asarray(int(ref_data.get("frame_stride", 1)), dtype=np.int32),
        "features": ref_data["features"],
        "base_features": ref_data["base_features"],
        "angle_features": ref_data["angle_features"],
        "combined_features_raw": ref_data["combined_features_raw"],
        "feature_mean": ref_data["feature_mean"],
        "feature_std": ref_data["feature_std"],
        "phase_ids": ref_data["phase_ids"],
        "points": ref_data["points"],
        "raw_points": ref_data["raw_points"],
        "frame_indices": ref_data["frame_indices"],
        "time_s": ref_data["time_s"],
        "fps": np.asarray(ref_data["fps"], dtype=np.float32),
    }


def build_query_alignment_map(
    path: list[tuple[int, int]],
    ref_frame_indices: np.ndarray,
    qry_frame_indices: np.ndarray,
) -> Dict[int, Dict[str, int]]:
    align_map: Dict[int, Dict[str, int]] = {}
    total_steps = max(1, len(path))
    for step_idx, (ref_idx, qry_idx) in enumerate(path):
        query_frame = int(qry_frame_indices[qry_idx])
        align_map[query_frame] = {
            "ref_frame": int(ref_frame_indices[ref_idx]),
            "ref_seq_idx": int(ref_idx),
            "qry_seq_idx": int(qry_idx),
            "path_step": int(step_idx),
            "path_total": int(total_steps),
        }
    return align_map


def save_pose_template(
    ref_video: str,
    task_model: str,
    num_poses: int,
    smooth_window: int,
    template_path: str,
    frame_stride: int = 1,
    preview: bool = False,
) -> Dict[str, Any]:
    # 预处理标准动作视频并导出模板文件，供后续重复评分直接加载。
    ref_data = extract_pose_sequence(
        video_path=ref_video,
        task_model=task_model,
        num_poses=max(1, num_poses),
        smooth_window=max(1, smooth_window),
        frame_stride=max(1, frame_stride),
        preview=preview,
        preview_title="Reference Pose Preview",
    )
    ref_data = _ensure_baduanjin_features(ref_data)
    ensure_parent_dir(template_path)
    np.savez_compressed(
        template_path,
        **_template_payload(
            ref_video=ref_video,
            task_model=task_model,
            num_poses=num_poses,
            smooth_window=smooth_window,
            ref_data=ref_data,
        ),
    )
    return {
        "template_path": template_path,
        "reference_video": ref_video,
        "reference_length": int(ref_data["features"].shape[0]),
        "frame_stride": int(ref_data["frame_stride"]),
    }


def load_pose_template(template_path: str) -> Dict[str, Any]:
    # 读取已导出的标准动作模板。
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    with np.load(template_path, allow_pickle=False) as data:
        loaded = {
            "reference_video": str(data["reference_video"].item()),
            "task_model": str(data["task_model"].item()),
            "num_poses": int(data["num_poses"].item()),
            "smooth_window": int(data["smooth_window"].item()),
            "frame_stride": int(data["frame_stride"].item()) if "frame_stride" in data else 1,
            "features": data["features"],
            "base_features": data["base_features"] if "base_features" in data else data["features"],
            "angle_features": data["angle_features"] if "angle_features" in data else None,
            "combined_features_raw": data["combined_features_raw"] if "combined_features_raw" in data else None,
            "feature_mean": data["feature_mean"] if "feature_mean" in data else None,
            "feature_std": data["feature_std"] if "feature_std" in data else None,
            "phase_ids": data["phase_ids"] if "phase_ids" in data else None,
            "points": data["points"],
            "raw_points": data["raw_points"],
            "frame_indices": data["frame_indices"],
            "time_s": data["time_s"],
            "fps": float(data["fps"].item()),
        }
    return _ensure_baduanjin_features(loaded)


def _phase_maps_for_query(
    alignment_map: Dict[int, Dict[str, int]],
    ref_data: Dict[str, Any],
) -> tuple[Dict[int, str], Dict[int, str]]:
    frame_phase_map: Dict[int, str] = {}
    frame_cue_map: Dict[int, str] = {}
    for query_frame, info in alignment_map.items():
        phase = get_phase_definition(int(ref_data["phase_ids"][info["ref_seq_idx"]]))
        frame_phase_map[int(query_frame)] = phase.display_name
        frame_cue_map[int(query_frame)] = phase.cue
    return frame_phase_map, frame_cue_map


def finalize_scoring_outputs(
    ref_data: Dict[str, Any],
    qry_data: Dict[str, Any],
    ref_video: str,
    query_video: str,
    path: list[tuple[int, int]],
    dist: float,
    score_scale: float,
    hint_threshold: float,
    hint_min_interval: int,
    max_hints: int,
    out_json: str,
    out_plot: str,
    out_video: Optional[str],
    preview: bool,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    norm_dist = dist / max(1, len(path))
    score = distance_to_score(norm_dist, score_scale)

    hints, local_error = build_feedback(
        path=path,
        ref_points=ref_data["points"],
        qry_points=qry_data["points"],
        hint_threshold=hint_threshold,
        hint_min_interval=hint_min_interval,
        max_hints=max_hints,
        ref_phase_ids=ref_data["phase_ids"],
        qry_phase_ids=qry_data["phase_ids"],
        ref_angles=ref_data["angle_features"],
        qry_angles=qry_data["angle_features"],
    )

    for hint in hints:
        q_idx = int(hint["query_index"])
        hint["query_frame"] = int(qry_data["frame_indices"][q_idx])
        hint["query_time_s"] = float(qry_data["time_s"][q_idx])
        hint["ref_time_s"] = float(ref_data["time_s"][int(hint["ref_index"])])

    ensure_parent_dir(out_json)
    if progress_callback:
        progress_callback(55, "正在生成分阶段 DTW 图")
    phase_plot_dir = os.path.join(os.path.dirname(os.path.abspath(out_plot)), "phase_plots")
    phase_plots = save_phase_plots(
        ref_data=ref_data,
        qry_data=qry_data,
        path=path,
        hints=hints,
        out_dir=phase_plot_dir,
        score_scale=score_scale,
    )
    result = {
        "mode": "baduanjin_pose_dtw_weighted",
        "reference_video": ref_video,
        "query_video": query_video,
        "feature": "pose33_xy_normalized_plus_joint_angles_phase_weighted",
        "reference_length": int(ref_data["features"].shape[0]),
        "query_length": int(qry_data["features"].shape[0]),
        "reference_frame_stride": int(ref_data.get("frame_stride", 1)),
        "query_frame_stride": int(qry_data.get("frame_stride", 1)),
        "dtw_distance": float(dist),
        "alignment_path_length": int(len(path)),
        "normalized_dtw_distance": float(norm_dist),
        "score_0_100": float(score),
        "score_scale": float(score_scale),
        "hint_threshold": float(hint_threshold),
        "hint_min_interval": int(hint_min_interval),
        "max_hints": int(max_hints),
        "hint_count": int(len(hints)),
        "reference_phases": ref_data["phase_rows"],
        "query_phases": qry_data["phase_rows"],
        "phase_plots": phase_plots,
        "hints": hints,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if progress_callback:
        progress_callback(62, "正在生成整体 DTW 曲线")
    save_plot(
        ref_data=ref_data,
        qry_data=qry_data,
        path=path,
        hints=hints,
        out_png=out_plot,
        score=score,
        norm_dist=norm_dist,
    )

    frame_hint_map: Dict[int, str] = {}
    for hint in hints:
        frame_hint_map[int(hint["query_frame"])] = str(hint["message"])

    frame_error_map: Dict[int, float] = {}
    for idx, frame_id in enumerate(qry_data["frame_indices"]):
        err = float(local_error[idx])
        if np.isfinite(err):
            frame_error_map[int(frame_id)] = err

    frame_pose_map: Dict[int, np.ndarray] = {}
    for idx, frame_id in enumerate(qry_data["frame_indices"]):
        frame_pose_map[int(frame_id)] = qry_data["raw_points"][idx]

    ref_pose_map: Dict[int, np.ndarray] = {}
    for idx, frame_id in enumerate(ref_data["frame_indices"]):
        ref_pose_map[int(frame_id)] = ref_data["raw_points"][idx]

    alignment_map = build_query_alignment_map(
        path=path,
        ref_frame_indices=ref_data["frame_indices"],
        qry_frame_indices=qry_data["frame_indices"],
    )
    frame_phase_map, frame_cue_map = _phase_maps_for_query(alignment_map=alignment_map, ref_data=ref_data)

    if out_video or preview:
        if progress_callback:
            progress_callback(72, "正在渲染对比视频")
        render_feedback_video(
            ref_video=ref_video,
            query_video=query_video,
            output_video=out_video,
            score=score,
            frame_hint_map=frame_hint_map,
            frame_error_map=frame_error_map,
            frame_pose_map=frame_pose_map,
            ref_pose_map=ref_pose_map,
            alignment_map=alignment_map,
            frame_phase_map=frame_phase_map,
            frame_cue_map=frame_cue_map,
            preview=preview,
            progress_callback=progress_callback,
            progress_range=(72, 92),
            compare_panel_height=540,
        )
    elif progress_callback:
        progress_callback(92, "评估结果已生成")

    return {
        "result": result,
        "norm_dist": float(norm_dist),
        "score": float(score),
        "hint_count": int(len(hints)),
    }


def run_dtw_scoring(
    ref_video: str,
    query_video: str,
    task_model: str,
    num_poses: int,
    smooth_window: int,
    score_scale: float,
    hint_threshold: float,
    hint_min_interval: int,
    max_hints: int,
    out_json: str,
    out_plot: str,
    out_video: Optional[str] = None,
    frame_stride: int = 1,
    preview: bool = False,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    if progress_callback:
        progress_callback(8, "正在提取标准视频姿态")
    ref_data = extract_pose_sequence(
        video_path=ref_video,
        task_model=task_model,
        num_poses=max(1, num_poses),
        smooth_window=max(1, smooth_window),
        frame_stride=max(1, frame_stride),
        preview=preview,
        preview_title="Reference Pose Preview",
    )
    if progress_callback:
        progress_callback(25, "正在提取待测视频姿态")
    qry_data = extract_pose_sequence(
        video_path=query_video,
        task_model=task_model,
        num_poses=max(1, num_poses),
        smooth_window=max(1, smooth_window),
        frame_stride=max(1, frame_stride),
        preview=preview,
        preview_title="Query Pose Preview",
    )
    ref_data = _ensure_baduanjin_features(ref_data)
    qry_data = _ensure_baduanjin_features(qry_data)
    if progress_callback:
        progress_callback(42, "正在执行 DTW 对齐")
    dist, path = dtw_distance_multidim(ref_data["features"], qry_data["features"])
    if progress_callback:
        progress_callback(50, "正在生成评分结果")
    return finalize_scoring_outputs(
        ref_data=ref_data,
        qry_data=qry_data,
        ref_video=ref_video,
        query_video=query_video,
        path=path,
        dist=dist,
        score_scale=score_scale,
        hint_threshold=hint_threshold,
        hint_min_interval=hint_min_interval,
        max_hints=max_hints,
        out_json=out_json,
        out_plot=out_plot,
        out_video=out_video,
        preview=preview,
        progress_callback=progress_callback,
    )


def run_dtw_scoring_from_template(
    template_path: str,
    query_video: str,
    out_json: str,
    out_plot: str,
    out_video: Optional[str] = None,
    preview: bool = False,
    score_scale: float = 8.0,
    hint_threshold: float = 0.18,
    hint_min_interval: int = 8,
    max_hints: int = 40,
    query_frame_stride: Optional[int] = None,
    query_smooth_window: Optional[int] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    # 直接加载模板并只处理待测视频，适合重复评分场景。
    ref_data = load_pose_template(template_path)
    if progress_callback:
        progress_callback(12, "模板已加载，正在提取待测视频姿态")
    qry_data = extract_pose_sequence(
        video_path=query_video,
        task_model=ref_data["task_model"],
        num_poses=max(1, ref_data["num_poses"]),
        smooth_window=max(1, int(query_smooth_window or ref_data["smooth_window"])),
        frame_stride=max(1, int(query_frame_stride or ref_data.get("frame_stride", 1))),
        preview=preview,
        preview_title="Query Pose Preview",
    )
    qry_data = _ensure_baduanjin_features(qry_data)
    if progress_callback:
        progress_callback(42, "正在执行 DTW 对齐")
    dist, path = dtw_distance_multidim(ref_data["features"], qry_data["features"])
    if progress_callback:
        progress_callback(50, "正在生成评分结果")
    summary = finalize_scoring_outputs(
        ref_data=ref_data,
        qry_data=qry_data,
        ref_video=ref_data["reference_video"],
        query_video=query_video,
        path=path,
        dist=dist,
        score_scale=score_scale,
        hint_threshold=hint_threshold,
        hint_min_interval=hint_min_interval,
        max_hints=max_hints,
        out_json=out_json,
        out_plot=out_plot,
        out_video=out_video,
        preview=preview,
        progress_callback=progress_callback,
    )
    summary["template_path"] = template_path
    return summary


def _load_or_prepare_reference(
    template_path: Optional[str],
    ref_video: Optional[str],
    task_model: str,
    num_poses: int,
    smooth_window: int,
    frame_stride: int,
) -> Dict[str, Any]:
    if template_path and os.path.exists(template_path):
        return load_pose_template(template_path)
    if not ref_video:
        raise ValueError("Either template_path or ref_video must be provided for camera coaching.")

    ref_data = extract_pose_sequence(
        video_path=ref_video,
        task_model=task_model,
        num_poses=max(1, num_poses),
        smooth_window=max(1, smooth_window),
        frame_stride=max(1, frame_stride),
        preview=False,
    )
    ref_data = _ensure_baduanjin_features(ref_data)
    ref_data["reference_video"] = ref_video
    ref_data["task_model"] = task_model
    ref_data["num_poses"] = max(1, num_poses)
    ref_data["smooth_window"] = max(1, smooth_window)
    if template_path:
        ensure_parent_dir(template_path)
        np.savez_compressed(
            template_path,
            **_template_payload(
                ref_video=ref_video,
                task_model=task_model,
                num_poses=num_poses,
                smooth_window=smooth_window,
                ref_data=ref_data,
            ),
        )
    return ref_data


def _resolve_capture_source(source: Union[int, str]) -> Union[int, str]:
    if isinstance(source, int):
        return source
    try:
        return int(source)
    except (TypeError, ValueError):
        return str(source)


def run_camera_coach(
    template_path: Optional[str],
    ref_video: Optional[str],
    camera_source: Union[int, str],
    task_model: str,
    num_poses: int,
    smooth_window: int,
    score_scale: float,
    hint_threshold: float,
    hint_min_interval: int,
    max_hints: int,
    ref_search_window: int,
    frame_stride: int = 1,
    camera_width: Optional[int] = None,
    camera_height: Optional[int] = None,
    camera_mirror: bool = True,
    out_json: Optional[str] = None,
    out_video: Optional[str] = None,
    preview: bool = True,
    max_frames: Optional[int] = None,
    stop_checker: Optional[Callable[[], bool]] = None,
    pause_checker: Optional[Callable[[], bool]] = None,
    frame_callback: Optional[Callable[[np.ndarray], None]] = None,
) -> Dict[str, Any]:
    capture_stride = max(1, int(frame_stride))
    ref_data = _load_or_prepare_reference(
        template_path=template_path,
        ref_video=ref_video,
        task_model=task_model,
        num_poses=num_poses,
        smooth_window=smooth_window,
        frame_stride=frame_stride,
    )
    ref_video_path = ref_data["reference_video"]

    cap = cv2.VideoCapture(_resolve_capture_source(camera_source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open capture source: {camera_source}")
    if camera_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_width))
    if camera_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_height))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    record_compare_video = bool(out_video)
    ref_cap = cv2.VideoCapture(ref_video_path) if record_compare_video and ref_video_path else None
    if ref_cap is not None and not ref_cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Cannot open video: {ref_video_path}")

    writer = None
    writer_kind: Optional[str] = None
    actual_out_video = out_video
    if out_video:
        ensure_parent_dir(out_video)

    current_ref_seq_idx = 0
    current_ref_frame_idx = -1
    current_ref_frame: Optional[np.ndarray] = None
    pose_window: Deque[np.ndarray] = deque(maxlen=max(3, smooth_window * 3))
    prev_center = None
    frame_idx = 0
    active_hint = ""
    active_hint_left = 0
    keep_frames = max(1, int(round(fps * 0.8)))
    running_scores: list[float] = []
    live_hints: list[dict[str, Any]] = []
    processed_idx = 0
    last_hint_processed_idx = -10**9
    latest_phase_name = ""
    latest_phase_cue = ""
    latest_local_err = float("nan")
    latest_part = ""

    with create_pose_landmarker(task_model=ref_data["task_model"], num_poses=ref_data["num_poses"]) as landmarker:
        while True:
            if stop_checker is not None and stop_checker():
                break
            while pause_checker is not None and pause_checker():
                if stop_checker is not None and stop_checker():
                    break
                time.sleep(0.05)
            if stop_checker is not None and stop_checker():
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % capture_stride != 0:
                frame_idx += 1
                continue
            if camera_mirror:
                frame = cv2.flip(frame, 1)

            query_view = frame.copy()
            timestamp_ms = int((frame_idx * 1000.0) / fps)
            norm_pts, raw_pts, prev_center = detect_pose_in_frame(
                landmarker=landmarker,
                frame=frame,
                timestamp_ms=timestamp_ms,
                prev_center=prev_center,
            )

            if raw_pts is not None and norm_pts is not None:
                draw_pose_skeleton(query_view, raw_pts)
                pose_window.append(norm_pts)
                current_points, _, current_feature, current_angles = build_current_feature(
                    recent_points=pose_window,
                    smooth_window=smooth_window,
                    feature_mean=ref_data["feature_mean"],
                    feature_std=ref_data["feature_std"],
                )

                search_start = current_ref_seq_idx
                search_end = min(ref_data["features"].shape[0], search_start + max(10, int(ref_search_window)))
                best_ref_idx = search_start
                best_dist = float("inf")
                for ref_idx in range(search_start, search_end):
                    weighted_feature = weight_single_feature(current_feature, int(ref_data["phase_ids"][ref_idx]))
                    dist = float(np.linalg.norm(weighted_feature - ref_data["features"][ref_idx]))
                    if dist < best_dist:
                        best_dist = dist
                        best_ref_idx = ref_idx

                current_ref_seq_idx = best_ref_idx
                phase = get_phase_definition(int(ref_data["phase_ids"][current_ref_seq_idx]))
                latest_phase_name = phase.display_name
                latest_phase_cue = phase.cue
                instant_score = distance_to_score(best_dist, score_scale)
                running_scores.append(instant_score)
                latest_local_err = float(
                    np.mean(np.linalg.norm(current_points - ref_data["points"][current_ref_seq_idx], axis=1))
                )
                active_hint, _, latest_part = build_live_feedback(
                    ref_points=ref_data["points"][current_ref_seq_idx],
                    qry_points=current_points,
                    hint_threshold=hint_threshold,
                    phase_id=int(ref_data["phase_ids"][current_ref_seq_idx]),
                    ref_angles=ref_data["angle_features"][current_ref_seq_idx],
                    qry_angles=current_angles,
                )
                active_hint_left = keep_frames if active_hint else max(0, active_hint_left - 1)
                if (
                    active_hint
                    and len(live_hints) < int(max_hints)
                    and (processed_idx - last_hint_processed_idx) >= int(hint_min_interval)
                ):
                    live_hints.append(
                        {
                            "query_frame": int(frame_idx),
                            "query_time_s": float(frame_idx / fps),
                            "phase_name": latest_phase_name,
                            "cue": latest_phase_cue,
                            "part": latest_part,
                            "message": active_hint,
                            "score": float(instant_score),
                        }
                    )
                    last_hint_processed_idx = processed_idx
            else:
                draw_text_block(
                    query_view,
                    ["未检测到完整人体姿态。", "请站在画面中央，并保持全身入镜。"],
                    x=20,
                    y=18,
                )

            if ref_cap is not None:
                target_ref_frame = int(ref_data["frame_indices"][current_ref_seq_idx])
                current_ref_frame_idx, current_ref_frame = get_aligned_reference_frame(
                    ref_cap=ref_cap,
                    target_ref_frame=target_ref_frame,
                    current_ref_frame_idx=current_ref_frame_idx,
                    current_ref_frame=current_ref_frame,
                )
                if current_ref_frame is None:
                    ref_view = np.zeros_like(query_view)
                else:
                    ref_view = current_ref_frame.copy()
                    draw_pose_skeleton(ref_view, ref_data["raw_points"][current_ref_seq_idx])
            else:
                ref_view = np.zeros_like(query_view)

            current_score = float(np.mean(running_scores[-60:])) if running_scores else 0.0
            hint_text = active_hint if active_hint_left > 0 else ""
            if active_hint_left > 0:
                active_hint_left -= 1

            live_preview_frame = compose_live_query_frame(
                qry_frame=query_view,
                score=current_score,
                current_local_err=latest_local_err,
                active_hint=hint_text,
                phase_name=latest_phase_name,
                phase_cue=latest_phase_cue,
            )
            if frame_callback is not None:
                frame_callback(live_preview_frame)

            output_frame = None
            if record_compare_video:
                output_frame = compose_compare_frame(
                    ref_frame=ref_view,
                    qry_frame=query_view,
                    score=current_score,
                    current_local_err=latest_local_err,
                    active_hint=hint_text,
                    align_info={
                        "ref_frame": int(ref_data["frame_indices"][current_ref_seq_idx]),
                        "qry_frame": int(frame_idx),
                        "ref_seq_idx": int(current_ref_seq_idx),
                        "qry_seq_idx": int(frame_idx),
                        "path_step": int(current_ref_seq_idx),
                        "path_total": int(ref_data["features"].shape[0]),
                    },
                    phase_name=latest_phase_name,
                    phase_cue=latest_phase_cue,
                    max_panel_height=480,
                )
                if output_frame.shape[1] % 2 != 0:
                    output_frame = cv2.copyMakeBorder(
                        output_frame,
                        0,
                        0,
                        0,
                        1,
                        cv2.BORDER_CONSTANT,
                        value=(16, 16, 16),
                    )
                if output_frame.shape[0] % 2 != 0:
                    output_frame = cv2.copyMakeBorder(
                        output_frame,
                        0,
                        1,
                        0,
                        0,
                        cv2.BORDER_CONSTANT,
                        value=(16, 16, 16),
                    )

            if writer is None and out_video and output_frame is not None:
                cv_writer = cv2.VideoWriter(
                    actual_out_video or out_video,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (output_frame.shape[1], output_frame.shape[0]),
                )
                if cv_writer.isOpened():
                    writer = cv_writer
                    writer_kind = "cv2"
                else:
                    cv_writer.release()
                    try:
                        import imageio.v2 as imageio

                        writer = imageio.get_writer(
                            actual_out_video or out_video,
                            fps=max(1, int(round(fps))),
                            codec="libx264",
                            macro_block_size=2,
                        )
                        writer_kind = "imageio"
                    except Exception:
                        fallback_path = str(Path(out_video).with_suffix(".avi"))
                        cv_writer = cv2.VideoWriter(
                            fallback_path,
                            cv2.VideoWriter_fourcc(*"MJPG"),
                            fps,
                            (output_frame.shape[1], output_frame.shape[0]),
                        )
                        if cv_writer.isOpened():
                            writer = cv_writer
                            writer_kind = "cv2"
                            actual_out_video = fallback_path
                        else:
                            cap.release()
                            if ref_cap is not None:
                                ref_cap.release()
                            raise RuntimeError(f"Cannot open video writer: {out_video}")

            if writer is not None and output_frame is not None:
                if writer_kind == "imageio":
                    writer.append_data(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
                else:
                    writer.write(output_frame)
            if preview:
                if not preview_frame("Baduanjin Camera Coach", live_preview_frame):
                    preview = False
                    close_preview_windows()
                    break
            processed_idx += 1
            frame_idx += 1

    if writer is not None:
        if writer_kind == "imageio":
            writer.close()
        else:
            writer.release()
    cap.release()
    if ref_cap is not None:
        ref_cap.release()
    if preview:
        close_preview_windows()

    summary = {
        "mode": "baduanjin_camera_coach",
        "reference_video": ref_video_path,
        "camera_source": str(camera_source),
        "matched_frames": int(len(running_scores)),
        "avg_score_0_100": float(np.mean(running_scores)) if running_scores else 0.0,
        "final_phase_name": latest_phase_name,
        "final_phase_cue": latest_phase_cue,
        "final_part": latest_part,
        "hint_count": int(len(live_hints)),
        "hints": live_hints,
        "template_path": template_path,
        "output_video": actual_out_video,
    }
    if out_json:
        ensure_parent_dir(out_json)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

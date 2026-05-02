from __future__ import annotations

"""算法主流程。

本文件把模板生成、离线评估、实时跟练三条链路串起来，是整个算法端的入口。
"""

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
    build_substage_metadata,
    default_baduanjin_rule_config,
    get_phase_definition,
    get_substage_definition,
    get_substage_by_key,
    phase_metadata_rows,
    weight_single_feature,
)
from fitness_action_eval.dtw import distance_to_score, dtw_distance_multidim
from fitness_action_eval.feedback import build_feedback, build_live_feedback, part_errors
from fitness_action_eval.pose import (
    LiveStreamPoseDetector,
    build_current_feature,
    build_pose_feature_bundle,
    create_pose_landmarker,
    detect_pose_in_frame,
    extract_pose_sequence,
    pose_quality_summary,
)
from fitness_action_eval.visualization import (
    close_preview_windows,
    compose_compare_frame,
    compose_error_frame,
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
    """确保输出文件的父目录存在。"""

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _ensure_baduanjin_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """补齐八段锦专项需要的阶段与子阶段特征。"""

    rule_config = data.get("rule_config") or default_baduanjin_rule_config()
    data["rule_config"] = rule_config
    if "angle_features" not in data or "combined_features_raw" not in data or "feature_mean" not in data or "feature_std" not in data:
        feature_bundle = build_pose_feature_bundle(data["points"], point_confidence=data.get("point_confidence"))
        data["angle_features"] = feature_bundle["angle_features"]
        data["combined_features_raw"] = feature_bundle["combined_features_raw"]
        data["feature_mean"] = feature_bundle["feature_mean"]
        data["feature_std"] = feature_bundle["feature_std"]
        data["base_features"] = feature_bundle["base_features"]
        data["point_confidence"] = feature_bundle["point_confidence"]
        data["point_confidence_source"] = data.get("point_confidence_source", "mediapipe")
        data["feature_confidence_weights"] = feature_bundle["feature_confidence_weights"]
    else:
        data["base_features"] = data.get("base_features", data["features"])
        if data.get("point_confidence") is None:
            data["point_confidence"] = np.ones((data["points"].shape[0], 33), dtype=np.float32)
            data["point_confidence_source"] = data.get("point_confidence_source", "legacy_default")
        else:
            data["point_confidence_source"] = data.get("point_confidence_source", "mediapipe")
        if data.get("feature_confidence_weights") is None:
            feature_bundle = build_pose_feature_bundle(data["points"], point_confidence=data["point_confidence"])
            data["feature_confidence_weights"] = feature_bundle["feature_confidence_weights"]
    if not data.get("pose_quality"):
        processed_frames = int(data.get("frame_indices", np.asarray([])).shape[0])
        data["pose_quality"] = pose_quality_summary(
            data["point_confidence"],
            processed_frames=processed_frames,
            skipped_frames=0,
        )
    data["pose_quality"]["confidence_source"] = data.get("point_confidence_source", "mediapipe")

    data["phase_ids"] = build_phase_ids(
        int(data["points"].shape[0]),
        time_s=data.get("time_s"),
        rule_config=rule_config,
    ).astype(np.int32)

    data["phase_rows"] = phase_metadata_rows(data["phase_ids"], data["time_s"], rule_config=rule_config)
    state_points = data.get("raw_points", data["points"])
    substage_metadata = build_substage_metadata(data["phase_ids"], data["time_s"], points=state_points, rule_config=rule_config)
    data["substage_keys"] = substage_metadata["keys"]
    data["substage_names"] = substage_metadata["names"]
    data["substage_cues"] = substage_metadata["cues"]
    data["substage_rows"] = substage_metadata["rows"]
    data["features"] = apply_phase_feature_weights(data["base_features"], data["phase_ids"], rule_config=rule_config)
    return data


def _apply_template_feature_stats(qry_data: Dict[str, Any], ref_data: Dict[str, Any]) -> Dict[str, Any]:
    """把待测序列投影到模板特征标准化空间。"""

    if qry_data.get("combined_features_raw") is None:
        feature_bundle = build_pose_feature_bundle(qry_data["points"], point_confidence=qry_data.get("point_confidence"))
        qry_data["angle_features"] = feature_bundle["angle_features"]
        qry_data["combined_features_raw"] = feature_bundle["combined_features_raw"]
        qry_data["point_confidence"] = feature_bundle["point_confidence"]
        qry_data["feature_confidence_weights"] = feature_bundle["feature_confidence_weights"]
    elif qry_data.get("feature_confidence_weights") is None:
        feature_bundle = build_pose_feature_bundle(qry_data["points"], point_confidence=qry_data.get("point_confidence"))
        qry_data["point_confidence"] = feature_bundle["point_confidence"]
        qry_data["feature_confidence_weights"] = feature_bundle["feature_confidence_weights"]
    feature_mean = ref_data["feature_mean"]
    feature_std = np.maximum(ref_data["feature_std"], 1e-6)
    qry_data["feature_mean"] = feature_mean
    qry_data["feature_std"] = feature_std
    qry_data["base_features"] = ((qry_data["combined_features_raw"] - feature_mean) / feature_std).astype(np.float32)
    qry_data["features"] = qry_data["base_features"]
    qry_data["rule_config"] = ref_data.get("rule_config") or default_baduanjin_rule_config()
    return qry_data


def _template_payload(ref_video: str, task_model: str, num_poses: int, smooth_window: int, ref_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {
        "reference_video": np.asarray(ref_video),
        "task_model": np.asarray(task_model),
        "rule_config_json": np.asarray(json.dumps(ref_data.get("rule_config") or default_baduanjin_rule_config(), ensure_ascii=False)),
        "num_poses": np.asarray(int(num_poses), dtype=np.int32),
        "smooth_window": np.asarray(int(smooth_window), dtype=np.int32),
        "frame_stride": np.asarray(int(ref_data.get("frame_stride", 1)), dtype=np.int32),
        "features": ref_data["features"],
        "base_features": ref_data["base_features"],
        "angle_features": ref_data["angle_features"],
        "combined_features_raw": ref_data["combined_features_raw"],
        "feature_mean": ref_data["feature_mean"],
        "feature_std": ref_data["feature_std"],
        "point_confidence": ref_data.get("point_confidence", np.ones((ref_data["points"].shape[0], 33), dtype=np.float32)),
        "point_confidence_source": np.asarray(ref_data.get("point_confidence_source", "mediapipe")),
        "feature_confidence_weights": ref_data.get(
            "feature_confidence_weights",
            np.ones_like(ref_data["features"], dtype=np.float32),
        ),
        "pose_quality_json": np.asarray(json.dumps(ref_data.get("pose_quality", {}), ensure_ascii=False)),
        "phase_ids": ref_data["phase_ids"],
        "substage_keys": ref_data["substage_keys"],
        "substage_names": ref_data["substage_names"],
        "substage_cues": ref_data["substage_cues"],
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
    rule_config: Optional[Dict[str, Any]] = None,
    preview: bool = False,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """从标准视频生成可复用模板。"""

    # 预处理标准动作视频并导出模板文件，供后续重复评分直接加载。
    if progress_callback:
        progress_callback(5, "开始生成模板")
    ref_data = extract_pose_sequence(
        video_path=ref_video,
        task_model=task_model,
        num_poses=max(1, num_poses),
        smooth_window=max(1, smooth_window),
        frame_stride=max(1, frame_stride),
        preview=preview,
        preview_title="Reference Pose Preview",
        progress_callback=progress_callback,
        progress_range=(8, 88),
        progress_message="正在提取标准视频姿态",
    )
    if progress_callback:
        progress_callback(92, "正在生成阶段特征")
    ref_data["rule_config"] = rule_config or default_baduanjin_rule_config()
    ref_data = _ensure_baduanjin_features(ref_data)
    if progress_callback:
        progress_callback(96, "正在保存模板文件")
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
    if progress_callback:
        progress_callback(100, "模板生成完成")
    return {
        "template_path": template_path,
        "reference_video": ref_video,
        "reference_length": int(ref_data["features"].shape[0]),
        "frame_stride": int(ref_data["frame_stride"]),
    }


def load_pose_template(template_path: str) -> Dict[str, Any]:
    """读取已经导出的标准动作模板。"""

    # 读取已导出的标准动作模板。
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    with np.load(template_path, allow_pickle=False) as data:
        rule_config = default_baduanjin_rule_config()
        if "rule_config_json" in data:
            try:
                rule_config = json.loads(str(data["rule_config_json"].item()))
            except (TypeError, json.JSONDecodeError):
                rule_config = default_baduanjin_rule_config()
        has_point_confidence = "point_confidence" in data
        confidence_source = (
            str(data["point_confidence_source"].item())
            if "point_confidence_source" in data
            else ("mediapipe" if has_point_confidence else "legacy_default")
        )
        loaded = {
            "reference_video": str(data["reference_video"].item()),
            "task_model": str(data["task_model"].item()),
            "rule_config": rule_config,
            "num_poses": int(data["num_poses"].item()),
            "smooth_window": int(data["smooth_window"].item()),
            "frame_stride": int(data["frame_stride"].item()) if "frame_stride" in data else 1,
            "features": data["features"],
            "base_features": data["base_features"] if "base_features" in data else data["features"],
            "angle_features": data["angle_features"] if "angle_features" in data else None,
            "combined_features_raw": data["combined_features_raw"] if "combined_features_raw" in data else None,
            "feature_mean": data["feature_mean"] if "feature_mean" in data else None,
            "feature_std": data["feature_std"] if "feature_std" in data else None,
            "point_confidence": data["point_confidence"] if has_point_confidence else None,
            "point_confidence_source": confidence_source,
            "feature_confidence_weights": data["feature_confidence_weights"] if "feature_confidence_weights" in data else None,
            "phase_ids": data["phase_ids"] if "phase_ids" in data else None,
            "substage_keys": data["substage_keys"] if "substage_keys" in data else None,
            "substage_names": data["substage_names"] if "substage_names" in data else None,
            "substage_cues": data["substage_cues"] if "substage_cues" in data else None,
            "points": data["points"],
            "raw_points": data["raw_points"],
            "frame_indices": data["frame_indices"],
            "time_s": data["time_s"],
            "fps": float(data["fps"].item()),
        }
        if "pose_quality_json" in data:
            try:
                loaded["pose_quality"] = json.loads(str(data["pose_quality_json"].item()))
            except (TypeError, json.JSONDecodeError):
                loaded["pose_quality"] = {}
    return _ensure_baduanjin_features(loaded)


def _phase_maps_for_query(
    alignment_map: Dict[int, Dict[str, int]],
    ref_data: Dict[str, Any],
) -> tuple[Dict[int, str], Dict[int, str]]:
    frame_phase_map: Dict[int, str] = {}
    frame_cue_map: Dict[int, str] = {}
    for query_frame, info in alignment_map.items():
        phase = get_phase_definition(int(ref_data["phase_ids"][info["ref_seq_idx"]]), rule_config=ref_data.get("rule_config"))
        frame_phase_map[int(query_frame)] = phase.display_name
        frame_cue_map[int(query_frame)] = phase.cue
    return frame_phase_map, frame_cue_map


PART_SCORE_NAMES = {
    "head_neck": "头颈",
    "shoulders": "肩臂",
    "hands": "手部",
    "torso": "躯干",
    "waist": "腰胯",
    "hips": "髋部",
    "knees": "膝部",
    "feet": "脚步",
}

PHASE_SCORE_SCALE_MULTIPLIER = {
    5: 1.45,
    6: 1.55,
}

PART_SCORE_SCALE_MULTIPLIER = {
    "hands": 2.0,
}


def _confidence_weighted_cost(
    ref_feature: np.ndarray,
    qry_feature: np.ndarray,
    ref_weight: Optional[np.ndarray] = None,
    qry_weight: Optional[np.ndarray] = None,
) -> float:
    diff = ref_feature - qry_feature
    if ref_weight is None or qry_weight is None:
        return float(np.linalg.norm(diff))
    weight = np.sqrt(np.maximum(ref_weight * qry_weight, 0.0))
    active = max(float(np.mean(weight)), 1e-6)
    return float(np.linalg.norm(diff * weight) / np.sqrt(active))


def _score_scale_for_phase(phase_id: int, base_score_scale: float) -> float:
    return float(base_score_scale) * float(PHASE_SCORE_SCALE_MULTIPLIER.get(int(phase_id), 1.0))


def _adaptive_distance_score(norm_dist: float, score_scale: float, phase_id: Optional[int] = None) -> float:
    linear_score = distance_to_score(norm_dist, score_scale)
    if phase_id not in PHASE_SCORE_SCALE_MULTIPLIER:
        return linear_score
    soft_tail = min(25.0, 100.0 * float(np.exp(-float(norm_dist) / max(1e-6, float(score_scale)))))
    return float(max(linear_score, soft_tail))


def _unique_sorted_indices(indices: list[int]) -> np.ndarray:
    if not indices:
        return np.asarray([], dtype=np.int32)
    return np.asarray(sorted(set(int(idx) for idx in indices)), dtype=np.int32)


def _local_normalized_dtw(
    ref_data: Dict[str, Any],
    qry_data: Dict[str, Any],
    ref_indices: list[int],
    qry_indices: list[int],
) -> Optional[float]:
    ref_idx = _unique_sorted_indices(ref_indices)
    qry_idx = _unique_sorted_indices(qry_indices)
    if ref_idx.size < 2 or qry_idx.size < 2:
        return None

    ref_weights = ref_data.get("feature_confidence_weights")
    qry_weights = qry_data.get("feature_confidence_weights")
    try:
        dist, local_path = dtw_distance_multidim(
            ref_data["features"][ref_idx],
            qry_data["features"][qry_idx],
            window_ratio=0.35,
            a_weights=ref_weights[ref_idx] if ref_weights is not None else None,
            b_weights=qry_weights[qry_idx] if qry_weights is not None else None,
        )
    except RuntimeError:
        return None
    return float(dist / max(1, len(local_path)))


def _bend_touch_goal_cost(
    ref_pts: np.ndarray,
    qry_pts: np.ndarray,
    ref_angles: Optional[np.ndarray],
    qry_angles: Optional[np.ndarray],
) -> float:
    ref_left_gap = min(float(np.linalg.norm(ref_pts[15] - ref_pts[idx])) for idx in (27, 29, 31))
    ref_right_gap = min(float(np.linalg.norm(ref_pts[16] - ref_pts[idx])) for idx in (28, 30, 32))
    qry_left_gap = min(float(np.linalg.norm(qry_pts[15] - qry_pts[idx])) for idx in (27, 29, 31))
    qry_right_gap = min(float(np.linalg.norm(qry_pts[16] - qry_pts[idx])) for idx in (28, 30, 32))
    gap_err = (abs(qry_left_gap - ref_left_gap) + abs(qry_right_gap - ref_right_gap)) / 2.0

    ref_hip_center = (ref_pts[23] + ref_pts[24]) / 2.0
    qry_hip_center = (qry_pts[23] + qry_pts[24]) / 2.0
    ref_hand_y = float(((ref_pts[15, 1] + ref_pts[16, 1]) / 2.0) - ref_hip_center[1])
    qry_hand_y = float(((qry_pts[15, 1] + qry_pts[16, 1]) / 2.0) - qry_hip_center[1])
    vertical_err = abs(qry_hand_y - ref_hand_y)

    hip_err = 0.0
    knee_err = 0.0
    if ref_angles is not None and qry_angles is not None:
        hip_err = float(np.mean(np.abs(qry_angles[[4, 5]] - ref_angles[[4, 5]])))
        knee_err = float(np.mean(np.abs(qry_angles[[6, 7]] - ref_angles[[6, 7]])))

    return float(18.0 * ((0.35 * gap_err) + (0.25 * vertical_err) + (0.25 * hip_err) + (0.15 * knee_err)))


def _bend_touch_robust_distance(
    path: list[tuple[int, int]],
    ref_data: Dict[str, Any],
    qry_data: Dict[str, Any],
    ref_indices: list[int],
    qry_indices: list[int],
) -> Optional[float]:
    ref_set = set(int(idx) for idx in ref_indices)
    qry_set = set(int(idx) for idx in qry_indices)
    costs = [
        _bend_touch_goal_cost(
            ref_data["points"][i],
            qry_data["points"][j],
            ref_data["angle_features"][i] if ref_data.get("angle_features") is not None else None,
            qry_data["angle_features"][j] if qry_data.get("angle_features") is not None else None,
        )
        for i, j in path
        if int(i) in ref_set and int(j) in qry_set
    ]
    if not costs:
        return None
    cost_arr = np.asarray(costs, dtype=np.float32)
    return float(np.percentile(cost_arr, 50))


def _mean_point_confidence(data: Dict[str, Any], indices: list[int]) -> Optional[float]:
    point_confidence = data.get("point_confidence")
    if point_confidence is None or not indices:
        return None
    idx = np.asarray(indices, dtype=np.int32)
    idx = idx[(idx >= 0) & (idx < len(point_confidence))]
    if idx.size == 0:
        return None
    return float(np.mean(point_confidence[idx]))


def _weighted_phase_score(phase_scores: list[dict[str, Any]], fallback: float) -> float:
    total_weight = sum(max(1, int(row.get("alignment_count", 0))) for row in phase_scores)
    if total_weight <= 0:
        return float(fallback)
    score_sum = sum(float(row.get("score_0_100", 0.0)) * max(1, int(row.get("alignment_count", 0))) for row in phase_scores)
    return float(score_sum / total_weight)


def build_score_breakdowns(
    path: list[tuple[int, int]],
    ref_data: Dict[str, Any],
    qry_data: Dict[str, Any],
    score_scale: float,
    hint_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    phase_acc: Dict[int, Dict[str, Any]] = {}
    substage_acc: Dict[tuple[int, str], Dict[str, Any]] = {}
    part_acc: Dict[str, Dict[str, float]] = {}
    ref_weights = ref_data.get("feature_confidence_weights")
    qry_weights = qry_data.get("feature_confidence_weights")

    for i, j in path:
        phase_id = int(ref_data["phase_ids"][i])
        phase = get_phase_definition(phase_id, rule_config=ref_data.get("rule_config"))
        local_cost = _confidence_weighted_cost(
            ref_data["features"][i],
            qry_data["features"][j],
            ref_weights[i] if ref_weights is not None else None,
            qry_weights[j] if qry_weights is not None else None,
        )
        phase_row = phase_acc.setdefault(
            phase_id,
            {
                "phase_id": phase_id,
                "phase_name": phase.display_name,
                "cue": phase.cue,
                "distance_sum": 0.0,
                "alignment_count": 0,
                "ref_indices": [],
                "qry_indices": [],
            },
        )
        phase_row["distance_sum"] += local_cost
        phase_row["alignment_count"] += 1
        phase_row["ref_indices"].append(int(i))
        phase_row["qry_indices"].append(int(j))

        substage_key = str(ref_data["substage_keys"][i]) if ref_data.get("substage_keys") is not None else ""
        substage_name = str(ref_data["substage_names"][i]) if ref_data.get("substage_names") is not None else ""
        substage_cue = str(ref_data["substage_cues"][i]) if ref_data.get("substage_cues") is not None else ""
        if substage_key:
            substage_row = substage_acc.setdefault(
                (phase_id, substage_key),
                {
                    "phase_id": phase_id,
                    "phase_name": phase.display_name,
                    "substage_key": substage_key,
                    "substage_name": substage_name,
                    "cue": substage_cue,
                    "distance_sum": 0.0,
                    "alignment_count": 0,
                    "ref_indices": [],
                    "qry_indices": [],
                },
            )
            substage_row["distance_sum"] += local_cost
            substage_row["alignment_count"] += 1
            substage_row["ref_indices"].append(int(i))
            substage_row["qry_indices"].append(int(j))

        p_err = part_errors(
            ref_pts=ref_data["points"][i],
            qry_pts=qry_data["points"][j],
            ref_angles=ref_data["angle_features"][i] if ref_data.get("angle_features") is not None else None,
            qry_angles=qry_data["angle_features"][j] if qry_data.get("angle_features") is not None else None,
            phase_id=phase_id,
            substage_key=substage_key,
            rule_config=ref_data.get("rule_config"),
        )
        for part, info in p_err.items():
            row = part_acc.setdefault(
                part,
                {"part_error": 0.0, "point_error": 0.0, "angle_error": 0.0, "alignment_count": 0},
            )
            row["part_error"] += float(info["score"])
            row["point_error"] += float(info["point_error"])
            row["angle_error"] += float(info["angle_error"])
            row["alignment_count"] += 1

    phase_scores: list[dict[str, Any]] = []
    for row in sorted(phase_acc.values(), key=lambda item: item["phase_id"]):
        count = max(1, int(row["alignment_count"]))
        norm_dist = float(row["distance_sum"] / count)
        ref_indices = row.pop("ref_indices")
        qry_indices = row.pop("qry_indices")
        local_norm_dist = _local_normalized_dtw(ref_data, qry_data, ref_indices, qry_indices)
        if local_norm_dist is not None:
            norm_dist = local_norm_dist
        phase_score_scale = _score_scale_for_phase(int(row["phase_id"]), score_scale)
        phase_scores.append(
            {
                "phase_id": int(row["phase_id"]),
                "phase_name": str(row["phase_name"]),
                "cue": str(row["cue"]),
                "alignment_count": count,
                "normalized_distance": norm_dist,
                "score_0_100": _adaptive_distance_score(norm_dist, phase_score_scale, int(row["phase_id"])),
                "score_scale": phase_score_scale,
                "reference_mean_confidence": _mean_point_confidence(ref_data, ref_indices),
                "query_mean_confidence": _mean_point_confidence(qry_data, qry_indices),
                "reference_start_time_s": float(ref_data["time_s"][min(ref_indices)]),
                "reference_end_time_s": float(ref_data["time_s"][max(ref_indices)]),
                "query_start_time_s": float(qry_data["time_s"][min(qry_indices)]),
                "query_end_time_s": float(qry_data["time_s"][max(qry_indices)]),
            }
        )

    substage_scores: list[dict[str, Any]] = []
    for row in sorted(substage_acc.values(), key=lambda item: (item["phase_id"], min(item["ref_indices"]))):
        count = max(1, int(row["alignment_count"]))
        norm_dist = float(row["distance_sum"] / count)
        ref_indices = row.pop("ref_indices")
        qry_indices = row.pop("qry_indices")
        local_norm_dist = _local_normalized_dtw(ref_data, qry_data, ref_indices, qry_indices)
        if local_norm_dist is not None:
            norm_dist = local_norm_dist
        substage_score_scale = _score_scale_for_phase(int(row["phase_id"]), score_scale)
        if int(row["phase_id"]) == 6 and str(row["substage_key"]) == "bend_touch":
            bend_touch_dist = _bend_touch_robust_distance(path, ref_data, qry_data, ref_indices, qry_indices)
            if bend_touch_dist is not None:
                norm_dist = bend_touch_dist
        substage_scores.append(
            {
                "phase_id": int(row["phase_id"]),
                "phase_name": str(row["phase_name"]),
                "substage_key": str(row["substage_key"]),
                "substage_name": str(row["substage_name"]),
                "cue": str(row["cue"]),
                "alignment_count": count,
                "normalized_distance": norm_dist,
                "score_0_100": _adaptive_distance_score(norm_dist, substage_score_scale, int(row["phase_id"])),
                "score_scale": substage_score_scale,
                "reference_mean_confidence": _mean_point_confidence(ref_data, ref_indices),
                "query_mean_confidence": _mean_point_confidence(qry_data, qry_indices),
                "reference_start_time_s": float(ref_data["time_s"][min(ref_indices)]),
                "reference_end_time_s": float(ref_data["time_s"][max(ref_indices)]),
                "query_start_time_s": float(qry_data["time_s"][min(qry_indices)]),
                "query_end_time_s": float(qry_data["time_s"][max(qry_indices)]),
            }
        )

    base_part_score_scale = max(0.05, float(hint_threshold) * 3.0)
    part_scores: list[dict[str, Any]] = []
    for part, row in part_acc.items():
        count = max(1, int(row["alignment_count"]))
        part_error = float(row["part_error"] / count)
        part_score_scale = base_part_score_scale * float(PART_SCORE_SCALE_MULTIPLIER.get(part, 1.0))
        part_scores.append(
            {
                "part": part,
                "part_name": PART_SCORE_NAMES.get(part, part),
                "alignment_count": count,
                "part_error": part_error,
                "point_error": float(row["point_error"] / count),
                "angle_error": float(row["angle_error"] / count),
                "score_0_100": distance_to_score(part_error, part_score_scale),
                "score_scale": part_score_scale,
            }
        )
    part_scores.sort(key=lambda item: item["score_0_100"])
    return phase_scores, substage_scores, part_scores


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
    timing_events: Optional[list[Dict[str, Any]]] = None,
    output_frame_stride: int = 1,
) -> Dict[str, Any]:
    """汇总离线评估结果，并按需生成 JSON、DTW 图和对比视频。"""

    finalize_started = time.perf_counter()
    timing_stages: list[Dict[str, Any]] = list(timing_events or [])

    def record_stage(stage: str, started_at: float, **extra: Any) -> None:
        row: Dict[str, Any] = {"stage": stage, "elapsed_s": round(time.perf_counter() - started_at, 3)}
        row.update(extra)
        timing_stages.append(row)

    stage_started = time.perf_counter()
    norm_dist = dist / max(1, len(path))
    global_dtw_score = distance_to_score(norm_dist, score_scale)

    hints, local_error = build_feedback(
        path=path,
        ref_points=ref_data["points"],
        qry_points=qry_data["points"],
        hint_threshold=hint_threshold,
        hint_min_interval=hint_min_interval,
        max_hints=max_hints,
        ref_phase_ids=ref_data["phase_ids"],
        qry_phase_ids=qry_data["phase_ids"],
        ref_substage_keys=ref_data.get("substage_keys"),
        ref_substage_names=ref_data.get("substage_names"),
        ref_substage_cues=ref_data.get("substage_cues"),
        ref_angles=ref_data["angle_features"],
        qry_angles=qry_data["angle_features"],
        rule_config=ref_data.get("rule_config"),
    )

    for hint in hints:
        q_idx = int(hint["query_index"])
        hint["query_frame"] = int(qry_data["frame_indices"][q_idx])
        hint["query_time_s"] = float(qry_data["time_s"][q_idx])
        hint["ref_time_s"] = float(ref_data["time_s"][int(hint["ref_index"])])

    phase_scores, substage_scores, part_scores = build_score_breakdowns(
        path=path,
        ref_data=ref_data,
        qry_data=qry_data,
        score_scale=score_scale,
        hint_threshold=hint_threshold,
    )
    score = _weighted_phase_score(phase_scores, fallback=global_dtw_score)
    record_stage("score_breakdown_and_hints", stage_started, hint_count=int(len(hints)))

    ensure_parent_dir(out_json)
    if progress_callback:
        progress_callback(55, "正在生成分阶段 DTW 图")
    stage_started = time.perf_counter()
    phase_plot_dir = os.path.join(os.path.dirname(os.path.abspath(out_plot)), "phase_plots")
    phase_plots = save_phase_plots(
        ref_data=ref_data,
        qry_data=qry_data,
        path=path,
        hints=hints,
        out_dir=phase_plot_dir,
        score_scale=score_scale,
    )
    record_stage("phase_plot_export", stage_started, plot_count=int(len(phase_plots)))
    result = {
        "mode": "baduanjin_pose_dtw_weighted",
        "reference_video": ref_video,
        "query_video": query_video,
        "feature": "pose33_xy_normalized_plus_joint_angles_phase_weighted",
        "rule_config": ref_data.get("rule_config") or default_baduanjin_rule_config(),
        "reference_length": int(ref_data["features"].shape[0]),
        "query_length": int(qry_data["features"].shape[0]),
        "reference_frame_stride": int(ref_data.get("frame_stride", 1)),
        "query_frame_stride": int(qry_data.get("frame_stride", 1)),
        "dtw_distance": float(dist),
        "alignment_path_length": int(len(path)),
        "normalized_dtw_distance": float(norm_dist),
        "score_0_100": float(score),
        "global_dtw_score_0_100": float(global_dtw_score),
        "score_method": "phase_local_dtw_weighted_average",
        "score_scale": float(score_scale),
        "hint_threshold": float(hint_threshold),
        "hint_min_interval": int(hint_min_interval),
        "max_hints": int(max_hints),
        "hint_count": int(len(hints)),
        "reference_phases": ref_data["phase_rows"],
        "query_phases": qry_data["phase_rows"],
        "reference_substages": ref_data.get("substage_rows", []),
        "query_substages": qry_data.get("substage_rows", []),
        "phase_plots": phase_plots,
        "phase_scores": phase_scores,
        "substage_scores": substage_scores,
        "part_scores": part_scores,
        "pose_quality": {
            "reference": ref_data.get("pose_quality", {}),
            "query": qry_data.get("pose_quality", {}),
        },
        "hints": hints,
    }
    if progress_callback:
        progress_callback(62, "正在生成整体 DTW 曲线")
    stage_started = time.perf_counter()
    save_plot(
        ref_data=ref_data,
        qry_data=qry_data,
        path=path,
        hints=hints,
        out_png=out_plot,
        score=score,
        norm_dist=norm_dist,
    )
    record_stage("overall_plot_export", stage_started)

    frame_hint_map: Dict[int, str] = {}
    for hint in hints:
        frame_hint_map[int(hint["query_frame"])] = str(hint["message"])

    frame_error_map: Dict[int, float] = {}
    for idx, frame_id in enumerate(qry_data["frame_indices"]):
        err = float(local_error[idx])
        if np.isfinite(err):
            frame_error_map[int(frame_id)] = err

    frame_pose_map: Dict[int, np.ndarray] = {}
    frame_confidence_map: Dict[int, np.ndarray] = {}
    for idx, frame_id in enumerate(qry_data["frame_indices"]):
        frame_pose_map[int(frame_id)] = qry_data["raw_points"][idx]
        if qry_data.get("point_confidence") is not None and len(qry_data["point_confidence"]) > idx:
            frame_confidence_map[int(frame_id)] = qry_data["point_confidence"][idx]

    ref_pose_map: Dict[int, np.ndarray] = {}
    ref_confidence_map: Dict[int, np.ndarray] = {}
    for idx, frame_id in enumerate(ref_data["frame_indices"]):
        ref_pose_map[int(frame_id)] = ref_data["raw_points"][idx]
        if ref_data.get("point_confidence") is not None and len(ref_data["point_confidence"]) > idx:
            ref_confidence_map[int(frame_id)] = ref_data["point_confidence"][idx]

    alignment_map = build_query_alignment_map(
        path=path,
        ref_frame_indices=ref_data["frame_indices"],
        qry_frame_indices=qry_data["frame_indices"],
    )
    frame_phase_map, frame_cue_map = _phase_maps_for_query(alignment_map=alignment_map, ref_data=ref_data)

    if out_video or preview:
        if progress_callback:
            progress_callback(72, "正在渲染对比视频")
        stage_started = time.perf_counter()
        render_feedback_video(
            ref_video=ref_video,
            query_video=query_video,
            output_video=out_video,
            score=score,
            frame_hint_map=frame_hint_map,
            frame_error_map=frame_error_map,
            frame_pose_map=frame_pose_map,
            frame_confidence_map=frame_confidence_map,
            ref_pose_map=ref_pose_map,
            ref_confidence_map=ref_confidence_map,
            alignment_map=alignment_map,
            frame_phase_map=frame_phase_map,
            frame_cue_map=frame_cue_map,
            preview=preview,
            progress_callback=progress_callback,
            progress_range=(72, 92),
            compare_panel_height=540,
            output_frame_stride=max(1, int(output_frame_stride)),
        )
        record_stage(
            "feedback_video_render",
            stage_started,
            enabled=bool(out_video),
            preview=bool(preview),
            output_frame_stride=max(1, int(output_frame_stride)),
        )
    elif progress_callback:
        progress_callback(92, "评估结果已生成")

    result["runtime_profile"] = {
        "stages": timing_stages,
        "total_recorded_s": round(sum(float(item.get("elapsed_s", 0.0)) for item in timing_stages), 3),
        "finalize_outputs_elapsed_s": round(time.perf_counter() - finalize_started, 3),
        "includes_video": bool(out_video or preview),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

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
    timing_stages: list[Dict[str, Any]] = []
    stage_started = time.perf_counter()
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
    timing_stages.append(
        {
            "stage": "reference_pose_extract",
            "elapsed_s": round(time.perf_counter() - stage_started, 3),
            "sequence_length": int(ref_data["features"].shape[0]),
        }
    )
    if progress_callback:
        progress_callback(25, "正在提取待测视频姿态")
    stage_started = time.perf_counter()
    qry_data = extract_pose_sequence(
        video_path=query_video,
        task_model=task_model,
        num_poses=max(1, num_poses),
        smooth_window=max(1, smooth_window),
        frame_stride=max(1, frame_stride),
        preview=preview,
        preview_title="Query Pose Preview",
    )
    timing_stages.append(
        {
            "stage": "query_pose_extract",
            "elapsed_s": round(time.perf_counter() - stage_started, 3),
            "sequence_length": int(qry_data["features"].shape[0]),
        }
    )
    ref_data = _ensure_baduanjin_features(ref_data)
    qry_data = _apply_template_feature_stats(qry_data, ref_data)
    qry_data = _ensure_baduanjin_features(qry_data)
    if progress_callback:
        progress_callback(42, "正在执行 DTW 对齐")
    stage_started = time.perf_counter()
    dist, path = dtw_distance_multidim(
        ref_data["features"],
        qry_data["features"],
        a_weights=ref_data.get("feature_confidence_weights"),
        b_weights=qry_data.get("feature_confidence_weights"),
    )
    timing_stages.append(
        {
            "stage": "dtw_alignment",
            "elapsed_s": round(time.perf_counter() - stage_started, 3),
            "alignment_path_length": int(len(path)),
        }
    )
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
        timing_events=timing_stages,
        output_frame_stride=max(1, int(frame_stride)),
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
    query_task_model: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """直接加载模板并处理测试视频，适合重复评估。"""

    # 直接加载模板并只处理待测视频，适合重复评分场景。
    timing_stages: list[Dict[str, Any]] = []
    stage_started = time.perf_counter()
    ref_data = load_pose_template(template_path)
    timing_stages.append(
        {
            "stage": "template_load",
            "elapsed_s": round(time.perf_counter() - stage_started, 3),
            "sequence_length": int(ref_data["features"].shape[0]),
        }
    )
    if progress_callback:
        progress_callback(12, "模板已加载，正在提取待测视频姿态")
    stage_started = time.perf_counter()
    qry_data = extract_pose_sequence(
        video_path=query_video,
        task_model=query_task_model or ref_data["task_model"],
        num_poses=max(1, ref_data["num_poses"]),
        smooth_window=max(1, int(query_smooth_window or ref_data["smooth_window"])),
        frame_stride=max(1, int(query_frame_stride or ref_data.get("frame_stride", 1))),
        preview=preview,
        preview_title="Query Pose Preview",
    )
    timing_stages.append(
        {
            "stage": "query_pose_extract",
            "elapsed_s": round(time.perf_counter() - stage_started, 3),
            "sequence_length": int(qry_data["features"].shape[0]),
        }
    )
    stage_started = time.perf_counter()
    qry_data = _apply_template_feature_stats(qry_data, ref_data)
    qry_data = _ensure_baduanjin_features(qry_data)
    if progress_callback:
        progress_callback(42, "正在执行 DTW 对齐")
    dist, path = dtw_distance_multidim(
        ref_data["features"],
        qry_data["features"],
        a_weights=ref_data.get("feature_confidence_weights"),
        b_weights=qry_data.get("feature_confidence_weights"),
    )
    if progress_callback:
        progress_callback(50, "正在生成评分结果")
    timing_stages.append(
        {
            "stage": "dtw_alignment",
            "elapsed_s": round(time.perf_counter() - stage_started, 3),
            "alignment_path_length": int(len(path)),
        }
    )
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
        timing_events=timing_stages,
        output_frame_stride=max(1, int(query_frame_stride or ref_data.get("frame_stride", 1))),
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
        ref_data = load_pose_template(template_path)
        return ref_data
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


def _is_video_file_source(source: Union[int, str]) -> bool:
    if isinstance(source, int):
        return False
    text = str(source).strip()
    if text.isdigit():
        return False
    return Path(text).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}


def _score_live_pose_sequence(
    ref_data: Dict[str, Any],
    points_seq: list[np.ndarray],
    raw_points_seq: list[np.ndarray],
    point_confidence_seq: Optional[list[np.ndarray]],
    frame_indices: list[int],
    time_s: list[float],
    fps: float,
    score_scale: float,
    hint_threshold: float,
    hint_min_interval: int,
    max_hints: int,
    processed_frames: Optional[int] = None,
    skipped_frames: int = 0,
    final_score_stride: int = 3,
) -> Dict[str, Any]:
    """实时会话结束后，对整段采集姿态执行与离线一致的全局 DTW 评分。"""

    if len(points_seq) < 10:
        return {
            "matched_frames": int(len(points_seq)),
            "avg_score_0_100": 0.0,
            "score_0_100": 0.0,
            "normalized_dtw_distance": 0.0,
            "dtw_distance": 0.0,
            "alignment_path_length": 0,
            "hint_count": 0,
            "hints": [],
            "phase_scores": [],
            "substage_scores": [],
            "part_scores": [],
            "pose_quality": {
                "reference": ref_data.get("pose_quality", {}),
                "query": pose_quality_summary(
                    np.asarray(point_confidence_seq or [], dtype=np.float32),
                    processed_frames=int(processed_frames or len(points_seq)),
                    skipped_frames=int(skipped_frames),
                ),
            },
        }

    max_final_score_frames = 600
    score_stride = max(1, int(final_score_stride))
    if len(points_seq) > max_final_score_frames:
        score_stride = max(score_stride, int(np.ceil(len(points_seq) / max_final_score_frames)))
    if score_stride > 1:
        sample_idx = np.arange(0, len(points_seq), score_stride, dtype=np.int32)
        if sample_idx[-1] != len(points_seq) - 1:
            sample_idx = np.append(sample_idx, len(points_seq) - 1)
        points_seq = [points_seq[int(idx)] for idx in sample_idx]
        raw_points_seq = [raw_points_seq[int(idx)] for idx in sample_idx]
        frame_indices = [frame_indices[int(idx)] for idx in sample_idx]
        time_s = [time_s[int(idx)] for idx in sample_idx]
        if point_confidence_seq and len(point_confidence_seq) >= int(sample_idx[-1]) + 1:
            point_confidence_seq = [point_confidence_seq[int(idx)] for idx in sample_idx]

    points = np.asarray(points_seq, dtype=np.float32)
    if point_confidence_seq and len(point_confidence_seq) == len(points_seq):
        point_confidence = np.asarray(point_confidence_seq, dtype=np.float32)
    else:
        point_confidence = np.ones((points.shape[0], 33), dtype=np.float32)
    qry_data: Dict[str, Any] = {
        "points": points,
        "raw_points": np.asarray(raw_points_seq, dtype=np.float32),
        "point_confidence": point_confidence,
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "time_s": np.asarray(time_s, dtype=np.float32),
        "fps": float(fps),
        "frame_stride": 1,
        "final_score_stride": int(score_stride),
        "pose_quality": pose_quality_summary(
            point_confidence,
            processed_frames=int(processed_frames or len(points_seq)),
            skipped_frames=int(skipped_frames),
        ),
    }
    feature_bundle = build_pose_feature_bundle(qry_data["points"], point_confidence=qry_data["point_confidence"])
    qry_data["angle_features"] = feature_bundle["angle_features"]
    qry_data["combined_features_raw"] = feature_bundle["combined_features_raw"]
    qry_data["feature_confidence_weights"] = feature_bundle["feature_confidence_weights"]
    qry_data = _apply_template_feature_stats(qry_data, ref_data)
    qry_data = _ensure_baduanjin_features(qry_data)

    dist, path = dtw_distance_multidim(
        ref_data["features"],
        qry_data["features"],
        a_weights=ref_data.get("feature_confidence_weights"),
        b_weights=qry_data.get("feature_confidence_weights"),
    )
    norm_dist = dist / max(1, len(path))
    global_dtw_score = distance_to_score(norm_dist, score_scale)
    hints, _ = build_feedback(
        path=path,
        ref_points=ref_data["points"],
        qry_points=qry_data["points"],
        hint_threshold=hint_threshold,
        hint_min_interval=hint_min_interval,
        max_hints=max_hints,
        ref_phase_ids=ref_data["phase_ids"],
        qry_phase_ids=qry_data["phase_ids"],
        ref_substage_keys=ref_data.get("substage_keys"),
        ref_substage_names=ref_data.get("substage_names"),
        ref_substage_cues=ref_data.get("substage_cues"),
        ref_angles=ref_data["angle_features"],
        qry_angles=qry_data["angle_features"],
        rule_config=ref_data.get("rule_config"),
    )
    for hint in hints:
        q_idx = int(hint["query_index"])
        hint["query_frame"] = int(qry_data["frame_indices"][q_idx])
        hint["query_time_s"] = float(qry_data["time_s"][q_idx])
        hint["ref_time_s"] = float(ref_data["time_s"][int(hint["ref_index"])])
    phase_scores, substage_scores, part_scores = build_score_breakdowns(
        path=path,
        ref_data=ref_data,
        qry_data=qry_data,
        score_scale=score_scale,
        hint_threshold=hint_threshold,
    )
    score = _weighted_phase_score(phase_scores, fallback=global_dtw_score)

    return {
        "matched_frames": int(qry_data["features"].shape[0]),
        "avg_score_0_100": float(score),
        "score_0_100": float(score),
        "global_dtw_score_0_100": float(global_dtw_score),
        "score_method": "phase_local_dtw_weighted_average",
        "normalized_dtw_distance": float(norm_dist),
        "dtw_distance": float(dist),
        "alignment_path_length": int(len(path)),
        "hint_count": int(len(hints)),
        "hints": hints,
        "phase_scores": phase_scores,
        "substage_scores": substage_scores,
        "part_scores": part_scores,
        "pose_quality": {
            "reference": ref_data.get("pose_quality", {}),
            "query": qry_data.get("pose_quality", {}),
        },
    }


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
    camera_task_model: Optional[str] = None,
    camera_width: Optional[int] = None,
    camera_height: Optional[int] = None,
    camera_mirror: bool = True,
    out_json: Optional[str] = None,
    out_video: Optional[str] = None,
    out_error_frames_dir: Optional[str] = None,
    preview: bool = True,
    max_frames: Optional[int] = None,
    stop_checker: Optional[Callable[[], bool]] = None,
    frame_callback: Optional[Callable[[np.ndarray], None]] = None,
    state_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    realtime_playback: bool = True,
    final_score_stride: int = 1,
) -> Dict[str, Any]:
    """运行实时跟练主循环。

    关键点：
    1. 参考动作来自模板或标准视频。
    2. 输入源可以是摄像头，也可以是本地视频模拟摄像头。
    3. 实时模式优先使用异步姿态检测，只消费最新结果，降低卡顿感。
    """

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
    live_task_model = camera_task_model or ref_data["task_model"]

    cap = cv2.VideoCapture(_resolve_capture_source(camera_source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open capture source: {camera_source}")
    if camera_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_width))
    if camera_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_height))

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        fps = float(raw_fps)
    except (TypeError, ValueError):
        fps = 30.0
    if not np.isfinite(fps) or fps <= 0 or fps > 120:
        fps = 30.0
    total_capture_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    record_compare_video = bool(out_video)
    record_error_frames = bool(out_error_frames_dir)
    ref_cap = cv2.VideoCapture(ref_video_path) if (record_compare_video or record_error_frames) and ref_video_path else None
    if ref_cap is not None and not ref_cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Cannot open video: {ref_video_path}")

    writer = None
    writer_kind: Optional[str] = None
    actual_out_video = out_video
    if out_video:
        ensure_parent_dir(out_video)
    if out_error_frames_dir:
        Path(out_error_frames_dir).mkdir(parents=True, exist_ok=True)

    current_ref_seq_idx = 0
    current_ref_frame_idx = -1
    current_ref_frame: Optional[np.ndarray] = None
    pose_window: Deque[np.ndarray] = deque(maxlen=max(3, smooth_window * 3))
    prev_center = None
    frame_idx = 0
    active_hint = ""
    active_hint_left = 0
    keep_frames = max(1, int(round(fps * 0.8)))
    live_hints: list[dict[str, Any]] = []
    error_frames: list[dict[str, Any]] = []
    query_points_seq: list[np.ndarray] = []
    query_raw_points_seq: list[np.ndarray] = []
    query_point_confidence_seq: list[np.ndarray] = []
    query_frame_indices: list[int] = []
    query_time_s: list[float] = []
    saved_error_keys: set[str] = set()
    processed_idx = 0
    invalid_pose_frames = 0
    last_hint_processed_idx = -10**9
    latest_phase_name = ""
    latest_phase_cue = ""
    latest_substage_key = ""
    latest_substage_name = ""
    latest_substage_cue = ""
    latest_local_err = float("nan")
    latest_part = ""
    ref_times = np.asarray(ref_data.get("time_s"), dtype=np.float32)
    ref_length = int(ref_data["features"].shape[0])
    session_start_time = time.perf_counter()
    use_video_timeline = bool(realtime_playback and _is_video_file_source(camera_source))
    use_async_live_stream = True
    live_score_stride = 3
    preview_update_stride = 1
    preview_min_interval_s = 0.12
    state_update_interval_s = 0.50
    last_preview_push_at = -1.0
    last_state_push_at = -1.0
    preview_sent_count = 0
    state_sent_count = 0
    skipped_capture_frames = 0
    last_timestamp_ms = -1

    detector_context = (
        LiveStreamPoseDetector(task_model=live_task_model, num_poses=ref_data["num_poses"])
        if use_async_live_stream
        else create_pose_landmarker(task_model=live_task_model, num_poses=ref_data["num_poses"])
    )
    with detector_context as landmarker:
        while True:
            if stop_checker is not None and stop_checker():
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            if use_video_timeline:
                # 用本地视频模拟摄像头时，要尽量贴近真实实时流：
                # 既不能比墙钟时间更快，也要主动跳过过期帧，避免越积越慢。
                wall_elapsed_s = time.perf_counter() - session_start_time
                target_capture_frame = int(wall_elapsed_s * fps)
                if total_capture_frames > 0:
                    target_capture_frame = min(target_capture_frame, total_capture_frames - 1)
                while frame_idx + capture_stride < target_capture_frame:
                    if not cap.grab():
                        break
                    skipped_capture_frames += 1
                    frame_idx += 1
            ok, frame = cap.read()
            if not ok:
                break
            if use_video_timeline:
                frame_time_s = float(frame_idx / fps)
                wall_elapsed_s = time.perf_counter() - session_start_time
                if frame_time_s > wall_elapsed_s:
                    time.sleep(min(0.05, frame_time_s - wall_elapsed_s))
            if frame_idx % capture_stride != 0:
                frame_idx += 1
                continue
            if camera_width or camera_height:
                target_width = int(camera_width or frame.shape[1])
                target_height = int(camera_height or frame.shape[0])
                if frame.shape[1] != target_width or frame.shape[0] != target_height:
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            if camera_mirror:
                frame = cv2.flip(frame, 1)

            query_view = frame.copy()
            query_overlay_view: Optional[np.ndarray] = None
            current_frame_idx = int(frame_idx)
            current_time_s = float(frame_idx / fps) if use_video_timeline else max(0.0, time.perf_counter() - session_start_time)
            if use_async_live_stream:
                # 异步模式下持续提交新帧，只消费最近一次完成的推理结果。
                timestamp_ms = int(round(current_time_s * 1000.0))
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms
                landmarker.submit_frame(
                    frame=frame,
                    timestamp_ms=timestamp_ms,
                    frame_idx=current_frame_idx,
                    frame_time_s=current_time_s,
                )
                latest_result = landmarker.pop_latest_result()
                if latest_result is None:
                    wall_elapsed_s = max(time.perf_counter() - session_start_time, 1e-6)
                    completed_frames = processed_idx + 1
                    if frame_callback is not None and (
                        preview_sent_count == 0
                        or (
                            completed_frames % preview_update_stride == 0
                            and (wall_elapsed_s - last_preview_push_at) >= preview_min_interval_s
                        )
                    ):
                        preview_sent_count += 1
                        last_preview_push_at = wall_elapsed_s
                        frame_callback(query_view)
                    if state_callback is not None and (
                        state_sent_count == 0 or (wall_elapsed_s - last_state_push_at) >= state_update_interval_s
                    ):
                        state_sent_count += 1
                        last_state_push_at = wall_elapsed_s
                        state_callback(
                            {
                                "query_frame": int(current_frame_idx),
                                "query_time_s": float(current_time_s),
                                "ref_frame": int(ref_data["frame_indices"][current_ref_seq_idx]) if ref_length else 0,
                                "ref_time_s": float(ref_data["time_s"][current_ref_seq_idx]) if ref_times.size else 0.0,
                                "phase_name": latest_phase_name,
                                "phase_cue": latest_phase_cue,
                                "substage_key": latest_substage_key,
                                "substage_name": latest_substage_name,
                                "substage_cue": latest_substage_cue,
                                "part": "pose",
                                "message": "正在识别人体姿态，请保持全身入镜。",
                                "local_error": latest_local_err,
                                "confidence": {
                                    "mean": 0.0,
                                    "min": 0.0,
                                    "valid_points": 0,
                                    "total_points": 33,
                                },
                                "perf": {
                                    "processed_fps": float(completed_frames / wall_elapsed_s),
                                    "preview_fps": float(preview_sent_count / wall_elapsed_s),
                                    "state_fps": float(state_sent_count / wall_elapsed_s),
                                    "timeline_lag_ms": 0.0,
                                    "skipped_frames": int(skipped_capture_frames),
                                },
                            }
                        )
                    frame_idx += 1
                    continue
                query_view = latest_result["frame"]
                current_frame_idx = int(latest_result["frame_idx"])
                current_time_s = float(latest_result["frame_time_s"])
                norm_pts = latest_result["norm_pts"]
                raw_pts = latest_result["raw_pts"]
                point_confidence = latest_result["point_confidence"]
            else:
                timestamp_ms = int((frame_idx * 1000.0) / fps)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms
                norm_pts, raw_pts, point_confidence, prev_center = detect_pose_in_frame(
                    landmarker=landmarker,
                    frame=frame,
                    timestamp_ms=timestamp_ms,
                    prev_center=prev_center,
                )

            if raw_pts is None or norm_pts is None or point_confidence is None:
                invalid_pose_frames += 1
                completed_frames = processed_idx + 1
                wall_elapsed_s = max(time.perf_counter() - session_start_time, 1e-6)
                no_pose_message = "未检测到完整人体姿态，请站在画面中央，并保持全身入镜。"
                if state_callback is not None and (
                    state_sent_count == 0 or (wall_elapsed_s - last_state_push_at) >= state_update_interval_s
                ):
                    state_sent_count += 1
                    last_state_push_at = wall_elapsed_s
                    state_callback(
                        {
                            "query_frame": int(current_frame_idx),
                            "query_time_s": float(current_time_s),
                            "ref_frame": int(ref_data["frame_indices"][current_ref_seq_idx]) if ref_length else 0,
                            "ref_time_s": float(ref_data["time_s"][current_ref_seq_idx]) if ref_times.size else 0.0,
                            "phase_name": latest_phase_name,
                            "phase_cue": latest_phase_cue,
                            "substage_key": latest_substage_key,
                            "substage_name": latest_substage_name,
                            "substage_cue": latest_substage_cue,
                            "part": "pose",
                            "message": no_pose_message,
                            "local_error": latest_local_err,
                            "confidence": {
                                "mean": 0.0,
                                "min": 0.0,
                                "valid_points": 0,
                                "total_points": 33,
                            },
                            "perf": {
                                "processed_fps": float(completed_frames / wall_elapsed_s),
                                "preview_fps": float(preview_sent_count / wall_elapsed_s),
                                "state_fps": float(state_sent_count / wall_elapsed_s),
                                "timeline_lag_ms": 0.0,
                                "skipped_frames": int(skipped_capture_frames),
                            },
                        }
                    )
                if frame_callback is not None and (
                    preview_sent_count == 0
                    or (
                        completed_frames % preview_update_stride == 0
                        and (wall_elapsed_s - last_preview_push_at) >= preview_min_interval_s
                    )
                ):
                    preview_sent_count += 1
                    last_preview_push_at = wall_elapsed_s
                    frame_callback(query_view)
                frame_idx += 1
                continue

            if raw_pts is not None and norm_pts is not None and point_confidence is not None:
                accepted_hint: Optional[dict[str, Any]] = None
                pose_window.append(norm_pts)
                current_points, _, current_feature, current_angles = build_current_feature(
                    recent_points=pose_window,
                    smooth_window=smooth_window,
                    feature_mean=ref_data["feature_mean"],
                    feature_std=ref_data["feature_std"],
                )
                query_points_seq.append(current_points.copy())
                query_raw_points_seq.append(raw_pts.copy())
                query_point_confidence_seq.append(point_confidence.copy())
                query_frame_indices.append(int(current_frame_idx))
                query_time_s.append(float(current_time_s))

                should_score_frame = (
                    processed_idx == 0
                    or processed_idx % live_score_stride == 0
                    or not latest_phase_name
                    or active_hint_left <= 0
                )
                if should_score_frame:
                    elapsed_s = current_time_s if use_video_timeline else time.perf_counter() - session_start_time
                    if ref_times.size:
                        target_time_s = min(float(elapsed_s), float(ref_times[-1]))
                        target_ref_seq_idx = int(np.searchsorted(ref_times, target_time_s, side="left"))
                        target_ref_seq_idx = int(np.clip(target_ref_seq_idx, 0, ref_length - 1))
                    else:
                        target_ref_seq_idx = min(current_ref_seq_idx + 1, ref_length - 1)
                        target_time_s = float(target_ref_seq_idx)

                    search_radius = max(3, int(ref_search_window))
                    search_start = max(0, target_ref_seq_idx - search_radius)
                    search_end = min(ref_length, target_ref_seq_idx + search_radius + 1)
                    candidate_features = ref_data["features"][search_start:search_end]
                    candidate_phase_ids = np.asarray(ref_data["phase_ids"][search_start:search_end], dtype=np.int32)
                    if candidate_features.size:
                        candidate_dists = np.empty(candidate_features.shape[0], dtype=np.float32)
                        weighted_current_cache: dict[int, np.ndarray] = {}
                        for phase_id in np.unique(candidate_phase_ids):
                            phase_int = int(phase_id)
                            weighted_current = weighted_current_cache.get(phase_int)
                            if weighted_current is None:
                                weighted_current = weight_single_feature(
                                    current_feature,
                                    phase_int,
                                    rule_config=ref_data.get("rule_config"),
                                )
                                weighted_current_cache[phase_int] = weighted_current
                            mask = candidate_phase_ids == phase_int
                            candidate_dists[mask] = np.linalg.norm(candidate_features[mask] - weighted_current, axis=1)
                        current_ref_seq_idx = int(search_start + int(np.argmin(candidate_dists)))
                    else:
                        current_ref_seq_idx = int(target_ref_seq_idx)

                    target_phase_id = int(ref_data["phase_ids"][current_ref_seq_idx])
                    phase = get_phase_definition(target_phase_id, rule_config=ref_data.get("rule_config"))
                    if ref_data.get("substage_keys") is not None and len(ref_data["substage_keys"]) > current_ref_seq_idx:
                        substage_key = str(ref_data["substage_keys"][current_ref_seq_idx])
                        substage = get_substage_by_key(target_phase_id, substage_key)
                        if substage is None:
                            substage = get_substage_definition(target_phase_id, 0.0)
                    else:
                        phase_times = ref_times[ref_data["phase_ids"] == target_phase_id] if ref_times.size else np.asarray([], dtype=np.float32)
                        if phase_times.size:
                            phase_start_s = float(phase_times[0])
                            phase_end_s = float(phase_times[-1])
                            phase_progress = (float(target_time_s) - phase_start_s) / max(1e-6, phase_end_s - phase_start_s)
                        else:
                            phase_progress = 0.0
                        substage = get_substage_definition(target_phase_id, phase_progress)
                    latest_phase_name = f"{phase.display_name} - {substage.name}"
                    latest_phase_cue = substage.cue
                    latest_substage_key = substage.key
                    latest_substage_name = substage.name
                    latest_substage_cue = substage.cue
                    latest_local_err = float(
                        np.mean(np.linalg.norm(current_points - ref_data["points"][current_ref_seq_idx], axis=1))
                    )
                    active_hint, _, latest_part = build_live_feedback(
                        ref_points=ref_data["points"][current_ref_seq_idx],
                        qry_points=current_points,
                        hint_threshold=hint_threshold,
                        phase_id=target_phase_id,
                        substage_key=latest_substage_key,
                        ref_angles=ref_data["angle_features"][current_ref_seq_idx],
                        qry_angles=current_angles,
                        rule_config=ref_data.get("rule_config"),
                    )
                    active_hint_left = keep_frames if active_hint else max(0, active_hint_left - live_score_stride)
                    if (
                        active_hint
                        and len(live_hints) < int(max_hints)
                        and (processed_idx - last_hint_processed_idx) >= int(hint_min_interval)
                    ):
                        accepted_hint = {
                            "query_frame": int(current_frame_idx),
                            "query_time_s": float(current_time_s),
                            "phase_name": latest_phase_name,
                            "cue": latest_phase_cue,
                            "substage_key": latest_substage_key,
                            "substage_name": latest_substage_name,
                            "substage_cue": latest_substage_cue,
                            "part": latest_part,
                            "message": active_hint,
                            "local_error": float(latest_local_err),
                            "ref_frame": int(ref_data["frame_indices"][current_ref_seq_idx]),
                            "ref_time_s": float(ref_data["time_s"][current_ref_seq_idx]) if ref_times.size else 0.0,
                        }
                        live_hints.append(accepted_hint)
                        last_hint_processed_idx = processed_idx
            else:
                invalid_pose_frames += 1
                accepted_hint = None
                draw_text_block(
                    query_view,
                    ["未检测到完整人体姿态。", "请站在画面中央，并保持全身入镜。"],
                    x=20,
                    y=18,
                )

            if ref_cap is not None and (record_compare_video or accepted_hint is not None):
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
                    ref_confidence = (
                        ref_data["point_confidence"][current_ref_seq_idx]
                        if ref_data.get("point_confidence") is not None
                        and len(ref_data["point_confidence"]) > current_ref_seq_idx
                        else None
                    )
                    draw_pose_skeleton(ref_view, ref_data["raw_points"][current_ref_seq_idx], ref_confidence)
            else:
                ref_view = np.zeros_like(query_view)

            if accepted_hint is not None and record_error_frames and out_error_frames_dir:
                error_key = f"{latest_phase_name}|{latest_part}|{active_hint}"
                if error_key in saved_error_keys:
                    accepted_hint = None
                else:
                    saved_error_keys.add(error_key)
            if accepted_hint is not None and record_error_frames and out_error_frames_dir:
                error_index = len(error_frames) + 1
                image_name = f"error_{error_index:03d}.jpg"
                image_path = str(Path(out_error_frames_dir) / image_name)
                if query_overlay_view is None:
                    query_overlay_view = query_view.copy()
                    if raw_pts is not None:
                        draw_pose_skeleton(query_overlay_view, raw_pts, point_confidence)
                error_frame = compose_error_frame(
                    ref_frame=ref_view,
                    qry_frame=query_overlay_view,
                    phase_name=latest_phase_name,
                    part=latest_part,
                    local_error=latest_local_err,
                    active_hint=active_hint,
                    query_time_s=float(current_time_s),
                )
                if cv2.imwrite(image_path, error_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
                    accepted_hint["error_frame_path"] = image_path
                    error_frames.append(
                        {
                            "image_path": image_path,
                            "query_frame": accepted_hint["query_frame"],
                            "query_time_s": accepted_hint["query_time_s"],
                            "ref_frame": accepted_hint["ref_frame"],
                            "ref_time_s": accepted_hint["ref_time_s"],
                            "phase_name": latest_phase_name,
                            "substage_key": latest_substage_key,
                            "substage_name": latest_substage_name,
                            "substage_cue": latest_substage_cue,
                            "part": latest_part,
                            "message": active_hint,
                            "local_error": float(latest_local_err),
                        }
                    )

            hint_text = active_hint if active_hint_left > 0 else ""
            if active_hint_left > 0:
                active_hint_left -= 1
            completed_frames = processed_idx + 1
            wall_elapsed_s = max(time.perf_counter() - session_start_time, 1e-6)
            timeline_lag_ms = max(0.0, (wall_elapsed_s - float(current_time_s)) * 1000.0) if use_video_timeline else 0.0
            should_push_state = state_callback is not None and (
                accepted_hint is not None
                or state_sent_count == 0
                or (wall_elapsed_s - last_state_push_at) >= state_update_interval_s
            )
            if should_push_state and state_callback is not None:
                state_sent_count += 1
                last_state_push_at = wall_elapsed_s
                state_callback(
                    {
                        "query_frame": int(current_frame_idx),
                        "query_time_s": float(current_time_s),
                        "ref_frame": int(ref_data["frame_indices"][current_ref_seq_idx]) if ref_length else 0,
                        "ref_time_s": float(ref_data["time_s"][current_ref_seq_idx]) if ref_times.size else 0.0,
                        "phase_name": latest_phase_name,
                        "phase_cue": latest_phase_cue,
                        "substage_key": latest_substage_key,
                        "substage_name": latest_substage_name,
                        "substage_cue": latest_substage_cue,
                        "part": latest_part,
                        "message": hint_text,
                        "local_error": latest_local_err,
                        "confidence": {
                            "mean": float(np.mean(point_confidence)),
                            "min": float(np.min(point_confidence)),
                            "valid_points": int(np.sum(point_confidence >= 0.40)),
                            "total_points": int(point_confidence.shape[0]),
                        },
                        "perf": {
                            "processed_fps": float(completed_frames / wall_elapsed_s),
                            "preview_fps": float(preview_sent_count / wall_elapsed_s),
                            "state_fps": float(state_sent_count / wall_elapsed_s),
                            "timeline_lag_ms": float(timeline_lag_ms),
                            "skipped_frames": int(skipped_capture_frames),
                        },
                    }
                )

            should_push_preview = frame_callback is not None and (
                accepted_hint is not None
                or preview_sent_count == 0
                or (
                    completed_frames % preview_update_stride == 0
                    and (wall_elapsed_s - last_preview_push_at) >= preview_min_interval_s
                )
            )
            if should_push_preview and frame_callback is not None:
                if query_overlay_view is None:
                    query_overlay_view = query_view.copy()
                    if raw_pts is not None:
                        draw_pose_skeleton(query_overlay_view, raw_pts, point_confidence)
                live_preview_frame = compose_live_query_frame(
                    qry_frame=query_overlay_view,
                    score=0.0,
                    current_local_err=latest_local_err,
                    active_hint=hint_text,
                    phase_name=latest_phase_name,
                    phase_cue=latest_phase_cue,
                    confidence=point_confidence,
                )
                preview_sent_count += 1
                last_preview_push_at = wall_elapsed_s
                frame_callback(live_preview_frame)

            output_frame = None
            if record_compare_video:
                if query_overlay_view is None:
                    query_overlay_view = query_view.copy()
                    if raw_pts is not None:
                        draw_pose_skeleton(query_overlay_view, raw_pts, point_confidence)
                output_frame = compose_compare_frame(
                    ref_frame=ref_view,
                    qry_frame=query_overlay_view,
                    score=0.0,
                    current_local_err=latest_local_err,
                    active_hint=hint_text,
                    align_info={
                        "ref_frame": int(ref_data["frame_indices"][current_ref_seq_idx]),
                        "qry_frame": int(current_frame_idx),
                        "ref_seq_idx": int(current_ref_seq_idx),
                        "qry_seq_idx": int(current_frame_idx),
                        "path_step": int(current_ref_seq_idx),
                        "path_total": int(ref_data["features"].shape[0]),
                    },
                    phase_name=latest_phase_name,
                    phase_cue=latest_phase_cue,
                    ref_confidence=(
                        ref_data["point_confidence"][current_ref_seq_idx]
                        if ref_data.get("point_confidence") is not None
                        and len(ref_data["point_confidence"]) > current_ref_seq_idx
                        else None
                    ),
                    qry_confidence=point_confidence,
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

    total_wall_elapsed_s = max(time.perf_counter() - session_start_time, 1e-6)
    final_query_time_s = float(query_time_s[-1]) if query_time_s else 0.0
    final_perf = {
        "processed_fps": float(processed_idx / total_wall_elapsed_s),
        "preview_fps": float(preview_sent_count / total_wall_elapsed_s),
        "state_fps": float(state_sent_count / total_wall_elapsed_s),
        "timeline_lag_ms": float(max(0.0, (total_wall_elapsed_s - final_query_time_s) * 1000.0) if use_video_timeline else 0.0),
        "skipped_frames": int(skipped_capture_frames),
        "wall_elapsed_s": float(total_wall_elapsed_s),
    }

    global_score = _score_live_pose_sequence(
        ref_data=ref_data,
        points_seq=query_points_seq,
        raw_points_seq=query_raw_points_seq,
        point_confidence_seq=query_point_confidence_seq,
        frame_indices=query_frame_indices,
        time_s=query_time_s,
        fps=fps,
        score_scale=score_scale,
        hint_threshold=hint_threshold,
        hint_min_interval=hint_min_interval,
        max_hints=max_hints,
        processed_frames=processed_idx,
        skipped_frames=invalid_pose_frames,
        final_score_stride=final_score_stride,
    )
    final_hints = global_score["hints"] or live_hints
    summary = {
        "mode": "baduanjin_camera_coach",
        "reference_video": ref_video_path,
        "camera_source": str(camera_source),
        "matched_frames": int(global_score["matched_frames"]),
        "collected_frames": int(len(query_points_seq)),
        "final_score_stride": int(max(1, final_score_stride)),
        "avg_score_0_100": float(global_score["avg_score_0_100"]),
        "score_0_100": float(global_score["score_0_100"]),
        "global_dtw_score_0_100": float(global_score.get("global_dtw_score_0_100", global_score["score_0_100"])),
        "score_method": str(global_score.get("score_method", "")),
        "normalized_dtw_distance": float(global_score["normalized_dtw_distance"]),
        "dtw_distance": float(global_score["dtw_distance"]),
        "alignment_path_length": int(global_score["alignment_path_length"]),
        "final_phase_name": latest_phase_name,
        "final_phase_cue": latest_phase_cue,
        "final_substage_key": latest_substage_key,
        "final_substage_name": latest_substage_name,
        "final_substage_cue": latest_substage_cue,
        "final_part": latest_part,
        "hint_count": int(len(final_hints)),
        "hints": final_hints,
        "live_hints": live_hints,
        "error_frame_count": int(len(error_frames)),
        "error_frames": error_frames,
        "phase_scores": global_score.get("phase_scores", []),
        "substage_scores": global_score.get("substage_scores", []),
        "part_scores": global_score.get("part_scores", []),
        "pose_quality": global_score.get("pose_quality", {}),
        "template_path": template_path,
        "output_video": actual_out_video,
        "perf": final_perf,
        "runtime_state": {
            "query_frame": int(query_frame_indices[-1]) if query_frame_indices else 0,
            "query_time_s": final_query_time_s,
            "ref_frame": int(ref_data["frame_indices"][current_ref_seq_idx]) if ref_length else 0,
            "ref_time_s": float(ref_data["time_s"][current_ref_seq_idx]) if ref_times.size else 0.0,
            "phase_name": latest_phase_name,
            "phase_cue": latest_phase_cue,
            "substage_key": latest_substage_key,
            "substage_name": latest_substage_name,
            "substage_cue": latest_substage_cue,
            "part": latest_part,
            "message": "",
            "local_error": latest_local_err,
            "perf": final_perf,
        },
    }
    if out_json:
        ensure_parent_dir(out_json)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

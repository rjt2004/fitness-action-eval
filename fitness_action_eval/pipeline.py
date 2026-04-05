import json
import os
from typing import Any, Dict, Optional

import numpy as np

from fitness_action_eval.dtw import distance_to_score, dtw_distance_multidim
from fitness_action_eval.feedback import build_feedback
from fitness_action_eval.pose import extract_pose_sequence
from fitness_action_eval.visualization import render_feedback_video, save_plot


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


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
    preview: bool = False,
) -> Dict[str, Any]:
    # 预处理标准动作视频并导出模板文件，供后续重复评分直接加载。
    ref_data = extract_pose_sequence(
        video_path=ref_video,
        task_model=task_model,
        num_poses=max(1, num_poses),
        smooth_window=max(1, smooth_window),
        preview=preview,
        preview_title="Reference Pose Preview",
    )
    ensure_parent_dir(template_path)
    np.savez_compressed(
        template_path,
        reference_video=np.asarray(ref_video),
        task_model=np.asarray(task_model),
        num_poses=np.asarray(int(num_poses), dtype=np.int32),
        smooth_window=np.asarray(int(smooth_window), dtype=np.int32),
        features=ref_data["features"],
        points=ref_data["points"],
        raw_points=ref_data["raw_points"],
        frame_indices=ref_data["frame_indices"],
        time_s=ref_data["time_s"],
        fps=np.asarray(ref_data["fps"], dtype=np.float32),
    )
    return {
        "template_path": template_path,
        "reference_video": ref_video,
        "reference_length": int(ref_data["features"].shape[0]),
    }


def load_pose_template(template_path: str) -> Dict[str, Any]:
    # 读取已导出的标准动作模板。
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    with np.load(template_path, allow_pickle=False) as data:
        return {
            "reference_video": str(data["reference_video"].item()),
            "task_model": str(data["task_model"].item()),
            "num_poses": int(data["num_poses"].item()),
            "smooth_window": int(data["smooth_window"].item()),
            "features": data["features"],
            "points": data["points"],
            "raw_points": data["raw_points"],
            "frame_indices": data["frame_indices"],
            "time_s": data["time_s"],
            "fps": float(data["fps"].item()),
        }


def finalize_scoring_outputs(
    ref_data: Dict[str, Any],
    qry_data: Dict[str, Any],
    ref_video: str,
    query_video: str,
    path: list[tuple[int, int]],
    dist: float,
    score_scale: float,
    hint_threshold: float,
    out_json: str,
    out_plot: str,
    out_video: Optional[str],
    preview: bool,
) -> Dict[str, Any]:
    norm_dist = dist / max(1, len(path))
    score = distance_to_score(norm_dist, score_scale)

    hints, local_error = build_feedback(
        path=path,
        ref_points=ref_data["points"],
        qry_points=qry_data["points"],
        hint_threshold=hint_threshold,
        hint_min_interval=8,
        max_hints=40,
    )

    for hint in hints:
        q_idx = int(hint["query_index"])
        hint["query_frame"] = int(qry_data["frame_indices"][q_idx])
        hint["query_time_s"] = float(qry_data["time_s"][q_idx])
        hint["ref_time_s"] = float(ref_data["time_s"][int(hint["ref_index"])])

    ensure_parent_dir(out_json)
    result = {
        "mode": "scheme_b_template_dtw",
        "reference_video": ref_video,
        "query_video": query_video,
        "feature": "pose33_xy_normalized",
        "reference_length": int(ref_data["features"].shape[0]),
        "query_length": int(qry_data["features"].shape[0]),
        "dtw_distance": float(dist),
        "alignment_path_length": int(len(path)),
        "normalized_dtw_distance": float(norm_dist),
        "score_0_100": float(score),
        "score_scale": float(score_scale),
        "hint_threshold": float(hint_threshold),
        "hint_count": int(len(hints)),
        "hints": hints,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

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

    if out_video or preview:
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
            preview=preview,
        )

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
    preview: bool = False,
) -> Dict[str, Any]:
    ref_data = extract_pose_sequence(
        video_path=ref_video,
        task_model=task_model,
        num_poses=max(1, num_poses),
        smooth_window=max(1, smooth_window),
        preview=preview,
        preview_title="Reference Pose Preview",
    )
    qry_data = extract_pose_sequence(
        video_path=query_video,
        task_model=task_model,
        num_poses=max(1, num_poses),
        smooth_window=max(1, smooth_window),
        preview=preview,
        preview_title="Query Pose Preview",
    )
    dist, path = dtw_distance_multidim(ref_data["features"], qry_data["features"])
    return finalize_scoring_outputs(
        ref_data=ref_data,
        qry_data=qry_data,
        ref_video=ref_video,
        query_video=query_video,
        path=path,
        dist=dist,
        score_scale=score_scale,
        hint_threshold=hint_threshold,
        out_json=out_json,
        out_plot=out_plot,
        out_video=out_video,
        preview=preview,
    )



def run_dtw_scoring_from_template(
    template_path: str,
    query_video: str,
    out_json: str,
    out_plot: str,
    out_video: Optional[str] = None,
    preview: bool = False,
    score_scale: float = 6.0,
    hint_threshold: float = 0.22,
) -> Dict[str, Any]:
    # 直接加载模板并只处理待测视频，适合重复评分场景。
    ref_data = load_pose_template(template_path)
    qry_data = extract_pose_sequence(
        video_path=query_video,
        task_model=ref_data["task_model"],
        num_poses=max(1, ref_data["num_poses"]),
        smooth_window=max(1, ref_data["smooth_window"]),
        preview=preview,
        preview_title="Query Pose Preview",
    )
    dist, path = dtw_distance_multidim(ref_data["features"], qry_data["features"])
    summary = finalize_scoring_outputs(
        ref_data=ref_data,
        qry_data=qry_data,
        ref_video=ref_data["reference_video"],
        query_video=query_video,
        path=path,
        dist=dist,
        score_scale=score_scale,
        hint_threshold=hint_threshold,
        out_json=out_json,
        out_plot=out_plot,
        out_video=out_video,
        preview=preview,
    )
    summary["template_path"] = template_path
    return summary

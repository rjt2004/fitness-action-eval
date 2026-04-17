from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fitness_action_eval.baduanjin import (
    FEEDBACK_PART_GROUPS,
    PART_TO_ANGLE_NAMES,
    build_baduanjin_hint_text,
    get_phase_definition,
)


def part_errors(
    ref_pts: np.ndarray,
    qry_pts: np.ndarray,
    ref_angles: Optional[np.ndarray] = None,
    qry_angles: Optional[np.ndarray] = None,
    phase_id: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    # 统计各身体部位与模板之间的平均偏差及方向，并融合关节角度误差。
    out: Dict[str, Dict[str, float]] = {}
    angle_index = {
        "left_shoulder": 0,
        "right_shoulder": 1,
        "left_elbow": 2,
        "right_elbow": 3,
        "left_hip": 4,
        "right_hip": 5,
        "left_knee": 6,
        "right_knee": 7,
    }
    phase = get_phase_definition(phase_id) if phase_id is not None else None

    for part, idxs in FEEDBACK_PART_GROUPS.items():
        delta = qry_pts[idxs] - ref_pts[idxs]
        dxy = np.linalg.norm(delta, axis=1)
        point_err = float(np.mean(dxy))
        dx = float(np.mean(delta[:, 0]))
        dy = float(np.mean(delta[:, 1]))

        angle_names = PART_TO_ANGLE_NAMES.get(part, [])
        if ref_angles is not None and qry_angles is not None and angle_names:
            angle_ids = [angle_index[name] for name in angle_names if name in angle_index]
            angle_err = float(np.mean(np.abs(qry_angles[angle_ids] - ref_angles[angle_ids])))
        else:
            angle_err = 0.0

        merged_err = (0.8 * point_err) + (0.2 * angle_err)
        if phase is not None:
            merged_err *= float(phase.point_importance.get(part, 1.0))
        out[part] = {
            "score": merged_err,
            "point_error": point_err,
            "angle_error": angle_err,
            "dx": dx,
            "dy": dy,
        }
    return out


def build_live_feedback(
    ref_points: np.ndarray,
    qry_points: np.ndarray,
    hint_threshold: float,
    phase_id: Optional[int] = None,
    ref_angles: Optional[np.ndarray] = None,
    qry_angles: Optional[np.ndarray] = None,
) -> Tuple[str, float, Optional[str]]:
    # 针对单帧实时比对生成当前提示。
    p_err = part_errors(
        ref_pts=ref_points,
        qry_pts=qry_points,
        ref_angles=ref_angles,
        qry_angles=qry_angles,
        phase_id=phase_id,
    )

    if phase_id is not None:
        phase = get_phase_definition(phase_id)
        part_order = list(phase.feedback_priority)
        effective_threshold = hint_threshold * float(phase.feedback_threshold_scale)
    else:
        part_order = list(p_err.keys())
        effective_threshold = hint_threshold

    best_part = max(part_order, key=lambda name: p_err.get(name, {"score": -1.0})["score"])
    best_info = p_err[best_part]
    point_err = float(np.mean(np.linalg.norm(qry_points - ref_points, axis=1)))
    if best_info["score"] < effective_threshold:
        return "", point_err, best_part

    message = build_baduanjin_hint_text(phase_id=phase_id, part=best_part, dx=best_info["dx"], dy=best_info["dy"])
    return message, point_err, best_part


def build_feedback(
    path: List[Tuple[int, int]],
    ref_points: np.ndarray,
    qry_points: np.ndarray,
    hint_threshold: float,
    hint_min_interval: int,
    max_hints: int,
    ref_phase_ids: Optional[np.ndarray] = None,
    qry_phase_ids: Optional[np.ndarray] = None,
    ref_angles: Optional[np.ndarray] = None,
    qry_angles: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    # 沿着 DTW 对齐路径计算局部误差，并在关键阶段生成八段锦中文提示。
    q_len = qry_points.shape[0]
    local_sum = np.zeros((q_len,), dtype=np.float32)
    local_cnt = np.zeros((q_len,), dtype=np.int32)
    hints: List[Dict[str, Any]] = []
    last_hint_q = -10**9

    for i, j in path:
        phase_id = int(ref_phase_ids[i]) if ref_phase_ids is not None else None
        p_err = part_errors(
            ref_pts=ref_points[i],
            qry_pts=qry_points[j],
            ref_angles=ref_angles[i] if ref_angles is not None else None,
            qry_angles=qry_angles[j] if qry_angles is not None else None,
            phase_id=phase_id,
        )
        local_err = float(np.mean(np.linalg.norm(qry_points[j] - ref_points[i], axis=1)))
        local_sum[j] += local_err
        local_cnt[j] += 1

        if len(hints) >= max_hints:
            continue
        if j - last_hint_q < hint_min_interval:
            continue

        if phase_id is not None:
            phase = get_phase_definition(phase_id)
            part_candidates = list(phase.feedback_priority)
            effective_threshold = hint_threshold * float(phase.feedback_threshold_scale)
        else:
            phase = None
            part_candidates = list(p_err.keys())
            effective_threshold = hint_threshold

        part = max(part_candidates, key=lambda name: p_err.get(name, {"score": -1.0})["score"])
        info = p_err[part]
        if info["score"] < effective_threshold:
            continue

        hints.append(
            {
                "ref_index": int(i),
                "query_index": int(j),
                "query_phase_id": int(qry_phase_ids[j]) if qry_phase_ids is not None else None,
                "phase_id": int(phase_id) if phase_id is not None else None,
                "phase_name": phase.display_name if phase is not None else "通用动作",
                "cue": phase.cue if phase is not None else "",
                "part": part,
                "part_error": float(info["score"]),
                "point_error": float(info["point_error"]),
                "angle_error": float(info["angle_error"]),
                "message": build_baduanjin_hint_text(
                    phase_id=phase_id,
                    part=part,
                    dx=float(info["dx"]),
                    dy=float(info["dy"]),
                ),
            }
        )
        last_hint_q = j

    local_error = np.full((q_len,), np.nan, dtype=np.float32)
    mask = local_cnt > 0
    local_error[mask] = local_sum[mask] / local_cnt[mask]
    return hints, local_error

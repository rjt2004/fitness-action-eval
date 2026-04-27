from __future__ import annotations

"""评分后的局部偏差分析与提示生成。"""

from typing import Any, Dict, List, Mapping, Optional, Tuple

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
    rule_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, float]]:
    """统计各身体部位相对模板的偏差。

    当前误差由两部分组成：
    1. 关键点几何位置误差
    2. 关节角度误差

    最终按 0.8 / 0.2 融合，并叠加八段锦各阶段的部位权重。
    """

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
    phase = get_phase_definition(phase_id, rule_config=rule_config) if phase_id is not None else None

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
    substage_key: Optional[str] = None,
    ref_angles: Optional[np.ndarray] = None,
    qry_angles: Optional[np.ndarray] = None,
    rule_config: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, float, Optional[str]]:
    """为实时模式生成单帧提示。"""

    p_err = part_errors(
        ref_pts=ref_points,
        qry_pts=qry_points,
        ref_angles=ref_angles,
        qry_angles=qry_angles,
        phase_id=phase_id,
        rule_config=rule_config,
    )

    if phase_id is not None:
        phase = get_phase_definition(phase_id, rule_config=rule_config)
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

    message = build_baduanjin_hint_text(
        phase_id=phase_id,
        part=best_part,
        dx=best_info["dx"],
        dy=best_info["dy"],
        substage_key=substage_key,
        rule_config=rule_config,
    )
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
    ref_substage_keys: Optional[np.ndarray] = None,
    ref_substage_names: Optional[np.ndarray] = None,
    ref_substage_cues: Optional[np.ndarray] = None,
    ref_angles: Optional[np.ndarray] = None,
    qry_angles: Optional[np.ndarray] = None,
    rule_config: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """沿着 DTW 对齐路径生成离线提示和每帧局部误差。"""

    q_len = qry_points.shape[0]
    local_sum = np.zeros((q_len,), dtype=np.float32)
    local_cnt = np.zeros((q_len,), dtype=np.int32)
    hints: List[Dict[str, Any]] = []
    last_hint_q = -10**9

    for i, j in path:
        phase_id = int(ref_phase_ids[i]) if ref_phase_ids is not None else None
        substage_key = str(ref_substage_keys[i]) if ref_substage_keys is not None else None
        substage_name = str(ref_substage_names[i]) if ref_substage_names is not None else ""
        substage_cue = str(ref_substage_cues[i]) if ref_substage_cues is not None else ""
        p_err = part_errors(
            ref_pts=ref_points[i],
            qry_pts=qry_points[j],
            ref_angles=ref_angles[i] if ref_angles is not None else None,
            qry_angles=qry_angles[j] if qry_angles is not None else None,
            phase_id=phase_id,
            rule_config=rule_config,
        )
        local_err = float(np.mean(np.linalg.norm(qry_points[j] - ref_points[i], axis=1)))
        local_sum[j] += local_err
        local_cnt[j] += 1

        if len(hints) >= max_hints:
            continue
        if j - last_hint_q < hint_min_interval:
            continue

        if phase_id is not None:
            phase = get_phase_definition(phase_id, rule_config=rule_config)
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
                "phase_name": (
                    f"{phase.display_name} - {substage_name}"
                    if phase is not None and substage_name
                    else (phase.display_name if phase is not None else "通用动作")
                ),
                "cue": substage_cue or (phase.cue if phase is not None else ""),
                "substage_key": substage_key,
                "substage_name": substage_name,
                "substage_cue": substage_cue,
                "part": part,
                "part_error": float(info["score"]),
                "point_error": float(info["point_error"]),
                "angle_error": float(info["angle_error"]),
                "message": build_baduanjin_hint_text(
                    phase_id=phase_id,
                    part=part,
                    dx=float(info["dx"]),
                    dy=float(info["dy"]),
                    substage_key=substage_key,
                    rule_config=rule_config,
                ),
            }
        )
        last_hint_q = j

    local_error = np.full((q_len,), np.nan, dtype=np.float32)
    mask = local_cnt > 0
    local_error[mask] = local_sum[mask] / local_cnt[mask]
    return hints, local_error

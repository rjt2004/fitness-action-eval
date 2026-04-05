from typing import Any, Dict, List, Tuple

import numpy as np


PART_GROUPS: Dict[str, List[int]] = {
    "left_arm": [11, 13, 15],
    "right_arm": [12, 14, 16],
    "left_leg": [23, 25, 27],
    "right_leg": [24, 26, 28],
    "torso": [11, 12, 23, 24],
}


def part_errors(ref_pts: np.ndarray, qry_pts: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
    # 统计各身体部位与模板之间的平均偏差及方向，用于生成反馈提示。
    out: Dict[str, Tuple[float, float, float]] = {}
    for part, idxs in PART_GROUPS.items():
        delta = qry_pts[idxs] - ref_pts[idxs]
        dxy = np.linalg.norm(delta, axis=1)
        mean_err = float(np.mean(dxy))
        dx = float(np.mean(delta[:, 0]))
        dy = float(np.mean(delta[:, 1]))
        out[part] = (mean_err, dx, dy)
    return out


def build_hint_text(part: str, dx: float, dy: float) -> str:
    # 根据偏差主方向生成自然语言提示文本。
    name_map = {
        "left_arm": "左臂",
        "right_arm": "右臂",
        "left_leg": "左腿",
        "right_leg": "右腿",
        "torso": "躯干",
    }
    part_name = name_map.get(part, part)
    if abs(dx) >= abs(dy):
        direction = "偏右" if dx > 0 else "偏左"
        return f"{part_name}位置{direction}，请向模板轨迹方向调整。"
    direction = "偏低" if dy > 0 else "偏高"
    return f"{part_name}位置{direction}，请调整高度并保持动作节奏。"


def build_feedback(
    path: List[Tuple[int, int]],
    ref_points: np.ndarray,
    qry_points: np.ndarray,
    hint_threshold: float,
    hint_min_interval: int,
    max_hints: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    # 沿着 DTW 对齐路径计算局部误差，并在误差较大时抽取关键提示点。
    q_len = qry_points.shape[0]
    local_sum = np.zeros((q_len,), dtype=np.float32)
    local_cnt = np.zeros((q_len,), dtype=np.int32)
    hints: List[Dict[str, Any]] = []
    last_hint_q = -10**9

    for i, j in path:
        ref_pts = ref_points[i]
        qry_pts = qry_points[j]
        local_err = float(np.mean(np.linalg.norm(qry_pts - ref_pts, axis=1)))
        local_sum[j] += local_err
        local_cnt[j] += 1

        if len(hints) >= max_hints:
            continue
        if j - last_hint_q < hint_min_interval:
            continue

        p_err = part_errors(ref_pts, qry_pts)
        part = max(p_err.keys(), key=lambda x: p_err[x][0])
        err, dx, dy = p_err[part]
        if err < hint_threshold:
            continue

        hints.append(
            {
                "ref_index": int(i),
                "query_index": int(j),
                "part": part,
                "part_error": float(err),
                "message": build_hint_text(part, dx, dy),
            }
        )
        last_hint_q = j

    local_error = np.full((q_len,), np.nan, dtype=np.float32)
    mask = local_cnt > 0
    local_error[mask] = local_sum[mask] / local_cnt[mask]
    return hints, local_error

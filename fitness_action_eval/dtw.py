from __future__ import annotations

"""DTW 对齐与分数映射工具。

这里的 DTW 使用带宽约束来限制搜索区域，避免长视频直接做全矩阵对齐时
计算量过大，同时仍允许标准动作和测试动作存在一定节奏差异。
"""

from typing import List, Tuple

import numpy as np


def dtw_distance_multidim(
    a: np.ndarray,
    b: np.ndarray,
    window_ratio: float = 0.12,
) -> Tuple[float, List[Tuple[int, int]]]:
    """计算两个多维序列的 DTW 距离和对齐路径。"""

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("dtw_distance_multidim expects (T,D) sequences.")
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf"), []

    # 带宽至少覆盖长度差，并根据比例额外留出节奏漂移空间。
    band = int(max(abs(n - m), round(max(n, m) * max(0.01, window_ratio))))
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    trace = np.full((n + 1, m + 1, 2), -1, dtype=np.int32)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        center = int(round(((i - 1) * m) / max(1, n - 1))) + 1 if n > 1 else 1
        j_start = max(1, center - band)
        j_end = min(m, center + band)
        ai = a[i - 1]
        for j in range(j_start, j_end + 1):
            cost = float(np.linalg.norm(ai - b[j - 1]))
            up = dp[i - 1, j]
            left = dp[i, j - 1]
            diag = dp[i - 1, j - 1]
            if diag <= up and diag <= left:
                dp[i, j] = cost + diag
                trace[i, j] = (i - 1, j - 1)
            elif up <= left:
                dp[i, j] = cost + up
                trace[i, j] = (i - 1, j)
            else:
                dp[i, j] = cost + left
                trace[i, j] = (i, j - 1)

    if not np.isfinite(dp[n, m]):
        raise RuntimeError("DTW failed to find a valid alignment path. Consider increasing the band width.")

    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        prev_i, prev_j = trace[i, j]
        if prev_i < 0 or prev_j < 0:
            break
        i, j = int(prev_i), int(prev_j)
    path.reverse()
    return float(dp[n, m]), path


def distance_to_score(norm_dist: float, score_scale: float) -> float:
    """把归一化距离映射到 0-100 分，距离越小分数越高。"""

    score = 100.0 * (1.0 - (norm_dist / max(1e-6, score_scale)))
    return float(np.clip(score, 0.0, 100.0))

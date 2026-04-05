from typing import List, Tuple

import numpy as np


def dtw_distance_multidim(a: np.ndarray, b: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    # 使用动态时间规整对两段动作序列进行对齐，兼容动作速度快慢不同的情况。
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("dtw_distance_multidim expects (T,D) sequences.")
    n, m = a.shape[0], b.shape[0]
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    trace = np.zeros((n + 1, m + 1, 2), dtype=np.int32)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(ai - b[j - 1]))
            choices = (dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            k = int(np.argmin(choices))
            if k == 0:
                prev = (i - 1, j)
            elif k == 1:
                prev = (i, j - 1)
            else:
                prev = (i - 1, j - 1)
            dp[i, j] = cost + choices[k]
            trace[i, j] = prev

    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = int(trace[i, j, 0]), int(trace[i, j, 1])
    path.reverse()
    return float(dp[n, m]), path


def distance_to_score(norm_dist: float, score_scale: float) -> float:
    # 将归一化 DTW 距离映射到 0-100 分，距离越小得分越高。
    score = 100.0 * (1.0 - (norm_dist / max(1e-6, score_scale)))
    return float(np.clip(score, 0.0, 100.0))


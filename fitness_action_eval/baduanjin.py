from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Mapping, Sequence

import numpy as np


ANGLE_NAMES: List[str] = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
]

ANGLE_SPECS: List[tuple[str, int, int, int]] = [
    ("left_shoulder", 13, 11, 23),
    ("right_shoulder", 14, 12, 24),
    ("left_elbow", 15, 13, 11),
    ("right_elbow", 16, 14, 12),
    ("left_hip", 11, 23, 25),
    ("right_hip", 12, 24, 26),
    ("left_knee", 23, 25, 27),
    ("right_knee", 24, 26, 28),
]

POINT_GROUPS: Dict[str, List[int]] = {
    "head_neck": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "shoulders": [11, 12],
    "elbows": [13, 14],
    "hands": [15, 16, 17, 18, 19, 20, 21, 22],
    "torso": [11, 12, 23, 24],
    "waist": [11, 12, 23, 24, 25, 26],
    "hips": [23, 24],
    "knees": [25, 26],
    "feet": [27, 28, 29, 30, 31, 32],
}

FEEDBACK_PART_GROUPS: Dict[str, List[int]] = {
    "head_neck": POINT_GROUPS["head_neck"],
    "shoulders": POINT_GROUPS["shoulders"] + POINT_GROUPS["elbows"],
    "hands": POINT_GROUPS["hands"],
    "torso": POINT_GROUPS["torso"],
    "waist": POINT_GROUPS["waist"],
    "hips": POINT_GROUPS["hips"],
    "knees": POINT_GROUPS["knees"],
    "feet": POINT_GROUPS["feet"],
}

PART_TO_ANGLE_NAMES: Dict[str, List[str]] = {
    "head_neck": [],
    "shoulders": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
    "hands": ["left_elbow", "right_elbow"],
    "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "waist": ["left_hip", "right_hip", "left_knee", "right_knee"],
    "hips": ["left_hip", "right_hip"],
    "knees": ["left_knee", "right_knee"],
    "feet": ["left_knee", "right_knee"],
}


@dataclass(frozen=True)
class BaduanjinPhase:
    phase_id: int
    key: str
    display_name: str
    cue: str
    duration_weight: float
    point_importance: Mapping[str, float]
    angle_importance: Mapping[str, float]
    feedback_priority: Sequence[str]
    hint_templates: Mapping[str, str]
    feedback_threshold_scale: float = 1.0


BADUANJIN_PHASES: List[BaduanjinPhase] = [
    BaduanjinPhase(
        phase_id=0,
        key="prepare",
        display_name="预备势",
        cue="立身中正，沉肩松腕，呼吸自然。",
        duration_weight=8.0,
        point_importance={"torso": 1.3, "waist": 1.2, "feet": 1.2, "shoulders": 1.2},
        angle_importance={"left_hip": 1.2, "right_hip": 1.2, "left_knee": 1.2, "right_knee": 1.2},
        feedback_priority=("torso", "shoulders", "feet"),
        hint_templates={
            "torso": "预备势先站稳中线，头顶上领，躯干不要前倾或后仰。",
            "shoulders": "预备势注意沉肩垂肘，双臂自然下垂，不要耸肩。",
            "feet": "双脚先摆正并站稳，重心放在两脚之间，再进入起势。",
        },
    ),
    BaduanjinPhase(
        phase_id=1,
        key="raise_sky",
        display_name="双手托天理三焦",
        cue="两掌上托，沉肩拔背，上下对拉。",
        duration_weight=15.0,
        point_importance={"shoulders": 1.6, "hands": 1.8, "torso": 1.3, "feet": 1.1},
        angle_importance={"left_shoulder": 1.8, "right_shoulder": 1.8, "left_elbow": 1.5, "right_elbow": 1.5},
        feedback_priority=("hands", "shoulders", "torso"),
        hint_templates={
            "hands": "双手上托还不够，掌根继续向上撑，肘尽量伸展。",
            "shoulders": "托天时要沉肩展胸，别把肩膀提起来。",
            "torso": "上托时身体要立直并有拔长感，不要塌腰。",
        },
    ),
    BaduanjinPhase(
        phase_id=2,
        key="draw_bow",
        display_name="左右开弓似射雕",
        cue="马步沉稳，开肩展背，目视指尖。",
        duration_weight=29.0,
        point_importance={"hands": 1.8, "shoulders": 1.7, "waist": 1.3, "knees": 1.4, "feet": 1.2},
        angle_importance={
            "left_shoulder": 1.8,
            "right_shoulder": 1.8,
            "left_elbow": 1.7,
            "right_elbow": 1.7,
            "left_knee": 1.4,
            "right_knee": 1.4,
        },
        feedback_priority=("hands", "shoulders", "knees", "waist"),
        hint_templates={
            "hands": "开弓动作还不够舒展，前手继续推出，后手再向外拉开。",
            "shoulders": "开弓时要开肩展背，胸口打开，肩线尽量拉平。",
            "knees": "马步下沉不足，膝关节继续稳定屈伸，重心压稳。",
            "waist": "开弓时腰胯要稳住，避免身体左右晃动。",
        },
    ),
    BaduanjinPhase(
        phase_id=3,
        key="regulate_spleen",
        display_name="调理脾胃须单举",
        cue="一掌上托一掌下按，拉长身体两侧。",
        duration_weight=22.0,
        point_importance={"hands": 1.8, "shoulders": 1.5, "torso": 1.3, "waist": 1.2},
        angle_importance={"left_shoulder": 1.7, "right_shoulder": 1.7, "left_elbow": 1.4, "right_elbow": 1.4},
        feedback_priority=("hands", "shoulders", "torso"),
        hint_templates={
            "hands": "上下撑按还不够明显，一手继续上托，一手继续下按。",
            "shoulders": "单举时两侧肩线要舒展，避免肩部缩紧。",
            "torso": "脊柱要继续拔长，身体两侧拉开的感觉再明显一些。",
        },
    ),
    BaduanjinPhase(
        phase_id=4,
        key="look_back",
        display_name="五劳七伤往后瞧",
        cue="头颈缓转，目视后方，肩胯保持稳定。",
        duration_weight=23.0,
        point_importance={"head_neck": 1.8, "shoulders": 1.4, "torso": 1.2},
        angle_importance={"left_shoulder": 1.3, "right_shoulder": 1.3},
        feedback_priority=("head_neck", "shoulders", "torso"),
        hint_templates={
            "head_neck": "回头幅度偏小，头颈放松后顾，目光尽量看向后方。",
            "shoulders": "回头时肩不要跟着耸起，保持双肩放松下沉。",
            "torso": "往后瞧时躯干要稳住，不要带着上身整体扭偏。",
        },
    ),
    BaduanjinPhase(
        phase_id=5,
        key="shake_tail",
        display_name="摇头摆尾去心火",
        cue="松胯屈膝，以腰带身，头尾相引。",
        duration_weight=28.0,
        point_importance={"waist": 1.8, "torso": 1.5, "knees": 1.5, "head_neck": 1.3},
        angle_importance={"left_hip": 1.8, "right_hip": 1.8, "left_knee": 1.5, "right_knee": 1.5},
        feedback_priority=("waist", "knees", "head_neck", "torso"),
        hint_templates={
            "waist": "摆尾时松胯带腰还不够，先沉重心，再带动上身侧摆。",
            "knees": "屈膝下蹲不足，重心应再压低一些，保持两膝稳定。",
            "head_neck": "摇头与摆尾配合不够，头颈顺着腰胯方向自然带动。",
            "torso": "侧摆时躯干应更圆活，不要僵直卡住。",
        },
        feedback_threshold_scale=1.1,
    ),
    BaduanjinPhase(
        phase_id=6,
        key="touch_feet",
        display_name="两手攀足固肾腰",
        cue="折髋前俯，双手下探，腰背徐徐放松。",
        duration_weight=19.0,
        point_importance={"hands": 1.7, "waist": 1.8, "torso": 1.5, "knees": 1.2},
        angle_importance={"left_hip": 1.8, "right_hip": 1.8, "left_knee": 1.3, "right_knee": 1.3},
        feedback_priority=("waist", "hands", "torso", "knees"),
        hint_templates={
            "waist": "前俯幅度偏小，先从髋部折叠，再带动上身向前下引。",
            "hands": "两手下探还不够，顺着腿侧继续向脚面方向伸。",
            "torso": "俯身时腰背放松不够，脊柱应顺势延展下沉。",
            "knees": "下探时膝部要保持稳定，不要明显晃动或弯曲过多。",
        },
        feedback_threshold_scale=1.1,
    ),
    BaduanjinPhase(
        phase_id=7,
        key="punch",
        display_name="攒拳怒目增气力",
        cue="马步稳固，拳力前送，旋腕怒目。",
        duration_weight=16.0,
        point_importance={"hands": 1.8, "shoulders": 1.5, "knees": 1.4, "waist": 1.2},
        angle_importance={
            "left_elbow": 1.8,
            "right_elbow": 1.8,
            "left_shoulder": 1.5,
            "right_shoulder": 1.5,
            "left_knee": 1.3,
            "right_knee": 1.3,
        },
        feedback_priority=("hands", "shoulders", "knees"),
        hint_templates={
            "hands": "冲拳路线还不够饱满，拳再向前送，同时注意拧转发力。",
            "shoulders": "出拳时肩肘配合要顺，不要抬肩抢劲。",
            "knees": "马步根基不够稳，先把下盘压住再出拳。",
        },
        feedback_threshold_scale=1.1,
    ),
    BaduanjinPhase(
        phase_id=8,
        key="heel_raise",
        display_name="背后七颠百病消",
        cue="脚跟上提，下落轻震，周身放松。",
        duration_weight=12.0,
        point_importance={"feet": 1.8, "knees": 1.4, "torso": 1.2},
        angle_importance={"left_knee": 1.3, "right_knee": 1.3, "left_hip": 1.1, "right_hip": 1.1},
        feedback_priority=("feet", "knees", "torso"),
        hint_templates={
            "feet": "踮脚高度还不够，脚跟再向上提，落地时保持轻震。",
            "knees": "七颠时膝部要微屈缓冲，不要僵硬锁死。",
            "torso": "踮起与下落时上身要保持端正，不要前扑后仰。",
        },
    ),
    BaduanjinPhase(
        phase_id=9,
        key="closing",
        display_name="收势",
        cue="双手回按丹田，气息平稳，缓缓并步。",
        duration_weight=6.0,
        point_importance={"torso": 1.3, "hands": 1.3, "feet": 1.2},
        angle_importance={"left_hip": 1.1, "right_hip": 1.1},
        feedback_priority=("hands", "torso", "feet"),
        hint_templates={
            "hands": "收势时双手回落丹田要柔和，不要着急收回。",
            "torso": "收势阶段保持立身中正，气息慢慢放匀。",
            "feet": "收势并步要稳，脚下先站定再收回。",
        },
    ),
]

BADUANJIN_STANDARD_DURATION_S = 12 * 60 + 12

# Standard-video phase starts/ends supplied by manual annotation.
BADUANJIN_MANUAL_BOUNDARIES_S: List[tuple[float, float]] = [
    (0.0, 45.0),
    (45.0, 140.0),
    (140.0, 225.0),
    (225.0, 305.0),
    (305.0, 370.0),
    (370.0, 475.0),
    (475.0, 600.0),
    (600.0, 660.0),
    (660.0, 710.0),
    (710.0, 732.0),
]


def _build_phase_ids_from_manual_timing(time_s: np.ndarray) -> np.ndarray:
    if time_s.ndim != 1 or time_s.size == 0:
        return np.zeros((0,), dtype=np.int32)

    sequence_duration = float(time_s[-1])
    if sequence_duration <= 0:
        return np.zeros((time_s.shape[0],), dtype=np.int32)

    scale = sequence_duration / float(BADUANJIN_STANDARD_DURATION_S)
    scaled_starts = np.asarray([start for start, _ in BADUANJIN_MANUAL_BOUNDARIES_S], dtype=np.float32) * scale

    phase_ids = np.zeros((time_s.shape[0],), dtype=np.int32)
    for phase_idx in range(1, len(scaled_starts)):
        phase_ids[time_s >= scaled_starts[phase_idx]] = phase_idx
    return phase_ids


@lru_cache(maxsize=64)
def _build_phase_ids_by_weight(length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.int32)
    weights = np.asarray([phase.duration_weight for phase in BADUANJIN_PHASES], dtype=np.float32)
    weights = weights / float(np.sum(weights))
    boundaries = np.cumsum(weights) * float(length)
    boundaries = np.rint(boundaries).astype(np.int32)
    boundaries[-1] = length

    phase_ids = np.zeros((length,), dtype=np.int32)
    start = 0
    for phase_idx, end in enumerate(boundaries):
        end = int(np.clip(end, start + 1, length))
        phase_ids[start:end] = phase_idx
        start = end
    if start < length:
        phase_ids[start:] = len(BADUANJIN_PHASES) - 1
    return phase_ids


def build_phase_ids(length: int, time_s: np.ndarray | None = None) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.int32)
    if time_s is not None and getattr(time_s, "shape", (0,))[0] == length:
        phase_ids = _build_phase_ids_from_manual_timing(np.asarray(time_s, dtype=np.float32))
        if phase_ids.shape[0] == length and np.any(phase_ids):
            return phase_ids
    return _build_phase_ids_by_weight(length)


def get_phase_definition(phase_id: int) -> BaduanjinPhase:
    phase_id = int(np.clip(phase_id, 0, len(BADUANJIN_PHASES) - 1))
    return BADUANJIN_PHASES[phase_id]


def _safe_angle(points: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    ba = points[:, a] - points[:, b]
    bc = points[:, c] - points[:, b]
    dot = np.sum(ba * bc, axis=1)
    norm = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    cos_val = dot / np.maximum(norm, 1e-6)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return np.degrees(np.arccos(cos_val)).astype(np.float32)


def compute_joint_angle_sequence(points: np.ndarray) -> np.ndarray:
    if points.ndim != 3 or points.shape[1:] != (33, 2):
        raise ValueError("compute_joint_angle_sequence expects (T, 33, 2) points.")
    features = [_safe_angle(points, a, b, c) / 180.0 for _, a, b, c in ANGLE_SPECS]
    return np.stack(features, axis=1).astype(np.float32)


@lru_cache(maxsize=32)
def _phase_weight_vector(phase_id: int, point_dims: int = 33 * 2, angle_dims: int = len(ANGLE_NAMES)) -> np.ndarray:
    phase = get_phase_definition(phase_id)
    point_weights = np.ones((33, 2), dtype=np.float32)
    for group_name, weight in phase.point_importance.items():
        for idx in POINT_GROUPS.get(group_name, []):
            point_weights[idx, :] = np.maximum(point_weights[idx, :], float(weight))

    angle_weights = np.ones((angle_dims,), dtype=np.float32)
    angle_index = {name: idx for idx, name in enumerate(ANGLE_NAMES)}
    for name, weight in phase.angle_importance.items():
        idx = angle_index.get(name)
        if idx is not None:
            angle_weights[idx] = max(angle_weights[idx], float(weight))

    return np.concatenate([point_weights.reshape(point_dims), angle_weights], axis=0)


def apply_phase_feature_weights(base_features: np.ndarray, phase_ids: np.ndarray) -> np.ndarray:
    if base_features.ndim != 2:
        raise ValueError("apply_phase_feature_weights expects (T, D) features.")
    if phase_ids.shape[0] != base_features.shape[0]:
        raise ValueError("phase_ids length must equal feature length.")
    weighted = np.empty_like(base_features, dtype=np.float32)
    for idx in range(base_features.shape[0]):
        weighted[idx] = base_features[idx] * _phase_weight_vector(int(phase_ids[idx]), angle_dims=base_features.shape[1] - 66)
    return weighted


def weight_single_feature(feature: np.ndarray, phase_id: int) -> np.ndarray:
    if feature.ndim != 1:
        raise ValueError("weight_single_feature expects a 1D feature vector.")
    return feature * _phase_weight_vector(int(phase_id), angle_dims=feature.shape[0] - 66)


def phase_metadata_rows(phase_ids: np.ndarray, time_s: np.ndarray) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    if phase_ids.size == 0:
        return rows
    start = 0
    current = int(phase_ids[0])
    for idx in range(1, len(phase_ids) + 1):
        if idx == len(phase_ids) or int(phase_ids[idx]) != current:
            phase = get_phase_definition(current)
            rows.append(
                {
                    "phase_id": phase.phase_id,
                    "phase_name": phase.display_name,
                    "cue": phase.cue,
                    "start_seq_idx": int(start),
                    "end_seq_idx": int(idx - 1),
                    "start_time_s": float(time_s[start]),
                    "end_time_s": float(time_s[idx - 1]),
                }
            )
            if idx < len(phase_ids):
                start = idx
                current = int(phase_ids[idx])
    return rows


def _direction_phrase(part: str, dx: float, dy: float) -> str:
    if part in {"hands", "shoulders"}:
        if abs(dy) >= abs(dx):
            return "再向上舒展一些" if dy > 0 else "不要抬得过高，注意沉肩"
        return "左右展开幅度再清晰一些"
    if part in {"waist", "torso", "hips"}:
        return "保持中正并让腰胯带动动作"
    if part in {"knees", "feet"}:
        return "把下盘站稳，重心控制更平顺"
    if part == "head_neck":
        return "头颈放松，转动方向再明确一些"
    return "向标准动作再靠近一些"


def build_baduanjin_hint_text(phase_id: int | None, part: str, dx: float, dy: float) -> str:
    if phase_id is not None:
        phase = get_phase_definition(phase_id)
        if part in phase.hint_templates:
            return phase.hint_templates[part]
        return f"{phase.display_name}阶段请重点关注{part}，{_direction_phrase(part, dx, dy)}。"
    return f"{part}与标准动作仍有偏差，{_direction_phrase(part, dx, dy)}。"

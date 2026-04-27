from __future__ import annotations

"""八段锦专项规则与动作阶段定义。

本文件集中维护：
1. 关键关节角定义
2. 大阶段与子阶段配置
3. 各阶段的部位权重和提示模板
4. 子阶段状态机规则
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Sequence

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


@dataclass(frozen=True)
class BaduanjinSubStage:
    key: str
    name: str
    start_ratio: float
    end_ratio: float
    cue: str
    priority_parts: Sequence[str]
    hint_templates: Mapping[str, str]


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


BADUANJIN_SUBSTAGES: Dict[int, List[BaduanjinSubStage]] = {
    0: [
        BaduanjinSubStage("open_step", "开步调身", 0.00, 0.55, "左脚开步与肩同宽，膝胯放松，重心平稳。", ("feet", "knees", "waist"), {"feet": "开步时脚距与肩同宽，落脚要轻稳。", "knees": "开步后膝关节微松，不要僵直锁死。", "waist": "开步调身时腰胯保持中正，不要左右晃动。"}),
        BaduanjinSubStage("hold_ball", "抱球静立", 0.65, 1.00, "两掌抱于腹前，呼吸自然，蓄势进入下一式。", ("hands", "torso", "shoulders"), {"hands": "抱球时两掌圆撑，手臂不要贴得过紧。", "torso": "抱球阶段躯干保持中正稳定。", "shoulders": "抱球时继续沉肩坠肘，肩颈放松。"}),
    ],
    1: [
        BaduanjinSubStage("cross_raise", "交叉上举", 0.00, 0.35, "两手交叉上举，掌心逐渐上翻。", ("hands", "shoulders", "torso"), {"hands": "交叉上举时双手路线要从体前上行，不要偏离中线。", "shoulders": "上举时肩部放松，避免耸肩夹颈。", "torso": "上举过程中脊柱保持伸展，不要塌腰。"}),
        BaduanjinSubStage("support_sky", "上托伸展", 0.35, 0.70, "两掌上托，肘臂尽量伸展，形成上下对拉。", ("hands", "shoulders", "torso"), {"hands": "上托时掌根继续向上撑，手臂伸展再充分一些。", "shoulders": "托天定势要沉肩拔背，不要把肩膀顶起来。", "torso": "上托时身体向上拔长，胸背舒展。"}),
        BaduanjinSubStage("lower_return", "下落还原", 0.70, 1.00, "两臂经体侧缓缓下落，还原自然站立。", ("hands", "shoulders", "torso"), {"hands": "下落还原要缓慢连贯，不要突然放下。", "shoulders": "下落时肩臂继续放松，动作保持圆活。", "torso": "还原时躯干不要晃动，重心保持稳定。"}),
    ],
    2: [
        BaduanjinSubStage("horse_step", "开步马步", 0.00, 0.25, "开步下蹲成马步，重心下沉。", ("feet", "knees", "waist"), {"feet": "马步开立要稳，脚下不要虚浮。", "knees": "马步下蹲幅度不足，膝关节继续稳定屈伸。", "waist": "马步时腰胯要稳住，避免左右晃动。"}),
        BaduanjinSubStage("draw_bow_left", "左式开弓", 0.25, 0.50, "左手推出，右手回拉，开肩展背。", ("hands", "shoulders", "torso"), {"hands": "开弓时前手继续推出，后手向外拉开。", "shoulders": "开弓定势要开肩展背，肩线拉平。", "torso": "开弓时躯干保持正直，不要随手臂歪斜。"}),
        BaduanjinSubStage("draw_bow_right", "右式开弓", 0.50, 0.80, "右手推出，左手回拉，目随指尖。", ("hands", "shoulders", "head_neck"), {"hands": "换侧开弓时两手分拉要清晰。", "shoulders": "换侧后仍要保持肩背展开。", "head_neck": "目视推出方向，头颈不要低垂。"}),
    ],
    3: [
        BaduanjinSubStage("single_lift_right", "右掌上托", 0.00, 0.50, "一掌上托，一掌下按，拉长身体两侧。", ("hands", "shoulders", "torso"), {"hands": "单举时上下两掌方向要明确，一掌上托一掌下按。", "shoulders": "上托一侧肩部不要耸起，保持舒展。", "torso": "单举时身体侧线拉长，不要塌腰。"}),
        BaduanjinSubStage("single_lift_left", "左掌上托", 0.50, 1.00, "左右换势，上下对拉保持连贯。", ("hands", "shoulders", "torso"), {"hands": "换侧时两掌交替要连贯，上下撑按更清晰。", "shoulders": "换侧后肩线保持放松舒展。", "torso": "换侧时躯干不要左右偏倒。"}),
    ],
    4: [
        BaduanjinSubStage("press_down", "两掌下按", 0.00, 0.25, "两掌下按，肩臂放松。", ("hands", "shoulders", "torso"), {"hands": "下按时两掌要有向下沉按感。", "shoulders": "下按时肩部不要上提。", "torso": "下按阶段身体保持正直。"}),
        BaduanjinSubStage("look_left", "左顾后瞧", 0.25, 0.50, "头颈向左后方缓转，目视后方。", ("head_neck", "shoulders", "torso"), {"head_neck": "左顾后瞧时头颈转动幅度再明确一些。", "shoulders": "回头时肩不要跟着耸起。", "torso": "后瞧时躯干保持稳定，不要整身扭偏。"}),
        BaduanjinSubStage("look_right", "右顾后瞧", 0.50, 0.80, "头颈向右后方缓转，肩胯稳定。", ("head_neck", "shoulders", "torso"), {"head_neck": "右顾后瞧时目光尽量看向后方。", "shoulders": "右顾时双肩放松下沉。", "torso": "转头时躯干保持中正。"}),
        BaduanjinSubStage("return_forward", "回正还原", 0.80, 1.00, "头颈回正，两掌自然。", ("head_neck", "torso", "hands"), {"head_neck": "回正时头颈要缓慢，不要猛转。", "torso": "回正后身体中线稳定。", "hands": "双手自然下按，手臂放松。"}),
    ],
    5: [
        BaduanjinSubStage("horse_press", "马步扶膝", 0.00, 0.25, "马步下蹲，两手扶膝，重心沉稳。", ("knees", "waist", "feet"), {"knees": "马步扶膝时下蹲不足，膝关节继续稳定屈伸。", "waist": "扶膝时腰胯放松下沉。", "feet": "两脚站稳，重心不要漂移。"}),
        BaduanjinSubStage("waist_swing", "俯身摆动", 0.25, 0.82, "保持马步扶膝，以腰胯带动身体左右圆活摆动。", ("waist", "head_neck", "torso"), {"waist": "俯身摆动时用腰胯带动，不要只动肩膀。", "head_neck": "头颈顺着身体方向自然带动。", "torso": "侧摆时躯干要圆活，不要僵直。"}),
        BaduanjinSubStage("rise_return", "起身还原", 0.82, 1.00, "身体回正起身，重心平稳。", ("torso", "knees", "feet"), {"torso": "起身回正时躯干保持中线。", "knees": "起身时膝关节不要突然弹直。", "feet": "回正时脚下继续站稳。"}),
    ],
    6: [
        BaduanjinSubStage("raise_press", "上举准备", 0.00, 0.20, "两臂上举后下按，身体准备前俯。", ("hands", "shoulders", "torso"), {"hands": "上举下按路线要连贯。", "shoulders": "手臂上举时肩部放松。", "torso": "下按时躯干保持伸展。"}),
        BaduanjinSubStage("bend_touch", "俯身攀足", 0.20, 0.86, "折髋前俯，两手沿腿下摩并接近足部。", ("waist", "hands", "knees"), {"waist": "前俯时以髋带动，腰背放松下探。", "hands": "两手沿腿向足部方向继续下探。", "knees": "前俯时膝部不要过度弯曲或锁死。"}),
        BaduanjinSubStage("rise_spine", "起身还原", 0.86, 1.00, "两手沿腿上行，脊柱逐节起身。", ("torso", "waist", "hands"), {"torso": "起身时脊柱逐渐伸展，不要猛然抬头。", "waist": "起身还原时腰背控制再平稳。", "hands": "两手回收路线要贴合身体。"}),
    ],
    7: [
        BaduanjinSubStage("horse_step", "马步", 0.00, 0.18, "马步下蹲，下盘站稳。", ("knees", "feet", "hands"), {"knees": "马步下蹲不足，下盘再稳一些。", "feet": "两脚开立要稳定。", "hands": "马步开始时上肢放松，准备冲拳。"}),
        BaduanjinSubStage("punch_left", "左冲拳怒目", 0.18, 0.58, "左拳前冲，目光有神，力达拳面。", ("hands", "shoulders", "head_neck"), {"hands": "冲拳幅度不足，拳面继续向前送。", "shoulders": "冲拳时肩不要上耸，肩背要展开。", "head_neck": "怒目方向要随冲拳前视。"}),
        BaduanjinSubStage("punch_right", "右冲拳怒目", 0.58, 1.00, "右拳前冲，旋腕抓握回收。", ("hands", "shoulders", "waist"), {"hands": "换侧冲拳和旋腕抓握要更清晰。", "shoulders": "右冲拳时肩线保持平稳。", "waist": "冲拳时腰胯不要晃动。"}),
    ],
    8: [
        BaduanjinSubStage("lift_heel", "提踵上拔", 0.00, 0.55, "脚跟提起，身体向上拔伸。", ("feet", "torso", "knees"), {"feet": "提踵高度不足，脚跟继续向上提。", "torso": "提踵时身体向上拔伸，不要塌腰。", "knees": "提踵时膝关节保持自然伸展。"}),
        BaduanjinSubStage("drop_shake", "下落轻震", 0.55, 1.00, "脚跟下落轻震，周身放松。", ("feet", "knees", "torso"), {"feet": "下落时脚跟轻震，不要砸地过重。", "knees": "下落时膝关节保持缓冲。", "torso": "下落后身体放松回稳。"}),
    ],
    9: [
        BaduanjinSubStage("gather_qi", "捧气归丹", 0.00, 0.45, "两手体前捧起，缓缓回按丹田。", ("hands", "torso", "shoulders"), {"hands": "捧气归丹时双手路线要柔和圆顺。", "torso": "收势时身体保持中正安稳。", "shoulders": "收势过程中肩臂放松。"}),
        BaduanjinSubStage("close_step", "并步收功", 0.45, 1.00, "两手回落，并步站稳，气息平和。", ("feet", "hands", "torso"), {"feet": "并步收功时脚步要轻稳。", "hands": "双手回落丹田要柔和，不要急收。", "torso": "收功时身体端正，呼吸平稳。"}),
    ],
}

# Standard-video phase starts/ends supplied by manual annotation.
BADUANJIN_MANUAL_BOUNDARIES_S: List[tuple[float, float]] = [
    (20.0, 45.0),
    (45.0, 143.0),
    (143.0, 230.0),
    (230.0, 305.0),
    (305.0, 375.0),
    (375.0, 475.0),
    (475.0, 600.0),
    (600.0, 660.0),
    (660.0, 710.0),
    (710.0, 732.0),
]


def default_baduanjin_rule_config() -> Dict[str, Any]:
    """把内置八段锦规则导出成可持久化到模板的配置。"""

    return {
        "action_key": "baduanjin",
        "phase_mode": "manual_time",
        "standard_duration_s": float(BADUANJIN_STANDARD_DURATION_S),
        "manual_boundaries_s": [
            {"phase_id": idx, "start_s": float(start), "end_s": float(end)}
            for idx, (start, end) in enumerate(BADUANJIN_MANUAL_BOUNDARIES_S)
        ],
        "phases": [
            {
                "phase_id": int(phase.phase_id),
                "key": phase.key,
                "name": phase.display_name,
                "cue": phase.cue,
                "duration_weight": float(phase.duration_weight),
                "point_importance": dict(phase.point_importance),
                "angle_importance": dict(phase.angle_importance),
                "feedback_priority": list(phase.feedback_priority),
                "hint_templates": dict(phase.hint_templates),
                "feedback_threshold_scale": float(phase.feedback_threshold_scale),
                "substages": [
                    {
                        "key": substage.key,
                        "name": substage.name,
                        "start_ratio": float(substage.start_ratio),
                        "end_ratio": float(substage.end_ratio),
                        "cue": substage.cue,
                        "priority_parts": list(substage.priority_parts),
                        "hint_templates": dict(substage.hint_templates),
                    }
                    for substage in BADUANJIN_SUBSTAGES.get(int(phase.phase_id), [])
                ],
            }
            for phase in BADUANJIN_PHASES
        ],
    }


def _rule_phase_map(rule_config: Mapping[str, Any] | None) -> Dict[int, Mapping[str, Any]]:
    if not rule_config:
        return {}
    phases = rule_config.get("phases", [])
    if not isinstance(phases, list):
        return {}
    return {int(item.get("phase_id", idx)): item for idx, item in enumerate(phases) if isinstance(item, Mapping)}


def _phase_from_rule(phase_id: int, rule_config: Mapping[str, Any] | None) -> BaduanjinPhase | None:
    phase_row = _rule_phase_map(rule_config).get(int(phase_id))
    if not phase_row:
        return None
    return BaduanjinPhase(
        phase_id=int(phase_row.get("phase_id", phase_id)),
        key=str(phase_row.get("key", f"phase_{phase_id}")),
        display_name=str(phase_row.get("name", phase_row.get("display_name", f"阶段{phase_id}"))),
        cue=str(phase_row.get("cue", "")),
        duration_weight=float(phase_row.get("duration_weight", 1.0)),
        point_importance=dict(phase_row.get("point_importance", {})),
        angle_importance=dict(phase_row.get("angle_importance", {})),
        feedback_priority=tuple(phase_row.get("feedback_priority", FEEDBACK_PART_GROUPS.keys())),
        hint_templates=dict(phase_row.get("hint_templates", {})),
        feedback_threshold_scale=float(phase_row.get("feedback_threshold_scale", 1.0)),
    )


def _build_phase_ids_from_manual_timing(time_s: np.ndarray, rule_config: Mapping[str, Any] | None = None) -> np.ndarray:
    if time_s.ndim != 1 or time_s.size == 0:
        return np.zeros((0,), dtype=np.int32)

    sequence_duration = float(time_s[-1])
    if sequence_duration <= 0:
        return np.zeros((time_s.shape[0],), dtype=np.int32)

    standard_duration_s = float((rule_config or {}).get("standard_duration_s") or BADUANJIN_STANDARD_DURATION_S)
    boundaries = (rule_config or {}).get("manual_boundaries_s")
    if isinstance(boundaries, list) and boundaries:
        starts = [float(item.get("start_s", 0.0)) for item in boundaries if isinstance(item, Mapping)]
    else:
        starts = [start for start, _ in BADUANJIN_MANUAL_BOUNDARIES_S]
    scale = sequence_duration / max(1e-6, standard_duration_s)
    scaled_starts = np.asarray(starts, dtype=np.float32) * scale

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


def _build_phase_ids_by_rule_weight(length: int, rule_config: Mapping[str, Any]) -> np.ndarray:
    phases = list(_rule_phase_map(rule_config).values())
    if length <= 0 or not phases:
        return np.zeros((0,), dtype=np.int32)
    weights = np.asarray([float(item.get("duration_weight", 1.0)) for item in phases], dtype=np.float32)
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0:
        weights = np.ones((len(phases),), dtype=np.float32)
    weights = weights / float(np.sum(weights))
    boundaries = np.cumsum(weights) * float(length)
    boundaries = np.rint(boundaries).astype(np.int32)
    boundaries[-1] = length

    phase_ids = np.zeros((length,), dtype=np.int32)
    start = 0
    phase_keys = [int(item.get("phase_id", idx)) for idx, item in enumerate(phases)]
    for phase_id, end in zip(phase_keys, boundaries):
        end = int(np.clip(end, start + 1, length))
        phase_ids[start:end] = int(phase_id)
        start = end
    if start < length:
        phase_ids[start:] = int(phase_keys[-1])
    return phase_ids


def build_phase_ids(length: int, time_s: np.ndarray | None = None, rule_config: Mapping[str, Any] | None = None) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.int32)
    if time_s is not None and getattr(time_s, "shape", (0,))[0] == length:
        phase_ids = _build_phase_ids_from_manual_timing(np.asarray(time_s, dtype=np.float32), rule_config=rule_config)
        if phase_ids.shape[0] == length and np.any(phase_ids):
            return phase_ids
    if rule_config and _rule_phase_map(rule_config):
        return _build_phase_ids_by_rule_weight(length, rule_config)
    return _build_phase_ids_by_weight(length)


def get_phase_definition(phase_id: int, rule_config: Mapping[str, Any] | None = None) -> BaduanjinPhase:
    raw_phase_id = int(phase_id)
    rule_phase = _phase_from_rule(raw_phase_id, rule_config)
    if rule_phase is not None:
        return rule_phase
    phase_id = int(np.clip(raw_phase_id, 0, len(BADUANJIN_PHASES) - 1))
    return BADUANJIN_PHASES[phase_id]


def get_substage_definition(phase_id: int, progress: float) -> BaduanjinSubStage:
    phase_id = int(np.clip(phase_id, 0, len(BADUANJIN_PHASES) - 1))
    progress = float(np.clip(progress, 0.0, 1.0))
    substages = BADUANJIN_SUBSTAGES.get(phase_id, [])
    for substage in substages:
        if substage.start_ratio <= progress < substage.end_ratio:
            return substage
    return substages[-1] if substages else BaduanjinSubStage("general", "完整动作", 0.0, 1.0, get_phase_definition(phase_id).cue, (), {})


def get_substage_by_key(phase_id: int, substage_key: str | None) -> BaduanjinSubStage | None:
    if substage_key is None:
        return None
    for substage in BADUANJIN_SUBSTAGES.get(int(phase_id), []):
        if substage.key == substage_key:
            return substage
    return None


def _series_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=float(np.median(finite)))
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))
    if high - low < 1e-6:
        return np.full_like(values, 0.5, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _single_pose_metrics(points: np.ndarray) -> Dict[str, float]:
    left_wrist = points[15]
    right_wrist = points[16]
    left_shoulder = points[11]
    right_shoulder = points[12]
    left_hip = points[23]
    right_hip = points[24]
    left_ankle = points[27]
    right_ankle = points[28]
    nose = points[0]

    shoulder_center = (left_shoulder + right_shoulder) * 0.5
    hip_center = (left_hip + right_hip) * 0.5
    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
    torso_len = float(np.linalg.norm(shoulder_center - hip_center))
    scale = max(shoulder_width, torso_len * 0.75, 1e-4)
    wrist_center = (left_wrist + right_wrist) * 0.5

    left_extension = abs(float(left_wrist[0] - shoulder_center[0])) / scale
    right_extension = abs(float(right_wrist[0] - shoulder_center[0])) / scale
    return {
        "hand_height": float((hip_center[1] - wrist_center[1]) / scale),
        "hand_down": float((wrist_center[1] - shoulder_center[1]) / scale),
        "hand_span": abs(float(left_wrist[0] - right_wrist[0])) / scale,
        "foot_span": abs(float(left_ankle[0] - right_ankle[0])) / scale,
        "left_extension": left_extension,
        "right_extension": right_extension,
        "side_signal": left_extension - right_extension,
        "head_offset": float((nose[0] - shoulder_center[0]) / scale),
        "torso_offset": float((shoulder_center[0] - hip_center[0]) / scale),
        "hand_vertical_diff": float((right_wrist[1] - left_wrist[1]) / scale),
        "hip_height": float(-hip_center[1]),
    }


def _phase_metric_series(points: np.ndarray) -> Dict[str, np.ndarray]:
    metric_rows = [_single_pose_metrics(frame) for frame in points]
    metric_names = metric_rows[0].keys() if metric_rows else []
    raw = {name: np.asarray([row[name] for row in metric_rows], dtype=np.float32) for name in metric_names}
    normalized = {f"{name}_norm": _series_normalize(values) for name, values in raw.items()}
    raw.update(normalized)
    return raw


def _fallback_substage_key(phase_id: int, progress: float) -> str:
    return get_substage_definition(phase_id, progress).key


def _state_machine_substage_key(phase_id: int, metrics: Dict[str, np.ndarray], idx: int, progress: float) -> str:
    hand_height = float(metrics["hand_height_norm"][idx])
    hand_down = float(metrics["hand_down_norm"][idx])
    hand_span = float(metrics["hand_span_norm"][idx])
    foot_span = float(metrics["foot_span_norm"][idx])
    side_signal = float(metrics["side_signal"][idx])
    head_offset = float(metrics["head_offset"][idx])
    torso_offset = float(metrics["torso_offset"][idx])
    vertical_diff = float(metrics["hand_vertical_diff"][idx])
    hip_height = float(metrics["hip_height_norm"][idx])

    if phase_id == 0:
        if progress < 0.75:
            return "open_step"
        return "hold_ball"
    if phase_id == 1:
        if hand_height > 0.72:
            return "support_sky"
        return "cross_raise"
    if phase_id == 2:
        if progress < 0.08 and hand_span < 0.55:
            return "horse_step"
        # 标准视频为镜像画面，左右标签需要与画面方向反向映射。
        return "draw_bow_right" if side_signal >= 0 else "draw_bow_left"
    if phase_id == 3:
        # 标准视频为镜像画面，左右单举标签反向映射。
        return "single_lift_right" if vertical_diff > 0 else "single_lift_left"
    if phase_id == 4:
        if abs(head_offset) < 0.045:
            return "press_down" if progress < 0.28 else "return_forward"
        return "look_left" if head_offset < 0 else "look_right"
    if phase_id == 5:
        if progress < 0.10:
            return "horse_press"
        if progress > 0.90 and abs(torso_offset) < 0.18:
            return "rise_return"
        return "waist_swing"
    if phase_id == 6:
        if progress > 0.88 and hand_down < 0.55:
            return "rise_spine"
        if hand_height > 0.50 or hand_down < 0.34:
            return "raise_press"
        if progress > 0.84 and hand_down < 0.62:
            return "rise_spine"
        return "bend_touch"
    if phase_id == 7:
        if progress < 0.12:
            return "horse_step"
        return "punch_left" if side_signal >= 0 else "punch_right"
    if phase_id == 8:
        return "lift_heel" if hip_height > 0.48 else "drop_shake"
    if phase_id == 9:
        if progress < 0.48 or hand_height > 0.45:
            return "gather_qi"
        return "close_step"
    return _fallback_substage_key(phase_id, progress)


def _smooth_short_substage_runs(keys: List[str], min_run: int) -> List[str]:
    if len(keys) < 3 or min_run <= 1:
        return keys
    smoothed = list(keys)
    idx = 0
    while idx < len(smoothed):
        end = idx + 1
        while end < len(smoothed) and smoothed[end] == smoothed[idx]:
            end += 1
        if end - idx < min_run:
            replacement = None
            if idx > 0:
                replacement = smoothed[idx - 1]
            elif end < len(smoothed):
                replacement = smoothed[end]
            if replacement:
                for replace_idx in range(idx, end):
                    smoothed[replace_idx] = replacement
        idx = end
    return smoothed


def _debounce_alternating_keys(keys: List[str], left_key: str, right_key: str, min_run: int) -> List[str]:
    if not keys:
        return keys
    out = list(keys)
    current = next((key for key in keys if key in {left_key, right_key}), None)
    candidate = None
    candidate_start = 0
    candidate_count = 0
    for idx, key in enumerate(keys):
        if key not in {left_key, right_key}:
            out[idx] = key
            candidate = None
            candidate_count = 0
            continue
        if current is None:
            current = key
        if key == current:
            out[idx] = current
            candidate = None
            candidate_count = 0
            continue
        if candidate != key:
            candidate = key
            candidate_start = idx
            candidate_count = 1
        else:
            candidate_count += 1
        if candidate_count >= min_run:
            current = key
            for replace_idx in range(candidate_start, idx + 1):
                out[replace_idx] = current
            candidate = None
            candidate_count = 0
        else:
            out[idx] = current
    return out


def _split_raise_sky_lowering(keys: List[str]) -> List[str]:
    out = list(keys)
    if not out:
        return out

    runs: List[tuple[str, int, int]] = []
    run_start = 0
    for idx in range(1, len(out) + 1):
        if idx == len(out) or out[idx] != out[run_start]:
            runs.append((out[run_start], run_start, idx))
            run_start = idx

    for run_idx, (run_key, start, end) in enumerate(runs):
        run_len = end - start
        previous_key = runs[run_idx - 1][0] if run_idx > 0 else ""
        next_key = runs[run_idx + 1][0] if run_idx + 1 < len(runs) else ""
        if run_key == "support_sky" and next_key == "cross_raise" and run_len >= 6:
            lower_len = max(2, int(round(run_len * 0.22)))
            for replace_idx in range(max(start, end - lower_len), end):
                out[replace_idx] = "lower_return"
        if run_key == "cross_raise" and previous_key == "support_sky" and run_len >= 4:
            lower_len = max(2, int(round(run_len * 0.42)))
            for replace_idx in range(start, min(end, start + lower_len)):
                out[replace_idx] = "lower_return"
        if run_idx == len(runs) - 1 and run_key == "cross_raise" and previous_key in {"support_sky", "lower_return"}:
            for replace_idx in range(start, end):
                out[replace_idx] = "lower_return"
    return out


def _force_phase_start_substage(keys: List[str], phase_id: int) -> List[str]:
    substages = BADUANJIN_SUBSTAGES.get(int(phase_id), [])
    if not keys or not substages:
        return keys
    if int(phase_id) in {5, 6}:
        return keys
    out = list(keys)
    force_len = max(3, int(round(len(out) * 0.04)))
    force_len = min(force_len, max(1, int(round(len(out) * 0.10))), len(out))
    first_key = substages[0].key
    for idx in range(force_len):
        out[idx] = first_key
    return out


def _build_state_machine_substage_keys(phase_ids: np.ndarray, time_s: np.ndarray, points: np.ndarray) -> List[str]:
    keys = [""] * int(phase_ids.shape[0])
    for phase_id in sorted({int(item) for item in phase_ids.tolist()}):
        phase_indices = np.flatnonzero(phase_ids == phase_id)
        if phase_indices.size == 0:
            continue
        phase_points = points[phase_indices]
        metrics = _phase_metric_series(phase_points)
        start_time = float(time_s[phase_indices[0]])
        end_time = float(time_s[phase_indices[-1]])
        denom = max(1e-6, end_time - start_time)
        local_keys: List[str] = []
        for local_idx, global_idx in enumerate(phase_indices):
            progress = (float(time_s[global_idx]) - start_time) / denom
            key = _state_machine_substage_key(phase_id, metrics, local_idx, progress)
            if get_substage_by_key(phase_id, key) is None:
                key = _fallback_substage_key(phase_id, progress)
            local_keys.append(key)
        if phase_id == 1:
            local_keys = _smooth_short_substage_runs(local_keys, min_run=max(2, int(round(len(local_keys) * 0.015))))
            local_keys = _split_raise_sky_lowering(local_keys)
        elif phase_id == 2:
            local_keys = _debounce_alternating_keys(
                local_keys,
                "draw_bow_left",
                "draw_bow_right",
                min_run=max(6, int(round(len(local_keys) * 0.06))),
            )
        elif phase_id == 7:
            local_keys = _debounce_alternating_keys(
                local_keys,
                "punch_left",
                "punch_right",
                min_run=max(5, int(round(len(local_keys) * 0.04))),
            )
        if phase_id != 1:
            local_keys = _smooth_short_substage_runs(local_keys, min_run=max(2, int(round(len(local_keys) * 0.015))))
        local_keys = _force_phase_start_substage(local_keys, phase_id)
        for global_idx, key in zip(phase_indices, local_keys):
            keys[int(global_idx)] = key
    return keys


def build_substage_metadata(
    phase_ids: np.ndarray,
    time_s: np.ndarray,
    points: np.ndarray | None = None,
    rule_config: Mapping[str, Any] | None = None,
) -> Dict[str, object]:
    keys: List[str] = []
    names: List[str] = []
    cues: List[str] = []
    rows: List[Dict[str, float | int | str]] = []
    if phase_ids.size == 0:
        return {"keys": np.asarray(keys), "names": np.asarray(names), "cues": np.asarray(cues), "rows": rows}

    state_keys: List[str] | None = None
    if points is not None and getattr(points, "shape", (0,))[0] == phase_ids.shape[0]:
        state_keys = _build_state_machine_substage_keys(phase_ids, time_s, np.asarray(points, dtype=np.float32))

    for idx, phase_id_raw in enumerate(phase_ids):
        phase_id = int(phase_id_raw)
        substage = get_substage_by_key(phase_id, state_keys[idx]) if state_keys is not None else None
        if substage is None:
            mask = phase_ids == phase_id
            phase_times = time_s[mask]
            start_time = float(phase_times[0]) if phase_times.size else float(time_s[idx])
            end_time = float(phase_times[-1]) if phase_times.size else float(time_s[idx])
            denom = max(1e-6, end_time - start_time)
            progress = (float(time_s[idx]) - start_time) / denom
            substage = get_substage_definition(phase_id, progress)
        keys.append(substage.key)
        names.append(substage.name)
        cues.append(substage.cue)

    start = 0
    current_key = keys[0] if keys else ""
    for idx in range(1, len(keys) + 1):
        if idx == len(keys) or keys[idx] != current_key:
            phase_id = int(phase_ids[start])
            substage = get_substage_by_key(phase_id, current_key)
            start_time_s = float(time_s[start])
            end_time_s = float(time_s[idx - 1])
            if phase_id == 0 and time_s.size:
                active_start_s = float(BADUANJIN_MANUAL_BOUNDARIES_S[0][0]) * (float(time_s[-1]) / float(BADUANJIN_STANDARD_DURATION_S))
                if end_time_s < active_start_s:
                    if idx < len(keys):
                        start = idx
                        current_key = keys[idx]
                    continue
                start_time_s = max(start_time_s, active_start_s)
            rows.append(
                {
                    "phase_id": phase_id,
                    "phase_name": get_phase_definition(phase_id, rule_config=rule_config).display_name,
                    "substage_key": current_key,
                    "substage_name": substage.name if substage is not None else current_key,
                    "cue": substage.cue if substage is not None else get_phase_definition(phase_id, rule_config=rule_config).cue,
                    "start_seq_idx": int(start),
                    "end_seq_idx": int(idx - 1),
                    "start_time_s": start_time_s,
                    "end_time_s": end_time_s,
                    "source": "state_machine" if state_keys is not None else "ratio",
                }
            )
            if idx < len(keys):
                start = idx
                current_key = keys[idx]
    return {"keys": np.asarray(keys), "names": np.asarray(names), "cues": np.asarray(cues), "rows": rows}


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
    return _phase_weight_vector_from_phase(phase, point_dims=point_dims, angle_dims=angle_dims)


def _phase_weight_vector_from_phase(
    phase: BaduanjinPhase,
    point_dims: int = 33 * 2,
    angle_dims: int = len(ANGLE_NAMES),
) -> np.ndarray:
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


def apply_phase_feature_weights(
    base_features: np.ndarray,
    phase_ids: np.ndarray,
    rule_config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    if base_features.ndim != 2:
        raise ValueError("apply_phase_feature_weights expects (T, D) features.")
    if phase_ids.shape[0] != base_features.shape[0]:
        raise ValueError("phase_ids length must equal feature length.")
    weighted = np.empty_like(base_features, dtype=np.float32)
    for idx in range(base_features.shape[0]):
        if rule_config:
            phase = get_phase_definition(int(phase_ids[idx]), rule_config=rule_config)
            weights = _phase_weight_vector_from_phase(phase, angle_dims=base_features.shape[1] - 66)
        else:
            weights = _phase_weight_vector(int(phase_ids[idx]), angle_dims=base_features.shape[1] - 66)
        weighted[idx] = base_features[idx] * weights
    return weighted


def weight_single_feature(feature: np.ndarray, phase_id: int, rule_config: Mapping[str, Any] | None = None) -> np.ndarray:
    if feature.ndim != 1:
        raise ValueError("weight_single_feature expects a 1D feature vector.")
    if rule_config:
        phase = get_phase_definition(int(phase_id), rule_config=rule_config)
        weights = _phase_weight_vector_from_phase(phase, angle_dims=feature.shape[0] - 66)
    else:
        weights = _phase_weight_vector(int(phase_id), angle_dims=feature.shape[0] - 66)
    return feature * weights


def phase_metadata_rows(
    phase_ids: np.ndarray,
    time_s: np.ndarray,
    rule_config: Mapping[str, Any] | None = None,
) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    if phase_ids.size == 0:
        return rows
    start = 0
    current = int(phase_ids[0])
    for idx in range(1, len(phase_ids) + 1):
        if idx == len(phase_ids) or int(phase_ids[idx]) != current:
            phase = get_phase_definition(current, rule_config=rule_config)
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


def build_baduanjin_hint_text(
    phase_id: int | None,
    part: str,
    dx: float,
    dy: float,
    substage_key: str | None = None,
    rule_config: Mapping[str, Any] | None = None,
) -> str:
    if phase_id is not None:
        phase = get_phase_definition(phase_id, rule_config=rule_config)
        substage = get_substage_by_key(phase.phase_id, substage_key)
        if substage is not None:
            if part in substage.hint_templates:
                return f"{substage.name}：{substage.hint_templates[part]}"
            return f"{substage.name}阶段请重点关注{part}，{_direction_phrase(part, dx, dy)}。"
        if part in phase.hint_templates:
            return phase.hint_templates[part]
        return f"{phase.display_name}阶段请重点关注{part}，{_direction_phrase(part, dx, dy)}。"
    return f"{part}与标准动作仍有偏差，{_direction_phrase(part, dx, dy)}。"

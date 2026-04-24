from __future__ import annotations

"""姿态模型选项定义。

模板提取更偏向精度，因此默认使用 heavy；
实时跟练更偏向延迟，因此前端通常会显式选择 lite。
"""

from pathlib import Path


POSE_MODEL_OPTIONS: dict[str, dict[str, str]] = {
    "lite": {
        "label": "速度优先（Lite）",
        "task_file": "pose_landmarker_lite.task",
        "description": "适合实时跟练场景，延迟更低，但稳定性和精度相对弱一些。",
    },
    "full": {
        "label": "平衡模式（Full）",
        "task_file": "pose_landmarker_full.task",
        "description": "适合通用场景，在速度和精度之间做折中。",
    },
    "heavy": {
        "label": "精度优先（Heavy）",
        "task_file": "pose_landmarker_heavy.task",
        "description": "适合模板提取和离线分析，精度更高，但推理更慢。",
    },
}

FOLLOW_TEMPLATE_MODEL_KEY = "follow_template"
DEFAULT_TEMPLATE_MODEL_KEY = "heavy"
DEFAULT_RUNTIME_MODEL_KEY = FOLLOW_TEMPLATE_MODEL_KEY


def get_pose_model_label(model_key: str) -> str:
    """返回模型中文标签。"""

    return POSE_MODEL_OPTIONS.get(model_key, {}).get("label", model_key)


def get_pose_model_choices(include_follow_template: bool = False) -> list[tuple[str, str]]:
    """返回 Django / 前端表单可直接使用的选项列表。"""

    choices = []
    if include_follow_template:
        choices.append((FOLLOW_TEMPLATE_MODEL_KEY, "跟随模板模型"))
    choices.extend((key, meta["label"]) for key, meta in POSE_MODEL_OPTIONS.items())
    return choices


def is_valid_pose_model_key(model_key: str | None, include_follow_template: bool = False) -> bool:
    """判断传入的模型 key 是否有效。"""

    if not model_key:
        return False
    if include_follow_template and model_key == FOLLOW_TEMPLATE_MODEL_KEY:
        return True
    return model_key in POSE_MODEL_OPTIONS


def normalize_pose_model_key(
    model_key: str | None,
    *,
    default: str = DEFAULT_TEMPLATE_MODEL_KEY,
    include_follow_template: bool = False,
) -> str:
    """把前端传来的模型 key 规范化成系统支持的值。"""

    if is_valid_pose_model_key(model_key, include_follow_template=include_follow_template):
        return str(model_key)
    return default


def resolve_pose_model_path(model_key: str) -> str:
    """把模型 key 转成仓库内 `.task` 文件的绝对路径。"""

    normalized = normalize_pose_model_key(model_key, default=DEFAULT_TEMPLATE_MODEL_KEY)
    task_file = POSE_MODEL_OPTIONS[normalized]["task_file"]
    base_dir = Path(__file__).resolve().parents[1]
    return str(base_dir / task_file)

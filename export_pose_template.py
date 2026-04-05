import argparse

from fitness_action_eval.pipeline import save_pose_template


FIXED_TASK_MODEL = "pose_landmarker_lite.task"
FIXED_NUM_POSES = 1
FIXED_SMOOTH_WINDOW = 7


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="导出标准动作模板文件。")
    p.add_argument("--ref_video", required=True, help="标准动作参考视频。")
    p.add_argument("--template_path", required=True, help="导出的模板文件路径（.npz）。")
    p.add_argument("--preview", action="store_true", help="实时显示参考视频姿态提取预览。")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = save_pose_template(
        ref_video=args.ref_video,
        task_model=FIXED_TASK_MODEL,
        num_poses=FIXED_NUM_POSES,
        smooth_window=FIXED_SMOOTH_WINDOW,
        template_path=args.template_path,
        preview=bool(args.preview),
    )
    print("[CONFIG] task_model:", FIXED_TASK_MODEL)
    print("[CONFIG] num_poses:", FIXED_NUM_POSES)
    print("[OK] template:", summary["template_path"])
    print("[OK] reference_video:", summary["reference_video"])
    print("[OK] reference_length:", summary["reference_length"])


if __name__ == "__main__":
    main()

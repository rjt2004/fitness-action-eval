import argparse

from fitness_action_eval.pipeline import save_pose_template


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="导出八段锦标准动作模板文件。")
    p.add_argument("--ref_video", required=True, help="标准动作参考视频。")
    p.add_argument("--template_path", required=True, help="导出的模板文件路径（.npz）。")
    p.add_argument("--task_model", default="pose_landmarker_lite.task", help="MediaPipe 姿态模型路径。")
    p.add_argument("--num_poses", type=int, default=1, help="单帧最大检测人数。")
    p.add_argument("--smooth_window", type=int, default=7, help="姿态平滑窗口。")
    p.add_argument("--frame_stride", type=int, default=4, help="抽帧步长，长视频建议 3~5。")
    p.add_argument("--preview", action="store_true", help="实时显示参考视频姿态提取预览。")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = save_pose_template(
        ref_video=args.ref_video,
        task_model=args.task_model,
        num_poses=args.num_poses,
        smooth_window=args.smooth_window,
        template_path=args.template_path,
        frame_stride=args.frame_stride,
        preview=bool(args.preview),
    )
    print("[CONFIG] task_model:", args.task_model)
    print("[CONFIG] num_poses:", args.num_poses)
    print("[CONFIG] frame_stride:", args.frame_stride)
    print("[OK] template:", summary["template_path"])
    print("[OK] reference_video:", summary["reference_video"])
    print("[OK] reference_length:", summary["reference_length"])


if __name__ == "__main__":
    main()

import argparse

from fitness_action_eval.pipeline import run_camera_coach


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="八段锦实时摄像头/视频跟练纠错演示。")
    p.add_argument("--template_path", default="output/baduanjin_template.npz", help="参考模板文件路径。")
    p.add_argument("--ref_video", default=None, help="参考视频路径；若模板不存在则用它生成模板。")
    p.add_argument("--camera_source", default="0", help="摄像头编号或视频路径，默认 0。")
    p.add_argument("--task_model", default="pose_landmarker_lite.task", help="MediaPipe 姿态模型路径。")
    p.add_argument("--num_poses", type=int, default=1, help="单帧最大检测人数。")
    p.add_argument("--smooth_window", type=int, default=7, help="姿态平滑窗口。")
    p.add_argument("--frame_stride", type=int, default=4, help="生成模板时的抽帧步长。")
    p.add_argument("--score_scale", type=float, default=8.0, help="单帧距离到分数的映射尺度。")
    p.add_argument("--hint_threshold", type=float, default=0.18, help="实时提示触发阈值。")
    p.add_argument("--ref_search_window", type=int, default=90, help="参考序列前向搜索窗口。")
    p.add_argument("--camera_width", type=int, default=None, help="摄像头采集宽度。")
    p.add_argument("--camera_height", type=int, default=None, help="摄像头采集高度。")
    p.add_argument("--camera_mirror", action="store_true", help="是否对摄像头画面做水平镜像。")
    p.add_argument("--out_json", default="output/baduanjin_camera_summary.json", help="实时跟练摘要 JSON。")
    p.add_argument("--out_video", default="output/baduanjin_camera_overlay.mp4", help="实时跟练对比视频输出路径。")
    p.add_argument("--max_frames", type=int, default=None, help="最大处理帧数，调试时可设置。")
    p.add_argument("--preview", action="store_true", help="实时预览，按 q 退出。")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_camera_coach(
        template_path=args.template_path,
        ref_video=args.ref_video,
        camera_source=args.camera_source,
        task_model=args.task_model,
        num_poses=args.num_poses,
        smooth_window=args.smooth_window,
        score_scale=args.score_scale,
        hint_threshold=args.hint_threshold,
        ref_search_window=args.ref_search_window,
        frame_stride=args.frame_stride,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_mirror=bool(args.camera_mirror),
        out_json=args.out_json,
        out_video=args.out_video,
        preview=bool(args.preview),
        max_frames=args.max_frames,
    )
    print("[OK] live compare video:", args.out_video)
    print("[OK] live summary json:", args.out_json)
    print("[LIVE] average score:", f"{summary['avg_score_0_100']:.2f}")
    print("[LIVE] matched frames:", summary["matched_frames"])
    print("[LIVE] final phase:", summary["final_phase_name"])


if __name__ == "__main__":
    main()

import argparse

from fitness_action_eval.pipeline import run_dtw_scoring


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="基于八段锦姿态与 DTW 的连续动作评分演示。")
    p.add_argument("--ref_video", required=True, help="标准动作参考视频。")
    p.add_argument("--query_video", required=True, help="待评分动作视频。")
    p.add_argument("--task_model", default="pose_landmarker_lite.task", help="MediaPipe 姿态模型路径。")
    p.add_argument("--num_poses", type=int, default=1, help="单帧最大检测人数。")
    p.add_argument("--smooth_window", type=int, default=7, help="姿态平滑窗口。")
    p.add_argument("--frame_stride", type=int, default=4, help="抽帧步长，长视频建议 3~5。")
    p.add_argument("--score_scale", type=float, default=8.0, help="DTW 距离到分数的映射尺度。")
    p.add_argument("--hint_threshold", type=float, default=0.18, help="实时提示触发阈值。")
    p.add_argument("--hint_min_interval", type=int, default=8, help="相邻两次提示的最小间隔。")
    p.add_argument("--max_hints", type=int, default=40, help="最多输出的提示条数。")
    p.add_argument("--out_json", default="output/baduanjin_result.json", help="评分结果 JSON 输出路径。")
    p.add_argument("--out_plot", default="output/baduanjin_plot.png", help="DTW 对齐曲线图输出路径。")
    p.add_argument("--out_video", default="output/baduanjin_overlay.mp4", help="对比版评分视频输出路径。")
    p.add_argument(
        "--preview",
        action="store_true",
        help="实时显示参考/待测对比预览，按 q 可关闭窗口。",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_dtw_scoring(
        ref_video=args.ref_video,
        query_video=args.query_video,
        task_model=args.task_model,
        num_poses=args.num_poses,
        smooth_window=args.smooth_window,
        score_scale=args.score_scale,
        hint_threshold=args.hint_threshold,
        hint_min_interval=args.hint_min_interval,
        max_hints=args.max_hints,
        out_json=args.out_json,
        out_plot=args.out_plot,
        out_video=args.out_video,
        frame_stride=args.frame_stride,
        preview=bool(args.preview),
    )

    print("[CONFIG] task_model:", args.task_model)
    print("[CONFIG] num_poses:", args.num_poses)
    print("[CONFIG] frame_stride:", args.frame_stride)
    print("[OK] coaching video:", args.out_video)
    print("[DTW] normalized distance:", f"{summary['norm_dist']:.6f}")
    print("[DTW] score:", f"{summary['score']:.2f}")
    print("[DTW] hints:", summary["hint_count"])
    print("[OK] result json:", args.out_json)
    print("[OK] plot png:", args.out_plot)


if __name__ == "__main__":
    main()

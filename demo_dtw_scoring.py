import argparse

from fitness_action_eval.pipeline import run_dtw_scoring


TASK_MODEL = "pose_landmarker_lite.task"
NUM_POSES = 1
SMOOTH_WINDOW = 7
SCORE_SCALE = 6.0
HINT_THRESHOLD = 0.22
HINT_MIN_INTERVAL = 8
MAX_HINTS = 40


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="基于姿态与 DTW 的连续健身动作评分演示。")
    p.add_argument("--ref_video", required=True, help="标准动作参考视频。")
    p.add_argument("--query_video", required=True, help="待评分动作视频。")
    p.add_argument("--out_json", default="output/dtw_coach_result.json", help="评分结果 JSON 输出路径。")
    p.add_argument("--out_plot", default="output/dtw_coach_plot.png", help="DTW 对齐曲线图输出路径。")
    p.add_argument("--out_video", default="output/dtw_coach_overlay.mp4", help="对比版评分视频输出路径。")
    p.add_argument(
        "--preview",
        action="store_true",
        help="实时显示双窗口对比预览，按 q 可关闭窗口。",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_dtw_scoring(
        ref_video=args.ref_video,
        query_video=args.query_video,
        task_model=TASK_MODEL,
        num_poses=NUM_POSES,
        smooth_window=SMOOTH_WINDOW,
        score_scale=SCORE_SCALE,
        hint_threshold=HINT_THRESHOLD,
        hint_min_interval=HINT_MIN_INTERVAL,
        max_hints=MAX_HINTS,
        out_json=args.out_json,
        out_plot=args.out_plot,
        out_video=args.out_video,
        preview=bool(args.preview),
    )

    print("[CONFIG] task_model:", TASK_MODEL)
    print("[CONFIG] num_poses:", NUM_POSES)
    print("[OK] coaching video:", args.out_video)
    print("[DTW] normalized distance:", f"{summary['norm_dist']:.6f}")
    print("[DTW] score:", f"{summary['score']:.2f}")
    print("[DTW] hints:", summary["hint_count"])
    print("[OK] result json:", args.out_json)
    print("[OK] plot png:", args.out_plot)


if __name__ == "__main__":
    main()

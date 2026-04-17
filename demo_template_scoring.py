import argparse

from fitness_action_eval.pipeline import run_dtw_scoring_from_template


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="基于已导出八段锦模板的动作评分演示。")
    p.add_argument("--template_path", required=True, help="标准动作模板文件路径（.npz）。")
    p.add_argument("--query_video", required=True, help="待评分动作视频。")
    p.add_argument("--score_scale", type=float, default=8.0, help="DTW 距离到分数的映射尺度。")
    p.add_argument("--hint_threshold", type=float, default=0.18, help="实时提示触发阈值。")
    p.add_argument("--hint_min_interval", type=int, default=8, help="相邻两次提示的最小间隔。")
    p.add_argument("--max_hints", type=int, default=40, help="最多输出的提示条数。")
    p.add_argument("--out_json", default="output/template_result.json", help="评分结果 JSON 输出路径。")
    p.add_argument("--out_plot", default="output/template_plot.png", help="DTW 对齐曲线图输出路径。")
    p.add_argument("--out_video", default="output/template_overlay.mp4", help="对比版评分视频输出路径。")
    p.add_argument("--preview", action="store_true", help="实时显示双窗口对比预览，按 q 可关闭窗口。")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_dtw_scoring_from_template(
        template_path=args.template_path,
        query_video=args.query_video,
        out_json=args.out_json,
        out_plot=args.out_plot,
        out_video=args.out_video,
        preview=bool(args.preview),
        score_scale=args.score_scale,
        hint_threshold=args.hint_threshold,
        hint_min_interval=args.hint_min_interval,
        max_hints=args.max_hints,
    )
    print("[OK] template:", summary["template_path"])
    print("[OK] coaching video:", args.out_video)
    print("[DTW] normalized distance:", f"{summary['norm_dist']:.6f}")
    print("[DTW] score:", f"{summary['score']:.2f}")
    print("[DTW] hints:", summary["hint_count"])
    print("[OK] result json:", args.out_json)
    print("[OK] plot png:", args.out_plot)


if __name__ == "__main__":
    main()

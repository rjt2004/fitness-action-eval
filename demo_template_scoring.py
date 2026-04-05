import argparse

from fitness_action_eval.pipeline import run_dtw_scoring_from_template


FIXED_SCORE_SCALE = 6.0
FIXED_HINT_THRESHOLD = 0.22


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="基于已导出模板的动作评分演示。")
    p.add_argument("--template_path", required=True, help="标准动作模板文件路径（.npz）。")
    p.add_argument("--query_video", required=True, help="待评分动作视频。")
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
        score_scale=FIXED_SCORE_SCALE,
        hint_threshold=FIXED_HINT_THRESHOLD,
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

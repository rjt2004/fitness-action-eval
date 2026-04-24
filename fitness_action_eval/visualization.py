"""可视化与视频渲染工具。"""

import os
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 0 鼻子，1 左眼内侧，2 左眼，3 左眼外侧，4 右眼内侧，5 右眼，6 右眼外侧，7 左耳，8 右耳
# 9 嘴左侧，10 嘴右侧，11 左肩，12 右肩，13 左肘，14 右肘，15 左腕，16 右腕
# 17 左小指，18 右小指，19 左食指，20 右食指，21 左拇指，22 右拇指
# 23 左髋，24 右髋，25 左膝，26 右膝，27 左踝，28 右踝，29 左脚跟，30 右脚跟，31 左脚尖，32 右脚尖
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]


FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


@lru_cache(maxsize=8)
def get_chinese_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


@lru_cache(maxsize=1)
def get_matplotlib_font_path() -> Optional[str]:
    for font_path in FONT_CANDIDATES:
        if os.path.exists(font_path):
            return font_path
    return None


def configure_matplotlib_chinese(plt) -> None:
    font_path = get_matplotlib_font_path()
    if font_path:
        from matplotlib import font_manager

        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.sans-serif"] = [font_name]
        plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def draw_utf8_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int,
    color: Tuple[int, int, int],
    shadow_color: Optional[Tuple[int, int, int]] = None,
    shadow_offset: Tuple[int, int] = (2, 2),
) -> None:
    if not text:
        return
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = get_chinese_font(font_size)
    x, y = position
    if shadow_color is not None:
        draw.text((x + shadow_offset[0], y + shadow_offset[1]), text, font=font, fill=shadow_color)
    draw.text((x, y), text, font=font, fill=color)
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def measure_text(text: str, font_size: int) -> Tuple[int, int]:
    font = get_chinese_font(font_size)
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def draw_pose_skeleton(frame: np.ndarray, points_xy: np.ndarray) -> None:
    if points_xy.shape != (33, 2):
        return
    height, width = frame.shape[:2]
    pts_px: List[Tuple[int, int]] = []
    for x, y in points_xy:
        px = int(np.clip(x * width, 0, width - 1))
        py = int(np.clip(y * height, 0, height - 1))
        pts_px.append((px, py))

    for a, b in POSE_CONNECTIONS:
        if a < len(pts_px) and b < len(pts_px):
            cv2.line(frame, pts_px[a], pts_px[b], (0, 180, 255), 2, cv2.LINE_AA)

    for px, py in pts_px:
        cv2.circle(frame, (px, py), 3, (80, 255, 120), -1, cv2.LINE_AA)


def draw_text_block(
    frame: np.ndarray,
    lines: List[str],
    x: int,
    y: int,
    bg_color: Tuple[int, int, int] = (20, 20, 20),
    fg_color: Tuple[int, int, int] = (235, 235, 235),
) -> None:
    if not lines:
        return
    font_size = 22
    line_h = 32
    pad = 10
    width = 0
    for text in lines:
        text_w, _ = measure_text(text, font_size)
        width = max(width, text_w)
    panel_w = width + pad * 2
    panel_h = line_h * len(lines) + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0.0, frame)

    cy = y + pad
    for text in lines:
        draw_utf8_text(frame, text, (x + pad, cy), font_size, fg_color, shadow_color=(0, 0, 0))
        cy += line_h


def preview_frame(window_name: str, frame: np.ndarray, wait_ms: int = 1) -> bool:
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(wait_ms) & 0xFF
    return key != ord("q")


def close_preview_windows() -> None:
    cv2.destroyAllWindows()


def resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height == target_height:
        return frame
    scale = target_height / max(1, height)
    new_width = max(1, int(round(width * scale)))
    return cv2.resize(frame, (new_width, target_height), interpolation=cv2.INTER_LINEAR)


def pad_to_width(frame: np.ndarray, target_width: int) -> np.ndarray:
    _, width = frame.shape[:2]
    if width >= target_width:
        return frame
    pad_total = target_width - width
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return cv2.copyMakeBorder(frame, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(16, 16, 16))


def compose_compare_frame(
    ref_frame: np.ndarray,
    qry_frame: np.ndarray,
    score: float,
    current_local_err: float,
    active_hint: str,
    align_info: Optional[Dict[str, int]],
    phase_name: str = "",
    phase_cue: str = "",
    max_panel_height: int = 720,
) -> np.ndarray:
    target_height = min(max(ref_frame.shape[0], qry_frame.shape[0]), max_panel_height)
    ref_view = resize_to_height(ref_frame, target_height)
    qry_view = resize_to_height(qry_frame, target_height)
    target_width = max(ref_view.shape[1], qry_view.shape[1])
    ref_view = pad_to_width(ref_view, target_width)
    qry_view = pad_to_width(qry_view, target_width)
    canvas = np.concatenate([ref_view, qry_view], axis=1)

    divider_x = ref_view.shape[1]
    cv2.line(canvas, (divider_x, 0), (divider_x, canvas.shape[0] - 1), (0, 140, 255), 2, cv2.LINE_AA)
    draw_utf8_text(canvas, "参考动作", (20, 10), 28, (255, 220, 120), shadow_color=(0, 0, 0))
    draw_utf8_text(canvas, "待测动作", (divider_x + 20, 10), 28, (120, 255, 180), shadow_color=(0, 0, 0))

    lines = [f"最终得分：{score:.1f}/100"]
    if np.isfinite(current_local_err):
        lines.append(f"局部偏差：{current_local_err:.3f}")
    if phase_name:
        lines.append(f"当前动作：{phase_name}")
    if phase_cue:
        lines.append(f"动作要领：{phase_cue}")
    if active_hint:
        lines.append(f"实时提示：{active_hint}")
    if align_info is not None:
        progress = align_info["path_step"] + 1
        total = align_info["path_total"]
        lines.append(f"DTW 对齐进度：{progress}/{total}")
        lines.append(f"对齐序列：参考 {align_info['ref_seq_idx']} -> 待测 {align_info['qry_seq_idx']}")
        lines.append(f"对齐帧号：参考 {align_info['ref_frame']} -> 待测 {align_info['qry_frame']}")
    draw_text_block(canvas, lines, x=20, y=50)
    return canvas


def compose_live_query_frame(
    qry_frame: np.ndarray,
    score: float,
    current_local_err: float,
    active_hint: str,
    phase_name: str = "",
    phase_cue: str = "",
) -> np.ndarray:
    return qry_frame.copy()


def compose_error_frame(
    ref_frame: np.ndarray,
    qry_frame: np.ndarray,
    phase_name: str,
    part: str,
    local_error: float,
    active_hint: str,
    query_time_s: float,
) -> np.ndarray:
    target_height = min(max(ref_frame.shape[0], qry_frame.shape[0]), 520)
    ref_view = resize_to_height(ref_frame, target_height)
    qry_view = resize_to_height(qry_frame, target_height)
    target_width = max(ref_view.shape[1], qry_view.shape[1])
    ref_view = pad_to_width(ref_view, target_width)
    qry_view = pad_to_width(qry_view, target_width)
    canvas = np.concatenate([ref_view, qry_view], axis=1)
    divider_x = ref_view.shape[1]
    cv2.line(canvas, (divider_x, 0), (divider_x, canvas.shape[0] - 1), (0, 140, 255), 2, cv2.LINE_AA)
    draw_utf8_text(canvas, "标准动作", (20, 10), 28, (255, 220, 120), shadow_color=(0, 0, 0))
    draw_utf8_text(canvas, "实时动作", (divider_x + 20, 10), 28, (120, 255, 180), shadow_color=(0, 0, 0))

    lines = [
        f"动作阶段：{phase_name or '--'}",
        f"关注部位：{part or '--'}",
        f"局部偏差：{local_error:.3f}" if np.isfinite(local_error) else "局部偏差：--",
        f"触发时间：{query_time_s:.1f}s",
    ]
    if active_hint:
        lines.append(f"纠错提示：{active_hint}")
    draw_text_block(canvas, lines, x=20, y=52)
    return canvas


def save_plot(
    ref_data: Dict[str, np.ndarray],
    qry_data: Dict[str, np.ndarray],
    path: List[Tuple[int, int]],
    hints: List[Dict[str, object]],
    out_png: str,
    score: float,
    norm_dist: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Skip plot export because matplotlib is unavailable: {exc}")
        return
    configure_matplotlib_chinese(plt)

    ensure_parent_dir(out_png)
    ref_proxy = np.linalg.norm(ref_data["points"].reshape(ref_data["points"].shape[0], -1), axis=1)
    qry_proxy = np.linalg.norm(qry_data["points"].reshape(qry_data["points"].shape[0], -1), axis=1)

    plt.figure(figsize=(11, 5))
    plt.plot(ref_data["time_s"], ref_proxy, label="Reference", linewidth=2)
    plt.plot(qry_data["time_s"], qry_proxy, label="Query", linewidth=2)

    step = max(1, len(path) // 80)
    for i, j in path[::step]:
        plt.plot(
            [ref_data["time_s"][i], qry_data["time_s"][j]],
            [ref_proxy[i], qry_proxy[j]],
            color="gray",
            alpha=0.12,
            linewidth=0.8,
        )

    for hint in hints[:10]:
        q_idx = int(hint["query_index"])
        q_t = float(qry_data["time_s"][q_idx])
        plt.axvline(x=q_t, color="orange", alpha=0.18, linewidth=1)

    plt.title(f"DTW Pose Coaching | score={score:.1f} | norm_dist={norm_dist:.4f}")
    plt.xlabel("Time (s)")
    plt.ylabel("Pose Feature Norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_phase_plots(
    ref_data: Dict[str, np.ndarray],
    qry_data: Dict[str, np.ndarray],
    path: List[Tuple[int, int]],
    hints: List[Dict[str, object]],
    out_dir: str,
    score_scale: float,
) -> List[Dict[str, object]]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Skip phase plot export because matplotlib is unavailable: {exc}")
        return []
    configure_matplotlib_chinese(plt)

    from fitness_action_eval.dtw import distance_to_score

    os.makedirs(out_dir, exist_ok=True)

    ref_proxy = np.linalg.norm(ref_data["points"].reshape(ref_data["points"].shape[0], -1), axis=1)
    qry_proxy = np.linalg.norm(qry_data["points"].reshape(qry_data["points"].shape[0], -1), axis=1)
    ref_phase_rows = {int(row["phase_id"]): row for row in ref_data["phase_rows"]}
    qry_phase_rows = {int(row["phase_id"]): row for row in qry_data["phase_rows"]}
    phase_hints: Dict[int, List[Dict[str, object]]] = {}
    for hint in hints:
        phase_id = int(hint.get("phase_id", -1))
        phase_hints.setdefault(phase_id, []).append(hint)

    phase_plot_rows: List[Dict[str, object]] = []
    for phase_id in sorted(set(ref_phase_rows) & set(qry_phase_rows)):
        ref_row = ref_phase_rows[phase_id]
        qry_row = qry_phase_rows[phase_id]
        ref_start = int(ref_row["start_seq_idx"])
        ref_end = int(ref_row["end_seq_idx"])
        qry_start = int(qry_row["start_seq_idx"])
        qry_end = int(qry_row["end_seq_idx"])
        phase_path = [
            (ref_idx, qry_idx)
            for ref_idx, qry_idx in path
            if ref_start <= ref_idx <= ref_end and qry_start <= qry_idx <= qry_end
        ]
        if not phase_path:
            continue

        ref_zero = float(ref_data["time_s"][ref_start])
        qry_zero = float(qry_data["time_s"][qry_start])
        ref_phase_times = ref_data["time_s"][ref_start : ref_end + 1] - ref_zero
        qry_phase_times = qry_data["time_s"][qry_start : qry_end + 1] - qry_zero
        phase_times = [float(qry_data["time_s"][qry_idx] - qry_zero) for _, qry_idx in phase_path]
        local_distances = [
            float(np.linalg.norm(ref_data["features"][ref_idx] - qry_data["features"][qry_idx]))
            for ref_idx, qry_idx in phase_path
        ]
        local_norm_dist = float(np.mean(local_distances))
        local_score = float(distance_to_score(local_norm_dist, score_scale))
        peak_idx = int(np.argmax(local_distances))
        peak_time = phase_times[peak_idx]
        peak_error = local_distances[peak_idx]

        output_path = os.path.join(out_dir, f"phase_{phase_id:02d}.png")
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            figsize=(8.4, 5.6),
            sharex=False,
            gridspec_kw={"height_ratios": [1.1, 1.0]},
        )

        ax_top.plot(
            ref_phase_times,
            ref_proxy[ref_start : ref_end + 1],
            label="标准模板",
            color="#2563eb",
            linewidth=2,
        )
        ax_top.plot(
            qry_phase_times,
            qry_proxy[qry_start : qry_end + 1],
            label="测试动作",
            color="#16a34a",
            linewidth=2,
        )
        ax_top.set_title(f"{qry_row['phase_name']} | 分数={local_score:.1f} | 平均误差={local_norm_dist:.4f}")
        ax_top.set_ylabel("姿态代理值")
        ax_top.grid(alpha=0.18, linestyle="--")
        ax_top.legend(loc="upper right")

        ax_bottom.plot(
            phase_times,
            local_distances,
            label="局部对齐误差",
            color="#dc2626",
            linewidth=2,
        )
        ax_bottom.fill_between(phase_times, local_distances, color="#fca5a5", alpha=0.25)
        ax_bottom.axhline(
            y=local_norm_dist,
            color="#64748b",
            linestyle="--",
            linewidth=1.1,
            label=f"平均误差={local_norm_dist:.3f}",
        )
        ax_bottom.scatter(
            [peak_time],
            [peak_error],
            color="#991b1b",
            s=42,
            zorder=3,
            label=f"峰值误差={peak_error:.3f}",
        )

        for hint_idx, hint in enumerate(phase_hints.get(phase_id, [])[:6]):
            query_time_s = float(hint.get("query_time_s", 0.0)) - qry_zero
            ax_bottom.axvline(
                x=query_time_s,
                color="#f59e0b",
                alpha=0.35,
                linewidth=1.2,
                label="提示触发点" if hint_idx == 0 else None,
            )

        ax_bottom.set_xlabel("阶段相对时间（秒）")
        ax_bottom.set_ylabel("局部对齐误差")
        ax_bottom.grid(alpha=0.18, linestyle="--")
        ax_bottom.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        phase_plot_rows.append(
            {
                "phase_id": phase_id,
                "phase_name": str(qry_row["phase_name"]),
                "cue": str(qry_row.get("cue", "")),
                "image_path": output_path,
                "score": round(local_score, 2),
                "normalized_distance": round(local_norm_dist, 4),
                "alignment_points": len(phase_path),
            }
        )

    return phase_plot_rows


def get_aligned_reference_frame(
    ref_cap: Optional[cv2.VideoCapture],
    target_ref_frame: int,
    current_ref_frame_idx: int,
    current_ref_frame: Optional[np.ndarray],
) -> Tuple[int, Optional[np.ndarray]]:
    if ref_cap is None:
        return current_ref_frame_idx, current_ref_frame
    if target_ref_frame < current_ref_frame_idx:
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, target_ref_frame)
        current_ref_frame_idx = target_ref_frame - 1
        current_ref_frame = None
    while current_ref_frame_idx < target_ref_frame:
        ok_ref, fetched = ref_cap.read()
        if not ok_ref:
            break
        current_ref_frame_idx += 1
        current_ref_frame = fetched
    return current_ref_frame_idx, current_ref_frame


def render_feedback_video(
    ref_video: Optional[str],
    query_video: str,
    output_video: Optional[str],
    score: float,
    frame_hint_map: Dict[int, str],
    frame_error_map: Dict[int, float],
    frame_pose_map: Dict[int, np.ndarray],
    ref_pose_map: Optional[Dict[int, np.ndarray]] = None,
    alignment_map: Optional[Dict[int, Dict[str, int]]] = None,
    frame_phase_map: Optional[Dict[int, str]] = None,
    frame_cue_map: Optional[Dict[int, str]] = None,
    preview: bool = False,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    progress_range: Tuple[int, int] = (72, 92),
    compare_panel_height: int = 540,
) -> None:
    cap = cv2.VideoCapture(query_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {query_video}")

    use_compare_mode = bool(ref_video and alignment_map)
    ref_cap = None
    if use_compare_mode:
        ref_cap = cv2.VideoCapture(ref_video)
        if not ref_cap.isOpened():
            cap.release()
            raise FileNotFoundError(f"Cannot open video: {ref_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    writer = None
    writer_kind: Optional[str] = None
    if output_video:
        ensure_parent_dir(output_video)

    frame_idx = 0
    active_hint = ""
    active_hint_left = 0
    current_local_err = float("nan")
    keep_frames = max(1, int(round(fps * 1.0)))
    current_align_info: Optional[Dict[str, int]] = None
    current_ref_frame_idx = -1
    current_ref_frame: Optional[np.ndarray] = None
    current_phase_name = ""
    current_phase_cue = ""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress_start, progress_end = progress_range
    progress_span = max(0, progress_end - progress_start)

    while True:
        ok, query_frame = cap.read()
        if not ok:
            break

        if frame_idx in frame_hint_map:
            active_hint = frame_hint_map[frame_idx]
            active_hint_left = keep_frames
        if frame_idx in frame_error_map:
            current_local_err = frame_error_map[frame_idx]
        if frame_phase_map is not None and frame_idx in frame_phase_map:
            current_phase_name = frame_phase_map[frame_idx]
        if frame_cue_map is not None and frame_idx in frame_cue_map:
            current_phase_cue = frame_cue_map[frame_idx]

        query_view = query_frame.copy()
        if frame_idx in frame_pose_map:
            draw_pose_skeleton(query_view, frame_pose_map[frame_idx])

        if use_compare_mode and alignment_map is not None and frame_idx in alignment_map:
            current_align_info = dict(alignment_map[frame_idx])
            current_align_info["qry_frame"] = int(frame_idx)
            target_ref_frame = current_align_info["ref_frame"]
            current_ref_frame_idx, current_ref_frame = get_aligned_reference_frame(
                ref_cap=ref_cap,
                target_ref_frame=target_ref_frame,
                current_ref_frame_idx=current_ref_frame_idx,
                current_ref_frame=current_ref_frame,
            )

        hint_text = active_hint if active_hint_left > 0 else ""
        if active_hint_left > 0:
            active_hint_left -= 1

        if use_compare_mode:
            if current_ref_frame is None:
                ref_view = np.zeros_like(query_view)
            else:
                ref_view = current_ref_frame.copy()
                if current_align_info is not None and ref_pose_map is not None:
                    target_ref_frame = current_align_info["ref_frame"]
                    if target_ref_frame in ref_pose_map:
                        draw_pose_skeleton(ref_view, ref_pose_map[target_ref_frame])
            output_frame = compose_compare_frame(
                ref_frame=ref_view,
                qry_frame=query_view,
                score=score,
                current_local_err=current_local_err,
                active_hint=hint_text,
                align_info=current_align_info,
                phase_name=current_phase_name,
                phase_cue=current_phase_cue,
                max_panel_height=compare_panel_height,
            )
        else:
            lines = [f"最终得分：{score:.1f}/100"]
            if np.isfinite(current_local_err):
                lines.append(f"局部偏差：{current_local_err:.3f}")
            if current_phase_name:
                lines.append(f"当前动作：{current_phase_name}")
            if current_phase_cue:
                lines.append(f"动作要领：{current_phase_cue}")
            if hint_text:
                lines.append(f"实时提示：{hint_text}")
            draw_text_block(query_view, lines, x=20, y=18)
            output_frame = query_view

        if writer is None and output_video:
            try:
                writer = imageio.get_writer(
                    output_video,
                    fps=fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=1,
                    ffmpeg_log_level="error",
                )
                writer_kind = "imageio"
            except Exception:
                writer = cv2.VideoWriter(
                    output_video,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (output_frame.shape[1], output_frame.shape[0]),
                )
                writer_kind = "cv2"
                if not writer.isOpened():
                    cap.release()
                    if ref_cap is not None:
                        ref_cap.release()
                    raise RuntimeError(f"Cannot open video writer: {output_video}")

        if writer is not None:
            if writer_kind == "imageio":
                writer.append_data(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
            else:
                writer.write(output_frame)
        if preview:
            window_name = "DTW Compare Preview" if use_compare_mode else "Feedback Preview"
            should_continue = preview_frame(window_name, output_frame)
            if not should_continue:
                preview = False
                close_preview_windows()

        frame_idx += 1
        if progress_callback and total_frames > 0 and frame_idx % 30 == 0:
            rendered_ratio = min(1.0, frame_idx / max(1, total_frames))
            progress = progress_start + int(round(progress_span * rendered_ratio))
            progress_callback(progress, "正在生成对比视频")

    if writer is not None:
        if writer_kind == "imageio":
            writer.close()
        else:
            writer.release()
    cap.release()
    if ref_cap is not None:
        ref_cap.release()
    if preview:
        close_preview_windows()

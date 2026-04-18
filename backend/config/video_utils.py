from __future__ import annotations

import subprocess
from pathlib import Path


def transcode_to_browser_mp4(input_path: str, output_path: str) -> str:
    from imageio_ffmpeg import get_ffmpeg_exe

    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    command = [
        get_ffmpeg_exe(),
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-preset",
        "veryfast",
        "-an",
        str(dst),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return str(dst)

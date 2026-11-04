"""Acquire a video (file or URL), hash it,
normalise it and extract 16 kHz audio."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.ingestion.metadata import IngestionError


def is_url(text: str) -> bool:
    return text.lower().startswith(
        ("http://", "https://")
    )


def require_tools() -> None:

    missing = [
        t
        for t in ("ffmpeg", "ffprobe")
        if shutil.which(t) is None
    ]

    if missing:
        raise IngestionError(
            "Required tool(s) not found on PATH: "
            + ", ".join(missing)
        )


def sha256_file(
    path: str | Path,
    chunk: int = 1 << 20
) -> str:

    h = hashlib.sha256()

    with open(path, "rb") as fh:

        while block := fh.read(chunk):
            h.update(block)

    return h.hexdigest()


def make_video_id(
    sha256: str,
    requested: str | None = None
) -> str:

    """Use caller's id or first 12 SHA-256 chars."""

    if requested:

        if not requested.replace(
            "_", ""
        ).replace(
            "-", ""
        ).isalnum():

            raise IngestionError(
                "video_id may only contain letters, "
                "digits, '_' and '-'"
            )

        return requested

    return sha256[:12]


def download_url(
    url: str,
    dest_dir: str | Path
) -> Path:

    """Download a URL using yt-dlp."""

    try:
        import yt_dlp

    except ImportError as exc:

        raise IngestionError(
            "yt-dlp is not installed "
            "(pip install yt-dlp)"
        ) from exc

    opts = {
        "outtmpl": str(
            Path(dest_dir)
            / "download.%(ext)s"
        ),
        "quiet": True,
        "merge_output_format": "mp4",
        "noplaylist": True
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    files = sorted(
        Path(dest_dir).glob("download.*")
    )

    if not files:
        raise IngestionError(
            f"Download produced no file for {url}"
        )

    return files[0]


def acquire_source(
    source: str,
    work_dir: str | Path | None = None
) -> Path:

    """Return local path for file path or URL."""

    if is_url(source):

        work = (
            Path(work_dir)
            if work_dir
            else Path(
                tempfile.mkdtemp(
                    prefix="cinecomic_dl_"
                )
            )
        )

        work.mkdir(
            parents=True,
            exist_ok=True
        )

        return download_url(
            source,
            work
        )

    path = Path(source)

    if not path.is_file():
        raise IngestionError(
            f"Input file not found: {path}"
        )

    if path.stat().st_size == 0:
        raise IngestionError(
            f"Input file is empty: {path}"
        )

    return path


def needs_normalising(
    src: Path,
    info: dict,
    icfg: dict
) -> tuple[bool, list[str]]:

    """Decide whether source must be re-encoded."""

    reasons = []

    if src.suffix.lower() != ".mp4":
        reasons.append(
            "container is not mp4"
        )

    if info["vfr"] and icfg["force_cfr"]:
        reasons.append(
            "variable frame rate"
        )

    if info["rotation"] != 0:
        reasons.append(
            f"rotation tag {info['rotation']}"
        )

    max_d = icfg["max_duration_s"]

    if max_d and info["duration_s"] > max_d:
        reasons.append(
            f"longer than max_duration_s={max_d}"
        )

    return bool(reasons), reasons


def normalise_video(
    src: Path,
    dst: Path,
    info: dict,
    icfg: dict
) -> None:

    """Re-encode to CFR, upright H.264/AAC MP4."""

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(src)
    ]

    max_d = icfg["max_duration_s"]

    if max_d and info["duration_s"] > max_d:
        cmd += [
            "-t",
            str(max_d)
        ]

    cmd += [
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?"
    ]

    if icfg["force_cfr"]:
        cmd += [
            "-vf",
            f"fps={info['fps_fraction']}"
        ]

    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        str(icfg["x264_preset"]),
        "-crf",
        str(icfg["x264_crf"]),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(dst)
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace"
    )

    if proc.returncode != 0:

        raise IngestionError(
            "ffmpeg re-encode failed: "
            + proc.stderr.strip()[:300]
        )


def copy_video(
    src: Path,
    dst: Path
) -> None:

    shutil.copy2(src, dst)


def extract_audio(
    video: Path,
    wav: Path,
    sample_rate: int
) -> None:

    """16 kHz mono 16-bit PCM WAV."""

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(wav)
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace"
    )

    if proc.returncode != 0:

        raise IngestionError(
            "Audio extraction failed: "
            + proc.stderr.strip()[:300]
        )
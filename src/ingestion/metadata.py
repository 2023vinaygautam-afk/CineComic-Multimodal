"""ffprobe / OpenCV helpers: describe a video and run the Phase 1 checks."""

from __future__ import annotations

import json
import random
import subprocess
import wave
from pathlib import Path


class IngestionError(RuntimeError):
    """Raised when a video cannot be accepted
    (missing, corrupt, too short...)."""


def _run(cmd: list[str]) -> subprocess.CompletedProcess:

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )


def _fraction(text: str | None) -> float:
    """'30000/1001' -> 29.97...; '0/0' or None -> 0.0"""

    if not text:
        return 0.0

    if "/" in text:
        num, den = text.split("/", 1)
        den_f = float(den)

        return (
            float(num) / den_f
            if den_f
            else 0.0
        )

    return float(text)


def probe(path: str | Path) -> dict:
    """Run ffprobe and return parsed JSON."""

    proc = _run([
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ])

    if proc.returncode != 0:
        raise IngestionError(
            f"ffprobe could not read '{path}' "
            f"(file corrupt or truncated?): "
            f"{proc.stderr.strip()[:200]}"
        )

    try:
        return json.loads(proc.stdout)

    except json.JSONDecodeError as exc:
        raise IngestionError(
            f"ffprobe returned invalid JSON for '{path}'"
        ) from exc


def _rotation(vstream: dict) -> int:
    """Rotation in degrees from the legacy 'rotate'
    tag or display matrix."""

    tag = vstream.get(
        "tags",
        {}
    ).get("rotate")

    if tag is not None:
        return int(float(tag)) % 360

    for side in vstream.get(
        "side_data_list",
        []
    ):
        if "rotation" in side:
            return int(
                round(float(side["rotation"]))
            ) % 360

    return 0


def describe(
    path: str | Path,
    vfr_tolerance: float = 0.02
) -> dict:

    """Summarise a video file.
    Width/height are displayed size after rotation.
    """

    info = probe(path)

    vstreams = [
        s
        for s in info.get("streams", [])
        if s.get("codec_type") == "video"
    ]

    astreams = [
        s
        for s in info.get("streams", [])
        if s.get("codec_type") == "audio"
    ]

    if not vstreams:
        raise IngestionError(
            f"'{path}' has no video stream"
        )

    v = vstreams[0]

    rate = _fraction(
        v.get("r_frame_rate")
    )

    avg = _fraction(
        v.get("avg_frame_rate")
    )

    fps = avg or rate

    if fps <= 0:
        raise IngestionError(
            "Could not determine the frame rate"
        )

    vfr = bool(
        rate
        and avg
        and abs(rate - avg) / rate
        > vfr_tolerance
    )

    duration = float(
        info.get("format", {}).get("duration")
        or v.get("duration")
        or 0
    )

    if duration <= 0:
        raise IngestionError(
            "Could not determine the duration"
        )

    width = int(v.get("width", 0))
    height = int(v.get("height", 0))

    rotation = _rotation(v)

    if rotation in (90, 270):
        width, height = height, width

    if width <= 0 or height <= 0:
        raise IngestionError(
            "Invalid video resolution"
        )

    nb = v.get("nb_frames")

    n_frames = (
        int(nb)
        if nb and str(nb).isdigit()
        else None
    )

    return {
        "duration_s": duration,
        "fps": fps,
        "fps_fraction": (
            v.get("avg_frame_rate")
            if avg
            else v.get("r_frame_rate")
        ),
        "width": width,
        "height": height,
        "rotation": rotation,
        "video_codec": v.get("codec_name"),
        "n_frames_ffprobe": n_frames,
        "n_frames": (
            n_frames
            if n_frames is not None
            else int(round(duration * fps))
        ),
        "n_frames_estimated": n_frames is None,
        "vfr": vfr,
        "has_audio": bool(astreams),
        "audio_codec": (
            astreams[0].get("codec_name")
            if astreams
            else None
        ),
    }


def decode_errors(path: str | Path) -> list[str]:
    """Decode the whole file to nowhere;
    every stderr line counts as an error."""

    proc = _run([
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "null",
        "-"
    ])

    return [
        line
        for line in proc.stderr.splitlines()
        if line.strip()
    ]


def opencv_frame_count(path: str | Path) -> int:

    import cv2

    cap = cv2.VideoCapture(str(path))

    try:

        if not cap.isOpened():
            raise IngestionError(
                f"OpenCV cannot open '{path}'"
            )

        return int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

    finally:
        cap.release()


def random_frame_check(
    path: str | Path,
    duration_s: float,
    n: int,
    seed: int,
    black_mean: float
) -> dict:

    """Seek to n random timestamps;
    a frame is bad if unreadable or almost black.
    """

    import cv2

    rng = random.Random(seed)

    times = sorted(
        rng.uniform(
            0.0,
            max(duration_s - 0.2, 0.0)
        )
        for _ in range(n)
    )

    bad = []

    cap = cv2.VideoCapture(str(path))

    try:

        for t in times:

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                t * 1000.0
            )

            ok, frame = cap.read()

            if (
                not ok
                or frame is None
                or float(frame.mean()) < black_mean
            ):
                bad.append(round(t, 3))

    finally:
        cap.release()

    return {
        "checked": n,
        "bad": len(bad),
        "bad_times_s": bad
    }


def wav_duration(path: str | Path) -> float:

    with wave.open(str(path), "rb") as wf:
        return (
            wf.getnframes()
            / float(wf.getframerate())
        )


def wav_format(
    path: str | Path
) -> tuple[int, int]:

    """Return (sample_rate, channels) of a WAV file."""

    with wave.open(str(path), "rb") as wf:
        return (
            wf.getframerate(),
            wf.getnchannels()
        )
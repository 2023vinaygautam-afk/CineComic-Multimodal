"""Shared fixtures:
tiny synthetic videos generated with FFmpeg
(no sample files needed).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from src.common.utils import (
    close_logger,
    load_config
)


ROOT = Path(
    __file__
).resolve().parents[1]


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not installed"
)


def make_video(
    path: Path,
    duration: float = 2.0,
    size: str = "320x240",
    fps: int = 25,
    audio: bool = True,
    in_opts=None,
    vf: str | None = None,
    extra=None
) -> Path:

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error"
    ] + list(in_opts or [])

    cmd += [
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:"
        f"size={size}:rate={fps}"
    ]

    if audio:

        cmd += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:"
            f"duration={duration}"
        ]

    if vf:
        cmd += [
            "-vf",
            vf
        ]

    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p"
    ]

    cmd += (
        ["-c:a", "aac"]
        if audio
        else ["-an"]
    )

    cmd += (
        list(extra or [])
        + [str(path)]
    )

    subprocess.run(
        cmd,
        check=True,
        capture_output=True
    )

    return path


@pytest.fixture
def cfg(tmp_path):

    c = load_config(
        ROOT
        / "configs"
        / "phase1.yaml"
    )

    c["data_dir"] = str(
        tmp_path / "data"
    )

    c["results_dir"] = str(
        tmp_path / "results"
    )

    c["log_dir"] = str(
        tmp_path / "logs"
    )

    c["ingestion"][
        "random_frame_checks"
    ] = 10

    return c


@pytest.fixture
def make(tmp_path):

    def _make(
        name="in.mp4",
        **kw
    ):

        return make_video(
            tmp_path / name,
            **kw
        )

    return _make


@pytest.fixture
def run_ingest(cfg):

    from src.ingestion.video_ingestion import ingest

    def _run(
        source,
        video_id="t001",
        force=False
    ):

        result = ingest(
            cfg,
            str(source),
            video_id,
            force=force
        )

        import logging

        close_logger(
            logging.getLogger(
                f"cinecomic.{video_id}"
            )
        )

        return result

    return _run
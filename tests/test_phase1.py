"""Phase 1 unit tests
(plus Phase 0 config / ffmpeg checks).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.common.utils import (
    ConfigError,
    load_config
)

from src.ingestion import metadata as md
from src.ingestion import video_loader as vl

from tests.conftest import (
    ROOT,
    needs_ffmpeg
)


pytestmark = needs_ffmpeg


def _meta(
    cfg,
    vid="t001"
):

    return json.loads(
        (
            Path(cfg["data_dir"])
            / "metadata"
            / f"{vid}.json"
        ).read_text()
    )


# ---- Phase 0 checks --------------------------------------------------


def test_ffmpeg_callable():

    assert subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True
    ).returncode == 0


def test_config_loader_rejects_unknown_keys(
    tmp_path
):

    bad = tmp_path / "bad.yaml"

    bad.write_text(
        "seed: 1\n"
        "not_a_key: 5\n"
    )

    with pytest.raises(ConfigError):
        load_config(bad)

    nested = tmp_path / "bad2.yaml"

    nested.write_text(
        "ingestion:\n"
        " typo_key: 1\n"
    )

    with pytest.raises(ConfigError):
        load_config(nested)


def test_default_config_loads():

    cfg = load_config(
        ROOT
        / "configs"
        / "phase1.yaml"
    )

    assert (
        cfg["ingestion"]
        ["audio_sample_rate"]
        == 16000
    )


# ---- Phase 1 behaviour ----------------------------------------------


def test_happy_path_passes_gate(
    cfg,
    make,
    run_ingest
):

    ok, vid = run_ingest(
        make()
    )

    assert ok and vid == "t001"

    meta = _meta(cfg)

    assert (
        meta["has_audio"]
        and meta["width"] == 320
        and meta["height"] == 240
    )

    assert (
        abs(meta["fps"] - 25)
        < 0.01
        and meta["n_frames"] == 50
    )

    wav = Path(
        meta["audio_path"]
    )

    assert md.wav_format(wav) == (
        16000,
        1
    )

    assert (
        abs(
            md.wav_duration(wav)
            - meta["duration_s"]
        )
        <= cfg["gates"]
        ["audio_video_diff_s"]
    )

    metrics = json.loads(
        (
            Path(cfg["data_dir"])
            / "metadata"
            / "t001.metrics.json"
        ).read_text()
    )

    assert (
        metrics["passed"] is True
        and metrics["decode_errors"] == 0
    )


def test_default_video_id_is_sha256_prefix(
    cfg,
    make,
    run_ingest,
    tmp_path
):

    src = make()

    ok, vid = run_ingest(
        src,
        video_id=None
    )

    assert (
        ok
        and vid == vl.sha256_file(src)[:12]
    )


def test_truncated_file_rejected(
    cfg,
    make,
    run_ingest,
    tmp_path
):

    good = make(
        "good.mp4"
    )

    data = good.read_bytes()

    broken = (
        tmp_path
        / "broken.mp4"
    )

    broken.write_bytes(
        data[:len(data) // 2]
    )

    with pytest.raises(
        md.IngestionError
    ):
        run_ingest(broken)


def test_missing_and_empty_files_rejected(
    run_ingest,
    tmp_path
):

    with pytest.raises(
        md.IngestionError
    ):
        run_ingest(
            tmp_path / "nope.mp4"
        )

    empty = (
        tmp_path
        / "empty.mp4"
    )

    empty.write_bytes(
        b""
    )

    with pytest.raises(
        md.IngestionError
    ):
        run_ingest(empty)


def test_no_audio_sets_flag_and_skips_wav(
    cfg,
    make,
    run_ingest
):

    ok, _ = run_ingest(
        make(audio=False)
    )

    meta = _meta(cfg)

    assert (
        ok
        and meta["has_audio"] is False
        and meta["audio_path"] is None
    )

    assert not (
        Path(cfg["data_dir"])
        / "audio"
        / "t001.wav"
    ).exists()


def test_rotated_video_is_upright(
    cfg,
    make,
    run_ingest
):

    base = make(
        "base.mp4",
        size="320x240"
    )

    rotated = (
        base.parent
        / "rotated.mp4"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-display_rotation",
            "90",
            "-i",
            str(base),
            "-c",
            "copy",
            str(rotated)
        ],
        check=True,
        capture_output=True
    )

    assert (
        md.describe(rotated)
        ["rotation"]
        in (90, 270)
    )

    assert (
        md.describe(rotated)
        ["height"]
        == 320
    )

    ok, _ = run_ingest(
        rotated
    )

    meta = _meta(cfg)

    assert (
        ok
        and meta["normalised"]
        and meta["source_rotation"]
        in (90, 270)
    )

    final = md.describe(
        Path(meta["raw_path"])
    )

    assert (
        final["rotation"] == 0
        and final["width"] == 240
        and final["height"] == 320
    )


def test_variable_frame_rate_becomes_cfr(
    cfg,
    make,
    run_ingest
):

    vfr = make(
        "vfr.mp4",
        duration=4,
        fps=30,
        vf="select='not(mod(n,3))+not(mod(n,7))'",
        extra=[
            "-fps_mode",
            "vfr"
        ]
    )

    if not md.describe(vfr)["vfr"]:

        pytest.skip(
            "could not synthesise a VFR file "
            "with this FFmpeg build"
        )

    ok, _ = run_ingest(vfr)

    meta = _meta(cfg)

    assert (
        ok
        and meta["source_vfr"] is True
        and meta["vfr"] is False
        and meta["normalised"]
    )


def test_max_duration_is_enforced(
    cfg,
    make,
    run_ingest
):

    cfg["ingestion"][
        "max_duration_s"
    ] = 1

    ok, _ = run_ingest(
        make(duration=3)
    )

    assert (
        ok
        and _meta(cfg)
        ["duration_s"]
        <= 1.1
    )


def test_too_short_video_rejected(
    cfg,
    make,
    run_ingest
):

    cfg["ingestion"][
        "min_duration_s"
    ] = 5

    with pytest.raises(
        md.IngestionError
    ):

        run_ingest(
            make(duration=2)
        )


def test_cache_skips_second_run_and_force_reruns(
    cfg,
    make,
    run_ingest
):

    src = make()

    run_ingest(src)

    mp = (
        Path(cfg["data_dir"])
        / "metadata"
        / "t001.metrics.json"
    )

    first = mp.stat().st_mtime_ns

    run_ingest(src)

    assert (
        mp.stat().st_mtime_ns
        == first
    )

    run_ingest(
        src,
        force=True
    )

    assert (
        mp.stat().st_mtime_ns
        > first
    )


def test_same_input_same_metadata(
    cfg,
    make,
    run_ingest
):

    src = make()

    run_ingest(
        src,
        "a1"
    )

    run_ingest(
        src,
        "a2"
    )

    a = _meta(
        cfg,
        "a1"
    )

    b = _meta(
        cfg,
        "a2"
    )

    for key in (
        "duration_s",
        "fps",
        "width",
        "height",
        "n_frames",
        "audio_duration_s"
    ):

        assert a[key] == b[key]


def test_gate_flags_black_video(
    cfg,
    make,
    run_ingest,
    tmp_path
):

    black = (
        tmp_path
        / "black.mp4"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=2:r=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(black)
        ],
        check=True,
        capture_output=True
    )

    ok, _ = run_ingest(
        black
    )

    assert ok is False
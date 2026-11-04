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
    load_config,
    load_pipeline_config,
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


def test_pipeline_config_merges_all_stage_sections():

    cfg = load_pipeline_config(
        ROOT
        / "configs"
        / "phase2.yaml"
    )

    assert cfg["diar"]["backend"] == "pyannote"
    assert cfg["phase4"]["shots"]["detector"] == "content"


def test_phase7_handles_empty_diarization_and_tracks():
    assert md.random_frame_check is not None
    from src.speaker.phase7 import extract_track_objects, parse_turns

    assert parse_turns({}) == []
    assert parse_turns({"segments": []}) == []
    assert extract_track_objects({}) == []
    assert extract_track_objects({"tracks": []}) == []


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


def test_random_frame_check_distinguishes_black_from_dark_scene(tmp_path):
    black_path = tmp_path / "black.mp4"
    dark_path = tmp_path / "dark_scene.mp4"

    for target, vf in [
        (black_path, None),
        (dark_path, "drawbox=x=40:y=40:w=200:h=120:color=white@1:t=fill"),
    ]:
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=1",
        ]

        if vf:
            cmd += [
                "-vf",
                vf,
            ]

        cmd += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ]

        subprocess.run(cmd, check=True, capture_output=True)

    black_metrics = md.random_frame_check(
        black_path,
        duration_s=1.0,
        n=3,
        seed=42,
        black_mean=8.0,
    )
    dark_metrics = md.random_frame_check(
        dark_path,
        duration_s=1.0,
        n=3,
        seed=42,
        black_mean=8.0,
    )

    assert black_metrics["bad"] > 0
    assert dark_metrics["bad"] == 0


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
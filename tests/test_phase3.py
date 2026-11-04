"""Phase 3 diarization checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.audio.diarization import DiarStage
from src.common.utils import load_config

from tests.conftest import ROOT


@pytest.fixture
def diar_cfg(tmp_path):
    cfg = load_config(ROOT / "configs" / "phase3.yaml")
    cfg["data_dir"] = str(tmp_path / "data")
    cfg["results_dir"] = str(tmp_path / "results")
    cfg["log_dir"] = str(tmp_path / "logs")
    return cfg


def test_phase3_config_loads():
    cfg = load_config(ROOT / "configs" / "phase3.yaml")
    assert cfg["diar"]["backend"] == "pyannote"
    assert cfg["diar_gates"]["der_max"] == 0.25


def test_main_registers_diarization_stage():
    import main

    assert "diarization" in main.STAGES


def test_diar_stage_runs_and_uses_mock_backend(diar_cfg, tmp_path):
    data_dir = Path(diar_cfg["data_dir"])
    (data_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (data_dir / "subtitles").mkdir(parents=True, exist_ok=True)
    (data_dir / "diarization").mkdir(parents=True, exist_ok=True)

    (data_dir / "metadata" / "demo.json").write_text(
        json.dumps({
            "audio_path": str(tmp_path / "demo.wav"),
            "audio_duration_s": 2.0,
            "has_audio": True,
        }),
        encoding="utf-8",
    )

    (data_dir / "subtitles" / "demo.json").write_text(
        json.dumps({
            "segments": [
                {"segment_id": "s0", "start": 0.0, "end": 1.0},
                {"segment_id": "s1", "start": 1.0, "end": 2.0},
            ]
        }),
        encoding="utf-8",
    )

    def fake_backend(wav, dcfg, log=None):
        return [
            {"start": 0.0, "end": 1.0, "speaker": "A"},
            {"start": 1.0, "end": 2.0, "speaker": "B"},
        ]

    stage = DiarStage(diar_cfg, "demo", diarize_fn=fake_backend)
    stage.run()

    metrics = stage.evaluate()
    ok, reason = stage.gate(metrics)

    assert ok is True
    assert reason.startswith("structure OK")
    assert (data_dir / "diarization" / "demo.raw.json").exists()
    assert (data_dir / "diarization" / "demo.rttm").exists()
    assert metrics["n_turns"] == 2

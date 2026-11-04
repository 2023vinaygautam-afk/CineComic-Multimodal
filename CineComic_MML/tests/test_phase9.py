"""Phase 9 Multimodal Emotion Fusion unit tests."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from src.common.utils import load_config
from src.emotion.fusion import fuse_modalities, LABELS
from src.emotion.text import TextEmotionClassifier
from src.evaluation.emotion_metrics import evaluate_emotion_predictions
from src.emotion.phase9 import Phase9Stage
from tests.conftest import ROOT


def test_phase9_config_loads():
    cfg = load_config(ROOT / "configs" / "phase9.yaml")
    assert "phase9" in cfg
    p9 = cfg["phase9"]
    assert "text_model" in p9
    assert "audio_model" in p9
    assert "face_model" in p9
    assert "weights" in p9
    assert set(p9["weights"].keys()) == {"text", "audio", "face"}


def test_fusion_with_all_modalities():
    probs = {
        "text": {lbl: 0.1 for lbl in LABELS},
        "audio": {lbl: 0.1 for lbl in LABELS},
        "face": {lbl: 0.1 for lbl in LABELS},
    }
    # Give joy higher probability in text and audio
    probs["text"]["joy"] = 0.4
    probs["audio"]["joy"] = 0.4
    probs["face"]["anger"] = 0.4

    weights = {"text": 0.5, "audio": 0.3, "face": 0.2}
    label, conf, modalities, fused_probs = fuse_modalities(probs, weights)

    assert label == "joy"
    assert "text" in modalities and "audio" in modalities and "face" in modalities
    assert abs(sum(fused_probs.values()) - 1.0) < 0.01


def test_fusion_with_missing_modalities():
    # Only text is available; face and audio are None (e.g. speaker off screen, silent audio)
    probs = {
        "text": {"anger": 0.8, "disgust": 0.0, "fear": 0.0, "joy": 0.1, "neutral": 0.1, "sadness": 0.0, "surprise": 0.0},
        "audio": None,
        "face": None,
    }
    weights = {"text": 0.45, "audio": 0.30, "face": 0.25}
    label, conf, modalities, fused_probs = fuse_modalities(probs, weights)

    assert label == "anger"
    assert modalities == ["text"]
    assert abs(sum(fused_probs.values()) - 1.0) < 0.01
    assert fused_probs["anger"] >= 0.75


def test_masked_modality_does_not_contribute():
    # If audio is masked (None), its weight must not skew the distribution
    probs1 = {
        "text": {"joy": 0.9, "anger": 0.1, "disgust": 0.0, "fear": 0.0, "neutral": 0.0, "sadness": 0.0, "surprise": 0.0},
        "audio": None,
        "face": None,
    }
    weights = {"text": 0.5, "audio": 0.5, "face": 0.0}
    label1, conf1, mods1, _ = fuse_modalities(probs1, weights)

    assert label1 == "joy"
    assert conf1 >= 0.85
    assert mods1 == ["text"]


def test_evaluate_emotion_metrics():
    sample_segments = [
        {
            "segment_id": 1,
            "fused": {"label": "joy", "conf": 0.85, "modalities": ["text", "audio"]},
        },
        {
            "segment_id": 2,
            "fused": {"label": "anger", "conf": 0.78, "modalities": ["text"]},
        },
    ]
    metrics = evaluate_emotion_predictions(sample_segments)
    assert metrics["num_segments"] == 2
    assert metrics["coverage"] == 1.0
    assert metrics["emotion_distribution"]["joy"] == 1
    assert metrics["emotion_distribution"]["anger"] == 1
    assert metrics["mean_fused_confidence"] == round((0.85 + 0.78) / 2, 4)

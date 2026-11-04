"""
Evaluation metrics for CineComic-MML Phase 9: Multimodal Emotion Fusion.
Computes modality coverage, confidence statistics, class distributions,
inter-modality agreement, and optional ground-truth accuracy / F1 metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def evaluate_emotion_predictions(
    segments: List[Dict[str, Any]],
    gt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates Phase 9 multimodal emotion predictions.

    Args:
        segments: List of segment prediction dictionaries from Phase 9.
        gt: Optional ground-truth mapping segment_id -> emotion label.

    Returns:
        Dictionary of computed evaluation metrics.
    """
    n_segments = len(segments)
    if n_segments == 0:
        return {
            "num_segments": 0,
            "coverage": 0.0,
            "modality_counts": {"text": 0, "audio": 0, "face": 0, "all_three": 0},
            "emotion_distribution": {lbl: 0 for lbl in LABELS},
            "mean_fused_confidence": 0.0,
            "ground_truth_evaluated": False,
        }

    text_count = 0
    audio_count = 0
    face_count = 0
    all_three_count = 0
    active_segments = 0
    confidences = []
    emotion_dist = {lbl: 0 for lbl in LABELS}

    for s in segments:
        m_list = s.get("fused", {}).get("modalities", [])
        if "text" in m_list:
            text_count += 1
        if "audio" in m_list:
            audio_count += 1
        if "face" in m_list:
            face_count += 1
        if len(m_list) == 3:
            all_three_count += 1
        if len(m_list) > 0:
            active_segments += 1

        conf = s.get("fused", {}).get("conf", 0.0)
        confidences.append(conf)

        label = s.get("fused", {}).get("label")
        if label in emotion_dist:
            emotion_dist[label] += 1

    metrics: Dict[str, Any] = {
        "num_segments": n_segments,
        "coverage": round(active_segments / max(1, n_segments), 4),
        "modality_counts": {
            "text": text_count,
            "audio": audio_count,
            "face": face_count,
            "all_three": all_three_count,
        },
        "emotion_distribution": emotion_dist,
        "mean_fused_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "min_fused_confidence": round(float(np.min(confidences)), 4) if confidences else 0.0,
        "max_fused_confidence": round(float(np.max(confidences)), 4) if confidences else 0.0,
        "ground_truth_evaluated": False,
    }

    if gt and "labels" in gt:
        y_true = []
        y_pred = []
        for s in segments:
            sid = str(s.get("segment_id"))
            if sid in gt["labels"]:
                y_true.append(gt["labels"][sid])
                y_pred.append(s.get("fused", {}).get("label", "neutral"))

        if y_true:
            metrics["ground_truth_evaluated"] = True
            metrics["gt_accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
            metrics["gt_macro_f1"] = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
            metrics["gt_weighted_f1"] = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)

    return metrics

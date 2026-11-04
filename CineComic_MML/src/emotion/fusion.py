"""
Multimodal Emotion Fusion module for CineComic-MML Phase 9.
Implements masked weighted late fusion with missing-modality handling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def fuse_modalities(
    probs: Dict[str, Optional[Dict[str, float]]],
    weights: Dict[str, float],
) -> Tuple[Optional[str], float, List[str], Dict[str, float]]:
    """
    Fuses emotion probability distributions from active modalities using masked weighted late fusion.

    Args:
        probs: Dictionary mapping modality name ("text", "audio", "face") to a
               dictionary of probabilities over LABELS, or None if masked/missing.
        weights: Dictionary mapping modality name to its fusion weight.

    Returns:
        (fused_label, fused_confidence, active_modalities, fused_probs_dict)
    """
    num = np.zeros(len(LABELS), dtype=np.float64)
    den = 0.0
    active_modalities: List[str] = []

    for m, p_dict in probs.items():
        if p_dict is None:
            # MASKED or missing modality contributes nothing
            continue

        w = float(weights.get(m, 1.0))
        if w <= 0:
            continue

        vec = np.array([float(p_dict.get(lbl, 0.0)) for lbl in LABELS], dtype=np.float64)
        total = vec.sum()
        if total > 0:
            vec /= total
        else:
            vec = np.zeros(len(LABELS), dtype=np.float64)
            vec[LABELS.index("neutral")] = 1.0

        num += w * vec
        den += w
        active_modalities.append(m)

    if den == 0.0:
        neutral_probs = {lbl: (1.0 if lbl == "neutral" else 0.0) for lbl in LABELS}
        return "neutral", 1.0, [], neutral_probs

    out = num / den
    # Ensure exact probability sum to 1.0
    out = out / out.sum()

    best_idx = int(out.argmax())
    best_label = LABELS[best_idx]
    best_conf = float(out[best_idx])

    fused_probs = {lbl: round(float(out[i]), 4) for i, lbl in enumerate(LABELS)}

    return best_label, round(best_conf, 4), active_modalities, fused_probs

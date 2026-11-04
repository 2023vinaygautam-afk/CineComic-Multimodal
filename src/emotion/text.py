"""
Text emotion classification for CineComic-MML Phase 9.
Outputs 7 MELD-compatible emotion classes:
["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from transformers import pipeline

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class TextEmotionClassifier:
    """Classifies dialogue text into MELD emotion probability distributions."""

    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base", device: str = "cpu"):
        self.model_name = model_name
        self.device = 0 if device == "cuda" else -1
        self._pipe = pipeline(
            "text-classification",
            model=model_name,
            device=self.device,
            top_k=None,
        )

    def predict(self, text: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Classifies input text.
        Returns None if text is empty or missing.
        """
        if not text or not text.strip():
            return None

        clean_text = text.strip()
        try:
            raw_results = self._pipe(clean_text)[0]
        except Exception:
            return None

        # raw_results is a list of {'label': str, 'score': float}
        probs = {lbl: 0.0 for lbl in LABELS}
        for item in raw_results:
            lbl = item["label"].lower()
            if lbl in probs:
                probs[lbl] = float(item["score"])

        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            probs["neutral"] = 1.0

        best_label = max(probs, key=probs.get)
        best_p = float(probs[best_label])

        return {
            "label": best_label,
            "p": round(best_p, 4),
            "probs": {k: round(v, 4) for k, v in probs.items()},
        }

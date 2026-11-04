"""
Audio emotion classification for CineComic-MML Phase 9.
Outputs 7 MELD-compatible emotion classes:
["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from transformers import pipeline

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

AUDIO_MAP = {
    "ang": "anger",
    "hap": "joy",
    "neu": "neutral",
    "sad": "sadness",
}


def load_wav_slice(wav_path: Path, start_s: float, end_s: float, target_sr: int = 16000) -> Optional[np.ndarray]:
    """Reads a chunk of audio from a mono WAV file as a float32 array in [-1.0, 1.0]."""
    if not wav_path.exists():
        return None

    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()

            start_frame = max(0, int(start_s * sr))
            end_frame = min(n_frames, int(end_s * sr))

            if end_frame <= start_frame:
                return None

            wf.setpos(start_frame)
            raw_bytes = wf.readframes(end_frame - start_frame)

            if sampwidth == 2:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)

            return samples
    except Exception:
        return None


class AudioEmotionClassifier:
    """Classifies speech audio segments into MELD emotion probability distributions."""

    def __init__(
        self,
        model_name: str = "superb/wav2vec2-base-superb-er",
        device: str = "cpu",
        min_duration_s: float = 0.25,
        min_rms: float = 0.002,
    ):
        self.model_name = model_name
        self.device = 0 if device == "cuda" else -1
        self.min_duration_s = min_duration_s
        self.min_rms = min_rms

        self._pipe = pipeline(
            "audio-classification",
            model=model_name,
            device=self.device,
            top_k=None,
        )

    def predict_slice(
        self,
        wav_path: Path,
        start_s: float,
        end_s: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Classifies audio in [start_s, end_s].
        Returns None (MASKED) if audio is too short, missing, or silent.
        """
        duration = end_s - start_s
        if duration < self.min_duration_s:
            return None

        samples = load_wav_slice(wav_path, start_s, end_s)
        if samples is None or len(samples) == 0:
            return None

        # Check RMS energy to mask non-speech / silence
        rms = np.sqrt(np.mean(samples ** 2))
        if rms < self.min_rms:
            return None

        try:
            raw_results = self._pipe(samples)
        except Exception:
            return None

        probs = {lbl: 0.0 for lbl in LABELS}
        for item in raw_results:
            raw_label = item["label"].lower()
            mapped = AUDIO_MAP.get(raw_label, "neutral")
            probs[mapped] += float(item["score"])

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

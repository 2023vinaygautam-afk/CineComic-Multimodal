"""
Face crop emotion classification for CineComic-MML Phase 9.
Outputs 7 MELD-compatible emotion classes:
["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

FACE_MAP = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "joy",
    "neutral": "neutral",
    "sad": "sadness",
    "surprise": "surprise",
}


class FaceEmotionClassifier:
    """Classifies face crops of speaking characters into MELD emotion distributions."""

    def __init__(
        self,
        model_name: str = "dima806/facial_emotions_image_detection",
        device: str = "cpu",
        min_face_size: int = 40,
        max_samples_per_segment: int = 4,
        margin: float = 0.15,
    ):
        self.model_name = model_name
        self.device = 0 if device == "cuda" else -1
        self.min_face_size = min_face_size
        self.max_samples_per_segment = max_samples_per_segment
        self.margin = margin

        self._pipe = pipeline(
            "image-classification",
            model=model_name,
            device=self.device,
            top_k=None,
        )

    def predict_crops(self, pil_crops: List[Image.Image]) -> Optional[Dict[str, Any]]:
        """Classifies a list of face crops and averages their probability vectors."""
        if not pil_crops:
            return None

        accum_probs = {lbl: 0.0 for lbl in LABELS}
        valid_count = 0

        for crop in pil_crops:
            try:
                raw_results = self._pipe(crop)
                crop_probs = {lbl: 0.0 for lbl in LABELS}
                for item in raw_results:
                    raw_lbl = item["label"].lower()
                    mapped = FACE_MAP.get(raw_lbl, "neutral")
                    crop_probs[mapped] += float(item["score"])

                total = sum(crop_probs.values())
                if total > 0:
                    for k in accum_probs:
                        accum_probs[k] += crop_probs[k] / total
                    valid_count += 1
            except Exception:
                continue

        if valid_count == 0:
            return None

        avg_probs = {k: v / valid_count for k, v in accum_probs.items()}
        total = sum(avg_probs.values())
        if total > 0:
            avg_probs = {k: v / total for k, v in avg_probs.items()}
        else:
            avg_probs["neutral"] = 1.0

        best_label = max(avg_probs, key=avg_probs.get)
        best_p = float(avg_probs[best_label])

        return {
            "label": best_label,
            "p": round(best_p, 4),
            "probs": {k: round(v, 4) for k, v in avg_probs.items()},
        }

    def extract_and_predict(
        self,
        video_path: Path,
        track_boxes: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts face crops directly from video frames for the specified track boxes,
        then computes averaged emotion prediction.
        Each track box entry has {'t': float, 'box': [x0, y0, x1, y1] or [x, y, w, h]}.
        """
        if not track_boxes or not video_path.exists():
            return None

        # Subsample up to max_samples_per_segment evenly
        step = max(1, len(track_boxes) // self.max_samples_per_segment)
        selected_boxes = track_boxes[::step][: self.max_samples_per_segment]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        crops = []

        try:
            for item in selected_boxes:
                t = item.get("t", 0.0)
                box = item.get("box")
                if not box or len(box) != 4:
                    continue

                # Handle [x0, y0, x1, y1] vs [x, y, w, h]
                # In phase 5, boxes are [x0, y0, x1, y1]
                x0, y0, x1, y1 = [int(v) for v in box]
                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0

                w = x1 - x0
                h = y1 - y0
                if w < self.min_face_size or h < self.min_face_size:
                    continue

                frame_idx = int(round(t * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # Apply margin
                H, W = frame.shape[:2]
                pad_x = int(w * self.margin)
                pad_y = int(h * self.margin)
                cx0 = max(0, x0 - pad_x)
                cy0 = max(0, y0 - pad_y)
                cx1 = min(W, x1 + pad_x)
                cy1 = min(H, y1 + pad_y)

                face_bgr = frame[cy0:cy1, cx0:cx1]
                if face_bgr.size == 0:
                    continue

                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                crops.append(Image.fromarray(face_rgb))
        finally:
            cap.release()

        return self.predict_crops(crops)

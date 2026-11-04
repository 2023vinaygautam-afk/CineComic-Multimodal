
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class FaceDetector:
    """InsightFace detector with ArcFace embeddings and basic face quality."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        model_root: str | None = None,
        det_size: tuple[int, int] = (640, 640),
        use_gpu: bool = False,
        min_face_px: int = 40,
        min_det_score: float = 0.5,
        ref_px: int = 112,
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "InsightFace is missing. Install it with: "
                "pip install insightface onnxruntime"
            ) from exc

        self.min_face_px = int(min_face_px)
        self.min_det_score = float(min_det_score)
        self.ref_px = int(ref_px)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )

        kwargs: dict[str, Any] = {"name": model_name}
        if model_root:
            kwargs["root"] = model_root

        self.app = FaceAnalysis(
            **kwargs,
            providers=providers,
        )
        self.app.prepare(
            ctx_id=0 if use_gpu else -1,
            det_size=tuple(det_size),
        )

    @staticmethod
    def _cosine_normalize(embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if norm > 1e-12:
            embedding = embedding / norm
        return embedding

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        """Return face boxes, scores, normalized embeddings and quality values."""
        if image_bgr is None or image_bgr.size == 0:
            return []

        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image with three color channels.")

        height, width = image_bgr.shape[:2]
        detections = self.app.get(image_bgr)
        results: list[dict[str, Any]] = []

        for face in detections:
            bbox = np.asarray(face.bbox, dtype=np.float32).reshape(4)
            x1, y1, x2, y2 = bbox.tolist()

            x1i = max(0, min(width, int(np.floor(x1))))
            y1i = max(0, min(height, int(np.floor(y1))))
            x2i = max(0, min(width, int(np.ceil(x2))))
            y2i = max(0, min(height, int(np.ceil(y2))))

            face_width = x2i - x1i
            face_height = y2i - y1i
            score = float(face.det_score)

            if face_width < self.min_face_px or face_height < self.min_face_px:
                continue
            if score < self.min_det_score:
                continue

            raw_embedding = getattr(face, "normed_embedding", None)
            if raw_embedding is None:
                raw_embedding = getattr(face, "embedding", None)
            if raw_embedding is None:
                continue

            embedding = self._cosine_normalize(raw_embedding)
            if embedding.size == 0 or not np.all(np.isfinite(embedding)):
                continue

            area = face_width * face_height
            size_quality = min(
                1.0,
                min(face_width, face_height) / max(self.ref_px, 1),
            )

            # A lightweight sharpness proxy; it is not a calibrated quality model.
            crop = image_bgr[y1i:y2i, x1i:x2i]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sharpness_quality = min(1.0, sharpness / 150.0)

            quality = float(
                np.clip(
                    0.6 * size_quality + 0.4 * sharpness_quality,
                    0.0,
                    1.0,
                )
            )

            landmarks = getattr(face, "kps", None)
            mouth_center = None
            if landmarks is not None:
                points = np.asarray(landmarks, dtype=np.float32)
                # InsightFace's five-point order: eyes, nose, mouth corners.
                if points.shape[0] >= 5:
                    mouth_center = (
                        points[3:5].mean(axis=0).astype(float).tolist()
                    )

            results.append(
                {
                    "bbox": [x1i, y1i, x2i, y2i],
                    "det_score": score,
                    "embedding": embedding.astype(np.float32),
                    "quality": quality,
                    "face_width": face_width,
                    "face_height": face_height,
                    "area": int(area),
                    "mouth_center": mouth_center,
                    "landmarks": (
                        np.asarray(landmarks, dtype=np.float32).tolist()
                        if landmarks is not None
                        else None
                    ),
                }
            )

        return results

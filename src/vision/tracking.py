
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


# ============================================================
# 1. Utility functions
# ============================================================

def cosine_similarity(a: Any, b: Any) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    if a is None or b is None:
        return -1.0

    try:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return -1.0

    if a.size == 0 or a.size != b.size:
        return -1.0

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return -1.0

    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))

    if denominator <= 1e-12:
        return -1.0

    similarity = float(np.dot(a, b) / denominator)

    return float(np.clip(similarity, -1.0, 1.0))


def bbox_iou(a: Any, b: Any) -> float:
    """
    Calculate Intersection over Union (IoU).

    Bounding box format:
        [x1, y1, x2, y2]
    """
    try:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return 0.0

    if a.size != 4 or b.size != 4:
        return 0.0

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0

    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    # Reject malformed boxes.
    if ax2 <= ax1 or ay2 <= ay1:
        return 0.0

    if bx2 <= bx1 or by2 <= by1:
        return 0.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - intersection

    if union <= 1e-12:
        return 0.0

    return float(intersection / union)


def normalized_embedding(embedding: Any) -> np.ndarray:
    """
    Normalize an embedding to unit length.

    Returns an empty array if the input is missing, invalid,
    non-finite, or has effectively zero magnitude.
    """
    if embedding is None:
        return np.empty(0, dtype=np.float32)

    try:
        embedding = np.asarray(
            embedding, dtype=np.float32
        ).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=np.float32)

    if embedding.size == 0:
        return np.empty(0, dtype=np.float32)

    if not np.all(np.isfinite(embedding)):
        return np.empty(0, dtype=np.float32)

    norm = float(np.linalg.norm(embedding))

    if norm <= 1e-12:
        return np.empty(0, dtype=np.float32)

    return embedding / norm


def valid_detection(detection: dict[str, Any]) -> bool:
    """Check whether a detection has usable tracking information."""
    if not isinstance(detection, dict):
        return False

    try:
        score = float(detection.get("det_score", 0.0))
    except (TypeError, ValueError):
        return False

    if not np.isfinite(score):
        return False

    bbox = detection.get("bbox")

    try:
        bbox_array = np.asarray(
            bbox, dtype=np.float32
        ).reshape(-1)
    except (TypeError, ValueError):
        return False

    if bbox_array.size != 4:
        return False

    if not np.all(np.isfinite(bbox_array)):
        return False

    x1, y1, x2, y2 = bbox_array

    if x2 <= x1 or y2 <= y1:
        return False

    embedding = normalized_embedding(detection.get("embedding"))

    return embedding.size > 0


# ============================================================
# 2. FaceTrack data structure
# ============================================================

@dataclass
class FaceTrack:
    """Stores detections belonging to one face track."""

    track_id: str
    shot_id: str
    first_time: float
    last_time: float

    detections: list[dict[str, Any]] = field(default_factory=list)
    embedding: np.ndarray | None = None

    quality_sum: float = 0.0
    hits: int = 0

    def add(
        self,
        detection: dict[str, Any],
        timestamp: float,
        ema: float,
    ) -> None:
        """Add a detection and update the track embedding."""

        item = dict(detection)
        item["bbox"] = [
            float(value) for value in item["bbox"]
        ]
        item["det_score"] = float(item["det_score"])
        item["timestamp"] = float(timestamp)
        item["track_id"] = self.track_id

        new_embedding = normalized_embedding(item["embedding"])

        if new_embedding.size == 0:
            raise ValueError(
                "FaceTrack.add received an invalid embedding."
            )

        if self.embedding is None:
            self.embedding = new_embedding.copy()
        else:
            if self.embedding.size != new_embedding.size:
                raise ValueError(
                    "Embedding dimensions differ within the same track."
                )

            alpha = float(np.clip(ema, 0.0, 1.0))

            updated_embedding = (
                (1.0 - alpha) * self.embedding
                + alpha * new_embedding
            )

            normalized_updated = normalized_embedding(
                updated_embedding
            )

            # If the weighted vectors cancel out, preserve the
            # previous valid embedding.
            if normalized_updated.size > 0:
                self.embedding = normalized_updated

        self.last_time = float(timestamp)
        self.detections.append(item)

        try:
            quality = float(item.get("quality", 0.0))
        except (TypeError, ValueError):
            quality = 0.0

        if not np.isfinite(quality):
            quality = 0.0

        self.quality_sum += quality
        self.hits += 1

    @property
    def mean_quality(self) -> float:
        """Return average detection quality for this track."""
        if self.hits == 0:
            return 0.0

        return self.quality_sum / self.hits


# ============================================================
# 3. FaceTracker
# ============================================================

class FaceTracker:
    """
    Face tracker using Hungarian assignment.

    Matching uses:
        1. Bounding-box IoU
        2. Face-embedding cosine similarity

    Tracks are isolated by shot_id. Faces cannot be linked
    across different shots.
    """

    def __init__(
        self,
        max_gap_s: float = 1.0,
        iou_weight: float = 0.5,
        iou_min: float = 0.1,
        emb_gate: float = 0.5,
        emb_veto: float = 0.1,
        min_score: float = 0.3,
        ema: float = 0.3,
    ) -> None:
        if max_gap_s < 0:
            raise ValueError("max_gap_s must be non-negative.")

        if not 0.0 <= iou_weight <= 1.0:
            raise ValueError("iou_weight must be between 0 and 1.")

        if not -1.0 <= emb_gate <= 1.0:
            raise ValueError("emb_gate must be between -1 and 1.")

        if not -1.0 <= emb_veto <= 1.0:
            raise ValueError("emb_veto must be between -1 and 1.")

        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1.")

        if not 0.0 <= ema <= 1.0:
            raise ValueError("ema must be between 0 and 1.")

        if not 0.0 <= iou_min <= 1.0:
            raise ValueError("iou_min must be between 0 and 1.")

        self.max_gap_s = float(max_gap_s)
        self.iou_weight = float(iou_weight)
        self.iou_min = float(iou_min)
        self.emb_gate = float(emb_gate)
        self.emb_veto = float(emb_veto)
        self.min_score = float(min_score)
        self.ema = float(ema)

        self.tracks: list[FaceTrack] = []
        self._next_id = 1

    def _new_track(
        self,
        shot_id: str,
        timestamp: float,
        detection: dict[str, Any],
    ) -> FaceTrack:
        """Create a new face track."""
        track_id = f"{shot_id}_T{self._next_id:04d}"
        self._next_id += 1

        track = FaceTrack(
            track_id=track_id,
            shot_id=shot_id,
            first_time=timestamp,
            last_time=timestamp,
        )

        track.add(detection, timestamp, self.ema)
        self.tracks.append(track)

        return track

    def update(
        self,
        shot_id: str,
        timestamp: float,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Process detections from one frame.

        Args:
            shot_id: ID of the current shot.
            timestamp: Frame timestamp in seconds.
            detections: Face detections with bbox, det_score,
                        embedding, and optional quality.

        Returns:
            Valid detections with track_id attached.
        """
        shot_id = str(shot_id)

        timestamp = float(timestamp)

        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite.")

        if not isinstance(detections, list):
            raise TypeError("detections must be a list.")

        # Validate detections before association.
        valid_detections: list[dict[str, Any]] = []

        for detection in detections:
            if not isinstance(detection, dict):
                continue

            if not valid_detection(detection):
                continue

            try:
                score = float(detection["det_score"])
            except (KeyError, TypeError, ValueError):
                continue

            if score < self.min_score:
                continue

            item = dict(detection)
            item["embedding"] = normalized_embedding(
                item["embedding"]
            )

            valid_detections.append(item)

        # Only tracks from the same shot and within the time
        # window may be considered for association.
        active_tracks = [
            track
            for track in self.tracks
            if (
                track.shot_id == shot_id
                and 0.0 <= timestamp - track.last_time
                <= self.max_gap_s
            )
        ]

        assigned_detection_indices: set[int] = set()

        if active_tracks and valid_detections:
            invalid_cost = 1e6

            costs = np.full(
                (len(active_tracks), len(valid_detections)),
                invalid_cost,
                dtype=np.float64,
            )

            for track_index, track in enumerate(active_tracks):
                previous_detection = track.detections[-1]

                for detection_index, detection in enumerate(
                    valid_detections
                ):
                    iou = bbox_iou(
                        previous_detection["bbox"],
                        detection["bbox"],
                    )

                    similarity = cosine_similarity(
                        track.embedding,
                        detection["embedding"],
                    )

                    # Reject incompatible face embeddings.
                    if similarity < self.emb_veto:
                        continue

                    # When spatial overlap is weak, require a
                    # stronger embedding match.
                    if (
                        iou < self.iou_min
                        and similarity < self.emb_gate
                    ):
                        continue

                    spatial_cost = 1.0 - iou
                    embedding_cost = 1.0 - similarity

                    embedding_weight = 1.0 - self.iou_weight

                    costs[track_index, detection_index] = (
                        self.iou_weight * spatial_cost
                        + embedding_weight * embedding_cost
                    )

            # Globally optimize one-to-one assignments.
            rows, columns = linear_sum_assignment(costs)

            for row, column in zip(
                rows.tolist(), columns.tolist()
            ):
                # Invalid pairs were assigned a large sentinel cost.
                if costs[row, column] >= invalid_cost:
                    continue

                track = active_tracks[row]
                detection = valid_detections[column]

                track.add(
                    detection=detection,
                    timestamp=timestamp,
                    ema=self.ema,
                )

                detection["track_id"] = track.track_id
                assigned_detection_indices.add(column)

        # Every unmatched valid detection starts a new track.
        output: list[dict[str, Any]] = []

        for index, detection in enumerate(valid_detections):
            if index not in assigned_detection_indices:
                track = self._new_track(
                    shot_id=shot_id,
                    timestamp=timestamp,
                    detection=detection,
                )
                detection["track_id"] = track.track_id

            # Do not expose embedding arrays in the returned
            # frame-level detections unless downstream code needs them.
            output_detection = {
                key: value
                for key, value in detection.items()
                if key != "embedding"
            }
            output_detection["bbox"] = [
                float(value) for value in detection["bbox"]
            ]
            output.append(output_detection)

        return output

    # ========================================================
    # 4. Final track summaries
    # ========================================================

    def finish(self) -> list[dict[str, Any]]:
        """
        Return track summaries suitable for JSON serialization.

        Individual detection embeddings are omitted. The mean
        track embedding is retained for identity matching.
        """
        summaries: list[dict[str, Any]] = []

        for track in self.tracks:
            embeddings = [
                normalized_embedding(detection.get("embedding"))
                for detection in track.detections
                if detection.get("embedding") is not None
            ]

            embeddings = [
                embedding
                for embedding in embeddings
                if embedding.size > 0
            ]

            if embeddings:
                dimensions = {embedding.size for embedding in embeddings}

                if len(dimensions) == 1:
                    mean_embedding = normalized_embedding(
                        np.mean(np.stack(embeddings), axis=0)
                    ).tolist()
                else:
                    # This should not happen for a single model,
                    # but avoid crashing if inconsistent data is supplied.
                    mean_embedding = (
                        track.embedding.tolist()
                        if track.embedding is not None
                        else []
                    )
            else:
                mean_embedding = []

            detection_summaries = []

            for detection in track.detections:
                item = {
                    key: value
                    for key, value in detection.items()
                    if key != "embedding"
                }

                # Ensure common NumPy values become JSON-compatible.
                if isinstance(item.get("bbox"), np.ndarray):
                    item["bbox"] = item["bbox"].tolist()

                detection_summaries.append(item)

            summaries.append(
                {
                    "track_id": track.track_id,
                    "shot_id": track.shot_id,
                    "start_time": float(track.first_time),
                    "end_time": float(track.last_time),
                    "num_detections": int(track.hits),
                    "mean_quality": float(track.mean_quality),
                    "embedding": mean_embedding,
                    "detections": detection_summaries,
                }
            )

        return summaries

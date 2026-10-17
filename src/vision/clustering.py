
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize(a)
    b = _normalize(b)

    if a.size != b.size or a.size == 0:
        return 1.0

    return float(np.clip(1.0 - np.dot(a, b), 0.0, 2.0))


def _tracks_overlap(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    tolerance: float = 0.06,
) -> bool:
    """Whether two tracks contain detections at similar times in one shot."""
    if str(track_a.get("shot_id")) != str(track_b.get("shot_id")):
        return False

    times_a = [
        float(d["timestamp"])
        for d in track_a.get("detections", [])
        if "timestamp" in d
    ]
    times_b = [
        float(d["timestamp"])
        for d in track_b.get("detections", [])
        if "timestamp" in d
    ]

    if not times_a or not times_b:
        return False

    times_b.sort()
    j = 0

    for ta in sorted(times_a):
        while j < len(times_b) and times_b[j] < ta - tolerance:
            j += 1

        if j < len(times_b) and abs(times_b[j] - ta) <= tolerance:
            return True

    return False


def cluster_tracks(
    tracks: list[dict[str, Any]],
    threshold: float = 0.55,
    min_track_faces: int = 2,
    min_track_quality: float = 0.25,
    min_id_conf: float = 0.40,
    overlap_tolerance: float = 0.06,
) -> list[dict[str, Any]]:
    """
    Assign identity labels to tracks using constrained agglomerative clustering.

    Tracks with insufficient detections or quality remain UNKNOWN.
    Tracks that overlap in the same shot cannot share an identity.
    """
    output = [dict(track) for track in tracks]

    eligible_indices: list[int] = []

    for index, track in enumerate(output):
        embedding = track.get("embedding")
        num_faces = int(track.get("num_detections", 0))
        quality = float(track.get("mean_quality", 0.0))

        if (
            embedding is not None
            and len(embedding) > 0
            and num_faces >= min_track_faces
            and quality >= min_track_quality
        ):
            eligible_indices.append(index)
        else:
            track["character_id"] = "UNKNOWN"
            track["identity_confidence"] = 0.0

    if not eligible_indices:
        return output

    embeddings = [
        _normalize(np.asarray(output[i]["embedding"], dtype=np.float32))
        for i in eligible_indices
    ]

    n = len(eligible_indices)
    distances = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            distance = _cosine_distance(embeddings[i], embeddings[j])
            distances[i, j] = distance
            distances[j, i] = distance

    # Constrained average-linkage clustering:
    # any two clusters containing overlapping tracks cannot merge.
    clusters: dict[int, set[int]] = {i: {i} for i in range(n)}
    next_cluster_id = n

    cannot_link: set[tuple[int, int]] = set()

    for i in range(n):
        track_i = output[eligible_indices[i]]

        for j in range(i + 1, n):
            track_j = output[eligible_indices[j]]

            if _tracks_overlap(track_i, track_j, overlap_tolerance):
                cannot_link.add((i, j))

    while True:
        cluster_ids = list(clusters.keys())
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")

        for pos, cluster_a_id in enumerate(cluster_ids):
            cluster_a = clusters[cluster_a_id]

            for cluster_b_id in cluster_ids[pos + 1:]:
                cluster_b = clusters[cluster_b_id]

                # Cannot merge clusters if any member pair violates cannot-link.
                violates = any(
                    (min(a, b), max(a, b)) in cannot_link
                    for a in cluster_a
                    for b in cluster_b
                )
                if violates:
                    continue

                pair_distances = [
                    distances[a, b]
                    for a in cluster_a
                    for b in cluster_b
                ]
                avg_distance = float(np.mean(pair_distances))

                if avg_distance < best_distance:
                    best_distance = avg_distance
                    best_pair = (cluster_a_id, cluster_b_id)

        if best_pair is None or best_distance > threshold:
            break

        a_id, b_id = best_pair
        merged = clusters.pop(a_id) | clusters.pop(b_id)
        clusters[next_cluster_id] = merged
        next_cluster_id += 1

    # Stable IDs: order clusters by their earliest observed timestamp.
    ordered_clusters = sorted(
        clusters.values(),
        key=lambda members: min(
            float(output[eligible_indices[m]].get("start_time", 0.0))
            for m in members
        ),
    )

    for identity_number, members in enumerate(ordered_clusters, start=1):
        character_id = f"CHAR_{identity_number:02d}"

        # Confidence is a conservative similarity-to-cluster-centroid score.
        centroid = _normalize(
            np.mean([embeddings[m] for m in members], axis=0)
        )

        member_similarities = [
            1.0 - _cosine_distance(embeddings[m], centroid)
            for m in members
        ]

        confidence = float(
            np.clip(np.mean(member_similarities), 0.0, 1.0)
        )

        if confidence < min_id_conf:
            character_id = "UNKNOWN"

        for member in members:
            track = output[eligible_indices[member]]
            track["character_id"] = character_id
            track["identity_confidence"] = confidence

    return output

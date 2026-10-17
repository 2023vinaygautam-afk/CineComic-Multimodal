
import numpy as np

from src.vision.tracking import FaceTracker


def make_detection(
    bbox,
    embedding,
    det_score=0.95,
    quality=0.9,
):
    return {
        "bbox": bbox,
        "embedding": np.asarray(embedding, dtype=np.float32),
        "det_score": det_score,
        "quality": quality,
    }


def test_same_face_continues_same_track():
    tracker = FaceTracker()

    first = make_detection(
        [10, 10, 80, 80],
        [1.0, 0.0, 0.0],
    )

    second = make_detection(
        [12, 11, 82, 81],
        [0.99, 0.01, 0.0],
    )

    result1 = tracker.update("shot_1", 0.0, [first])
    result2 = tracker.update("shot_1", 0.2, [second])

    assert result1[0]["track_id"] == result2[0]["track_id"]

    summaries = tracker.finish()

    assert len(summaries) == 1
    assert summaries[0]["num_detections"] == 2


def test_faces_are_not_linked_across_shots():
    tracker = FaceTracker()

    detection1 = make_detection(
        [10, 10, 80, 80],
        [1.0, 0.0, 0.0],
    )

    detection2 = make_detection(
        [10, 10, 80, 80],
        [1.0, 0.0, 0.0],
    )

    result1 = tracker.update("shot_1", 1.0, [detection1])
    result2 = tracker.update("shot_2", 1.1, [detection2])

    assert result1[0]["track_id"] != result2[0]["track_id"]
    assert len(tracker.finish()) == 2


def test_low_confidence_detection_is_ignored():
    tracker = FaceTracker(min_score=0.5)

    detection = make_detection(
        [10, 10, 80, 80],
        [1.0, 0.0, 0.0],
        det_score=0.2,
    )

    result = tracker.update("shot_1", 0.0, [detection])

    assert result == []
    assert tracker.finish() == []


def test_invalid_detection_is_ignored():
    tracker = FaceTracker()

    detection = {
        "bbox": [10, 10, 10, 80],
        "embedding": [1.0, 0.0, 0.0],
        "det_score": 0.9,
    }

    result = tracker.update("shot_1", 0.0, [detection])

    assert result == []
    assert tracker.finish() == []


def test_track_expires_after_maximum_gap():
    tracker = FaceTracker(max_gap_s=0.5)

    first = make_detection(
        [10, 10, 80, 80],
        [1.0, 0.0, 0.0],
    )

    second = make_detection(
        [10, 10, 80, 80],
        [1.0, 0.0, 0.0],
    )

    result1 = tracker.update("shot_1", 0.0, [first])
    result2 = tracker.update("shot_1", 1.0, [second])

    assert result1[0]["track_id"] != result2[0]["track_id"]
    assert len(tracker.finish()) == 2

"""Shot boundary detection with PySceneDetect."""

from __future__ import annotations

from scenedetect import AdaptiveDetector, ContentDetector, detect


def detect_shots(video_path, duration_s: float, fps: float, detector="content",
                 threshold=27.0, min_scene_len_s=0.5) -> list[dict]:
    """Return [{'shot_id', 'start', 'end'}] in seconds."""
    min_len = max(1, int(round(min_scene_len_s * fps)))

    if detector == "adaptive":
        det = AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_len)
    else:
        det = ContentDetector(threshold=threshold, min_scene_len=min_len)

    scenes = detect(str(video_path), det)
    if not scenes:
        return [{"shot_id": 0, "start": 0.0, "end": float(duration_s)}]

    shots = [{"shot_id": k, "start": s.get_seconds(), "end": e.get_seconds()}
             for k, (s, e) in enumerate(scenes)]
    shots[0]["start"] = 0.0
    shots[-1]["end"] = float(duration_s)
    return shots

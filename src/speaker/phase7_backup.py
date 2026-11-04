"""
CineComic_MML - Phase 7
Speaker-to-character matching using diarization, face tracks,
temporal overlap, face confidence, and mouth-region motion.

Run:
    python -m src.speaker.phase7 --config configs/phase7.yaml \
        --video-id a13eb5bedd6d

Important:
This is a heuristic active-speaker baseline, not a trained lip-reading
model. Validate its assignments manually before treating them as correct.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


LOGGER = logging.getLogger("phase7")


# ---------------------------------------------------------
# File and configuration utilities
# ---------------------------------------------------------

def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return data


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    temporary_path.replace(path)


def expand_template(template: str, video_id: str) -> Path:
    return Path(template.format(video_id=video_id))


def first_value(item: dict, names: tuple[str, ...], default=None):
    for name in names:
        if name in item and item[name] is not None:
            return item[name]
    return default


def as_float(value, default=None):
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return default


# ---------------------------------------------------------
# Diarization parsing
# ---------------------------------------------------------

def normalize_speaker(value: Any) -> str:
    if value is None:
        return "unknown"

    text = str(value).strip()

    if not text or text.lower() in {
        "none", "unknown", "null", "mixed", "overlap"
    }:
        return "unknown"

    return text


def parse_turns(data: dict) -> list[dict]:
    """
    Accepts:
      {"turns": [{"start": ..., "end": ..., "speaker": ...}]}

    Also supports:
      {"segments": [{"start": ..., "end": ..., "speaker": ...}]}
      {"segment_speakers": [...]}

    Segment-speaker assignments are used as a fallback when turns
    are unavailable.
    """
    raw_turns = first_value(
        data,
        ("turns", "speaker_turns", "diarization", "segments"),
        [],
    )

    if not isinstance(raw_turns, list):
        raw_turns = []

    turns = []

    for index, item in enumerate(raw_turns):
        if not isinstance(item, dict):
            continue

        start = as_float(
            first_value(item, ("start", "start_s", "start_time"))
        )
        end = as_float(
            first_value(item, ("end", "end_s", "end_time"))
        )

        speaker = normalize_speaker(
            first_value(
                item,
                ("speaker", "speaker_id", "label", "speaker_label"),
            )
        )

        if start is None or end is None or end <= start:
            continue

        turns.append({
            "turn_id": str(
                first_value(item, ("turn_id", "segment_id", "id"), index)
            ),
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": str(item.get("text", "")),
        })

    # Some diarization artifacts keep timestamps in turns and speaker
    # labels separately. Align those entries by segment_id.
    if not turns:
        segment_speakers = data.get("segment_speakers", [])
        segments = data.get("segments", [])

        if isinstance(segment_speakers, list) and isinstance(segments, list):
            speakers_by_id = {}

            for item in segment_speakers:
                if isinstance(item, dict):
                    key = str(
                        first_value(item, ("segment_id", "id"), "")
                    )
                    speakers_by_id[key] = normalize_speaker(
                        first_value(
                            item,
                            ("speaker", "speaker_id", "label"),
                        )
                    )

            for index, item in enumerate(segments):
                if not isinstance(item, dict):
                    continue

                start = as_float(item.get("start"))
                end = as_float(item.get("end"))

                if start is None or end is None or end <= start:
                    continue

                key = str(item.get("segment_id", item.get("id", index)))

                turns.append({
                    "turn_id": key,
                    "start": start,
                    "end": end,
                    "speaker": speakers_by_id.get(key, "unknown"),
                    "text": str(item.get("text", "")),
                })

    turns.sort(key=lambda turn: turn["start"])

    if not turns:
        raise ValueError(
            "No valid diarization turns found. Inspect the input JSON "
            "and adapt parse_turns() to its schema."
        )

    return turns


# ---------------------------------------------------------
# Face-track parsing
# ---------------------------------------------------------

def get_bbox(item: dict):
    """
    Supports:
      bbox: [x1, y1, x2, y2]
      bbox: {x1, y1, x2, y2}
      bbox: {x, y, w, h}
      x1/y1/x2/y2 or x/y/w/h
    """
    bbox = first_value(item, ("bbox", "box", "face_bbox"))

    if isinstance(bbox, dict):
        x1 = as_float(bbox.get("x1"))
        y1 = as_float(bbox.get("y1"))
        x2 = as_float(bbox.get("x2"))
        y2 = as_float(bbox.get("y2"))

        if None not in (x1, y1, x2, y2):
            return [x1, y1, x2, y2]

        x = as_float(bbox.get("x"))
        y = as_float(bbox.get("y"))
        w = as_float(first_value(bbox, ("w", "width")))
        h = as_float(first_value(bbox, ("h", "height")))

        if None not in (x, y, w, h):
            return [x, y, x + w, y + h]

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        values = [as_float(value) for value in bbox[:4]]
        if all(value is not None for value in values):
            return values

    x1 = as_float(item.get("x1"))
    y1 = as_float(item.get("y1"))
    x2 = as_float(item.get("x2"))
    y2 = as_float(item.get("y2"))

    if None not in (x1, y1, x2, y2):
        return [x1, y1, x2, y2]

    return None


def extract_track_objects(data: dict) -> list[dict]:
    """
    Finds tracks in common JSON formats:
      {"tracks": [...]}
      {"tracks": {"T0001": {...}, ...}}

    Detection entries are retained with their parent track ID.
    """
    raw_tracks = data.get("tracks", data.get("track_summaries", []))

    if isinstance(raw_tracks, dict):
        entries = []

        for key, value in raw_tracks.items():
            if isinstance(value, dict):
                value = dict(value)
                value.setdefault("track_id", key)
                entries.append(value)

        raw_tracks = entries

    if not isinstance(raw_tracks, list):
        raise ValueError(
            "Face-track JSON has no supported 'tracks' collection."
        )

    normalized_tracks = []

    for index, track in enumerate(raw_tracks):
        if not isinstance(track, dict):
            continue

        track_id = str(
            first_value(
                track,
                ("track_id", "id", "character_track_id"),
                f"T{index + 1:04d}",
            )
        )

        raw_detections = first_value(
            track,
            ("detections", "observations", "frames", "samples"),
            [],
        )

        # Some files represent one detection directly as a track object.
        if not isinstance(raw_detections, list):
            raw_detections = []

        if not raw_detections and get_bbox(track) is not None:
            raw_detections = [track]

        detections = []

        for detection in raw_detections:
            if not isinstance(detection, dict):
                continue

            bbox = get_bbox(detection)
            if bbox is None:
                continue

            timestamp = as_float(
                first_value(
                    detection,
                    ("t", "time", "timestamp", "time_s", "t_sec"),
                )
            )

            frame_index = as_float(
                first_value(
                    detection,
                    ("frame_index", "frame_idx", "frame"),
                )
            )

            # If the detection has no time but the track does, use it.
            if timestamp is None:
                timestamp = as_float(
                    first_value(
                        track,
                        ("t", "time", "timestamp", "time_s"),
                    )
                )

            confidence = as_float(
                first_value(
                    detection,
                    ("det_score", "score", "confidence", "quality"),
                ),
                0.5,
            )

            detections.append({
                "t": timestamp,
                "frame_index": frame_index,
                "bbox": bbox,
                "confidence": max(0.0, min(1.0, confidence)),
            })

        # Use frame index + video FPS later if timestamp is unavailable.
        normalized_tracks.append({
            "track_id": track_id,
            "start": as_float(
                first_value(track, ("start", "start_s", "start_time"))
            ),
            "end": as_float(
                first_value(track, ("end", "end_s", "end_time"))
            ),
            "identity": first_value(
                track,
                ("character_id", "identity_id", "identity", "person_id"),
            ),
            "track_confidence": as_float(
                first_value(
                    track,
                    ("track_confidence", "mean_quality", "confidence"),
                ),
                0.5,
            ),
            "detections": detections,
        })

    return normalized_tracks


# ---------------------------------------------------------
# Temporal overlap
# ---------------------------------------------------------

def overlap_seconds(a_start: float, a_end: float,
                    b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def overlap_ratio(turn: dict, track: dict) -> float:
    start = track.get("start")
    end = track.get("end")

    if start is None or end is None:
        times = [
            item["t"]
            for item in track["detections"]
            if item["t"] is not None
        ]

        if not times:
            return 0.0

        start, end = min(times), max(times)

    intersection = overlap_seconds(
        turn["start"], turn["end"], start, end
    )

    turn_duration = max(1e-6, turn["end"] - turn["start"])
    track_duration = max(1e-6, end - start)

    # Penalize a very brief intersection. This ratio is not a probability.
    return min(
        1.0,
        intersection / min(turn_duration, track_duration),
    )


# ---------------------------------------------------------
# Mouth-motion feature
# ---------------------------------------------------------

def sample_times(start: float, end: float, sample_fps: float) -> list[float]:
    duration = end - start

    if duration <= 0:
        return []

    count = max(2, int(duration * sample_fps) + 1)

    # Avoid repeatedly seeking exactly to the endpoints.
    return np.linspace(start, end, count).tolist()


def crop_mouth(frame: np.ndarray, bbox: list[float],
               mouth_fraction: float) -> np.ndarray | None:
    height, width = frame.shape[:2]

    x1, y1, x2, y2 = [int(round(value)) for value in bbox]

    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))

    face_width = x2 - x1
    face_height = y2 - y1

    if face_width < 12 or face_height < 12:
        return None

    # Approximate the lower face/mouth area. This is not a landmark model.
    mouth_y1 = y1 + int(face_height * (1.0 - mouth_fraction))
    mouth_y2 = y2

    crop = frame[mouth_y1:mouth_y2, x1:x2]

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 32), interpolation=cv2.INTER_AREA)

    return gray


def compute_mouth_motion(
    video_path: Path,
    track: dict,
    turn: dict,
    fps: float,
    sample_fps: float,
    minimum_samples: int,
    mouth_fraction: float,
) -> dict:
    """
    Measures normalized frame-to-frame change in the lower-face crop
    for detections belonging to a track during a speaker turn.

    Important: head motion, camera motion, and lighting can also change
    this value. It is only a weak active-speaker cue.
    """
    detections = track["detections"]

    if not detections:
        return {
            "motion_score": 0.0,
            "num_samples": 0,
            "available": False,
        }

    # Prefer real timestamps. Convert frame indices using video FPS
    # only if timestamps are missing.
    usable = []

    for detection in detections:
        t = detection["t"]

        if t is None and detection["frame_index"] is not None and fps > 0:
            t = detection["frame_index"] / fps

        if t is None:
            continue

        if turn["start"] <= t <= turn["end"]:
            usable.append((t, detection))

    if len(usable) < minimum_samples:
        return {
            "motion_score": 0.0,
            "num_samples": len(usable),
            "available": False,
        }

    usable.sort(key=lambda pair: pair[0])

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    crops = []

    try:
        for timestamp, detection in usable:
            frame_number = max(0, int(round(timestamp * fps)))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()

            if not ok or frame is None:
                continue

            crop = crop_mouth(
                frame,
                detection["bbox"],
                mouth_fraction,
            )

            if crop is not None:
                crops.append(crop)
    finally:
        cap.release()

    if len(crops) < minimum_samples:
        return {
            "motion_score": 0.0,
            "num_samples": len(crops),
            "available": False,
        }

    differences = []

    for previous, current in zip(crops[:-1], crops[1:]):
        difference = np.mean(
            cv2.absdiff(previous, current)
        ) / 255.0

        differences.append(float(difference))

    if not differences:
        return {
            "motion_score": 0.0,
            "num_samples": len(crops),
            "available": False,
        }

    # Robust summary. It is a motion magnitude, not speech probability.
    motion = float(np.median(differences))

    return {
        "motion_score": motion,
        "num_samples": len(crops),
        "available": True,
    }


# ---------------------------------------------------------
# Matching engine
# ---------------------------------------------------------

def minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    low = min(values)
    high = max(values)

    if high - low < 1e-8:
        return [0.0 for _ in values]

    return [(value - low) / (high - low) for value in values]


def run_phase7(config_path: Path, video_id: str) -> dict:
    started = time.perf_counter()

    with config_path.open("r", encoding="utf-8-sig") as file:
        config_data = yaml.safe_load(file) or {}

    config = config_data.get("phase7", config_data)

    video_path = expand_template(
        config["video_template"], video_id
    )
    diarization_path = expand_template(
        config["diarization_template"], video_id
    )
    tracks_path = expand_template(
        config["tracks_template"], video_id
    )
    metadata_path = expand_template(
        config["metadata_template"], video_id
    )
    output_path = expand_template(
        config["output_template"], video_id
    )
    metrics_path = expand_template(
        config["metrics_template"], video_id
    )

    for path in (video_path, diarization_path, tracks_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    if output_path.exists() and not config.get("overwrite", True):
        LOGGER.info("Output exists and overwrite=false: %s", output_path)
        return read_json(output_path)

    diarization_data = read_json(diarization_path)
    tracks_data = read_json(tracks_path)

    metadata = (
        read_json(metadata_path)
        if metadata_path.exists()
        else {}
    )

    fps = as_float(
        first_value(metadata, ("fps", "frame_rate")),
        0.0,
    )

    # Fallback to video metadata when the metadata JSON lacks FPS.
    if fps <= 0:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = as_float(cap.get(cv2.CAP_PROP_FPS), 0.0)
        cap.release()

    if fps <= 0:
        raise ValueError(
            "Could not determine video FPS. Add fps to data/metadata JSON."
        )

    turns = parse_turns(diarization_data)
    tracks = extract_track_objects(tracks_data)

    LOGGER.info("Video ID: %s", video_id)
    LOGGER.info("Speaker turns: %d", len(turns))
    LOGGER.info("Face tracks: %d", len(tracks))
    LOGGER.info("Video FPS: %.3f", fps)

    min_overlap = float(config.get("minimum_turn_overlap", 0.30))
    min_track_conf = float(config.get("minimum_track_confidence", 0.30))
    min_score = float(config.get("minimum_match_score", 0.45))

    sample_fps = float(config.get("sample_fps", 3))
    minimum_samples = int(config.get("minimum_samples", 3))
    mouth_fraction = float(config.get("mouth_region_fraction", 0.45))
    unknown_label = str(config.get("unknown_label", "unknown"))

    matches = []
    motion_cache = {}

    for turn_index, turn in enumerate(turns, start=1):
        LOGGER.info(
            "Matching turn %d/%d: %s (%.2f-%.2f s)",
            turn_index,
            len(turns),
            turn["speaker"],
            turn["start"],
            turn["end"],
        )

        if turn["speaker"] == "unknown":
            matches.append({
                **turn,
                "character_id": unknown_label,
                "match_score": 0.0,
                "decision": "unknown_speaker",
                "candidates": [],
            })
            continue

        candidates = []

        for track in tracks:
            track_conf = max(
                0.0,
                min(1.0, float(track["track_confidence"])),
            )

            if track_conf < min_track_conf:
                continue

            temporal = overlap_ratio(turn, track)

            if temporal < min_overlap:
                continue

            cache_key = (
                turn["turn_id"],
                track["track_id"],
            )

            if cache_key not in motion_cache:
                motion_cache[cache_key] = compute_mouth_motion(
                    video_path=video_path,
                    track=track,
                    turn=turn,
                    fps=fps,
                    sample_fps=sample_fps,
                    minimum_samples=minimum_samples,
                    mouth_fraction=mouth_fraction,
                )

            motion_result = motion_cache[cache_key]

            candidates.append({
                "track_id": track["track_id"],
                "existing_character_id": track["identity"],
                "temporal_overlap": temporal,
                "track_confidence": track_conf,
                "motion_score": motion_result["motion_score"],
                "motion_available": motion_result["available"],
                "motion_samples": motion_result["num_samples"],
            })

        if not candidates:
            matches.append({
                **turn,
                "character_id": unknown_label,
                "match_score": 0.0,
                "decision": "no_candidate_track",
                "candidates": [],
            })
            continue

        # Normalize motion scores among candidates for this turn.
        raw_motion = [candidate["motion_score"] for candidate in candidates]
        normalized_motion = minmax_normalize(raw_motion)

        for candidate, motion_normalized in zip(
            candidates, normalized_motion
        ):
            # If motion could not be measured, do not treat it as evidence.
            if candidate["motion_available"]:
                motion_component = motion_normalized
                weights = (0.45, 0.25, 0.30)
                score = (
                    weights[0] * candidate["temporal_overlap"]
                    + weights[1] * candidate["track_confidence"]
                    + weights[2] * motion_component
                )
            else:
                # No mouth-motion evidence: use weaker temporal/track evidence.
                score = (
                    0.65 * candidate["temporal_overlap"]
                    + 0.35 * candidate["track_confidence"]
                )

            candidate["match_score"] = round(float(score), 4)

        candidates.sort(
            key=lambda candidate: candidate["match_score"],
            reverse=True,
        )

        best = candidates[0]
        second_score = (
            candidates[1]["match_score"]
            if len(candidates) > 1
            else 0.0
        )

        # A small margin prevents arbitrary choices between near-ties.
        margin = best["match_score"] - second_score

        if (
            best["match_score"] >= min_score
            and (len(candidates) == 1 or margin >= 0.03)
        ):
            character_id = best["existing_character_id"]

            if not character_id or str(character_id).lower() in {
                "unknown", "none", "null"
            }:
                # Stable character ID derived from the face-track ID.
                safe_track = re.sub(
                    r"[^A-Za-z0-9_-]", "_", best["track_id"]
                )
                character_id = f"CHAR_{safe_track}"

            decision = "matched"
        else:
            character_id = unknown_label
            decision = "low_confidence_or_ambiguous"

        matches.append({
            **turn,
            "character_id": character_id,
            "matched_track_id": (
                best["track_id"]
                if character_id != unknown_label
                else None
            ),
            "match_score": best["match_score"],
            "score_margin": round(float(margin), 4),
            "decision": decision,
            "candidates": candidates,
        })

    # Aggregate speaker-to-character evidence across turns.
    speaker_votes = defaultdict(lambda: defaultdict(float))
    speaker_turn_counts = defaultdict(int)

    for match in matches:
        speaker = match["speaker"]
        speaker_turn_counts[speaker] += 1

        if match["character_id"] != unknown_label:
            speaker_votes[speaker][match["character_id"]] += (
                match["match_score"]
            )

    speaker_mapping = []

    for speaker, votes in sorted(speaker_votes.items()):
        if votes:
            character_id = max(votes, key=votes.get)
            total_weight = sum(votes.values())

            speaker_mapping.append({
                "speaker_id": speaker,
                "character_id": character_id,
                "evidence_weight": round(float(total_weight), 4),
                "matched_turns": sum(
                    1 for match in matches
                    if match["speaker"] == speaker
                    and match["character_id"] == character_id
                ),
                "total_turns": speaker_turn_counts[speaker],
            })

    total_turns = len(matches)
    matched_turns = sum(
        1 for match in matches
        if match["character_id"] != unknown_label
    )

    runtime = time.perf_counter() - started

    output = {
        "video_id": video_id,
        "phase": 7,
        "method": (
            "temporal overlap + face-track confidence + "
            "heuristic mouth-region motion"
        ),
        "video_fps": fps,
        "num_speaker_turns": total_turns,
        "num_face_tracks": len(tracks),
        "num_matched_turns": matched_turns,
        "num_unknown_turns": total_turns - matched_turns,
        "match_rate": round(
            matched_turns / total_turns if total_turns else 0.0,
            4,
        ),
        "speaker_mapping": speaker_mapping,
        "turn_matches": matches,
        "runtime_seconds": round(runtime, 3),
        "ground_truth_evaluated": False,
        "limitations": [
            "Mouth-region motion is a heuristic, not lip-reading.",
            "Head movement and camera motion can look like mouth motion.",
            "Speaker diarization errors propagate into this phase.",
            "Manual labels are required to measure actual matching accuracy.",
        ],
    }

    metrics = {
        "video_id": video_id,
        "phase": 7,
        "num_speaker_turns": total_turns,
        "num_face_tracks": len(tracks),
        "num_matched_turns": matched_turns,
        "num_unknown_turns": total_turns - matched_turns,
        "match_rate": output["match_rate"],
        "runtime_seconds": round(runtime, 3),
        "ground_truth_evaluated": False,
    }

    write_json(output_path, output)
    write_json(metrics_path, metrics)

    LOGGER.info("Phase 7 complete.")
    LOGGER.info("Matched turns: %d/%d", matched_turns, total_turns)
    LOGGER.info("Output: %s", output_path)
    LOGGER.info("Metrics: %s", metrics_path)
    LOGGER.info("Runtime: %.2f seconds", runtime)

    return output


# ---------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: Active speaker to face-track matching"
    )
    parser.add_argument(
        "--config",
        default="configs/phase7.yaml",
        help="Path to Phase 7 YAML config",
    )
    parser.add_argument(
        "--video-id",
        required=True,
        help="Video ID, e.g. a13eb5bedd6d",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    run_phase7(Path(args.config), args.video_id)


if __name__ == "__main__":
    main()
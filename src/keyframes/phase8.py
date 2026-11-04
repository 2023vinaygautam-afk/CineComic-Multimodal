
"""
CineComic_MML - Phase 8: Multimodal Keyframe Selection

Inputs:
    data/frames/{video_id}/
    data/alignment/{video_id}.json
    data/speaker/{video_id}.json
    data/shots/{video_id}.json                 (optional)
    data/embeddings/clip_img_{video_id}.npy

Outputs:
    data/keyframes/{video_id}.json
    results/phase8/{video_id}_metrics.json

Missing timestamps are kept as None; they are never fabricated as zero.
Missing semantic or speaker evidence is not invented.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import yaml
except ImportError:
    yaml = None


LOG = logging.getLogger("phase8")


# ============================================================
# 1. CONFIGURATION AND JSON UTILITIES
# ============================================================

def load_config(path: Path) -> dict:
    """Load a YAML configuration, falling back to JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")

    content = path.read_text(encoding="utf-8-sig")

    if yaml is not None:
        data = yaml.safe_load(content)
    else:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Install PyYAML using: python -m pip install pyyaml"
            ) from exc

    if not isinstance(data, dict):
        raise ValueError("Configuration must contain a mapping.")

    return data


def read_json(path: Path, required: bool = False) -> Any:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input missing: {path}")
        LOG.warning("Optional input not found: %s", path)
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        if required:
            raise ValueError(
                f"Could not parse JSON {path}: {exc}"
            ) from exc

        LOG.warning("Could not read %s: %s", path, exc)
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def first(mapping: Any, keys: tuple[str, ...], default=None):
    if not isinstance(mapping, dict):
        return default

    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]

    return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def clamp01(value: Any) -> float:
    return max(0.0, min(1.0, as_float(value)))


def norm01(values: list[float]) -> list[float]:
    """Normalize values to [0, 1]."""
    if not values:
        return []

    low = min(values)
    high = max(values)

    if high - low < 1e-9:
        return [0.5 if high > 0 else 0.0 for _ in values]

    return [
        clamp01((value - low) / (high - low))
        for value in values
    ]


# ============================================================
# 2. FRAME DISCOVERY AND TIMESTAMPS
# ============================================================

def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def parse_time(value: Any) -> float | None:
    """
    Parse seconds or HH:MM:SS timestamps.

    Invalid, negative, NaN and infinite values return None.
    """
    try:
        if isinstance(value, str):
            value = value.strip()

            if not value:
                return None

            if ":" in value:
                result = 0.0
                for part in value.split(":"):
                    result = result * 60.0 + float(part)
            else:
                result = float(value)

        elif isinstance(value, (int, float)):
            result = float(value)

        else:
            return None

    except (TypeError, ValueError):
        return None

    if not math.isfinite(result) or result < 0:
        return None

    return result


def frame_time(
    path: Path,
    metadata: dict | None = None,
) -> float | None:
    """Extract a genuine timestamp without inventing one."""

    if isinstance(metadata, dict):
        value = parse_time(first(
            metadata,
            (
                "t",
                "time",
                "timestamp",
                "time_seconds",
                "timestamp_seconds",
                "timestamp_sec",
                "start_time",
                "start",
            ),
        ))

        if value is not None:
            return value

    # Examples: frame_t12.5.jpg, frame_time_12.5.jpg
    match = re.search(
        r"(?:^|[_-])t(?:ime)?[_-]?(\d+(?:\.\d+)?)",
        path.stem,
        re.IGNORECASE,
    )

    if match:
        return parse_time(match.group(1))

    # Numeric filename means timestamp only when the whole stem is numeric.
    if re.fullmatch(r"\d+(?:\.\d+)?", path.stem):
        return parse_time(path.stem)

    # Names such as s01_f009 do not tell us the timestamp.
    return None


def extract_frame_records(
    frame_dir: Path,
    shots: Any,
) -> list[dict]:
    paths = list_images(frame_dir)

    if isinstance(shots, dict):
        shot_rows = first(
            shots,
            ("frames", "candidate_frames", "sampled_frames"),
            [],
        )
        if not shot_rows:
            shot_rows = first(shots, ("shots",), [])
    elif isinstance(shots, list):
        shot_rows = shots
    else:
        shot_rows = []

    metadata_by_name = {}

    if isinstance(shot_rows, list):
        for row in shot_rows:
            if not isinstance(row, dict):
                continue

            name = first(
                row,
                ("frame_id", "id", "filename", "file", "path"),
            )

            if name:
                metadata_by_name[Path(str(name)).stem] = row

    records = []

    for index, path in enumerate(paths):
        metadata = metadata_by_name.get(path.stem, {})

        frame_id = str(first(
            metadata,
            ("frame_id", "id"),
            path.stem,
        ))

        records.append({
            "frame_id": Path(frame_id).stem,
            "path": str(path),
            "t": frame_time(path, metadata),
            "timestamp_source": (
                "metadata_or_filename"
                if frame_time(path, metadata) is not None
                else "unavailable"
            ),
            "source_index": index,
            "shot_id": first(
                metadata,
                ("shot_id", "shot", "segment_id"),
            ),
            "metadata": metadata,
        })

    # Known timestamps first, then unknown timestamps.
    records.sort(
        key=lambda row: (
            row["t"] is None,
            row["t"] if row["t"] is not None else float("inf"),
            row["frame_id"],
        )
    )

    return records


# ============================================================
# 3. PHASE 6: SEMANTIC ALIGNMENT
# ============================================================

def get_segments(alignment: Any) -> list[dict]:
    """Support Phase 6's alignment key and common alternatives."""
    if isinstance(alignment, list):
        rows = alignment

    elif isinstance(alignment, dict):
        rows = first(
            alignment,
            (
                "alignment",
                "segments",
                "alignments",
                "results",
                "items",
            ),
            [],
        )

        if not rows and (
            "candidates" in alignment or "segment_id" in alignment
        ):
            rows = [alignment]

    else:
        rows = []

    if not isinstance(rows, list):
        return []

    return [row for row in rows if isinstance(row, dict)]


def get_candidate_rows(segment: dict) -> list[dict]:
    candidates = first(
        segment,
        ("candidates", "frames", "matches", "candidate_frames"),
        [],
    )

    if not isinstance(candidates, list):
        return []

    output = []

    for candidate in candidates:
        if isinstance(candidate, str):
            output.append({"frame_id": Path(candidate).stem})
        elif isinstance(candidate, dict):
            output.append(candidate)

    return output


def segment_text(segment: dict) -> str:
    return str(first(
        segment,
        ("text", "transcript", "utterance", "sentence", "caption"),
        "",
    ) or "")


def segment_id(segment: dict, index: int) -> str:
    return str(first(
        segment,
        ("segment_id", "id", "turn_id"),
        index + 1,
    ))


def candidate_frame_id(candidate: dict) -> str | None:
    value = first(
        candidate,
        ("frame_id", "id", "image_id", "filename", "file", "path"),
    )

    return Path(str(value)).stem if value is not None else None


def semantic_by_frame(
    segments: list[dict],
) -> tuple[dict[str, float], dict[str, list[str]], dict[str, list[str]]]:
    """
    Map actual Phase 6 candidate scores to frame IDs.

    Missing evidence remains missing; this function does not fabricate
    semantic matches for frames that have no alignment evidence.
    """
    raw_scores: dict[str, list[float]] = {}
    segment_ids: dict[str, list[str]] = {}
    texts: dict[str, list[str]] = {}

    for index, segment in enumerate(segments):
        sid = segment_id(segment, index)
        text = segment_text(segment)

        if segment.get("non_visual", False):
            continue

        for candidate in get_candidate_rows(segment):
            fid = candidate_frame_id(candidate)

            if not fid:
                continue

            similarity = first(
                candidate,
                (
                    "clip_sim",
                    "similarity",
                    "score",
                    "cosine_similarity",
                    "sim",
                ),
            )
            z_score = first(
                candidate,
                ("z", "z_score", "normalized_score"),
            )
            rank = candidate.get("rank")

            if z_score is not None:
                z = max(-20.0, min(20.0, as_float(z_score)))
                value = 1.0 / (1.0 + math.exp(-z))

            elif similarity is not None:
                value = clamp01(
                    (as_float(similarity) + 1.0) / 2.0
                )

            elif rank is not None:
                value = 1.0 / max(1.0, as_float(rank, 1.0))

            else:
                continue

            raw_scores.setdefault(fid, []).append(value)
            segment_ids.setdefault(fid, []).append(sid)

            if text:
                texts.setdefault(fid, []).append(text)

    averaged = {
        fid: float(np.mean(values))
        for fid, values in raw_scores.items()
    }

    if averaged:
        ids = list(averaged)
        normalized = norm01([averaged[fid] for fid in ids])
        averaged = {
            fid: normalized[index]
            for index, fid in enumerate(ids)
        }

    return averaged, segment_ids, texts


def alignment_frame_metadata(alignment: Any) -> dict[str, dict]:
    """Read genuine timestamps and shot IDs from Phase 6 candidates."""
    output = {}

    for segment in get_segments(alignment):
        for candidate in get_candidate_rows(segment):
            fid = candidate_frame_id(candidate)

            if not fid:
                continue

            metadata = output.setdefault(fid, {})

            timestamp = parse_time(first(
                candidate,
                (
                    "t",
                    "time",
                    "timestamp",
                    "time_seconds",
                    "timestamp_seconds",
                ),
            ))

            if timestamp is not None:
                metadata["t"] = timestamp

            shot_id = first(candidate, ("shot_id", "shot"))

            if shot_id is not None:
                metadata["shot_id"] = shot_id

    return output

# ============================================================
# 4. PHASE 6: CLIP EMBEDDING MAPPING
# ============================================================

def read_npy(path: Path):
    if not path.exists():
        LOG.warning("CLIP image embeddings not found: %s", path)
        return None

    try:
        array = np.load(path, allow_pickle=False)

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            LOG.warning(
                "Expected a 2D embedding matrix; got %s",
                array.shape,
            )
            return None

        array = array.astype(np.float32, copy=False)

        if not np.isfinite(array).all():
            LOG.warning("Embedding matrix contains NaN or infinity.")
            return None

        norms = np.linalg.norm(array, axis=1, keepdims=True)
        array = array / np.maximum(norms, 1e-12)

        return array

    except Exception as exc:
        LOG.warning("Cannot load embeddings %s: %s", path, exc)
        return None


def load_embeddings(
    config: dict,
    video_id: str,
    alignment: Any,
) -> dict[str, np.ndarray]:
    """
    Map CLIP embedding rows using image_embedding_index in Phase 6.

    Do not assume embedding row i corresponds to frame i in the full
    frames directory.
    """
    template = config.get(
        "image_embeddings_template",
        "data/embeddings/clip_img_{video_id}.npy",
    )
    path = Path(template.format(video_id=video_id))
    array = read_npy(path)

    if array is None:
        return {}

    segments = get_segments(alignment)

    if not segments:
        LOG.warning("No usable Phase 6 alignment records.")
        return {}

    embeddings: dict[str, np.ndarray] = {}

    for segment in segments:
        for candidate in get_candidate_rows(segment):
            fid = candidate_frame_id(candidate)
            row_index = candidate.get("image_embedding_index")

            if fid is None or row_index is None:
                continue

            try:
                row_index = int(row_index)
            except (TypeError, ValueError):
                LOG.warning(
                    "Invalid embedding index for %s: %r",
                    fid,
                    row_index,
                )
                continue

            if not 0 <= row_index < len(array):
                LOG.warning(
                    "Embedding index %d out of range for %s; "
                    "embedding rows available: %d",
                    row_index,
                    fid,
                    len(array),
                )
                continue

            embeddings[fid] = array[row_index]

    LOG.info(
        "CLIP embedding rows: %d; mapped unique frames: %d",
        len(array),
        len(embeddings),
    )

    if not embeddings:
        LOG.warning(
            "No embedding rows mapped. Check image_embedding_index "
            "in the Phase 6 alignment JSON."
        )

    return embeddings


# ============================================================
# 5. IMAGE QUALITY FEATURES
# ============================================================

def image_features(path: str) -> tuple[float, float, float]:
    """
    Return sharpness, exposure and saliency proxies in [0, 1].

    These are heuristic image-quality features, not model predictions.
    """
    if cv2 is None:
        return 0.5, 0.5, 0.5

    image = cv2.imread(path)

    if image is None:
        return 0.0, 0.0, 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharp_raw = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )
    sharp = clamp01(sharp_raw / 500.0)

    mean_brightness = float(np.mean(gray)) / 255.0
    exposure = clamp01(
        1.0 - abs(mean_brightness - 0.5) / 0.5
    )

    height, width = gray.shape[:2]
    crop = gray[
        int(height * 0.2):int(height * 0.8),
        int(width * 0.2):int(width * 0.8),
    ]

    contrast = (
        float(np.std(crop)) / 64.0
        if crop.size
        else 0.0
    )

    saliency = clamp01(
        0.55 * clamp01(contrast)
        + 0.45 * clamp01(sharp_raw / 350.0)
    )

    return sharp, exposure, saliency


# ============================================================
# 6. PHASE 7: SPEAKER / CHARACTER MATCHING
# ============================================================

def parse_turns(speaker_data: Any) -> list[dict]:
    """Read speaker turns and character assignments from Phase 7."""
    if isinstance(speaker_data, list):
        rows = speaker_data

    elif isinstance(speaker_data, dict):
        rows = first(
            speaker_data,
            ("turn_matches", "turns", "segments", "matches"),
            [],
        )

    else:
        rows = []

    if not isinstance(rows, list):
        return []

    output = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue

        start = parse_time(first(
            row,
            ("start", "start_time", "start_seconds", "t_start"),
        ))
        end = parse_time(first(
            row,
            ("end", "end_time", "end_seconds", "t_end"),
        ))

        if start is None or end is None or end < start:
            continue

        character_id = first(
            row,
            (
                "final_character_id",
                "visual_character_id",
                "character_id",
            ),
        )

        decision = str(first(
            row,
            ("decision", "status", "match_status"),
            "",
        )).lower()

        if (
            character_id is not None
            and str(character_id).lower() == "unknown"
        ):
            character_id = None

        output.append({
            "id": str(first(
                row,
                ("turn_id", "segment_id", "id"),
                index + 1,
            )),
            "speaker_id": first(
                row,
                ("speaker_id", "speaker", "diarization_speaker"),
            ),
            "character_id": character_id,
            "track_id": first(
                row,
                ("track_id", "visual_track_id"),
            ),
            "confidence": clamp01(first(
                row,
                ("score", "confidence", "match_score"),
                0.0,
            )),
            "start": start,
            "end": end,
            "decision": decision,
        })

    return output


def speaker_score(
    timestamp: float | None,
    turns: list[dict],
) -> tuple[float, list[dict], list[str], str | None]:
    """
    Find active speaker turns for a timestamp.

    If the frame timestamp is missing, do not assign a speaker.
    """
    if timestamp is None:
        return 0.0, [], [], None

    active = [
        turn for turn in turns
        if turn["start"] <= timestamp <= turn["end"]
    ]

    if not active:
        return 0.0, [], [], None

    assigned = [
        turn for turn in active
        if turn.get("character_id") not in (None, "")
        and turn.get("decision", "") not in (
            "rejected",
            "unmatched",
        )
    ]

    turn_ids = [turn["id"] for turn in active]

    if not assigned:
        return 0.0, active, turn_ids, None

    best_score = max(
        turn["confidence"] for turn in assigned
    )

    character_ids = list(dict.fromkeys(
        str(turn["character_id"])
        for turn in assigned
    ))

    character_id = (
        character_ids[0]
        if len(character_ids) == 1
        else None
    )

    return clamp01(best_score), active, turn_ids, character_id


# ============================================================
# 7. MULTIMODAL DIVERSITY AND OPTIONAL INPUTS
# ============================================================

def diversity_score(
    frame_id: str,
    selected: list[dict],
    embeddings: dict[str, np.ndarray],
    record_by_id: dict[str, dict],
) -> float:
    """Estimate difference from previously selected frames."""
    if not selected:
        return 1.0

    if (
        frame_id in embeddings
        and all(
            item["frame_id"] in embeddings
            for item in selected
        )
    ):
        similarities = [
            float(np.dot(
                embeddings[frame_id],
                embeddings[item["frame_id"]],
            ))
            for item in selected
        ]

        return clamp01(1.0 - max(similarities))

    timestamp = record_by_id[frame_id]["t"]
    selected_times = [
        item["t"] for item in selected
        if item.get("t") is not None
    ]

    # Unknown timestamps cannot support temporal-diversity estimates.
    if timestamp is None or not selected_times:
        return 0.5

    nearest = min(
        abs(timestamp - other_time)
        for other_time in selected_times
    )

    return clamp01(nearest / 8.0)


def load_optional(
    config: dict,
    key: str,
    default: str,
    video_id: str,
):
    path = Path(
        config.get(key, default).format(video_id=video_id)
    )
    return read_json(path)


def normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize nonnegative weights so that they sum to one."""
    clean = {
        key: max(0.0, as_float(value))
        for key, value in weights.items()
    }
    total = sum(clean.values())

    if total <= 0:
        return {
            key: 1.0 / len(clean)
            for key in clean
        } if clean else {}

    return {
        key: value / total
        for key, value in clean.items()
    }


def get_group_timestamp(
    record: dict,
    fallback_interval: float,
) -> float:
    """
    Use a genuine timestamp for temporal grouping when available.

    For a timestamp-less frame, use its stable source index only to group
    it; this value is not saved as a timestamp or used for speaker matching.
    """
    timestamp = record.get("t")

    if timestamp is not None:
        return timestamp

    return record["source_index"] * fallback_interval

# ============================================================
# 8. MAIN PHASE 8 PIPELINE
# ============================================================

def run_phase8(config_path: str, video_id: str) -> dict:
    started = time.perf_counter()

    cfg = load_config(Path(config_path))
    section = cfg.get("phase8", cfg)

    if not isinstance(section, dict):
        raise ValueError("The phase8 configuration must be a mapping.")

    def configured_path(key: str, default: str) -> Path:
        template = section.get(key, default)
        return Path(template.format(video_id=video_id))

    frame_dir = configured_path(
        "frames_dir_template",
        "data/frames/{video_id}",
    )
    alignment_path = configured_path(
        "alignment_template",
        "data/alignment/{video_id}.json",
    )
    speaker_path = configured_path(
        "speaker_template",
        "data/speaker/{video_id}.json",
    )
    shots_path = configured_path(
        "shots_template",
        "data/shots/{video_id}.json",
    )

    alignment = read_json(alignment_path)
    speaker_data = read_json(speaker_path)
    shots = read_json(shots_path)

    records = extract_frame_records(frame_dir, shots)

    if not records:
        raise FileNotFoundError(
            f"No candidate images found under {frame_dir}"
        )

    # --------------------------------------------------------
    # Phase 6: semantic alignment and image embeddings
    # --------------------------------------------------------
    segments = get_segments(alignment)
    semantic, segment_ids, texts = semantic_by_frame(segments)

    embeddings = load_embeddings(
        section,
        video_id,
        alignment,
    )

    # --------------------------------------------------------
    # Phase 7: speaker and character matching
    # --------------------------------------------------------
    turns = parse_turns(speaker_data)

    LOG.info("Phase 6 alignment segments: %d", len(segments))
    LOG.info("Frames with semantic evidence: %d", len(semantic))
    LOG.info("Phase 7 speaker turns loaded: %d", len(turns))

    # Enrich frame records with timestamps and shot metadata from Phase 6.
    frame_metadata = alignment_frame_metadata(alignment)

    for record in records:
        metadata = frame_metadata.get(record["frame_id"], {})

        # Prefer a real Phase 6 timestamp when one is available.
        if metadata.get("t") is not None:
            record["t"] = metadata["t"]
            record["timestamp_source"] = "phase6_alignment"

        if (
            record.get("shot_id") is None
            and metadata.get("shot_id") is not None
        ):
            record["shot_id"] = metadata["shot_id"]

    # --------------------------------------------------------
    # Extract image, semantic and speaker features
    # --------------------------------------------------------
    features = {}

    for record in records:
        fid = record["frame_id"]

        sharp, exposure, saliency = image_features(
            record["path"]
        )

        speaker_raw, active_turns, active_turn_ids, character_id = (
            speaker_score(record["t"], turns)
        )

        features[fid] = {
            "sharp": sharp,
            "exposure": exposure,
            "sal": saliency,
            "speaker_raw": speaker_raw,
            "character_id": character_id,
            "active_turns": active_turns,
            "active_turn_ids": active_turn_ids,
            "sem": semantic.get(fid, 0.0),
            "segment_ids": segment_ids.get(fid, []),
            "texts": texts.get(fid, []),
        }

    sharp_norm = norm01([
        features[row["frame_id"]]["sharp"]
        for row in records
    ])
    exposure_norm = norm01([
        features[row["frame_id"]]["exposure"]
        for row in records
    ])
    saliency_norm = norm01([
        features[row["frame_id"]]["sal"]
        for row in records
    ])
    speaker_norm = norm01([
        features[row["frame_id"]]["speaker_raw"]
        for row in records
    ])

    for index, record in enumerate(records):
        feature = features[record["frame_id"]]

        feature["quality"] = clamp01(
            0.65 * sharp_norm[index]
            + 0.35 * exposure_norm[index]
        )
        feature["sal"] = saliency_norm[index]

        feature["speaker"] = (
            speaker_norm[index]
            if turns and record["t"] is not None
            else 0.0
        )

        # A frame belongs to the dialogue branch if it has
        # semantic evidence or an active speaker turn.
        feature["branch"] = (
            "dialogue"
            if feature["segment_ids"] or feature["active_turns"]
            else "visual"
        )

    # --------------------------------------------------------
    # Motion proxy from consecutive available frames
    # --------------------------------------------------------
    motion = {}

    if cv2 is not None:
        previous = None

        for record in records:
            fid = record["frame_id"]
            image = cv2.imread(record["path"])

            if image is None:
                motion[fid] = 0.0
                continue

            small = cv2.resize(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                (96, 54),
            )

            if previous is None:
                motion[fid] = 0.0
            else:
                motion[fid] = (
                    float(np.mean(cv2.absdiff(small, previous)))
                    / 255.0
                )

            previous = small

    else:
        motion = {
            row["frame_id"]: 0.0
            for row in records
        }

    motion_norm = norm01([
        motion[row["frame_id"]]
        for row in records
    ])

    for index, record in enumerate(records):
        features[record["frame_id"]]["motion"] = motion_norm[index]

    # --------------------------------------------------------
    # Approximate frame duration
    # --------------------------------------------------------
    # Duration is computed only between records with genuine timestamps.
    known_times = [
        record["t"] for record in records
        if record["t"] is not None
    ]
    default_duration = (
        float(np.median(np.diff(sorted(set(known_times)))))
        if len(set(known_times)) > 1
        else 0.0
    )
    default_duration = max(0.0, default_duration)

    for index, record in enumerate(records):
        timestamp = record["t"]
        duration = default_duration

        if timestamp is not None:
            neighboring_differences = []

            for other in records:
                other_time = other["t"]

                if other_time is None or other is record:
                    continue

                difference = abs(other_time - timestamp)

                if difference > 1e-9:
                    neighboring_differences.append(difference)

            if neighboring_differences:
                duration = min(neighboring_differences)

        features[record["frame_id"]]["duration"] = clamp01(
            duration / 5.0
        )

    record_by_id = {
        row["frame_id"]: row
        for row in records
    }

    # --------------------------------------------------------
    # Group frames by dialogue segment, shot or visual interval
    # --------------------------------------------------------
    groups: dict[str, dict] = {}

    visual_group_seconds = max(
        1.0,
        as_float(section.get("visual_group_seconds", 5.0), 5.0),
    )

    for record in records:
        fid = record["frame_id"]
        feature = features[fid]
        segment_list = sorted(set(feature["segment_ids"]))
        shot_id = record.get("shot_id")

        if segment_list:
            group_key = (
                f"dialogue:{shot_id or 'unknown_shot'}:"
                f"{'|'.join(segment_list)}"
            )

        elif shot_id is not None:
            group_key = f"visual:shot:{shot_id}"

        else:
            group_time = get_group_timestamp(
                record,
                visual_group_seconds,
            )
            interval = int(group_time // visual_group_seconds)
            group_key = f"visual:interval:{interval}"

        group = groups.setdefault(
            group_key,
            {
                "group_id": group_key,
                "branch": feature["branch"],
                "frame_ids": [],
                "segment_ids": [],
                "character_ids": [],
            },
        )

        group["frame_ids"].append(fid)
        group["segment_ids"].extend(segment_list)

        if feature["character_id"]:
            group["character_ids"].append(
                feature["character_id"]
            )

    max_per_group = max(
        1,
        int(section.get("max_frames_per_group", 2)),
    )
    long_group_seconds = as_float(
        section.get("long_group_seconds", 8.0),
        8.0,
    )

    group_candidates = []

    for group in groups.values():
        ids = group["frame_ids"]
        times = [
            record_by_id[fid]["t"]
            for fid in ids
            if record_by_id[fid]["t"] is not None
        ]

        duration = (
            max(times) - min(times)
            if len(times) > 1
            else 0.0
        )

        group["quota"] = min(
            max_per_group,
            2 if duration >= long_group_seconds else 1,
        )

        group["segment_ids"] = sorted(
            set(group["segment_ids"])
        )
        group["character_ids"] = sorted(
            set(group["character_ids"])
        )

        group_candidates.append(group)

    # --------------------------------------------------------
    # Weighted greedy selection with diversity
    # --------------------------------------------------------
    weights = section.get("weights", {})

    dialogue_weights = normalized_weights({
        "sem": as_float(weights.get("sem", 0.35)),
        "spk": as_float(weights.get("spk", 0.25)),
        "q": as_float(weights.get("q", 0.20)),
        "sal": as_float(weights.get("sal", 0.10)),
        "div": as_float(weights.get("div", 0.10)),
    })

    visual_weights = normalized_weights({
        "q": as_float(weights.get("visual_q", 0.30)),
        "sal": as_float(weights.get("visual_sal", 0.30)),
        "motion": as_float(weights.get("motion", 0.20)),
        "duration": as_float(weights.get("duration", 0.10)),
        "div": as_float(weights.get("visual_div", 0.10)),
    })

    selected_panels = []

    for group in group_candidates:
        candidates = [
            record_by_id[fid]
            for fid in group["frame_ids"]
        ]

        chosen = []

        for _ in range(min(group["quota"], len(candidates))):
            best_record = None
            best_score = -1.0

            for record in candidates:
                if record in chosen:
                    continue

                fid = record["frame_id"]
                feature = features[fid]

                diversity = diversity_score(
                    fid,
                    chosen,
                    embeddings,
                    record_by_id,
                )

                if group["branch"] == "dialogue":
                    local_weights = dict(dialogue_weights)

                    # Do not count a missing semantic score as evidence.
                    if fid not in semantic:
                        local_weights["sem"] = 0.0

                    # Re-normalize remaining weights when evidence is absent.
                    local_weights = normalized_weights(local_weights)

                    score = (
                        local_weights.get("sem", 0.0)
                        * feature["sem"]
                        + local_weights.get("spk", 0.0)
                        * feature["speaker"]
                        + local_weights.get("q", 0.0)
                        * feature["quality"]
                        + local_weights.get("sal", 0.0)
                        * feature["sal"]
                        + local_weights.get("div", 0.0)
                        * diversity
                    )

                else:
                    score = (
                        visual_weights.get("q", 0.0)
                        * feature["quality"]
                        + visual_weights.get("sal", 0.0)
                        * feature["sal"]
                        + visual_weights.get("motion", 0.0)
                        * feature["motion"]
                        + visual_weights.get("duration", 0.0)
                        * feature["duration"]
                        + visual_weights.get("div", 0.0)
                        * diversity
                    )

                if score > best_score:
                    best_score = score
                    best_record = record

            if best_record is None:
                break

            chosen.append(best_record)

            fid = best_record["frame_id"]
            feature = features[fid]

            diversity = diversity_score(
                fid,
                chosen[:-1],
                embeddings,
                record_by_id,
            )

            active_speaker_ids = sorted(set(
                str(turn["speaker_id"])
                for turn in feature["active_turns"]
                if turn.get("speaker_id") is not None
            ))

            selected_panels.append({
                "frame_id": fid,
                "path": best_record["path"],
                "t": (
                    round(best_record["t"], 3)
                    if best_record["t"] is not None
                    else None
                ),
                "timestamp_source": best_record.get(
                    "timestamp_source",
                    "unavailable",
                ),
                "segment_ids": sorted(
                    set(group["segment_ids"])
                ),
                "character_id": feature["character_id"],
                "speaker_ids": active_speaker_ids,
                "group_id": group["group_id"],
                "scores": {
                    "sem": round(feature["sem"], 4),
                    "spk": round(feature["speaker"], 4),
                    "q": round(feature["quality"], 4),
                    "sal": round(feature["sal"], 4),
                    "motion": round(feature["motion"], 4),
                    "div": round(diversity, 4),
                    "total": round(clamp01(best_score), 4),
                },
                "branch": group["branch"],
            })

    # --------------------------------------------------------
    # Enforce page budget and restore chronological order
    # --------------------------------------------------------
    page_budget = max(
        1,
        int(section.get("page_budget", 12)),
    )

    if len(selected_panels) > page_budget:
        selected_panels = sorted(
            selected_panels,
            key=lambda panel: panel["scores"]["total"],
            reverse=True,
        )[:page_budget]

    selected_panels.sort(
        key=lambda panel: (
            panel["t"] is None,
            panel["t"] if panel["t"] is not None else float("inf"),
            panel["frame_id"],
        )
    )

    for index, panel in enumerate(selected_panels, start=1):
        panel["order"] = index

    # --------------------------------------------------------
    # Save results and diagnostic metrics
    # --------------------------------------------------------
    output_path = configured_path(
        "output_template",
        "data/keyframes/{video_id}.json",
    )

    metrics_path = configured_path(
        "metrics_template",
        "results/phase8/{video_id}_metrics.json",
    )

    runtime = time.perf_counter() - started

    frames_with_semantics = sum(
        1 for record in records
        if record["frame_id"] in semantic
    )

    frames_with_embeddings = sum(
        1 for record in records
        if record["frame_id"] in embeddings
    )

    selected_ids = [
        panel["frame_id"]
        for panel in selected_panels
    ]

    metrics = {
        "phase": 8,
        "video_id": video_id,
        "candidate_frames": len(records),
        "frames_with_real_timestamps": sum(
            record["t"] is not None
            for record in records
        ),
        "frames_without_real_timestamps": sum(
            record["t"] is None
            for record in records
        ),
        "alignment_segments": len(segments),
        "frames_with_semantic_scores": frames_with_semantics,
        "speaker_turns_loaded": len(turns),
        "embedding_rows_mapped": len(embeddings),
        "candidate_frames_with_embeddings": frames_with_embeddings,
        "selected_panels": len(selected_panels),
        "dialogue_panels": sum(
            panel["branch"] == "dialogue"
            for panel in selected_panels
        ),
        "visual_only_panels": sum(
            panel["branch"] == "visual"
            for panel in selected_panels
        ),
        "duplicate_frame_ids": (
            len(selected_ids) - len(set(selected_ids))
        ),
        "ground_truth_evaluated": False,
        "gate_passed": (
            len(selected_panels) > 0
            and len(selected_ids) == len(set(selected_ids))
        ),
        "gate_note": (
            "Operational sanity gate only. Keyframe accuracy requires "
            "human-labelled reference frames."
        ),
        "weights": {
            "dialogue": dialogue_weights,
            "visual": visual_weights,
        },
        "runtime_seconds": round(runtime, 4),
        "inputs": {
            "frames_dir": str(frame_dir),
            "alignment_json": str(alignment_path),
            "speaker_json": str(speaker_path),
            "image_embeddings": str(configured_path(
                "image_embeddings_template",
                "data/embeddings/clip_img_{video_id}.npy",
            )),
        },
    }

    write_json(output_path, {
        "phase": 8,
        "video_id": video_id,
        "selection_method": (
            "Weighted multimodal scoring with greedy diversity selection"
        ),
        "panels": selected_panels,
        "groups": group_candidates,
        "metrics_path": str(metrics_path),
    })

    write_json(metrics_path, metrics)

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    LOG.info("Phase 8 complete.")
    LOG.info("Candidate frames: %d", len(records))
    LOG.info(
        "Frames with real timestamps: %d",
        metrics["frames_with_real_timestamps"],
    )
    LOG.info(
        "Frames without real timestamps: %d",
        metrics["frames_without_real_timestamps"],
    )
    LOG.info("Alignment segments: %d", len(segments))
    LOG.info("Frames with semantic scores: %d", frames_with_semantics)
    LOG.info("Speaker turns loaded: %d", len(turns))
    LOG.info("Embedding rows mapped: %d", len(embeddings))
    LOG.info(
        "Candidate frames with embeddings: %d",
        frames_with_embeddings,
    )
    LOG.info("Selected panels: %d", len(selected_panels))
    LOG.info("Output: %s", output_path)
    LOG.info("Metrics: %s", metrics_path)
    LOG.info("Runtime: %.3f seconds", runtime)

    if not semantic:
        LOG.warning(
            "No Phase 6 semantic scores mapped. Check the alignment JSON."
        )

    if not embeddings:
        LOG.warning(
            "No CLIP embeddings mapped. Diversity uses a temporal fallback "
            "when genuine timestamps are available."
        )

    if not turns:
        LOG.warning(
            "No Phase 7 speaker turns loaded. Speaker evidence is unavailable."
        )

    if metrics["frames_without_real_timestamps"]:
        LOG.warning(
            "%d frames have no genuine timestamp. These frames will not "
            "receive speaker matches based on guessed timestamps.",
            metrics["frames_without_real_timestamps"],
        )

    return metrics


# ============================================================
# 9. COMMAND-LINE ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CineComic_MML Phase 8 keyframe selection"
    )

    parser.add_argument(
        "--config",
        default="configs/phase8.yaml",
        help="Path to the Phase 8 YAML configuration",
    )

    parser.add_argument(
        "--video-id",
        required=True,
        help="Video ID used by the previous pipeline phases",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    run_phase8(
        config_path=args.config,
        video_id=args.video_id,
    )


if __name__ == "__main__":
    main()

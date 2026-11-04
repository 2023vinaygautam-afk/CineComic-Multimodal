
"""
Phase 12: Dialogue-aware comic assembly and export.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

LOGGER = logging.getLogger("phase12")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS: dict[str, Any] = {
    "transcript_dir": "data/subtitles",
    "keyframes_dir": "data/keyframes",
    "layout_dir": "data/layout",
    "speaker_dir": "data/speaker",
    "emotion_dir": "data/emotions",
    "styled_dir": "data/outputs",
    "output_dir": "output",
    "bubble_margin": 18,
    "bubble_padding": 12,
    "bubble_max_width_ratio": 0.72,
    "bubble_max_height_ratio": 0.36,
    "bubble_font_size": 20,
    "bubble_min_font_size": 12,
    "bubble_fill": "#fffdf7",
    "bubble_outline": "#171717",
    "bubble_text": "#111111",
    "panel_border_width": 3,
    "page_width": 1200,
    "page_height": 1600,
    "page_margin": 28,
    "panel_gap": 18,
    "max_bubbles_per_panel": 2,
    "allow_duplicate_segments": False,
    "fallback_nearest_segment_seconds": 1.5,
    "export_pdf": True,
    "export_cbz": True,
    "export_html": True,
}


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Could not read JSON %s: %s", path, exc)
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = dict(DEFAULTS)
    if not path.exists():
        LOGGER.warning("Config not found: %s. Using defaults.", path)
        return config

    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            config.update(loaded)
    except ImportError:
        LOGGER.warning(
            "PyYAML is not installed. Install with: pip install pyyaml"
        )
    except Exception as exc:
        LOGGER.warning("Could not read config %s: %s", path, exc)

    return config


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def artifact_path(
    config: dict[str, Any],
    key: str,
    video_id: str,
) -> Path:
    base = resolve_path(str(config[key]))
    candidates = [
        base / f"{video_id}.json",
        base / video_id / f"{video_id}.json",
    ]

    found = first_existing(candidates)
    if found:
        return found

    if base.is_dir():
        matches = sorted(base.glob(f"{video_id}*.json"))
        if matches:
            return matches[0]

    return candidates[0]


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def first_value(
    record: dict[str, Any],
    keys: Iterable[str],
    default: Any = None,
) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return default


def unwrap_records(
    data: Any,
    preferred_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if not isinstance(data, dict):
        return []

    for key in preferred_keys:
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    for value in data.values():
        if isinstance(value, dict):
            for key in preferred_keys:
                records = value.get(key)
                if isinstance(records, list):
                    return [
                        item for item in records
                        if isinstance(item, dict)
                    ]

    return []


def normalize_id(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    return str(value)


def normalize_segment_ids(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (str, int, float)):
        return [str(value)]

    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict):
                candidate = first_value(
                    item,
                    ("segment_id", "segmentId", "id", "segment"),
                )
                if candidate is not None:
                    result.append(str(candidate))
            elif item is not None:
                result.append(str(item))
        return list(dict.fromkeys(result))

    return []


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_transcript(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    records = unwrap_records(
        data,
        ("segments", "transcript", "utterances", "subtitles", "results"),
    )

    if not records and isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) and any(
                field in value for field in ("text", "dialogue", "content")
            ):
                item = dict(value)
                item.setdefault("segment_id", key)
                records.append(item)

    transcript: dict[str, dict[str, Any]] = {}

    for index, record in enumerate(records):
        segment_id = first_value(
            record,
            ("segment_id", "segmentId", "id", "uid"),
            index,
        )
        start = to_float(
            first_value(record, ("start", "start_time", "startTime", "begin")),
            0.0,
        )
        end = to_float(
            first_value(record, ("end", "end_time", "endTime", "finish")),
            start,
        )
        text_value = first_value(
            record,
            ("text", "dialogue", "content", "sentence", "transcript"),
            "",
        )

        if isinstance(text_value, dict):
            text_value = first_value(
                text_value, ("text", "content"), ""
            )

        text_value = re.sub(
            r"\s+", " ", str(text_value or "")
        ).strip()

        if not text_value:
            continue

        transcript[str(segment_id)] = {
            "segment_id": str(segment_id),
            "start": float(start or 0.0),
            "end": float(end if end is not None else (start or 0.0)),
            "text": text_value,
            "raw": record,
        }

    LOGGER.info(
        "Loaded %d transcript segments from %s",
        len(transcript),
        path,
    )
    return transcript


def load_emotions(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    records = unwrap_records(
        data, ("segments", "emotions", "results", "predictions")
    )
    emotions: dict[str, dict[str, Any]] = {}

    if isinstance(data, dict) and not records:
        for segment_id, value in data.items():
            if not isinstance(value, dict):
                continue

            fused = as_dict(value.get("fused"))
            label = first_value(
                fused,
                ("label", "emotion"),
                first_value(
                    value,
                    ("label", "emotion", "predicted_label"),
                    None,
                ),
            )

            if label:
                confidence = to_float(
                    first_value(
                        fused,
                        ("conf", "confidence", "score"),
                        first_value(
                            value, ("confidence", "conf", "score"), 0.0
                        ),
                    ),
                    0.0,
                )
                emotions[str(segment_id)] = {
                    "label": str(label).lower(),
                    "confidence": float(confidence or 0.0),
                }

    for record in records:
        segment_id = first_value(
            record, ("segment_id", "segmentId", "id", "uid")
        )
        if segment_id is None:
            continue

        fused = as_dict(record.get("fused"))
        label = first_value(
            fused,
            ("label", "emotion"),
            first_value(
                record, ("label", "emotion", "predicted_label"), "neutral"
            ),
        )
        confidence = first_value(
            fused,
            ("conf", "confidence", "score"),
            first_value(record, ("confidence", "conf", "score"), 0.0),
        )

        emotions[str(segment_id)] = {
            "label": str(label or "neutral").lower(),
            "confidence": float(to_float(confidence, 0.0) or 0.0),
        }

    return emotions


def load_speaker_artifact(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    data = read_json(path)
    if not isinstance(data, dict):
        return {}, []

    segment_speakers: dict[str, dict[str, Any]] = {}
    turns: list[dict[str, Any]] = []

    for key in ("turn_matches", "speaker_turns", "turns", "matches", "segments"):
        value = data.get(key)
        if not isinstance(value, list):
            continue

        for item in value:
            if not isinstance(item, dict):
                continue

            if key in ("turn_matches", "speaker_turns", "turns", "matches"):
                turns.append(dict(item))

            segment_id = first_value(
                item,
                ("segment_id", "segmentId", "transcript_segment_id"),
            )
            speaker_id = first_value(
                item,
                ("speaker_id", "speaker", "speakerId", "track_id", "track"),
            )
            character_id = first_value(
                item,
                ("character_id", "character", "face_id", "person_id"),
            )

            if segment_id is not None and (
                speaker_id is not None or character_id is not None
            ):
                confidence = to_float(
                    first_value(
                        item, ("confidence", "score", "match_confidence"), 0.0
                    ),
                    0.0,
                )
                segment_speakers[str(segment_id)] = {
                    "speaker_id": normalize_id(speaker_id),
                    "character_id": normalize_id(character_id),
                    "confidence": float(confidence or 0.0),
                }

    for map_key in (
        "segment_speakers",
        "speaker_by_segment",
        "attributions",
        "segment_mapping",
    ):
        mapping = data.get(map_key)
        if not isinstance(mapping, dict):
            continue

        for segment_id, value in mapping.items():
            if isinstance(value, dict):
                speaker_id = first_value(
                    value, ("speaker_id", "speaker", "speakerId", "track_id")
                )
                character_id = first_value(
                    value, ("character_id", "character", "face_id", "person_id")
                )
                confidence = to_float(
                    first_value(value, ("confidence", "score"), 0.0), 0.0
                )
                segment_speakers[str(segment_id)] = {
                    "speaker_id": normalize_id(speaker_id),
                    "character_id": normalize_id(character_id),
                    "confidence": float(confidence or 0.0),
                }
            elif value is not None:
                segment_speakers[str(segment_id)] = {
                    "speaker_id": str(value),
                    "character_id": None,
                    "confidence": 0.0,
                }

    for turn in turns:
        turn["_start"] = to_float(
            first_value(turn, ("start", "start_time", "startTime", "turn_start"))
        )
        turn["_end"] = to_float(
            first_value(turn, ("end", "end_time", "endTime", "turn_end"))
        )
        turn["_speaker_id"] = normalize_id(
            first_value(turn, ("speaker_id", "speaker", "speakerId", "track_id", "track"))
        )
        turn["_character_id"] = normalize_id(
            first_value(turn, ("character_id", "character", "face_id", "person_id"))
        )
        turn["_confidence"] = float(
            to_float(
                first_value(turn, ("confidence", "score", "match_confidence")),
                0.0,
            ) or 0.0
        )

    return segment_speakers, turns


def speaker_for_segment(
    segment: dict[str, Any],
    segment_speakers: dict[str, dict[str, Any]],
    speaker_turns: list[dict[str, Any]],
) -> dict[str, Any]:
    segment_id = str(segment["segment_id"])

    if segment_id in segment_speakers:
        direct = segment_speakers[segment_id]
        if direct.get("speaker_id") or direct.get("character_id"):
            return direct

    start = float(segment["start"])
    end = float(segment["end"])
    best_score = 0.0
    best_turn: dict[str, Any] | None = None

    for turn in speaker_turns:
        turn_start = turn.get("_start")
        turn_end = turn.get("_end")

        if turn_start is None or turn_end is None:
            continue

        overlap = max(
            0.0,
            min(end, turn_end) - max(start, turn_start),
        )
        score = overlap / max(0.001, end - start)

        if score <= 0:
            distance = min(
                abs(start - turn_end),
                abs(turn_start - end),
            )
            if distance <= 0.25:
                score = 0.01

        if score > best_score:
            best_score = score
            best_turn = turn

    if best_turn:
        return {
            "speaker_id": best_turn.get("_speaker_id"),
            "character_id": best_turn.get("_character_id"),
            "confidence": best_turn.get("_confidence", 0.0),
        }

    return {
        "speaker_id": None,
        "character_id": None,
        "confidence": 0.0,
    }

def load_layout_panels(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, dict):
        return []

    panels: list[dict[str, Any]] = []

    for page_index, page in enumerate(as_list(data.get("pages")), 1):
        if not isinstance(page, dict):
            continue

        page_number = int(
            to_float(
                first_value(page, ("page", "page_number", "page_id")),
                page_index,
            ) or page_index
        )

        for panel_index, panel in enumerate(as_list(page.get("panels")), 1):
            if not isinstance(panel, dict):
                continue

            item = dict(panel)
            item.setdefault("page", page_number)
            item.setdefault(
                "panel_id",
                first_value(panel, ("panel_id", "id", "index"), panel_index),
            )
            panels.append(item)

    if not panels:
        records = unwrap_records(
            data, ("panels", "layout_panels", "panel_records")
        )
        for index, panel in enumerate(records, 1):
            item = dict(panel)
            item.setdefault("page", 1)
            item.setdefault(
                "panel_id",
                first_value(panel, ("panel_id", "id", "index"), index),
            )
            panels.append(item)

    LOGGER.info("Loaded %d layout panels from %s", len(panels), path)
    return panels


def load_keyframe_records(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    records = unwrap_records(
        data,
        (
            "panels",
            "keyframes",
            "selected_frames",
            "frames",
            "candidates",
            "results",
        ),
    )

    by_frame: dict[str, dict[str, Any]] = {}

    for record in records:
        frame_id = first_value(
            record,
            ("frame_id", "frameId", "keyframe_id", "id", "frame"),
        )
        if frame_id is not None:
            by_frame[str(frame_id)] = record

    if isinstance(data, dict):
        for key, value in data.items():
            if not isinstance(value, dict):
                continue

            if any(
                field in value
                for field in ("frame_id", "segment_ids", "timestamp")
            ):
                frame_id = first_value(
                    value,
                    ("frame_id", "frameId", "keyframe_id", "id"),
                    key,
                )
                by_frame[str(frame_id)] = value

    LOGGER.info("Loaded %d keyframe records from %s", len(by_frame), path)
    return by_frame


def join_keyframes(
    panels: list[dict[str, Any]],
    keyframes: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    matched = 0
    joined: list[dict[str, Any]] = []

    for original in panels:
        panel = dict(original)
        frame_id = first_value(
            panel, ("frame_id", "frameId", "keyframe_id", "keyframe")
        )
        frame = keyframes.get(str(frame_id)) if frame_id is not None else None

        if frame:
            matched += 1
            field_mapping = {
                "segment_ids": (
                    "segment_ids",
                    "segments",
                    "transcript_segment_ids",
                    "aligned_segment_ids",
                ),
                "speaker_ids": ("speaker_ids", "speakers"),
                "timestamp": ("timestamp", "time", "start", "frame_time"),
                "character_id": ("character_id", "character", "face_id"),
                "image": (
                    "panel_image",
                    "image",
                    "image_path",
                    "styled_image",
                    "path",
                ),
            }

            for target, keys in field_mapping.items():
                if panel.get(target) in (None, "", []):
                    value = first_value(frame, keys)
                    if value is not None:
                        panel[target] = value

        panel["segment_ids"] = normalize_segment_ids(
            first_value(
                panel,
                (
                    "segment_ids",
                    "segments",
                    "transcript_segment_ids",
                    "aligned_segment_ids",
                ),
                [],
            )
        )
        joined.append(panel)

    return joined, matched


def image_candidates(
    panel: dict[str, Any],
    video_id: str,
    config: dict[str, Any],
) -> list[Path]:
    candidates: list[Path] = []

    explicit = first_value(
        panel,
        (
            "panel_image",
            "image",
            "image_path",
            "styled_image",
            "path",
            "crop_path",
        ),
    )

    if explicit:
        path = Path(str(explicit))
        candidates.append(
            path if path.is_absolute() else PROJECT_ROOT / path
        )

    frame_id = first_value(
        panel, ("frame_id", "frameId", "keyframe_id", "keyframe")
    )
    panel_id = first_value(panel, ("panel_id", "id", "index"))

    for key in ("styled_dir", "layout_dir", "keyframes_dir"):
        base = resolve_path(str(config[key]))

        candidates.extend([
            base / video_id / f"{panel_id}.png",
            base / video_id / f"{frame_id}.png"
            if frame_id is not None else base / "__missing__.png",
            base / f"{video_id}_{panel_id}.png",
            base / f"{video_id}_{frame_id}.png"
            if frame_id is not None else base / "__missing__.png",
            base / f"{video_id}.png",
        ])

    candidates.extend([
        PROJECT_ROOT / "data" / "layout" / video_id
        / f"panel_{panel_id}.png",
        PROJECT_ROOT / "data" / "layout"
        / f"{video_id}_panel_{panel_id}.png",
        PROJECT_ROOT / "data" / "outputs" / video_id
        / f"{frame_id}.png"
        if frame_id is not None else PROJECT_ROOT / "__missing__.png",
        PROJECT_ROOT / "data" / "outputs" / video_id
        / f"frame_{frame_id}.png"
        if frame_id is not None else PROJECT_ROOT / "__missing__.png",
        PROJECT_ROOT / "data" / "styled_frames" / video_id
        / f"{frame_id}.png"
        if frame_id is not None else PROJECT_ROOT / "__missing__.png",
    ])

    unique: list[Path] = []
    seen: set[str] = set()

    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)

    return unique


def find_panel_image(
    panel: dict[str, Any],
    video_id: str,
    config: dict[str, Any],
) -> Path:
    candidates = image_candidates(panel, video_id, config)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    panel_id = first_value(panel, ("panel_id", "id", "index"), "?")
    frame_id = first_value(
        panel, ("frame_id", "frameId", "keyframe_id", "keyframe"), "?"
    )
    tried = "\n  ".join(str(path) for path in candidates[:12])

    raise FileNotFoundError(
        f"No image found for panel_id={panel_id}, frame_id={frame_id}, "
        f"video_id={video_id}.\nChecked candidates:\n  {tried}\n"
        "Correct the image path in the Phase 10 layout JSON or update "
        "styled_dir in phase12.yaml."
    )


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ["arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"]
        if bold
        else ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
    )

    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue

    return ImageFont.load_default()


def text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> float:
    try:
        return float(draw.textlength(text, font=font))
    except AttributeError:
        box = draw.textbbox((0, 0), text, font=font)
        return float(box[2] - box[0])


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = re.split(r"\s+", text.strip())
    lines: list[str] = []
    current = ""

    for word in words:
        if not word:
            continue

        candidate = word if not current else f"{current} {word}"

        if text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        if text_width(draw, word, font) > max_width:
            piece = ""

            for char in word:
                if piece and text_width(draw, piece + char, font) > max_width:
                    lines.append(piece)
                    piece = char
                else:
                    piece += char

            current = piece
        else:
            current = word

    if current:
        lines.append(current)

    return lines or [""]


def normalize_safe_zone(
    panel: dict[str, Any],
    width: int,
    height: int,
    margin: int,
) -> tuple[int, int, int, int]:
    raw = first_value(panel, ("safe_zone", "safe_bbox", "text_safe_zone"))

    if isinstance(raw, dict):
        x1 = to_float(first_value(raw, ("x1", "left", "x")), margin)
        y1 = to_float(first_value(raw, ("y1", "top", "y")), margin)
        x2 = to_float(first_value(raw, ("x2", "right")), width - margin)
        y2 = to_float(first_value(raw, ("y2", "bottom")), height - margin)

    elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
        x1, y1, x2, y2 = [
            to_float(value, 0.0) or 0.0 for value in raw[:4]
        ]

        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height

    else:
        x1, y1, x2, y2 = margin, margin, width - margin, height - margin

    left = max(margin, min(width - margin, int(x1)))
    top = max(margin, min(height - margin, int(y1)))
    right = max(left + 1, min(width - margin, int(x2)))
    bottom = max(top + 1, min(height - margin, int(y2)))

    return left, top, right, bottom


def draw_bubble(
    image: Image.Image,
    text: str,
    emotion: str,
    speaker_label: str | None,
    safe_zone: tuple[int, int, int, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    draw = ImageDraw.Draw(image)
    sx1, sy1, sx2, sy2 = safe_zone

    safe_w = max(1, sx2 - sx1)
    safe_h = max(1, sy2 - sy1)
    padding = max(4, int(config["bubble_padding"]))

    max_w = max(
        80,
        min(
            int(safe_w * float(config["bubble_max_width_ratio"])),
            safe_w,
        ),
    )
    max_h = max(
        45,
        min(
            int(safe_h * float(config["bubble_max_height_ratio"])),
            safe_h,
        ),
    )

    font_size = max(10, int(config["bubble_font_size"]))
    min_size = max(8, int(config["bubble_min_font_size"]))
    font = load_font(font_size)
    lines = wrap_text(draw, text, font, max_w - 2 * padding)

    def line_height(font_obj: ImageFont.ImageFont) -> int:
        try:
            box = draw.textbbox((0, 0), "Ag", font=font_obj)
            return max(1, box[3] - box[1] + 3)
        except Exception:
            return font_size + 4

    while font_size > min_size:
        height_per_line = line_height(font)
        widest_line = max(
            text_width(draw, line, font) for line in lines
        )

        if (
            len(lines) * height_per_line + 2 * padding <= max_h
            and widest_line <= max_w - 2 * padding
        ):
            break

        font_size -= 1
        font = load_font(font_size)
        lines = wrap_text(draw, text, font, max_w - 2 * padding)

    height_per_line = line_height(font)
    content_w = int(
        max(text_width(draw, line, font) for line in lines)
    )
    bubble_w = min(max_w, max(70, content_w + 2 * padding))
    bubble_h = min(
        max_h,
        max(38, len(lines) * height_per_line + 2 * padding),
    )

    x1 = max(
        sx1,
        min(sx2 - bubble_w, sx1 + (safe_w - bubble_w) // 2),
    )
    y1 = sy1
    x2 = x1 + bubble_w
    y2 = y1 + bubble_h

    overflow = (
        len(lines) * height_per_line + 2 * padding > max_h
        or text_width(draw, text, font) > max_w * 2.5
    )

    fill = str(config.get("bubble_fill", "#fffdf7"))
    outline = str(config.get("bubble_outline", "#171717"))
    text_color = str(config.get("bubble_text", "#111111"))
    outline_width = max(1, int(config.get("bubble_outline_width", 2)))
    radius = max(8, int(min(bubble_w, bubble_h) * 0.12))

    draw.rounded_rectangle(
        (x1, y1, x2, y2),
        radius=radius,
        fill=fill,
        outline=outline,
        width=outline_width,
    )

    y = y1 + padding

    for line in lines:
        draw.text(
            (x1 + padding, y),
            line,
            fill=text_color,
            font=font,
        )
        y += height_per_line

    tail_x = int(x1 + bubble_w * 0.28)
    tail_y = min(sy2 - 1, y2 + max(5, padding // 2))

    if tail_y > y2 and tail_x + 14 < sx2:
        draw.polygon(
            [
                (tail_x, y2 - 1),
                (tail_x + 14, y2 - 1),
                (tail_x + 4, tail_y),
            ],
            fill=fill,
        )
        draw.line(
            [
                (tail_x, y2 - 1),
                (tail_x + 4, tail_y),
                (tail_x + 14, y2 - 1),
            ],
            fill=outline,
            width=outline_width,
        )

    return {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "text": text,
        "lines": lines,
        "font_size": font_size,
        "speaker_label": speaker_label,
        "emotion": emotion,
        "text_overflow": bool(overflow),
        "attributed": bool(speaker_label),
        "placement": "top_safe_zone",
    }

def panel_segment_ids(
    panel: dict[str, Any],
    transcript: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    ids = normalize_segment_ids(
        first_value(
            panel,
            (
                "segment_ids",
                "segments",
                "transcript_segment_ids",
                "aligned_segment_ids",
            ),
            [],
        )
    )
    ids = [sid for sid in ids if sid in transcript]

    if ids:
        return ids

    timestamp = to_float(
        first_value(
            panel, ("timestamp", "time", "frame_time", "start_time")
        )
    )
    if timestamp is None:
        return []

    overlapping = [
        segment
        for segment in transcript.values()
        if segment["start"] <= timestamp <= segment["end"]
    ]
    if overlapping:
        return [overlapping[0]["segment_id"]]

    tolerance = float(
        config.get("fallback_nearest_segment_seconds", 1.5)
    )
    nearest: tuple[float, dict[str, Any]] | None = None

    for segment in transcript.values():
        midpoint = (segment["start"] + segment["end"]) / 2
        distance = abs(midpoint - timestamp)

        if nearest is None or distance < nearest[0]:
            nearest = (distance, segment)

    if nearest and nearest[0] <= tolerance:
        return [nearest[1]["segment_id"]]

    return []


def build_pages(
    page_records: list[dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
) -> list[Path]:
    page_paths: list[Path] = []
    page_w = int(config["page_width"])
    page_h = int(config["page_height"])
    margin = int(config["page_margin"])
    gap = int(config["panel_gap"])

    for page in page_records:
        panels = page["panels"]
        canvas = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(canvas)
        page_no = int(page["page"])

        count = max(1, len(panels))
        cols = 2 if count >= 4 else 1
        rows = math.ceil(count / cols)

        cell_w = (
            page_w - 2 * margin - (cols - 1) * gap
        ) // cols
        cell_h = (
            page_h - 2 * margin - (rows - 1) * gap
        ) // rows

        for index, panel in enumerate(panels):
            image_path = Path(panel["image"])

            with Image.open(image_path) as src:
                panel_image = src.convert("RGB")

            col = index % cols
            row = index // cols
            x = margin + col * (cell_w + gap)
            y = margin + row * (cell_h + gap)

            panel_image.thumbnail(
                (cell_w, cell_h),
                Image.Resampling.LANCZOS,
            )

            px = x + (cell_w - panel_image.width) // 2
            py = y + (cell_h - panel_image.height) // 2

            canvas.paste(panel_image, (px, py))
            draw.rectangle(
                (
                    px,
                    py,
                    px + panel_image.width - 1,
                    py + panel_image.height - 1,
                ),
                outline="#111111",
                width=2,
            )

        page_path = output_dir / f"page_{page_no:02d}.png"
        canvas.save(page_path)
        page_paths.append(page_path)

    return page_paths


def write_html(
    page_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    chunks = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>CineComic Export</title><style>",
        "body{font-family:Arial,sans-serif;background:#eee;"
        "margin:0;padding:24px;color:#111}",
        ".page{max-width:900px;margin:0 auto 32px;background:white;"
        "padding:16px;box-shadow:0 2px 8px #bbb}",
        ".panel{margin:12px 0}",
        ".panel img{display:block;max-width:100%;height:auto;"
        "margin:auto;border:2px solid #111}",
        ".dialogue{font-size:14px;margin:6px auto;max-width:90%;"
        "padding:8px;border:1px solid #777;border-radius:12px;"
        "background:#fffdf7}",
        "</style></head><body><h1>CineComic</h1>",
    ]

    for page in page_records:
        chunks.append(
            f"<section class='page'><h2>Page {int(page['page'])}</h2>"
        )

        for panel in page["panels"]:
            chunks.append("<div class='panel'>")
            image_rel = html.escape(Path(panel["image"]).as_posix())
            panel_id = html.escape(str(panel["panel_id"]))

            chunks.append(
                f"<img src='{image_rel}' alt='Panel {panel_id}'>"
            )

            for bubble in panel.get("bubbles", []):
                dialogue = html.escape(str(bubble.get("text", "")))
                chunks.append(
                    f"<div class='dialogue'>{dialogue}</div>"
                )

            chunks.append("</div>")

        chunks.append("</section>")

    chunks.append("</body></html>")
    output_path.write_text("\n".join(chunks), encoding="utf-8")


def export_pdf(
    page_images: list[Path],
    output_path: Path,
) -> None:
    if not page_images:
        LOGGER.warning("No page images available for PDF export.")
        return

    images: list[Image.Image] = []

    for path in page_images:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())

    try:
        first_image = images[0]
        remaining_images = images[1:]

        first_image.save(
            output_path,
            save_all=True,
            append_images=remaining_images,
            resolution=150.0,
        )
    finally:
        for image in images:
            image.close()


def export_cbz(
    page_images: list[Path],
    output_path: Path,
) -> None:
    with zipfile.ZipFile(
        output_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for index, path in enumerate(page_images, 1):
            archive.write(path, arcname=f"{index:03d}.png")


def run_phase12(
    video_id: str,
    config_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)

    paths = {
        "transcript": artifact_path(
            config, "transcript_dir", video_id
        ),
        "keyframes": artifact_path(
            config, "keyframes_dir", video_id
        ),
        "layout": artifact_path(
            config, "layout_dir", video_id
        ),
        "speaker": artifact_path(
            config, "speaker_dir", video_id
        ),
        "emotion": artifact_path(
            config, "emotion_dir", video_id
        ),
    }

    for name in ("transcript", "layout"):
        if not paths[name].exists():
            raise FileNotFoundError(
                f"Required Phase 12 {name} artifact not found: "
                f"{paths[name]}"
            )

    transcript = load_transcript(paths["transcript"])
    if not transcript:
        raise ValueError(
            f"No transcript segments found in {paths['transcript']}. "
            "Check the JSON schema and transcript field names."
        )

    panels = load_layout_panels(paths["layout"])
    if not panels:
        raise ValueError(
            f"No panels found in layout artifact: {paths['layout']}"
        )

    keyframes = load_keyframe_records(paths["keyframes"])
    panels, matched_frames = join_keyframes(panels, keyframes)

    emotions = load_emotions(paths["emotion"])
    segment_speakers, speaker_turns = load_speaker_artifact(
        paths["speaker"]
    )

    output_root = resolve_path(str(config["output_dir"]))
    output_dir = output_root / video_id
    panels_output_dir = output_dir / "panels"
    panels_output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[int, list[dict[str, Any]]] = {}

    for panel in panels:
        page_no = int(
            to_float(
                first_value(panel, ("page", "page_number")),
                1,
            ) or 1
        )
        grouped.setdefault(page_no, []).append(panel)

    for page_no in grouped:
        grouped[page_no].sort(
            key=lambda panel: int(
                to_float(
                    first_value(panel, ("panel_id", "id", "index")),
                    0,
                ) or 0
            )
        )

    page_records: list[dict[str, Any]] = []
    total_bubbles = 0
    unmatched_segments = 0
    segment_seen: set[str] = set()
    overflow_count = 0
    attribution_count = 0
    panels_without_segments = 0

    for page_no in sorted(grouped):
        panel_records: list[dict[str, Any]] = []

        for fallback_index, panel in enumerate(
            grouped[page_no], 1
        ):
            panel_id = first_value(
                panel,
                ("panel_id", "id", "index"),
                fallback_index,
            )

            image_path = find_panel_image(
                panel, video_id, config
            )

            with Image.open(image_path) as source:
                panel_image = source.convert("RGB").copy()

            width, height = panel_image.size
            safe_zone = normalize_safe_zone(
                panel,
                width,
                height,
                int(config["bubble_margin"]),
            )

            segment_ids = panel_segment_ids(
                panel, transcript, config
            )

            if not segment_ids:
                panels_without_segments += 1

            bubbles: list[dict[str, Any]] = []

            for segment_id in segment_ids:
                segment = transcript.get(str(segment_id))

                if not segment:
                    unmatched_segments += 1
                    LOGGER.warning(
                        "Panel %s references missing transcript segment %s",
                        panel_id,
                        segment_id,
                    )
                    continue

                if (
                    not bool(config.get("allow_duplicate_segments", False))
                    and segment_id in segment_seen
                ):
                    continue

                segment_seen.add(segment_id)

                speaker_info = speaker_for_segment(
                    segment,
                    segment_speakers,
                    speaker_turns,
                )

                speaker_id = speaker_info.get("speaker_id")
                character_id = speaker_info.get("character_id")

                # Use keyframe speaker metadata only when Phase 7 did
                # not provide a speaker for the transcript segment.
                panel_speakers = panel.get("speaker_ids")
                if (
                    not speaker_id
                    and isinstance(panel_speakers, list)
                    and panel_speakers
                ):
                    speaker_id = panel_speakers[0]

                if not character_id:
                    character_id = panel.get("character_id")

                speaker_label = (
                    str(character_id or speaker_id)
                    if (character_id or speaker_id)
                    else None
                )

                emotion = emotions.get(
                    str(segment_id),
                    {"label": "neutral", "confidence": 0.0},
                )

                bubble = draw_bubble(
                    panel_image,
                    segment["text"],
                    emotion["label"],
                    speaker_label,
                    safe_zone,
                    config,
                )

                bubble.update({
                    "segment_id": str(segment_id),
                    "start": segment["start"],
                    "end": segment["end"],
                    "emotion": emotion["label"],
                    "emotion_confidence": emotion["confidence"],
                    "speaker_id": speaker_id,
                    "character_id": character_id,
                    "speaker_confidence": speaker_info.get(
                        "confidence", 0.0
                    ),
                })

                bubbles.append(bubble)
                total_bubbles += 1
                overflow_count += int(bubble["text_overflow"])
                attribution_count += int(bubble["attributed"])

                if len(bubbles) >= int(
                    config["max_bubbles_per_panel"]
                ):
                    break

            border_width = max(
                1, int(config["panel_border_width"])
            )
            ImageDraw.Draw(panel_image).rectangle(
                (0, 0, width - 1, height - 1),
                outline="#111111",
                width=border_width,
            )

            panel_out = panels_output_dir / (
                f"page_{page_no:02d}_panel_{panel_id}.png"
            )
            panel_image.save(panel_out)

            panel_records.append({
                "panel_id": panel_id,
                "frame_id": first_value(
                    panel,
                    ("frame_id", "frameId", "keyframe_id"),
                ),
                "timestamp": first_value(
                    panel,
                    ("timestamp", "time", "frame_time"),
                ),
                "segment_ids": segment_ids,
                "source_image": str(image_path),
                "image": str(panel_out),
                "bubbles": bubbles,
            })

        page_records.append({
            "page": page_no,
            "panels": panel_records,
        })

    page_images = build_pages(
        page_records,
        output_dir,
        config,
    )

    html_path = output_dir / "page.html"
    pdf_path = output_dir / "comic.pdf"
    cbz_path = output_dir / "comic.cbz"

    if bool(config.get("export_html", True)):
        write_html(page_records, html_path)

    if bool(config.get("export_pdf", True)):
        export_pdf(page_images, pdf_path)

    if bool(config.get("export_cbz", True)):
        export_cbz(page_images, cbz_path)

    metrics = {
        "page_count": len(page_records),
        "panel_count": sum(
            len(page["panels"]) for page in page_records
        ),
        "bubble_count": total_bubbles,
        "transcript_segment_count": len(transcript),
        "keyframe_record_count": len(keyframes),
        "keyframe_layout_matches": matched_frames,
        "segments_missing_from_transcript": unmatched_segments,
        "segments_placed": len(segment_seen),
        "panels_without_segments": panels_without_segments,
        "text_overflow_count": overflow_count,
        "attribution_count": attribution_count,
        "speaker_turn_count": len(speaker_turns),
        "emotion_segment_count": len(emotions),
        "note": (
            "Speaker attribution depends on Phase 7 turn matches. "
            "Review unknown or low-confidence speakers manually."
        ),
    }

    page_json = {
        "video_id": video_id,
        "phase": 12,
        "stage": "dialogue_aware_page_assembly_export",
        "pages": page_records,
        "metrics": metrics,
        "source_artifacts": {
            key: str(value) for key, value in paths.items()
        },
        "exports": {
            "html": (
                str(html_path)
                if bool(config.get("export_html", True))
                else None
            ),
            "pdf": (
                str(pdf_path)
                if bool(config.get("export_pdf", True))
                else None
            ),
            "cbz": (
                str(cbz_path)
                if bool(config.get("export_cbz", True))
                else None
            ),
            "page_images": [str(path) for path in page_images],
        },
    }

    write_json(output_dir / "page.json", page_json)

    LOGGER.info(
        "Phase 12 keyframe join: %d layout panels, %d matched frame IDs",
        len(panels),
        matched_frames,
    )
    LOGGER.info(
        "Phase 12 completed: pages=%d panels=%d bubbles=%d "
        "panels_without_segments=%d",
        metrics["page_count"],
        metrics["panel_count"],
        total_bubbles,
        panels_without_segments,
    )

    return {
        "metrics": metrics,
        "exports": page_json["exports"],
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 12: dialogue-aware comic assembly and export"
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--config",
        default="configs/phase12.yaml",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(
            logging, args.log_level.upper(), logging.INFO
        ),
        format=(
            "%(asctime)s | %(levelname)-7s | "
            "%(name)s | %(message)s"
        ),
    )

    config_path = resolve_path(args.config)
    result = run_phase12(args.video_id, config_path)
    metrics = result["metrics"]

    print(
        f"Phase 12 completed. Pages: {metrics['page_count']}; "
        f"panels: {metrics['panel_count']}; "
        f"bubbles: {metrics['bubble_count']}"
    )
    print(f"Output directory: {result['output_dir']}")

    for name, path in result["exports"].items():
        if path:
            print(f"{name.upper()}: {path}")


if __name__ == "__main__":
    main()

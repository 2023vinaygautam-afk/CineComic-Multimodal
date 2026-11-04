from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.vision.face_detection import FaceDetector
from src.vision.tracking import FaceTracker
from src.vision.clustering import cluster_tracks


LOGGER = logging.getLogger("cinecomic.phase5")


# ============================================================
# 1. JSON AND CONFIGURATION UTILITIES
# ============================================================

def read_json(path: Path) -> Any:
    """Read JSON data from a file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    """Write JSON data, creating the parent directory if necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_config(path: Path) -> dict[str, Any]:
    """Load the YAML configuration."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def find_json_file(root: Path, folder: str, video_id: str) -> Path:
    """Find a Phase 2 or Phase 4 JSON file."""
    directory = root / folder

    candidates = [
        directory / f"{video_id}.json",
        directory / video_id / f"{video_id}.json",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find {folder} JSON for video '{video_id}'. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
    )


def as_list(
    data: Any,
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Extract a list of dictionaries from common JSON structures."""
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)

            if isinstance(value, list):
                return [
                    item for item in value
                    if isinstance(item, dict)
                ]

    raise ValueError(
        "Expected a list or one of these list fields: "
        + ", ".join(keys)
    )


def first_number(
    record: dict[str, Any],
    keys: tuple[str, ...],
    default: float | None = None,
) -> float | None:
    """Return the first numeric value found under the given keys."""
    for key in keys:
        value = record.get(key)

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

    return default


# ============================================================
# 2. VIDEO AND SHOT UTILITIES
# ============================================================

def get_shot_times(
    shot: dict[str, Any],
) -> tuple[float, float] | None:
    """Get the start and end timestamps of a shot."""
    start = first_number(
        shot,
        ("start_time", "start_s", "start", "start_seconds"),
    )

    end = first_number(
        shot,
        ("end_time", "end_s", "end", "end_seconds"),
    )

    if start is None or end is None or end <= start:
        return None

    return start, end


def get_video_path(root: Path, video_id: str) -> Path:
    """Resolve common video paths used by CineComic_MML."""
    candidates = [
        root / "Video" / f"{video_id}.mp4",
        root / "Video" / "Video1.mp4",
        root / "raw" / f"{video_id}.mp4",
        root / "raw" / "video1.mp4",
        root / "videos" / f"{video_id}.mp4",
        root / "Video1.mp4",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    search_dirs = [
        root / "Video",
        root / "raw",
        root / "videos",
    ]

    for directory in search_dirs:
        if not directory.is_dir():
            continue

        matches = [
            path
            for path in directory.glob(f"*{video_id}*")
            if path.is_file()
            and path.suffix.lower() in {
                ".mp4",
                ".mov",
                ".mkv",
                ".avi",
            }
        ]

        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not locate video '{video_id}' under {root}. "
        "Pass --video-path with the actual video file."
    )


# ============================================================
# 3. SUBTITLE WINDOW PROCESSING
# ============================================================

def normalize_windows(
    subtitles_data: Any,
    pad_s: float,
    duration_s: float,
) -> list[tuple[float, float]]:
    """
    Extract subtitle/dialogue time windows.

    Each valid subtitle interval is extended by pad_s on both sides,
    then clipped to the video duration.
    """
    if subtitles_data is None:
        return []

    if isinstance(subtitles_data, dict):
        segments = None

        for key in (
            "segments",
            "subtitles",
            "dialogue",
            "turns",
            "items",
        ):
            if isinstance(subtitles_data.get(key), list):
                segments = subtitles_data[key]
                break

        if segments is None:
            segments = []

    elif isinstance(subtitles_data, list):
        segments = subtitles_data

    else:
        segments = []

    windows: list[tuple[float, float]] = []

    for segment in segments:
        if not isinstance(segment, dict):
            continue

        start = first_number(
            segment,
            ("start", "start_time", "start_s", "start_seconds"),
        )

        end = first_number(
            segment,
            ("end", "end_time", "end_s", "end_seconds"),
        )

        if start is None or end is None or end <= start:
            continue

        start = max(0.0, start - pad_s)
        end = min(duration_s, end + pad_s)

        if end > start:
            windows.append((start, end))

    return windows


def merge_windows(
    windows: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping or touching time intervals."""
    if not windows:
        return []

    windows = sorted(windows)
    merged = [windows[0]]

    for start, end in windows[1:]:
        previous_start, previous_end = merged[-1]

        if start <= previous_end:
            merged[-1] = (
                previous_start,
                max(previous_end, end),
            )
        else:
            merged.append((start, end))

    return merged


def select_frame_times(
    duration_s: float,
    fps: float,
    windows: list[tuple[float, float]],
    shots: list[dict[str, Any]],
    pad_s: float,
) -> list[tuple[float, str]]:
    """
    Select timestamps from dialogue windows and shot intervals.

    Returns:
        List of (timestamp, shot_id) tuples.
    """
    intervals = list(windows)

    for shot in shots:
        shot_times = get_shot_times(shot)

        if shot_times is None:
            continue

        start, end = shot_times

        intervals.append(
            (
                max(0.0, start - pad_s),
                min(duration_s, end + pad_s),
            )
        )

    intervals = merge_windows(
        [
            (start, end)
            for start, end in intervals
            if end > start
        ]
    )

    if not intervals:
        intervals = [(0.0, duration_s)]

    # For example, fps=2 means a nominal sampling interval of 0.5 seconds.
    step = 1.0 / max(float(fps), 0.1)

    times: list[float] = []

    for start, end in intervals:
        timestamp = start

        while timestamp < end and timestamp < duration_s:
            times.append(timestamp)
            timestamp += step

    # Deduplicate timestamps after rounding to milliseconds.
    times = sorted({round(timestamp, 3) for timestamp in times})

    def shot_for_time(timestamp: float) -> str:
        for index, shot in enumerate(shots):
            bounds = get_shot_times(shot)

            if bounds is None:
                continue

            if bounds[0] <= timestamp < bounds[1]:
                return str(
                    shot.get(
                        "shot_id",
                        shot.get("id", f"shot_{index + 1:04d}"),
                    )
                )

        return "shot_unknown"

    return [
        (timestamp, shot_for_time(timestamp))
        for timestamp in times
    ]


def safe_filename(value: str) -> str:
    """Convert a string into a filesystem-friendly name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


# ============================================================
# 4. PHASE 5 PIPELINE
# ============================================================

def run_phase5(
    video_id: str,
    config: dict[str, Any],
    video_path: Path | None = None,
    data_root_override: Path | None = None,
) -> dict[str, Any]:
    """
    Run Phase 5:
      1. Load shot and subtitle information.
      2. Select video frames.
      3. Detect faces and extract face embeddings.
      4. Track faces across frames.
      5. Cluster tracks into candidate identities.
      6. Save detections, tracks, embeddings and summary.
    """
    root = data_root_override or Path(
        config.get("data_dir", "data")
    )

    results_root = Path(
        config.get("results_dir", "results")
    )

    output_faces = root / "faces" / f"{video_id}.json"
    output_tracks = root / "tracks" / f"{video_id}.json"
    output_summary = results_root / "phase5" / f"{video_id}_summary.json"
    output_embeddings = root / "embeddings" / f"faces_{video_id}.npy"

    if not config.get("force", False) and output_faces.exists() and output_tracks.exists() and output_summary.exists():
        LOGGER.info("[phase5] cache hit for %s, skipping", video_id)
        return read_json(output_summary)

    settings = config.get("phase5", {})
    detection_cfg = settings.get("detection", {})
    tracking_cfg = settings.get("tracking", {})
    identity_cfg = settings.get("identity", {})

    # --------------------------------------------------------
    # Load upstream outputs
    # --------------------------------------------------------

    shots_path = find_json_file(
        root,
        "shots",
        video_id,
    )

    subtitles_path = find_json_file(
        root,
        "subtitles",
        video_id,
    )

    LOGGER.info("Loading shots from: %s", shots_path)
    LOGGER.info("Loading subtitles from: %s", subtitles_path)

    shots = as_list(
        read_json(shots_path),
        ("shots", "scenes", "items"),
    )

    subtitles = read_json(subtitles_path)

    resolved_video = (
        video_path
        if video_path is not None
        else get_video_path(root, video_id)
    )

    if not resolved_video.is_file():
        raise FileNotFoundError(
            f"Video file does not exist: {resolved_video}"
        )

    LOGGER.info("Input video: %s", resolved_video)

    # --------------------------------------------------------
    # Open video and determine duration
    # --------------------------------------------------------

    capture = cv2.VideoCapture(str(resolved_video))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {resolved_video}"
        )

    try:
        source_fps = float(
            capture.get(cv2.CAP_PROP_FPS) or 0.0
        )

        frame_count = float(
            capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        )

        duration_s = (
            frame_count / source_fps
            if source_fps > 0 and frame_count > 0
            else 0.0
        )

        if duration_s <= 0:
            raise RuntimeError(
                "Could not determine video duration. "
                "Check the video file and its codec."
            )

        LOGGER.info(
            "Video metadata: duration=%.2fs, source_fps=%.2f, "
            "frame_count=%.0f",
            duration_s,
            source_fps,
            frame_count,
        )

        # ----------------------------------------------------
        # Select timestamps for face detection
        # ----------------------------------------------------

        pad_s = float(
            detection_cfg.get("window_pad_s", 0.5)
        )

        windows = normalize_windows(
            subtitles,
            pad_s,
            duration_s,
        )

        detection_fps = float(
            detection_cfg.get("det_fps", 5)
        )

        frame_times = select_frame_times(
            duration_s=duration_s,
            fps=detection_fps,
            windows=windows,
            shots=shots,
            pad_s=pad_s,
        )

        LOGGER.info(
            "Selected %d sampled frames at nominal %.2f FPS",
            len(frame_times),
            detection_fps,
        )

        if not frame_times:
            raise RuntimeError(
                "No frames were selected for Phase 5."
            )

        # ----------------------------------------------------
        # Initialize face detector
        # ----------------------------------------------------

        detector_size = tuple(
            detection_cfg.get("det_size", [640, 640])
        )

        LOGGER.info(
            "Initializing face detector: model=%s, det_size=%s, "
            "use_gpu=%s",
            detection_cfg.get("model_name", "buffalo_l"),
            detector_size,
            detection_cfg.get("use_gpu", False),
        )

        detector = FaceDetector(
            model_name=str(
                detection_cfg.get("model_name", "buffalo_l")
            ),
            model_root=detection_cfg.get("model_root"),
            det_size=detector_size,
            use_gpu=bool(
                detection_cfg.get("use_gpu", False)
            ),
            min_face_px=int(
                detection_cfg.get("min_face_px", 40)
            ),
            min_det_score=float(
                detection_cfg.get("min_det_score", 0.5)
            ),
            ref_px=int(
                detection_cfg.get("ref_px", 112)
            ),
        )

        LOGGER.info("Face detector initialized.")

        # ----------------------------------------------------
        # Initialize face tracker
        # ----------------------------------------------------

        tracker = FaceTracker(
            max_gap_s=float(
                tracking_cfg.get("max_gap_s", 1.0)
            ),
            iou_weight=float(
                tracking_cfg.get("iou_weight", 0.5)
            ),
            iou_min=float(
                tracking_cfg.get("iou_min", 0.1)
            ),
            emb_gate=float(
                tracking_cfg.get("emb_gate", 0.5)
            ),
            emb_veto=float(
                tracking_cfg.get("emb_veto", 0.1)
            ),
            min_score=float(
                tracking_cfg.get("min_score", 0.3)
            ),
            ema=float(
                tracking_cfg.get("ema", 0.3)
            ),
        )

        # ----------------------------------------------------
        # Process frames
        # ----------------------------------------------------

        all_detections: list[dict[str, Any]] = []
        face_embeddings: list[np.ndarray] = []

        pipeline_start = time.perf_counter()
        inference_times: list[float] = []
        frames_read = 0
        frames_failed = 0

        for frame_number, (timestamp, shot_id) in enumerate(
            frame_times,
            start=1,
        ):
            LOGGER.info(
                "Processing frame %d/%d at %.2f seconds "
                "(shot=%s)",
                frame_number,
                len(frame_times),
                timestamp,
                shot_id,
            )

            # Seek to the selected timestamp.
            capture.set(
                cv2.CAP_PROP_POS_MSEC,
                timestamp * 1000.0,
            )

            ok, frame = capture.read()

            if not ok or frame is None:
                frames_failed += 1

                LOGGER.warning(
                    "Could not read frame at %.3f seconds",
                    timestamp,
                )
                continue

            frames_read += 1

            # -----------------------------------------------
            # Face detection and landmark/embedding inference
            # -----------------------------------------------

            inference_start = time.perf_counter()

            LOGGER.info(
                "Starting face detection and "
                "landmark/embedding inference for frame %d",
                frame_number,
            )

            detections = detector.detect(frame)

            inference_elapsed = (
                time.perf_counter() - inference_start
            )

            inference_times.append(inference_elapsed)

            LOGGER.info(
                "Inference finished in %.2f seconds; "
                "detected %d faces",
                inference_elapsed,
                len(detections),
            )

            # -----------------------------------------------
            # Update face tracks
            # -----------------------------------------------

            assigned = tracker.update(
                shot_id,
                timestamp,
                detections,
            )

            for detection in assigned:
                embedding = np.asarray(
                    detection["embedding"],
                    dtype=np.float32,
                )

                face_embeddings.append(embedding)

                record = dict(detection)
                record["video_id"] = video_id
                record["timestamp"] = float(timestamp)
                record["shot_id"] = shot_id
                record["frame_number"] = frame_number
                record["embedding"] = embedding.tolist()

                all_detections.append(record)

            # Log progress every five frames and on the final frame.
            if (
                frame_number % 5 == 0
                or frame_number == len(frame_times)
            ):
                elapsed_total = (
                    time.perf_counter() - pipeline_start
                )

                LOGGER.info(
                    "Progress: %d/%d frames; %d detections; "
                    "elapsed %.1f seconds",
                    frame_number,
                    len(frame_times),
                    len(all_detections),
                    elapsed_total,
                )

        # ----------------------------------------------------
        # Finalize tracking and cluster identities
        # ----------------------------------------------------

        LOGGER.info("Finalizing face tracks.")

        track_summaries = tracker.finish()

        LOGGER.info(
            "Tracker produced %d track summaries.",
            len(track_summaries),
        )

        LOGGER.info("Clustering tracks into candidate identities.")

        clustered_tracks = cluster_tracks(
            track_summaries,
            threshold=float(
                identity_cfg.get("cluster_threshold", 0.55)
            ),
            min_track_faces=int(
                identity_cfg.get("min_track_faces", 2)
            ),
            min_track_quality=float(
                identity_cfg.get("min_track_quality", 0.25)
            ),
            min_id_conf=float(
                identity_cfg.get("min_id_conf", 0.40)
            ),
            overlap_tolerance=float(
                settings.get("eval", {}).get("tol_s", 0.06)
            ),
        )

        identity_by_track = {
            track["track_id"]: track.get(
                "character_id",
                "UNKNOWN",
            )
            for track in clustered_tracks
        }

        confidence_by_track = {
            track["track_id"]: track.get(
                "identity_confidence",
                0.0,
            )
            for track in clustered_tracks
        }

        for detection in all_detections:
            track_id = detection.get("track_id")

            detection["character_id"] = identity_by_track.get(
                track_id,
                "UNKNOWN",
            )

            detection["identity_confidence"] = (
                confidence_by_track.get(track_id, 0.0)
            )

        # ----------------------------------------------------
        # Prepare output paths
        # ----------------------------------------------------

        output_faces = (
            root / "faces" / f"{video_id}.json"
        )

        output_tracks = (
            root / "tracks" / f"{video_id}.json"
        )

        output_embeddings = (
            root / "embeddings" / f"faces_{video_id}.npy"
        )

        output_summary = (
            results_root / "phase5" / f"{video_id}_summary.json"
        )

        # ----------------------------------------------------
        # Save face embeddings
        # ----------------------------------------------------

        if face_embeddings:
            embedding_matrix = np.stack(
                face_embeddings
            ).astype(np.float32)
        else:
            embedding_matrix = np.empty(
                (0, 0),
                dtype=np.float32,
            )

        output_embeddings.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            output_embeddings,
            embedding_matrix,
        )

        LOGGER.info(
            "Saved face embeddings: %s, shape=%s",
            output_embeddings,
            embedding_matrix.shape,
        )

        # ----------------------------------------------------
        # Save face detections
        # ----------------------------------------------------

        write_json(
            output_faces,
            {
                "video_id": video_id,
                "video_path": str(resolved_video),
                "num_sampled_frames": len(frame_times),
                "frames_read": frames_read,
                "frames_failed": frames_failed,
                "num_face_detections": len(all_detections),
                "detections": all_detections,
            },
        )

        LOGGER.info(
            "Saved face detections: %s",
            output_faces,
        )

        # ----------------------------------------------------
        # Save track results
        # ----------------------------------------------------

        write_json(
            output_tracks,
            {
                "video_id": video_id,
                "num_tracks": len(clustered_tracks),
                "tracks": clustered_tracks,
            },
        )

        LOGGER.info(
            "Saved face tracks: %s",
            output_tracks,
        )

        # ----------------------------------------------------
        # Build and save summary
        # ----------------------------------------------------

        total_elapsed = (
            time.perf_counter() - pipeline_start
        )

        average_inference = (
            sum(inference_times) / len(inference_times)
            if inference_times
            else 0.0
        )

        summary = {
            "video_id": video_id,
            "video_path": str(resolved_video),
            "video_duration_seconds": duration_s,
            "source_fps": source_fps,
            "sampled_frames": len(frame_times),
            "frames_read": frames_read,
            "frames_failed": frames_failed,
            "face_detections": len(all_detections),
            "tracks": len(clustered_tracks),
            "known_identity_tracks": sum(
                track.get("character_id") != "UNKNOWN"
                for track in clustered_tracks
            ),
            "unknown_identity_tracks": sum(
                track.get("character_id") == "UNKNOWN"
                for track in clustered_tracks
            ),
            "average_inference_seconds": round(
                average_inference,
                3,
            ),
            "total_processing_seconds": round(
                total_elapsed,
                3,
            ),
            "outputs": {
                "faces": str(output_faces),
                "tracks": str(output_tracks),
                "embeddings": str(output_embeddings),
            },
            "ground_truth_evaluated": False,
            "note": (
                "Metrics requiring ground truth were not evaluated. "
                "Output counts are not accuracy metrics."
            ),
        }

        write_json(
            output_summary,
            summary,
        )

        LOGGER.info(
            "Phase 5 complete: %d detections, %d tracks, "
            "average inference %.2fs, total time %.1fs",
            len(all_detections),
            len(clustered_tracks),
            average_inference,
            total_elapsed,
        )

        LOGGER.info(
            "Summary saved to: %s",
            output_summary,
        )

        return summary

    finally:
        capture.release()
        LOGGER.info("Video capture released.")


# ============================================================
# 5. COMMAND-LINE INTERFACE
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CineComic-MML Phase 5 face detection "
            "and identity tracking"
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the Phase 5 YAML configuration",
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Video ID, for example a13eb5bedd6d",
    )

    parser.add_argument(
        "--video-path",
        default=None,
        help="Optional explicit path to the source video",
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional override for the data directory",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s: %(message)s"
        ),
    )

    config_path = Path(args.config)
    config = load_config(config_path)

    video_path = (
        Path(args.video_path)
        if args.video_path
        else None
    )

    data_root = (
        Path(args.data_dir)
        if args.data_dir
        else None
    )

    summary = run_phase5(
        video_id=args.video,
        config=config,
        video_path=video_path,
        data_root_override=data_root,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
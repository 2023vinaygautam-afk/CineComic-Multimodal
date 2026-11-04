"""
CineComic-MML Phase 9: Multimodal Emotion Fusion.

Inputs:
    data/subtitles/{video_id}.json
    data/audio/{video_id}.wav
    data/speaker/{video_id}.json
    data/tracks/{video_id}.json
    data/raw/{video_id}.mp4 or data/Video/*.mp4

Outputs:
    data/emotions/{video_id}.json
    results/phase9/{video_id}_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from src.common.base_stage import BaseStage
from src.common.utils import artifact_path, get_logger, load_config
from src.emotion.text import TextEmotionClassifier, LABELS
from src.emotion.audio import AudioEmotionClassifier
from src.emotion.face import FaceEmotionClassifier
from src.emotion.fusion import fuse_modalities
from src.evaluation.emotion_metrics import evaluate_emotion_predictions

LOGGER = logging.getLogger("cinecomic.phase9")


def match_segment_to_speaker_track(
    seg_start: float,
    seg_end: float,
    speaker_data: Dict[str, Any],
) -> Optional[str]:
    """
    Finds the matched face track_id for the given dialogue time interval
    from Phase 7 speaker matches.
    """
    turn_matches = speaker_data.get("turn_matches", [])
    best_track_id = None
    max_overlap = 0.0

    for turn in turn_matches:
        t_start = turn.get("start", 0.0)
        t_end = turn.get("end", 0.0)
        overlap = max(0.0, min(seg_end, t_end) - max(seg_start, t_start))
        if overlap > max_overlap:
            max_overlap = overlap
            matched_track = turn.get("matched_track_id")
            decision = turn.get("decision", "")
            if matched_track and decision != "unknown":
                best_track_id = matched_track

    return best_track_id


def get_track_boxes(
    track_id: str,
    tracks_data: Dict[str, Any],
    start_s: float,
    end_s: float,
    window_pad_s: float = 0.5,
) -> List[Dict[str, Any]]:
    """Retrieves bounding box detections for a track inside [start_s - pad, end_s + pad]."""
    tracks = tracks_data.get("tracks", [])
    matched_boxes = []

    for trk in tracks:
        if str(trk.get("track_id")) == str(track_id):
            detections = trk.get("detections", [])
            for det in detections:
                t = det.get("timestamp", 0.0)
                if (start_s - window_pad_s) <= t <= (end_s + window_pad_s):
                    bbox = det.get("bbox")
                    if bbox:
                        matched_boxes.append({"t": t, "box": bbox})
            break

    return matched_boxes


class Phase9Stage(BaseStage):
    """Phase 9 Multimodal Emotion Fusion Stage."""

    name = "emotion"

    requires = [
        "subtitles",
        "speaker",
        "tracks",
    ]

    def __init__(
        self,
        cfg: dict,
        video_id: str,
        logger=None,
    ):
        super().__init__(cfg, video_id, logger or LOGGER)
        self.p9_cfg = cfg.get("phase9", {})
        self._out_path = Path(
            self.p9_cfg.get(
                "output_template",
                f"data/emotions/{video_id}.json",
            ).format(video_id=video_id)
        )
        self._metrics_path = Path(
            self.p9_cfg.get(
                "metrics_template",
                f"results/phase9/{video_id}_metrics.json",
            ).format(video_id=video_id)
        )
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def artifact(self) -> Path:
        return self._out_path

    @property
    def metrics_path(self) -> Path:
        return self._metrics_path

    def run(self) -> None:
        """Executes the Phase 9 multimodal emotion extraction and fusion pipeline."""
        video_id = self.video_id
        log = self.log or LOGGER

        # 1. Resolve paths
        subs_path = Path(f"data/subtitles/{video_id}.json")
        wav_path = Path(f"data/audio/{video_id}.wav")
        speaker_path = Path(f"data/speaker/{video_id}.json")
        tracks_path = Path(f"data/tracks/{video_id}.json")

        if not subs_path.exists():
            raise FileNotFoundError(f"Required subtitle file missing: {subs_path}")

        subtitles_data = json.loads(subs_path.read_text(encoding="utf-8"))
        segments = subtitles_data.get("segments", [])

        speaker_data = {}
        if speaker_path.exists():
            speaker_data = json.loads(speaker_path.read_text(encoding="utf-8"))

        tracks_data = {}
        if tracks_path.exists():
            tracks_data = json.loads(tracks_path.read_text(encoding="utf-8"))

        # Find source video
        video_path = None
        candidate_video_paths = [
            Path(f"data/Video/Video1.mp4"),
            Path(f"data/raw/{video_id}.mp4"),
        ]
        for vp in candidate_video_paths:
            if vp.exists():
                video_path = vp
                break

        log.info(f"Phase 9: processing {len(segments)} segments for video {video_id}")

        # 2. Initialize classifiers
        weights = self.p9_cfg.get("weights", {"text": 0.45, "audio": 0.30, "face": 0.25})
        device = self.p9_cfg.get("device", "cpu")

        text_classifier = TextEmotionClassifier(
            model_name=self.p9_cfg.get("text_model", "j-hartmann/emotion-english-distilroberta-base"),
            device=device,
        )

        audio_classifier = None
        if wav_path.exists():
            audio_cfg = self.p9_cfg.get("audio", {})
            audio_classifier = AudioEmotionClassifier(
                model_name=self.p9_cfg.get("audio_model", "superb/wav2vec2-base-superb-er"),
                device=device,
                min_duration_s=audio_cfg.get("min_duration_s", 0.25),
                min_rms=audio_cfg.get("min_rms", 0.002),
            )

        face_classifier = None
        if video_path and video_path.exists():
            face_cfg = self.p9_cfg.get("face", {})
            face_classifier = FaceEmotionClassifier(
                model_name=self.p9_cfg.get("face_model", "dima806/facial_emotions_image_detection"),
                device=device,
                min_face_size=face_cfg.get("min_face_size", 40),
                max_samples_per_segment=face_cfg.get("max_samples_per_segment", 4),
                margin=face_cfg.get("margin", 0.15),
            )

        # 3. Process each segment across text, audio, face
        output_segments = []

        for idx, seg in enumerate(segments):
            seg_id = seg.get("segment_id", idx + 1)
            t_start = float(seg.get("start", 0.0))
            t_end = float(seg.get("end", 0.0))
            text = seg.get("text", "")

            log.info(f"Segment {seg_id}/{len(segments)}: [{t_start:.2f}s - {t_end:.2f}s] '{text[:30]}...'")

            # --- A. Text Branch ---
            text_result = text_classifier.predict(text)

            # --- B. Audio Branch ---
            audio_result = None
            if audio_classifier and wav_path.exists():
                audio_result = audio_classifier.predict_slice(wav_path, t_start, t_end)

            # --- C. Face Branch ---
            face_result = None
            if face_classifier and video_path:
                matched_track_id = match_segment_to_speaker_track(t_start, t_end, speaker_data)
                if matched_track_id:
                    track_boxes = get_track_boxes(matched_track_id, tracks_data, t_start, t_end)
                    if track_boxes:
                        face_result = face_classifier.extract_and_predict(video_path, track_boxes)

            # --- D. Multimodal Late Fusion ---
            probs_for_fusion = {
                "text": text_result["probs"] if text_result else None,
                "audio": audio_result["probs"] if audio_result else None,
                "face": face_result["probs"] if face_result else None,
            }

            fused_label, fused_conf, active_modalities, fused_probs = fuse_modalities(
                probs_for_fusion, weights
            )

            segment_payload = {
                "segment_id": seg_id,
                "start": t_start,
                "end": t_end,
                "text": text_result,
                "audio": audio_result,
                "face": face_result,
                "fused": {
                    "label": fused_label,
                    "conf": fused_conf,
                    "modalities": active_modalities,
                    "probs": fused_probs,
                },
            }
            output_segments.append(segment_payload)

        # 4. Save output artifact
        result_payload = {
            "video_id": video_id,
            "phase": 9,
            "stage": "emotion_fusion",
            "weights": weights,
            "num_segments": len(output_segments),
            "segments": output_segments,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.artifact.parent.mkdir(parents=True, exist_ok=True)
        self.artifact.write_text(
            json.dumps(result_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info(f"Phase 9 complete: saved {len(output_segments)} emotion segments to {self.artifact}")

    def evaluate(self, gt=None) -> dict:
        """Evaluates Phase 9 outputs against contract standards and optional ground truth."""
        if not self.artifact.exists():
            raise FileNotFoundError(f"Artifact {self.artifact} not found for evaluation.")

        data = json.loads(self.artifact.read_text(encoding="utf-8"))
        segments = data.get("segments", [])
        return evaluate_emotion_predictions(segments, gt)

    def gate(self, metrics: dict) -> tuple[bool, str]:
        """Validates that Phase 9 satisfies all quality gate thresholds."""
        gates = self.p9_cfg.get("gates", {})
        min_cov = gates.get("min_coverage", 0.80)
        coverage = metrics.get("coverage", 0.0)

        if coverage < min_cov:
            return False, f"Modality coverage {coverage:.3f} < {min_cov:.3f}"

        mean_conf = metrics.get("mean_fused_confidence", 0.0)
        if mean_conf < 0.20:
            return False, f"Mean fused confidence too low: {mean_conf:.3f}"

        return True, f"Structure OK: coverage={coverage:.2f}, mean_conf={mean_conf:.2f}"


def run_phase9(config_path: str, video_id: str, force: bool = False, gt_path: Optional[str] = None) -> bool:
    """Helper function to execute Phase 9."""
    cfg = load_config(config_path)
    gt = None
    if gt_path and Path(gt_path).exists():
        gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))

    stage = Phase9Stage(cfg, video_id)
    return stage.execute(gt=gt, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="CineComic-MML Phase 9: Multimodal Emotion Fusion")
    parser.add_argument("--config", default="configs/phase9.yaml", help="Path to config file")
    parser.add_argument("--video-id", required=True, help="Video ID (e.g. a13eb5bedd6d)")
    parser.add_argument("--force", action="store_true", help="Force re-execution ignoring cache")
    parser.add_argument("--gt", default=None, help="Path to ground truth JSON (optional)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    success = run_phase9(
        config_path=args.config,
        video_id=args.video_id,
        force=args.force,
        gt_path=args.gt,
    )
    print(f"[phase9] video_id={args.video_id} gate={'PASS' if success else 'FAIL'}")
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()


"""
CineComic_MML - Phase 6: CLIP Multimodal Alignment

Inputs:
    data/shots/{video_id}.json
    data/subtitles/{video_id}.json
    data/frames/{video_id}/.../*.jpg

Outputs:
    data/embeddings/clip_img_{video_id}.npy
    data/embeddings/clip_txt_{video_id}.npy
    data/alignment/{video_id}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import open_clip
from PIL import Image


LOG = logging.getLogger("phase6")


def load_json(path: Path) -> dict:
    """Load and validate a JSON artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return data


def save_json(path: Path, data: dict) -> None:
    """Save JSON, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def resolve_path(root: Path, template: str, video_id: str) -> Path:
    """Resolve a configured artifact path relative to project root."""
    return root / template.format(video_id=video_id)


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if "phase6" not in config:
        raise ValueError("Config must contain a 'phase6' section.")

    return config["phase6"]


def get_segments(subtitles: dict) -> list[dict]:
    """Read timestamped text segments from the Phase 2 artifact."""
    segments = subtitles.get("segments")

    if not isinstance(segments, list):
        raise ValueError(
            "Subtitle JSON must contain a 'segments' list."
        )

    cleaned = []

    for seg in segments:
        text = str(seg.get("text", "")).strip()

        if not text:
            continue

        if "segment_id" not in seg:
            raise ValueError(
                "Every transcript segment must have segment_id."
            )

        cleaned.append({
            "segment_id": str(seg["segment_id"]),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": text,
        })

    return cleaned


def get_frame_index(shots: dict) -> tuple[dict, dict]:
    """
    Return:
        frame_metadata: frame_id -> metadata
        segment_candidates: segment_id -> list of frame IDs

    Supports the Phase 4 structure:
        shots: [{shot_id, frames: [{frame_id, t, ...}]}]
        segment_candidates: {"1": ["s02_f003", ...]}
    """
    frame_metadata = {}

    for shot in shots.get("shots", []):
        shot_id = str(shot.get("shot_id", ""))

        for frame in shot.get("frames", []):
            frame_id = str(frame.get("frame_id", "")).strip()

            if not frame_id:
                continue

            metadata = dict(frame)
            metadata.setdefault("shot_id", shot_id)
            frame_metadata[frame_id] = metadata

    candidates = shots.get("segment_candidates", {})

    if not isinstance(candidates, dict):
        raise ValueError(
            "Phase 4 JSON must contain a segment_candidates object."
        )

    return frame_metadata, {
        str(segment_id): [str(fid) for fid in frame_ids]
        for segment_id, frame_ids in candidates.items()
    }


def find_frame_paths(frames_root: Path) -> dict[str, Path]:
    """
    Build an index from frame ID to image path.
    Searches nested shot folders and accepts jpg/jpeg/png/webp.
    """
    if not frames_root.exists():
        raise FileNotFoundError(
            f"Candidate-frame directory not found: {frames_root}"
        )

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    result = {}

    for path in sorted(frames_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            # Phase 4 frame IDs are normally the image filename stem.
            result.setdefault(path.stem, path)

    return result


def encode_texts(
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Encode text using the CLIP text encoder."""
    vectors = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        tokens = tokenizer(batch).to(device)

        with torch.inference_mode():
            features = model.encode_text(tokens)
            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)

        vectors.append(features.cpu().numpy().astype(np.float32))

    if not vectors:
        return np.empty((0, 512), dtype=np.float32)

    return np.concatenate(vectors, axis=0)


def encode_images(
    image_paths: list[Path],
    model,
    preprocess,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Encode candidate frames using the same CLIP model."""
    vectors = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        images = []

        for path in batch_paths:
            with Image.open(path) as image:
                images.append(preprocess(image.convert("RGB")))

        batch = torch.stack(images).to(device)

        with torch.inference_mode():
            features = model.encode_image(batch)
            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)

        vectors.append(features.cpu().numpy().astype(np.float32))

        LOG.info(
            "Encoded images: %d/%d",
            min(start + len(batch_paths), len(image_paths)),
            len(image_paths),
        )

    if not vectors:
        return np.empty((0, 512), dtype=np.float32)

    return np.concatenate(vectors, axis=0)


def z_scores(values: np.ndarray) -> np.ndarray:
    """Normalize similarity scores within one segment's candidates."""
    if len(values) == 0:
        return values

    std = float(values.std())

    if std < 1e-8:
        return np.zeros_like(values, dtype=np.float32)

    return ((values - values.mean()) / std).astype(np.float32)


def run_phase6(
    project_root: Path,
    video_id: str,
    config_path: Path,
) -> dict:
    started = time.perf_counter()
    cfg = load_config(config_path)

    shots_path = resolve_path(
        project_root, cfg["shots_template"], video_id
    )
    subtitles_path = resolve_path(
        project_root, cfg["subtitles_template"], video_id
    )
    frames_root = resolve_path(
        project_root, cfg["frames_dir_template"], video_id
    )

    image_output = resolve_path(
        project_root, cfg["image_embeddings_template"], video_id
    )
    text_output = resolve_path(
        project_root, cfg["text_embeddings_template"], video_id
    )
    alignment_output = resolve_path(
        project_root, cfg["alignment_template"], video_id
    )

    LOG.info("Loading Phase 4 candidate-frame index.")
    shots = load_json(shots_path)
    subtitles = load_json(subtitles_path)

    segments = get_segments(subtitles)
    frame_metadata, candidates = get_frame_index(shots)
    frame_paths = find_frame_paths(frames_root)

    if not segments:
        raise ValueError("No non-empty transcript segments found.")

    # Include every frame referenced by at least one segment.
    referenced_ids = sorted({
        frame_id
        for frame_ids in candidates.values()
        for frame_id in frame_ids
    })

    missing = [
        frame_id for frame_id in referenced_ids
        if frame_id not in frame_paths
    ]

    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} candidate frame image(s) are missing. "
            f"First missing IDs: {preview}. "
            f"Check Phase 4's frame IDs and directory."
        )

    # Keep a deterministic order so saved embeddings are reproducible.
    frame_ids = [
        frame_id for frame_id in referenced_ids
        if frame_id in frame_paths
    ]
    image_paths = [frame_paths[frame_id] for frame_id in frame_ids]
    frame_to_index = {
        frame_id: i for i, frame_id in enumerate(frame_ids)
    }

    device_name = str(cfg.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        LOG.warning("CUDA requested but unavailable; falling back to CPU.")
        device_name = "cpu"

    device = torch.device(device_name)
    batch_size = max(1, int(cfg.get("batch_size", 8)))

    LOG.info("Loading CLIP model on %s.", device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.get("model_name", "ViT-B-32"),
        pretrained=cfg.get("pretrained", "laion2b_s34b_b79k"),
    )
    tokenizer = open_clip.get_tokenizer(
        cfg.get("model_name", "ViT-B-32")
    )
    model = model.to(device).eval()

    # Encode each unique candidate image only once.
    LOG.info("Encoding %d candidate images.", len(image_paths))
    image_embeddings = encode_images(
        image_paths, model, preprocess, device, batch_size
    )

    # Optional context is not added here; use the segment text as-is.
    segment_texts = [seg["text"] for seg in segments]
    LOG.info("Encoding %d transcript segments.", len(segment_texts))
    text_embeddings = encode_texts(
        segment_texts, model, tokenizer, device, batch_size
    )

    image_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)

    np.save(image_output, image_embeddings)
    np.save(text_output, text_embeddings)

    text_to_index = {
        seg["segment_id"]: i for i, seg in enumerate(segments)
    }

    threshold = float(
        cfg.get("non_visual_similarity_threshold", 0.18)
    )
    alignment_segments = []
    segments_without_candidates = 0

    for seg in segments:
        segment_id = seg["segment_id"]
        segment_frame_ids = candidates.get(segment_id, [])

        valid_ids = [
            fid for fid in segment_frame_ids
            if fid in frame_to_index
        ]

        if not valid_ids:
            segments_without_candidates += 1
            alignment_segments.append({
                "segment_id": segment_id,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "candidates": [],
                "non_visual": None,
                "status": "no_candidates",
            })
            continue

        img_indices = [
            frame_to_index[fid] for fid in valid_ids
        ]
        text_idx = text_to_index[segment_id]

        # Both embeddings are L2-normalized, so dot product is cosine similarity.
        similarities = (
            image_embeddings[img_indices]
            @ text_embeddings[text_idx]
        ).astype(np.float32)

        normalized = z_scores(similarities)

        ranking = sorted(
            range(len(valid_ids)),
            key=lambda i: float(similarities[i]),
            reverse=True,
        )

        ranked_candidates = []

        for rank, local_index in enumerate(ranking, start=1):
            frame_id = valid_ids[local_index]
            meta = frame_metadata.get(frame_id, {})

            ranked_candidates.append({
                "frame_id": frame_id,
                "t": meta.get("t"),
                "shot_id": str(meta.get("shot_id", "")),
                "clip_sim": round(
                    float(similarities[local_index]), 6
                ),
                "z": round(float(normalized[local_index]), 6),
                "rank": rank,
                "image_embedding_index": img_indices[local_index],
            })

        best_similarity = float(similarities.max())

        alignment_segments.append({
            "segment_id": segment_id,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "candidates": ranked_candidates,
            "best_similarity": round(best_similarity, 6),
            "non_visual": best_similarity < threshold,
            "status": "ok",
        })

    result = {
        "video_id": video_id,
        "stage": "phase6_clip_alignment",
        "model": cfg.get("model_name", "ViT-B-32"),
        "pretrained": cfg.get("pretrained", "laion2b_s34b_b79k"),
        "device": str(device),
        "num_segments": len(segments),
        "num_unique_candidate_frames": len(frame_ids),
        "segments_without_candidates": segments_without_candidates,
        "non_visual_threshold": threshold,
        "image_embeddings": str(image_output),
        "text_embeddings": str(text_output),
        "alignment": alignment_segments,
        "runtime_seconds": round(
            time.perf_counter() - started, 3
        ),
        "ground_truth_evaluated": False,
        "note": (
            "Similarity scores are ranking signals, not calibrated "
            "probabilities. The non_visual flag is a heuristic and "
            "requires validation against human relevance labels."
        ),
    }

    save_json(alignment_output, result)

    LOG.info("Saved image embeddings: %s", image_output)
    LOG.info("Saved text embeddings: %s", text_output)
    LOG.info("Saved alignment results: %s", alignment_output)
    LOG.info(
        "Phase 6 complete: %d segments, %d images, %.1f seconds.",
        len(segments),
        len(frame_ids),
        result["runtime_seconds"],
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="CineComic_MML Phase 6 - CLIP alignment"
    )
    parser.add_argument(
        "--config", default="configs/phase6.yaml"
    )
    parser.add_argument(
        "--video", required=True,
        help="Existing video ID, e.g. a13eb5bedd6d"
    )
    parser.add_argument(
        "--root", default=".",
        help="Project root directory"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    project_root = Path(args.root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    try:
        result = run_phase6(
            project_root=project_root,
            video_id=args.video,
            config_path=config_path,
        )
        print("\nPhase 6 finished.")
        print("Segments:", result["num_segments"])
        print("Candidate frames:", result["num_unique_candidate_frames"])
        print("Segments without candidates:", result["segments_without_candidates"])
        print("Runtime:", result["runtime_seconds"], "seconds")
    except Exception:
        LOG.exception("Phase 6 failed.")
        raise


if __name__ == "__main__":
    main()

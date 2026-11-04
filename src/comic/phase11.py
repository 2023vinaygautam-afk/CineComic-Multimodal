
"""
CineComic-MML — Phase 11: Cartoonization

Input:
    data/layout/{video_id}.json
    Phase 10 panel images referenced by the JSON

Outputs:
    data/outputs/panels_styled/{video_id}/panel_XXXX.png
    data/outputs/panels_styled/{video_id}/phase11_results.json

Method:
    1. Bilateral filtering to smooth colors while retaining boundaries.
    2. K-means color quantization.
    3. Adaptive-threshold edge extraction.
    4. Combine quantized colors with comic-style edges.

Original Phase 10 images are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_logger(video_id: str) -> logging.Logger:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"phase11_{video_id}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler(
            log_dir / f"{video_id}_phase11.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def load_config(config_path: Path | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "data_dir": "data",
        "layout_dir": "data/layout",
        "output_dir": "data/outputs/panels_styled",
        "bilateral_d": 9,
        "sigma_color": 75,
        "sigma_space": 75,
        "color_clusters": 8,
        "kmeans_attempts": 1,
        "edge_block_size": 9,
        "edge_c": 9,
        "edge_strength": 1.0,
        "seed": 42,
        "cache": True,
        "save_metrics": True,
    }

    if config_path is None:
        return defaults

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required. Install it with: "
            "python -m pip install pyyaml"
        ) from exc

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Phase 11 YAML must contain a mapping.")

    if isinstance(loaded.get("phase11"), dict):
        loaded = {**loaded, **loaded["phase11"]}

    return {**defaults, **loaded}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def get_panels(layout: dict[str, Any]) -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = []

    pages = layout.get("pages", [])

    if isinstance(pages, list):
        for page in pages:
            if isinstance(page, dict):
                page_number = page.get("page")

                for panel in page.get("panels", []):
                    if isinstance(panel, dict):
                        record = dict(panel)
                        record.setdefault("page", page_number)
                        panels.append(record)

    if not panels and isinstance(layout.get("panels"), list):
        panels = [
            panel for panel in layout["panels"]
            if isinstance(panel, dict)
        ]

    if not panels:
        raise ValueError(
            "No panels found in Phase 10 layout JSON. "
            "Expected pages[].panels[] or panels[]."
        )

    return panels


def get_source_image_path(panel: dict[str, Any]) -> Path:
    """
    Prefer the cropped Phase 10 panel image.

    This intentionally does not cartoonize the original full video frame.
    """
    for key in ("panel_image", "cropped_image", "image_path"):
        value = panel.get(key)

        if isinstance(value, str) and value.strip():
            path = resolve_path(value)

            if path.is_file():
                return path

    raise FileNotFoundError(
        f"No readable panel image path for panel "
        f"{panel.get('panel_id', 'unknown')}. "
        "Check panel_image in the Phase 10 layout JSON."
    )


def config_fingerprint(config: dict[str, Any]) -> str:
    serialized = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def quantize_colors(
    image: np.ndarray,
    clusters: int,
    seed: int,
    attempts: int,
) -> np.ndarray:
    """Quantize BGR colors using deterministic OpenCV k-means."""
    height, width = image.shape[:2]

    pixels = image.reshape((-1, 3)).astype(np.float32)
    unique_count = len(np.unique(pixels, axis=0))

    k = max(1, min(int(clusters), unique_count, len(pixels)))

    if k == 1:
        mean_color = np.mean(pixels, axis=0).astype(np.uint8)
        return np.tile(mean_color, (height, width, 1))

    cv2.setRNGSeed(int(seed))

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1.0,
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        max(1, int(attempts)),
        cv2.KMEANS_PP_CENTERS,
    )

    centers = np.clip(centers, 0, 255).astype(np.uint8)
    quantized = centers[labels.flatten()]

    return quantized.reshape((height, width, 3))


def cartoonize(
    image: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Create a comic-style image while preserving input dimensions."""
    if image is None or image.size == 0:
        raise ValueError("Cannot cartoonize an empty image.")

    original_height, original_width = image.shape[:2]

    d = max(1, int(config["bilateral_d"]))
    sigma_color = max(1, int(config["sigma_color"]))
    sigma_space = max(1, int(config["sigma_space"]))

    # Smooth flat regions but retain prominent boundaries.
    smooth = cv2.bilateralFilter(
        image,
        d,
        sigma_color,
        sigma_space,
    )

    # Simplify the image's color palette.
    quantized = quantize_colors(
        smooth,
        clusters=int(config["color_clusters"]),
        seed=int(config["seed"]),
        attempts=int(config["kmeans_attempts"]),
    )

    # Extract bold, black cartoon-style boundaries.
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    block_size = int(config["edge_block_size"])

    # Adaptive threshold requires an odd block size greater than one.
    block_size = max(3, block_size)
    if block_size % 2 == 0:
        block_size += 1

    # Avoid invalid parameters for unusually tiny panels.
    smallest_dimension = min(gray.shape[:2])

    if smallest_dimension < block_size:
        block_size = (
            smallest_dimension
            if smallest_dimension % 2 == 1
            else smallest_dimension - 1
        )

    if block_size < 3:
        edges = np.full_like(gray, 255)
    else:
        edges = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            int(config["edge_c"]),
        )

    # Apply edge strength using a small erosion for thicker ink lines.
    edge_strength = float(config["edge_strength"])

    if edge_strength >= 1.5:
        kernel = np.ones((2, 2), dtype=np.uint8)
        edges = cv2.erode(edges, kernel, iterations=1)

    if edge_strength <= 0:
        edges = np.full_like(edges, 255)

    # Combine the flat color regions and comic edges.
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    styled = cv2.bitwise_and(quantized, edge_bgr)

    if styled.shape[:2] != (original_height, original_width):
        raise RuntimeError("Cartoonization changed image dimensions.")

    return styled


def calculate_metrics(
    original: np.ndarray,
    styled: np.ndarray,
    runtime_s: float,
) -> dict[str, Any]:
    """
    Compute inexpensive structural and runtime diagnostics.

    These are proxies; they do not replace face-detection or human review.
    """
    if original.shape != styled.shape:
        raise ValueError("Original and styled images have different shapes.")

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    styled_gray = cv2.cvtColor(styled, cv2.COLOR_BGR2GRAY)

    original_edges = cv2.Canny(original_gray, 100, 200) > 0
    styled_edges = cv2.Canny(styled_gray, 100, 200) > 0

    intersection = np.logical_and(
        original_edges, styled_edges
    ).sum()

    union = np.logical_or(
        original_edges, styled_edges
    ).sum()

    edge_overlap = (
        float(intersection / union) if union else 1.0
    )

    original_hist = cv2.calcHist(
        [original], [0, 1, 2], None,
        [8, 8, 8], [0, 256, 0, 256, 0, 256],
    )
    styled_hist = cv2.calcHist(
        [styled], [0, 1, 2], None,
        [8, 8, 8], [0, 256, 0, 256, 0, 256],
    )

    cv2.normalize(original_hist, original_hist, alpha=1, norm_type=cv2.NORM_L1)
    cv2.normalize(styled_hist, styled_hist, alpha=1, norm_type=cv2.NORM_L1)

    histogram_distance = float(
        cv2.compareHist(
            original_hist,
            styled_hist,
            cv2.HISTCMP_BHATTACHARYYA,
        )
    )

    return {
        "height": int(original.shape[0]),
        "width": int(original.shape[1]),
        "dimensions_preserved": original.shape == styled.shape,
        "edge_overlap_proxy": edge_overlap,
        "color_histogram_distance": histogram_distance,
        "runtime_s": float(runtime_s),
    }


def run_phase11(
    video_id: str,
    config_path: Path | None = None,
) -> dict[str, Any]:
    logger = setup_logger(video_id)
    config = load_config(config_path)

    layout_path = (
        PROJECT_ROOT
        / str(config["layout_dir"])
        / f"{video_id}.json"
    )

    layout = load_json(layout_path)
    panels = get_panels(layout)

    output_dir = (
        PROJECT_ROOT
        / str(config["output_dir"])
        / video_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = config_fingerprint(config)
    panel_results = []
    runtimes = []
    cache_hits = 0
    failures = []

    logger.info("Loaded %d panels from %s", len(panels), layout_path)

    for index, panel in enumerate(panels, start=1):
        panel_id = panel.get("panel_id", index)

        try:
            source_path = get_source_image_path(panel)
            original = cv2.imread(str(source_path), cv2.IMREAD_COLOR)

            if original is None:
                raise ValueError(
                    f"OpenCV could not read {source_path}"
                )

            # Stable output filename based on panel ID.
            safe_id = "".join(
                char if char.isalnum() or char in ("-", "_") else "_"
                for char in str(panel_id)
            )
            output_path = output_dir / f"panel_{safe_id}.png"

            cached = False
            runtime_s = 0.0

            if (
                bool(config["cache"])
                and output_path.is_file()
            ):
                styled_cached = cv2.imread(
                    str(output_path), cv2.IMREAD_COLOR
                )
                if (
                    styled_cached is not None
                    and styled_cached.shape == original.shape
                ):
                    cached = True
                    styled = styled_cached
                    cache_hits += 1
                else:
                    styled = None
            else:
                styled = None

            if styled is None:
                start_time = time.perf_counter()
                styled = cartoonize(original, config)
                runtime_s = time.perf_counter() - start_time

                if not cv2.imwrite(str(output_path), styled):
                    raise IOError(
                        f"Could not save styled panel to {output_path}"
                    )

            metrics = calculate_metrics(
                original,
                styled,
                runtime_s,
            )

            result = {
                "panel_id": panel_id,
                "page": panel.get("page"),
                "frame_id": panel.get("frame_id"),
                "source_image": str(source_path),
                "styled_image": str(output_path),
                "style": "opencv_bilateral_kmeans_edges_v1",
                "config_hash": fingerprint,
                "cached": cached,
                "params": {
                    "bilateral_d": int(config["bilateral_d"]),
                    "sigma_color": int(config["sigma_color"]),
                    "sigma_space": int(config["sigma_space"]),
                    "color_clusters": int(config["color_clusters"]),
                    "edge_block_size": int(config["edge_block_size"]),
                    "edge_c": int(config["edge_c"]),
                    "edge_strength": float(config["edge_strength"]),
                },
                "metrics": metrics,
            }

            panel_results.append(result)
            runtimes.append(runtime_s)

            logger.info(
                "Panel %s | cached=%s | runtime=%.3fs | edge overlap=%.3f",
                panel_id,
                cached,
                runtime_s,
                metrics["edge_overlap_proxy"],
            )

        except Exception as exc:
            logger.exception("Failed to style panel %s", panel_id)
            failures.append({
                "panel_id": panel_id,
                "error": str(exc),
            })

    if not panel_results:
        raise RuntimeError(
            "Phase 11 failed to style any panels. "
            "Check the Phase 10 panel_image paths."
        )

    metrics_summary = {
        "input_panels": len(panels),
        "successful_panels": len(panel_results),
        "failed_panels": len(failures),
        "cache_hits": cache_hits,
        "mean_runtime_s_uncached": (
            float(np.mean(runtimes)) if runtimes else 0.0
        ),
        "all_dimensions_preserved": all(
            p["metrics"]["dimensions_preserved"]
            for p in panel_results
        ),
        "mean_edge_overlap_proxy": float(np.mean([
            p["metrics"]["edge_overlap_proxy"]
            for p in panel_results
        ])),
        "mean_color_histogram_distance": float(np.mean([
            p["metrics"]["color_histogram_distance"]
            for p in panel_results
        ])),
        "face_preservation_test": (
            "Not measured: run face detection on original and styled panels "
            "to calculate the guide's face-preservation rate."
        ),
        "note": (
            "Edge overlap and color histogram distance are diagnostics, "
            "not proof of visual quality. Review the styled images."
        ),
    }

    output = {
        "video_id": video_id,
        "phase": 11,
        "stage": "cartoonization",
        "style": "opencv_bilateral_kmeans_edges_v1",
        "config_hash": fingerprint,
        "input_layout": str(layout_path),
        "output_directory": str(output_dir),
        "panels": panel_results,
        "failures": failures,
        "metrics": metrics_summary,
        "config": config,
    }

    if bool(config["save_metrics"]):
        save_json(
            output_dir / "phase11_results.json",
            output,
        )

    logger.info(
        "Phase 11 finished: %d/%d panels styled; %d failures.",
        len(panel_results),
        len(panels),
        len(failures),
    )
    logger.info("Output directory: %s", output_dir)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 11: cartoonize Phase 10 comic panels."
    )
    parser.add_argument(
        "--video-id",
        required=True,
        help="Video ID from Phase 1.",
    )
    parser.add_argument(
        "--config",
        default="configs/phase11.yaml",
        help="Path to Phase 11 YAML config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    result = run_phase11(args.video_id, config_path)

    print("\nPhase 11 completed.")
    print(f"Panels styled: {result['metrics']['successful_panels']}")
    print(f"Failures: {result['metrics']['failed_panels']}")
    print(
        "Mean uncached runtime per panel: "
        f"{result['metrics']['mean_runtime_s_uncached']:.3f}s"
    )
    print(f"Output: {result['output_directory']}")


if __name__ == "__main__":
    main()

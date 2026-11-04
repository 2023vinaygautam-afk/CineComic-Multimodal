"""Phase 0 shared utilities: config loader, seeding, logging, artifact paths."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
from pathlib import Path

import yaml


def discover_shared_ffmpeg_bin() -> str | None:
    """Locate the FFmpeg shared-library bin directory used by torchcodec.

    Prefer the known-good Gyan shared build on Windows and ignore broken or
    unrelated shared builds that may shadow the system ffmpeg.exe.
    """
    if os.name != "nt":
        return None

    def _is_viable_shared_bin(candidate: Path) -> bool:
        exe = candidate / "ffmpeg.exe"
        if not exe.exists():
            return False

        dlls = [
            item.name.lower()
            for item in candidate.iterdir()
            if item.is_file() and item.suffix.lower() == ".dll"
        ]
        if not any(name.startswith("av") or "ffmpeg" in name for name in dlls):
            return False

        try:
            subprocess.run(
                [str(exe), "-hide_banner", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=15,
            )
        except Exception:
            return False

        return True

    roots: list[Path] = []

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        roots.append(Path(local_app_data) / "Microsoft" / "WinGet" / "Packages")

    for base in (
        Path("C:/ffmpeg/bin"),
        Path("C:/Program Files/ffmpeg/bin"),
        Path("C:/Program Files (x86)/ffmpeg/bin"),
    ):
        roots.append(base)

    best: tuple[int, str] | None = None

    for root in roots:
        if not root.exists():
            continue

        if root.name.lower() == "packages":
            for package_dir in sorted(root.iterdir()):
                if not package_dir.is_dir() or "ffmpeg" not in package_dir.name.lower():
                    continue
                for candidate in sorted(package_dir.rglob("bin")):
                    if not _is_viable_shared_bin(candidate):
                        continue
                    score = 0
                    pack_name = package_dir.name.lower()
                    if "gyan" in pack_name and "shared" in pack_name:
                        score += 100
                    elif "shared" in pack_name:
                        score += 50
                    if "ffmpeg" in pack_name:
                        score += 10
                    candidate_key = (score, str(candidate))
                    if best is None or candidate_key > best:
                        best = candidate_key
        else:
            for candidate in sorted(root.rglob("bin")):
                if not _is_viable_shared_bin(candidate):
                    continue
                score = 10
                if "gyan" in str(candidate).lower() and "shared" in str(candidate).lower():
                    score += 100
                candidate_key = (score, str(candidate))
                if best is None or candidate_key > best:
                    best = candidate_key

    if best is not None:
        return best[1]

    return None


def ensure_shared_ffmpeg_path() -> None:
    """Add the FFmpeg shared-lib directory to PATH for torchcodec on Windows."""
    shared_bin = discover_shared_ffmpeg_bin()
    if not shared_bin:
        return

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if shared_bin not in path_entries:
        os.environ["PATH"] = shared_bin + os.pathsep + os.environ.get("PATH", "")


ensure_shared_ffmpeg_path()


class ConfigError(ValueError):
    """Raised when a config file contains unknown or invalid keys."""


# Allowed keys.
# A value of None means "any value";
# a dict means "nested section".
#
# Phase 2 adds:
#   - asr
#   - asr_gates
#
# Phase 3 adds:
#   - diar
#   - diar_gates

CONFIG_SCHEMA = {
    "data_dir": None,
    "results_dir": None,
    "log_dir": None,
    "layout_dir": None,
    "keyframes_dir": None,
    "subtitle_dir": None,
    "emotion_dir": None,
    "speaker_mapping_dir": None,
    "tracks_dir": None,
    "output_dir": None,
    "seed": None,

    # ------------------------------------------------------------------
    # Phase 1 - Video Ingestion
    # ------------------------------------------------------------------

    "ingestion": {
        "max_duration_s": None,
        "min_duration_s": None,
        "audio_sample_rate": None,
        "force_cfr": None,
        "vfr_tolerance": None,
        "random_frame_checks": None,
        "black_frame_mean": None,
        "max_bad_frame_fraction": None,
        "x264_crf": None,
        "x264_preset": None,
    },

    "gates": {
        "frame_count_rel_diff": None,
        "audio_video_diff_s": None,
        "decode_errors": None,
    },

    # ------------------------------------------------------------------
    # Phase 2 - Automatic Speech Recognition
    # ------------------------------------------------------------------

    "asr": {
        "backend": None,
        "model_size": None,
        "language": None,
        "device": None,
        "compute_type": None,
        "vad_filter": None,
        "beam_size": None,
        "condition_on_previous_text": None,

        "max_no_speech_prob": None,
        "min_avg_logprob": None,

        "hallucination_phrases": None,

        "max_words": None,
        "max_duration_s": None,
        "split_gap_s": None,

        "min_duration_s": None,
        "merge_gap_s": None,
        "merge_max_words": None,

        "timestamp_match_min_ratio": None,
        "nonspeech_overlap_ratio": None,
    },

    # ------------------------------------------------------------------
    # Phase 2 - ASR Quality Gates
    # ------------------------------------------------------------------

    "asr_gates": {
        "wer_max": None,
        "start_mae_s_max": None,
        "hallucination_rate_max": None,
    },

    # ------------------------------------------------------------------
    # Phase 3 - Speaker Diarization
    # ------------------------------------------------------------------

    "diar": {
        "backend": None,
        "model": None,
        "hf_token_env": None,
        "device": None,
        "min_speakers": None,
        "max_speakers": None,
        "merge_gap_s": None,
        "min_turn_s": None,
        "min_overlap_ratio": None,
        "mixed_second_ratio": None,
        "collar_s": None,
        "der_step_s": None,
        "use_pyannote_metrics": None,
    },

    "diar_gates": {
        "der_max": None,
        "attribution_acc_min": None,
    },

    # ------------------------------------------------------------------
    # Phase 4 - Shot Detection and Frame Pooling
    # ------------------------------------------------------------------

    "phase4": {
        "black_bar": {
            "n_samples": None,
            "thresh": None,
            "min_keep_frac": None,
        },
        "shots": {
            "detector": None,
            "threshold": None,
            "min_scene_len_s": None,
        },
        "sampling": {
            "fps": None,
            "window_pad_s": None,
            "min_per_segment": None,
            "max_candidates_per_segment": None,
            "jpeg_quality": None,
        },
        "quality": {
            "min_brightness": None,
            "blur_rel_thresh": None,
            "phash_hamming": None,
            "border_dark_thresh": None,
        },
        "gate": {
            "coverage": None,
            "max_black_border_rate": None,
            "boundary_f1": None,
            "boundary_tol_s": None,
            "crop_iou": None,
        },
        "contact_sheet": {
            "n_shots": None,
            "cols": None,
            "thumb_w": None,
        },
    },
    "phase5": None,
    "phase6": None,
    "phase7": None,
    "phase8": None,
    "phase9": None,
    "phase10": None,
    "phase11": None,
    "phase12": None,
}


def _validate(
    cfg: dict,
    schema: dict,
    prefix: str = ""
) -> None:
    """Validate configuration keys recursively."""

    for key, value in cfg.items():

        path = f"{prefix}{key}"

        if key not in schema:
            raise ConfigError(
                f"Unknown config key: '{path}'"
            )

        sub = schema[key]

        if isinstance(sub, dict):

            if not isinstance(value, dict):
                raise ConfigError(
                    f"Config key '{path}' must be a mapping"
                )

            _validate(
                value,
                sub,
                prefix=f"{path}."
            )


def merge_dicts(
    base: dict,
    override: dict,
) -> dict:
    """Deep-merge two nested dictionaries."""

    merged = dict(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_dicts(
                merged[key],
                value,
            )
        else:
            merged[key] = value

    return merged


def load_config(
    path: str | Path,
    schema: dict | None = None
) -> dict:
    """Load a YAML config and reject unknown keys."""

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as fh:

        cfg = yaml.safe_load(fh) or {}

    if not isinstance(cfg, dict):
        raise ConfigError(
            "Top level of the config must be a mapping"
        )

    _validate(
        cfg,
        schema or CONFIG_SCHEMA
    )

    return cfg


def load_pipeline_config(
    path: str | Path,
    schema: dict | None = None,
) -> dict:
    """Load a single config or merge all stage configs for one end-to-end run."""

    cfg_path = Path(path)
    config_root = cfg_path.parent
    explicit = load_config(cfg_path, schema)

    if not cfg_path.name.startswith("phase") or not cfg_path.name.endswith(".yaml"):
        return explicit

    merged: dict = {}

    for idx in range(1, 13):
        candidate = config_root / f"phase{idx}.yaml"

        if candidate.exists():
            merged = merge_dicts(
                merged,
                load_config(candidate, schema),
            )

    return merge_dicts(
        merged,
        explicit,
    )


def load_phase3_config(path: str | Path) -> dict:
    """Compatibility helper for phase 3 config files."""
    return load_config(path, CONFIG_SCHEMA)


def config_hash(cfg: dict) -> str:
    """
    Short stable hash of a config.

    Used for caching and in metrics files.
    """

    blob = json.dumps(
        cfg,
        sort_keys=True,
        default=str
    ).encode("utf-8")

    return hashlib.sha1(
        blob
    ).hexdigest()[:8]


def set_seed(seed: int) -> None:
    """
    Seed Python, NumPy and, if installed, PyTorch.
    """

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)

    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)

    except ImportError:
        pass


def get_logger(
    video_id: str,
    log_dir: str | Path = "logs"
) -> logging.Logger:
    """
    Logger that writes to the console
    and to logs/{video_id}.log.
    """

    logger = logging.getLogger(
        f"cinecomic.{video_id}"
    )

    if logger.handlers:
        return logger

    logger.setLevel(
        logging.INFO
    )

    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)-7s | "
        "%(name)s | "
        "%(message)s"
    )

    Path(log_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    file_handler = logging.FileHandler(
        Path(log_dir)
        / f"{video_id}.log",
        encoding="utf-8"
    )

    file_handler.setFormatter(fmt)

    console = logging.StreamHandler()

    console.setFormatter(fmt)

    logger.addHandler(
        file_handler
    )

    logger.addHandler(
        console
    )

    return logger


def close_logger(
    logger: logging.Logger
) -> None:
    """
    Release file handles.

    Needed on Windows and in tests.
    """

    for handler in list(
        logger.handlers
    ):

        handler.close()

        logger.removeHandler(
            handler
        )


def artifact_path(
    cfg: dict,
    kind: str,
    video_id: str,
    ext: str = "json"
) -> Path:
    """
    Return:

        data/{kind}/{video_id}.{ext}

    and create the parent folder.

    Example:

        artifact_path(
            cfg,
            "audio",
            "clip_001",
            "wav"
        )

    returns:

        data/audio/clip_001.wav
    """

    path = (
        Path(cfg["data_dir"])
        / kind
        / f"{video_id}.{ext}"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return path
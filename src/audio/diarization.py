
"""Phase 3 diarization stage."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from src.audio.diar_postprocess import (
    MIXED,
    UNKNOWN,
    attribute_segments,
    relabel_by_first_appearance,
    smooth_turns,
    validate_diarization,
)
from src.audio.rttm_io import read_rttm, write_rttm
from src.common.base_stage import BaseStage
from src.common.utils import artifact_path
from src.evaluation.diarization_metrics import (
    compute_der,
    der_pyannote,
    segment_attribution,
)

RAW_KEYS = ["backend", "model", "device", "min_speakers", "max_speakers"]

_MODEL_CACHE: dict = {}


class DiarError(RuntimeError):
    """Raised when diarization fails or returns invalid output."""


def raw_key(dcfg: dict, audio_duration_s: float) -> dict:
    """Build a configuration key for raw diarization results."""
    return {
        "audio_duration_s": round(audio_duration_s, 3),
        **{key: dcfg.get(key) for key in RAW_KEYS},
    }


def _diarize_pyannote(
    wav: str,
    dcfg: dict,
    log=None,
) -> list[dict]:
    """Run pyannote diarization and return normalized speaker turns."""

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarError(
            "pyannote.audio is not installed. "
            "Install it with: pip install pyannote.audio"
        ) from exc

    # Cache the loaded pipeline so it is not loaded repeatedly.
    key = (
        "pa",
        dcfg["model"],
        dcfg["device"],
    )

    if key not in _MODEL_CACHE:
        token_env = dcfg.get("hf_token_env", "HF_TOKEN")
        token = os.environ.get(token_env)

        if not token:
            raise DiarError(
                f"Environment variable {token_env} is empty. "
                f"Create a Hugging Face token and accept the terms "
                f"for {dcfg['model']} and pyannote/segmentation-3.0, "
                "then set the token in your environment."
            )

        if log:
            log.info(
                "Loading diarization pipeline '%s'",
                dcfg["model"],
            )

        try:
            # pyannote.audio 4.x uses `token`.
            try:
                pipe = Pipeline.from_pretrained(
                    dcfg["model"],
                    token=token,
                )
            except TypeError:
                # Compatibility with older pyannote versions.
                pipe = Pipeline.from_pretrained(
                    dcfg["model"],
                    use_auth_token=token,
                )

        except Exception as exc:
            raise DiarError(
                f"Could not load '{dcfg['model']}': "
                f"{type(exc).__name__}: {str(exc)[:200]}"
            ) from exc

        if pipe is None:
            raise DiarError(
                f"Pipeline '{dcfg['model']}' returned None. "
                "Check your Hugging Face token and model access."
            )

        # Apply the configured device if possible.
        device = dcfg.get("device")
        if device:
            try:
                import torch

                pipe.to(torch.device(device))
            except (AttributeError, TypeError, ValueError, RuntimeError):
                # Some pipeline implementations handle device placement
                # differently. Do not fail solely on this compatibility step.
                if log:
                    log.warning(
                        "Could not explicitly set pipeline device to %s",
                        device,
                    )

        _MODEL_CACHE[key] = pipe

    pipe = _MODEL_CACHE[key]

    # Run inference.
    try:
        try:
            output = pipe(str(wav))
        except TypeError:
            # Compatibility with versions accepting an input dictionary.
            output = pipe({"audio": str(wav)})

    except Exception as exc:
        raise DiarError(
            f"Diarization inference failed: "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        ) from exc

    # Newer pyannote versions can return a DiarizeOutput object.
    # Older versions can return an Annotation directly.
    annotation = getattr(
        output,
        "speaker_diarization",
        None,
    )

    if annotation is None:
        annotation = getattr(
            output,
            "diarization",
            None,
        )

    # Some versions return the Annotation itself.
    if annotation is None and hasattr(output, "itertracks"):
        annotation = output

    # Support dictionary-like outputs where applicable.
    if annotation is None and isinstance(output, dict):
        annotation = output.get("speaker_diarization")

        if annotation is None:
            annotation = output.get("diarization")

    if annotation is None or not hasattr(annotation, "itertracks"):
        available_attributes = [
            name
            for name in dir(output)
            if not name.startswith("_")
        ]

        raise DiarError(
            "Unsupported pyannote output structure. "
            f"Output type: {type(output).__name__}; "
            f"available attributes: {available_attributes[:30]}"
        )

    # Convert pyannote Annotation segments into the project's format.
    turns: list[dict] = []

    try:
        for segment, _, speaker in annotation.itertracks(
            yield_label=True
        ):
            start = float(segment.start)
            end = float(segment.end)

            # Ignore invalid or zero-duration segments.
            if end <= start:
                continue

            turns.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": str(speaker),
                }
            )

    except Exception as exc:
        raise DiarError(
            f"Could not extract speaker turns: "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        ) from exc

    # Keep segments in chronological order.
    turns.sort(
        key=lambda turn: (
            turn["start"],
            turn["end"],
        )
    )

    if log:
        log.info(
            "[diarization] extracted %d raw turns from %s",
            len(turns),
            type(output).__name__,
        )

    return turns


BACKENDS = {
    "pyannote": _diarize_pyannote,
}


def run_diarization(
    dcfg: dict,
    wav_path: str,
    log=None,
    diarize_fn=None,
):
    """Run diarization. Returns (raw_turns, runtime_s)."""

    fn = diarize_fn or BACKENDS.get(dcfg["backend"])

    if fn is None:
        raise DiarError(
            f"Unknown diar.backend '{dcfg['backend']}'. "
            f"Available backends: {list(BACKENDS)}"
        )

    start_time = time.time()

    turns = fn(
        str(wav_path),
        dcfg,
        log,
    )

    runtime_s = time.time() - start_time

    return turns, runtime_s


class DiarStage(BaseStage):
    """Phase 3 speaker diarization stage."""

    name = "diarization"

    requires = [
        "ingestion",
        "asr",
    ]

    config_keys = [
        "diar",
        "diar_gates",
        "asr",
        "seed",
    ]

    def __init__(
        self,
        cfg,
        video_id,
        logger=None,
        diarize_fn=None,
    ):
        super().__init__(
            cfg,
            video_id,
            logger,
        )

        self.diarize_fn = diarize_fn

        self._artifact = artifact_path(
            cfg,
            "diarization",
            video_id,
            "json",
        )

        self.rttm_path = artifact_path(
            cfg,
            "diarization",
            video_id,
            "rttm",
        )

        self.raw_path = artifact_path(
            cfg,
            "diarization",
            video_id,
            "raw.json",
        )

        self.ref_rttm_path = artifact_path(
            cfg,
            "diarization",
            video_id,
            "ref.rttm",
        )

        self.gt_path = artifact_path(
            cfg,
            "diarization",
            video_id,
            "gt.json",
        )

    @property
    def artifact(self) -> Path:
        """Return the main diarization artifact path."""
        return self._artifact

    def _load(
        self,
        kind: str,
        hint: str,
    ) -> dict:
        """Load a JSON artifact for a pipeline stage."""

        path = artifact_path(
            self.cfg,
            kind,
            self.video_id,
            "json",
        )

        if not path.exists():
            raise DiarError(
                f"{path} not found: {hint}"
            )

        return json.loads(
            path.read_text(encoding="utf-8")
        )

    def _raw_turns(
        self,
        meta: dict,
    ) -> tuple[list[dict], float, bool]:
        """Load cached raw turns if their configuration matches."""

        dcfg = self.cfg["diar"]

        key = raw_key(
            dcfg,
            meta.get("audio_duration_s") or 0.0,
        )

        if self.raw_path.exists():
            try:
                saved = json.loads(
                    self.raw_path.read_text(
                        encoding="utf-8"
                    )
                )

                if saved.get("raw_cfg") == key:
                    if self.log:
                        self.log.info(
                            "[diarization] reusing cached raw turns"
                        )

                    return (
                        saved["turns"],
                        saved.get("runtime_s", 0.0),
                        True,
                    )

            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
            ):
                pass

        return [], 0.0, False

    def run(self) -> None:
        """Execute diarization and write JSON and RTTM artifacts."""

        meta_path = artifact_path(
            self.cfg,
            "metadata",
            self.video_id,
            "json",
        )

        if not meta_path.exists():
            raise DiarError(
                f"{meta_path} not found: "
                "run Phase 1 (ingestion) first"
            )

        meta = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )

        wav = meta.get("audio_path")

        if not wav:
            raise DiarError(
                f"{meta_path} has no audio_path; "
                "diarization requires audio"
            )

        dcfg = self.cfg["diar"]

        # Execute the selected backend.
        raw_turns, runtime = run_diarization(
            dcfg,
            wav,
            self.log,
            self.diarize_fn,
        )

        # Apply the existing project's post-processing.
        raw_turns = smooth_turns(
            raw_turns,
            dcfg,
        )

        raw_turns = relabel_by_first_appearance(
            raw_turns,
        )

        payload = {
            "video_id": self.video_id,
            "backend": dcfg["backend"],
            "model": dcfg["model"],
            "audio_duration_s": (
                meta.get("audio_duration_s") or 0.0
            ),
            "runtime_s": round(runtime, 3),
            "raw_cfg": raw_key(
                dcfg,
                meta.get("audio_duration_s") or 0.0,
            ),
            "turns": raw_turns,
            "created": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        # Ensure output directories exist.
        self.raw_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.rttm_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.raw_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        write_rttm(
            raw_turns,
            self.video_id,
            self.rttm_path,
        )

        if self.log:
            self.log.info(
                "[diarization] wrote %d turns",
                len(raw_turns),
            )

    def evaluate(self, gt=None) -> dict:
        """Evaluate diarization and optionally compare with ground truth."""

        if not self.raw_path.exists():
            raise DiarError(
                f"{self.raw_path} not found. "
                "Run the diarization stage first."
            )

        payload = json.loads(
            self.raw_path.read_text(
                encoding="utf-8"
            )
        )

        turns = payload.get("turns", [])

        if not turns:
            return {
                "model": self.cfg["diar"]["model"],
                "n_turns": 0,
                "segments_labelled": 0,
                "der": None,
                "segment_accuracy": None,
                "attribution_accuracy": None,
                "ground_truth": False,
                "problems": [],
            }

        asr_artifact = artifact_path(
            self.cfg,
            "subtitles",
            self.video_id,
            "json",
        )

        if asr_artifact.exists():
            asr_data = json.loads(
                asr_artifact.read_text(
                    encoding="utf-8"
                )
            )
        else:
            asr_data = {
                "segments": [],
            }

        segment_speakers = attribute_segments(
            asr_data.get("segments", []),
            turns,
            self.cfg["diar"],
        )

        problems = validate_diarization(
            turns,
            segment_speakers,
            asr_data.get("segments", []),
        )

        metrics = {
            "model": self.cfg["diar"]["model"],
            "n_turns": len(turns),
            "segments_labelled": len(segment_speakers),
            "speaker_names": sorted(
                {
                    turn["speaker"]
                    for turn in turns
                }
            ),
            "problems": problems,
            "ground_truth": bool(gt),
        }

        if gt:
            ref_turns = (
                gt.get("turns")
                or read_rttm(
                    gt.get(
                        "rttm_path",
                        self.ref_rttm_path,
                    )
                )
            )

            if ref_turns:
                der = compute_der(
                    ref_turns,
                    turns,
                    collar=self.cfg["diar"]["collar_s"],
                    step=self.cfg["diar"]["der_step_s"],
                )

                metrics.update(
                    {
                        "der": der["der"],
                        "der_missed": der["der_missed"],
                        "der_false_alarm": der["der_false_alarm"],
                        "der_confusion": der["der_confusion"],
                    }
                )

                if self.cfg["diar"].get(
                    "use_pyannote_metrics"
                ):
                    metrics["der_pyannote"] = der_pyannote(
                        ref_turns,
                        turns,
                        collar=self.cfg["diar"]["collar_s"],
                    )

            if gt.get("segment_labels"):
                attrib = segment_attribution(
                    segment_speakers,
                    gt["segment_labels"],
                )

                metrics["segment_accuracy"] = (
                    attrib["segment_accuracy"]
                )

                metrics["attribution_accuracy"] = (
                    attrib["accuracy_assigned"]
                )

                metrics["coverage"] = (
                    attrib["coverage"]
                )

        return metrics

    def gate(
        self,
        metrics: dict,
    ) -> tuple[bool, str]:
        """Apply configured diarization quality gates."""

        gates = self.cfg["diar_gates"]
        failures = []

        der = metrics.get("der")

        if der is not None and der > gates["der_max"]:
            failures.append(
                f"DER {der:.3f} > {gates['der_max']}"
            )

        attribution_accuracy = metrics.get(
            "attribution_accuracy"
        )

        if (
            attribution_accuracy is not None
            and attribution_accuracy
            < gates["attribution_acc_min"]
        ):
            failures.append(
                f"attribution accuracy "
                f"{attribution_accuracy:.3f} < "
                f"{gates['attribution_acc_min']}"
            )

        if metrics.get("problems"):
            failures.extend(
                metrics["problems"][:3]
            )

        if failures:
            return False, "; ".join(failures)

        if not metrics.get("ground_truth"):
            return (
                True,
                "structure OK; no ground truth file "
                "(DER/attribution not evaluated)",
            )

        return True, "all checks passed"

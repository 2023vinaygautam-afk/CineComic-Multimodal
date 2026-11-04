"""Phase 2: speech recognition with Whisper."""

from __future__ import annotations

import inspect
import json
import time
from datetime import datetime
from pathlib import Path


try:
    import av
except Exception:  # pragma: no cover - optional dependency
    av = None
else:
    _av_open = av.open
    try:
        inspect.signature(_av_open)
    except (TypeError, ValueError):
        def _compat_av_open(*args, **kwargs):
            kwargs.pop("metadata_errors", None)
            return _av_open(*args, **kwargs)

        av.open = _compat_av_open

from src.audio.postprocess import (
    postprocess,
    validate_timeline,
)
from src.audio.srt_io import write_srt
from src.common.base_stage import BaseStage
from src.common.utils import artifact_path
from src.evaluation.asr_metrics import evaluate_against_gt


class AsrError(RuntimeError):
    pass


_MODEL_CACHE: dict = {}


def _transcribe_faster_whisper(
    wav: str,
    acfg: dict,
    log=None
):

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise AsrError(
            "faster-whisper is not installed "
            "(pip install faster-whisper)"
        ) from exc

    key = (
        "fw",
        acfg["model_size"],
        acfg["device"],
        acfg["compute_type"],
    )

    if key not in _MODEL_CACHE:

        if log:
            log.info(
                "loading faster-whisper model '%s'",
                acfg["model_size"]
            )

        try:
            _MODEL_CACHE[key] = WhisperModel(
                acfg["model_size"],
                device=acfg["device"],
                compute_type=acfg["compute_type"],
            )
        except Exception as exc:
            raise AsrError(
                f"could not load faster-whisper model "
                f"'{acfg['model_size']}': "
                f"{type(exc).__name__}: "
                f"{str(exc)[:200]}. "
                "Check your internet connection "
                "(first run downloads the model) "
                "or device settings."
            ) from exc

    model = _MODEL_CACHE[key]

    segments, info = model.transcribe(
        wav,
        language=acfg["language"],
        beam_size=acfg["beam_size"],
        word_timestamps=True,
        vad_filter=acfg["vad_filter"],
        condition_on_previous_text=(
            acfg["condition_on_previous_text"]
        ),
    )

    raw = []

    for seg in segments:

        raw.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text,
            "avg_logprob": float(
                seg.avg_logprob
            ),
            "no_speech_prob": float(
                seg.no_speech_prob
            ),
            "words": [
                {
                    "w": w.word,
                    "start": float(w.start),
                    "end": float(w.end),
                    "p": float(w.probability),
                }
                for w in (seg.words or [])
            ],
        })

    return raw, {
        "language": info.language,
        "language_probability": (
            info.language_probability
        ),
    }


def _transcribe_openai_whisper(
    wav: str,
    acfg: dict,
    log=None
):

    try:
        import whisper
    except ImportError as exc:
        raise AsrError(
            "openai-whisper is not installed "
            "(pip install openai-whisper)"
        ) from exc

    device = (
        None
        if acfg["device"] == "auto"
        else acfg["device"]
    )

    key = (
        "ow",
        acfg["model_size"],
        device,
    )

    if key not in _MODEL_CACHE:

        if log:
            log.info(
                "loading openai-whisper model '%s'",
                acfg["model_size"]
            )

        try:
            _MODEL_CACHE[key] = whisper.load_model(
                acfg["model_size"],
                device=device,
            )
        except Exception as exc:
            raise AsrError(
                f"could not load openai-whisper model "
                f"'{acfg['model_size']}': "
                f"{type(exc).__name__}: "
                f"{str(exc)[:200]}"
            ) from exc

    model = _MODEL_CACHE[key]

    if acfg["vad_filter"] and log:
        log.warning(
            "vad_filter is only supported by "
            "faster-whisper; ignored"
        )

    result = model.transcribe(
        wav,
        language=acfg["language"],
        word_timestamps=True,
        condition_on_previous_text=(
            acfg["condition_on_previous_text"]
        ),
        beam_size=acfg["beam_size"],
        fp16=(model.device.type == "cuda"),
    )

    raw = [
        {
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": s["text"],
            "avg_logprob": float(
                s["avg_logprob"]
            ),
            "no_speech_prob": float(
                s["no_speech_prob"]
            ),
            "words": [
                {
                    "w": w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "p": float(
                        w.get("probability", 1.0)
                    ),
                }
                for w in s.get("words", [])
            ],
        }
        for s in result["segments"]
    ]

    return raw, {
        "language": result.get("language")
    }


BACKENDS = {
    "faster-whisper": _transcribe_faster_whisper,
    "openai-whisper": _transcribe_openai_whisper,
}


def run_asr(
    acfg: dict,
    wav_path: str,
    audio_duration: float,
    log=None,
    transcribe_fn=None,
):

    fn = (
        transcribe_fn
        or BACKENDS.get(acfg["backend"])
    )

    if fn is None:
        raise AsrError(
            f"Unknown asr.backend "
            f"'{acfg['backend']}' "
            f"(use {list(BACKENDS)})"
        )

    t0 = time.time()

    raw, info = fn(
        str(wav_path),
        acfg,
        log
    )

    runtime = time.time() - t0

    segments, dropped = postprocess(
        raw,
        acfg,
        audio_duration
    )

    return (
        segments,
        dropped,
        info,
        runtime,
    )


class AsrStage(BaseStage):

    name = "asr"

    requires = ["ingestion"]

    config_keys = [
        "asr",
        "asr_gates",
        "seed",
    ]

    def __init__(
        self,
        cfg,
        video_id,
        logger=None,
        transcribe_fn=None,
    ):
        super().__init__(
            cfg,
            video_id,
            logger
        )

        self.transcribe_fn = transcribe_fn

        self._artifact = artifact_path(
            cfg,
            "subtitles",
            video_id,
            "json"
        )

        self.srt_path = artifact_path(
            cfg,
            "subtitles",
            video_id,
            "srt"
        )

        self.gt_path = artifact_path(
            cfg,
            "subtitles",
            video_id,
            "gt.json"
        )

    @property
    def artifact(self) -> Path:
        return self._artifact

    def run(self) -> None:

        meta_path = artifact_path(
            self.cfg,
            "metadata",
            self.video_id,
            "json"
        )

        if not meta_path.exists():
            raise AsrError(
                f"{meta_path} not found: "
                "run Phase 1 (ingestion) first"
            )

        meta = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )

        acfg = self.cfg["asr"]

        if not meta["has_audio"]:

            if self.log:
                self.log.info(
                    "no audio track: "
                    "writing an empty transcript"
                )

            segments = []
            dropped = []
            info = {"language": None}
            runtime = 0.0

        else:

            (
                segments,
                dropped,
                info,
                runtime,
            ) = run_asr(
                acfg,
                meta["audio_path"],
                meta["audio_duration_s"],
                self.log,
                self.transcribe_fn,
            )

        audio_dur = (
            meta.get("audio_duration_s")
            or 0.0
        )

        payload = {
            "video_id": self.video_id,
            "model": acfg["model_size"],
            "backend": acfg["backend"],
            "language": info.get("language"),
            "audio_duration_s": audio_dur,
            "runtime_s": round(runtime, 3),
            "rtf": (
                round(
                    runtime / audio_dur,
                    4
                )
                if audio_dur
                else None
            ),
            "n_dropped": len(dropped),
            "segments": segments,
            "dropped": dropped,
            "created": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        self.artifact.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        write_srt(
            segments,
            self.srt_path
        )

    def evaluate(self, gt=None) -> dict:

        data = json.loads(
            self.artifact.read_text(
                encoding="utf-8"
            )
        )

        segments = data["segments"]

        metrics = {
            "model": data["model"],
            "n_segments": len(segments),
            "n_dropped": data["n_dropped"],
            "rtf": data["rtf"],
            "timeline_problems": (
                validate_timeline(segments)
            ),
            "mean_words_per_segment": round(
                sum(
                    len(s["words"])
                    for s in segments
                )
                / len(segments),
                2
            )
            if segments
            else 0,
        }

        if (
            gt is None
            and self.gt_path.exists()
        ):
            gt = json.loads(
                self.gt_path.read_text(
                    encoding="utf-8"
                )
            )

        metrics["ground_truth"] = bool(gt)

        if gt:
            metrics.update(
                evaluate_against_gt(
                    segments,
                    gt,
                    self.cfg["asr"]
                )
            )

        return metrics

    def gate(
        self,
        metrics: dict
    ) -> tuple[bool, str]:

        g = self.cfg["asr_gates"]

        fails = list(
            metrics["timeline_problems"][:3]
        )

        if (
            metrics.get("wer") is not None
            and metrics["wer"] > g["wer_max"]
        ):
            fails.append(
                f"WER {metrics['wer']:.3f} "
                f"> {g['wer_max']}"
            )

        mae = metrics.get(
            "start_mae_s"
        )

        if (
            mae is not None
            and mae > g["start_mae_s_max"]
        ):
            fails.append(
                f"start-time MAE {mae}s "
                f"> {g['start_mae_s_max']}s"
            )

        hr = metrics.get(
            "hallucination_rate"
        )

        if (
            hr is not None
            and hr > g["hallucination_rate_max"]
        ):
            fails.append(
                f"hallucination rate {hr} "
                f"> {g['hallucination_rate_max']}"
            )

        if fails:
            return False, "; ".join(fails)

        if not metrics["ground_truth"]:
            return (
                True,
                "structure OK; no ground truth file "
                "(WER/MAE/hallucination not evaluated)",
            )

        return True, "all checks passed"
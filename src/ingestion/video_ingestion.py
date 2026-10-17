"""Phase 1 stage: ingest a video and produce raw mp4,
metadata JSON and 16 kHz WAV."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

from src.common.base_stage import BaseStage
from src.common.utils import artifact_path
from src.ingestion import metadata as md
from src.ingestion import video_loader as vl


class VideoIngestionStage(BaseStage):

    name = "ingestion"

    requires: list[str] = []


    def __init__(
        self,
        cfg,
        video_id,
        source_path: Path,
        sha256: str,
        source_label: str,
        logger=None
    ):

        super().__init__(
            cfg,
            video_id,
            logger
        )

        self.source_path = Path(source_path)
        self.sha256 = sha256
        self.source_label = source_label

        self.raw_path = artifact_path(
            cfg,
            "raw",
            video_id,
            "mp4"
        )

        self.wav_path = artifact_path(
            cfg,
            "audio",
            video_id,
            "wav"
        )

        self._artifact = artifact_path(
            cfg,
            "metadata",
            video_id,
            "json"
        )


    @property
    def artifact(self) -> Path:
        return self._artifact


    # ---- processing ---------------------------------------------------------

    def run(self) -> None:

        icfg = self.cfg["ingestion"]

        vl.require_tools()

        src_info = md.describe(
            self.source_path,
            icfg["vfr_tolerance"]
        )

        if src_info["duration_s"] < icfg["min_duration_s"]:

            raise md.IngestionError(
                f"Video too short "
                f"({src_info['duration_s']:.2f}s "
                f"< {icfg['min_duration_s']}s)"
            )

        src_errors = md.decode_errors(
            self.source_path
        )

        redo, reasons = vl.needs_normalising(
            self.source_path,
            src_info,
            icfg
        )

        if redo:

            self._info(
                "re-encoding source: %s",
                ", ".join(reasons)
            )

            vl.normalise_video(
                self.source_path,
                self.raw_path,
                src_info,
                icfg
            )

        else:

            vl.copy_video(
                self.source_path,
                self.raw_path
            )

        raw = md.describe(
            self.raw_path,
            icfg["vfr_tolerance"]
        )

        audio_path, audio_dur = None, None

        if raw["has_audio"]:

            vl.extract_audio(
                self.raw_path,
                self.wav_path,
                icfg["audio_sample_rate"]
            )

            audio_path = self.wav_path.as_posix()

            audio_dur = md.wav_duration(
                self.wav_path
            )

        else:

            self._info(
                "no audio stream: skipping audio extraction"
            )

            if self.wav_path.exists():
                self.wav_path.unlink()

        meta = {
            "video_id": self.video_id,
            "source": self.source_label,
            "sha256": self.sha256,
            "raw_path": self.raw_path.as_posix(),
            "duration_s": round(
                raw["duration_s"],
                3
            ),
            "fps": round(
                raw["fps"],
                5
            ),
            "fps_fraction": raw["fps_fraction"],
            "width": raw["width"],
            "height": raw["height"],
            "n_frames": raw["n_frames"],
            "video_codec": raw["video_codec"],
            "has_audio": raw["has_audio"],
            "audio_path": audio_path,
            "audio_duration_s": (
                round(audio_dur, 3)
                if audio_dur is not None
                else None
            ),
            "audio_sample_rate": (
                icfg["audio_sample_rate"]
                if audio_path
                else None
            ),
            "vfr": raw["vfr"],
            "normalised": redo,
            "normalise_reasons": reasons,
            "source_vfr": src_info["vfr"],
            "source_rotation": src_info["rotation"],
            "source_decode_errors": len(src_errors),
            "time_convention":
                "seconds (float) from start of CFR video; "
                "frame = round(t*fps)",
            "created":
                datetime.now().isoformat(
                    timespec="seconds"
                ),
        }

        self.artifact.write_text(
            json.dumps(
                meta,
                indent=2
            ),
            encoding="utf-8"
        )


    # ---- evaluation ---------------------------------------------------------

    def evaluate(self, gt=None) -> dict:

        icfg = self.cfg["ingestion"]

        meta = json.loads(
            self.artifact.read_text(
                encoding="utf-8"
            )
        )

        cv_frames = md.opencv_frame_count(
            self.raw_path
        )

        ref = max(
            meta["n_frames"],
            1
        )

        frame_diff = (
            abs(
                cv_frames
                - meta["n_frames"]
            )
            / ref
        )

        av_diff = None

        if meta["has_audio"]:

            av_diff = abs(
                meta["audio_duration_s"]
                - meta["duration_s"]
            )

        errs = max(
            meta["source_decode_errors"],
            len(
                md.decode_errors(
                    self.raw_path
                )
            )
        )

        frames = md.random_frame_check(
            self.raw_path,
            meta["duration_s"],
            icfg["random_frame_checks"],
            self.cfg["seed"],
            icfg["black_frame_mean"]
        )

        return {
            "opencv_frames": cv_frames,
            "ffprobe_frames": meta["n_frames"],
            "frame_count_rel_diff":
                round(frame_diff, 5),
            "audio_video_diff_s":
                (
                    None
                    if av_diff is None
                    else round(av_diff, 4)
                ),
            "decode_errors": errs,
            "random_frame_check": frames,
            "has_audio": meta["has_audio"],
        }


    def gate(
        self,
        metrics: dict
    ) -> tuple[bool, str]:

        g = self.cfg["gates"]
        icfg = self.cfg["ingestion"]

        fails = []

        if (
            metrics["frame_count_rel_diff"]
            > g["frame_count_rel_diff"]
        ):

            fails.append(
                f"frame count diff "
                f"{metrics['frame_count_rel_diff']:.4f} "
                f"> {g['frame_count_rel_diff']}"
            )

        diff = metrics["audio_video_diff_s"]

        if (
            diff is not None
            and diff > g["audio_video_diff_s"]
        ):

            fails.append(
                f"audio/video length diff "
                f"{diff}s > "
                f"{g['audio_video_diff_s']}s"
            )

        if (
            metrics["decode_errors"]
            > g["decode_errors"]
        ):

            fails.append(
                f"{metrics['decode_errors']} "
                f"decode error line(s)"
            )

        chk = metrics["random_frame_check"]

        if (
            chk["bad"]
            > icfg["max_bad_frame_fraction"]
            * chk["checked"]
        ):

            fails.append(
                f"{chk['bad']}/{chk['checked']} "
                "random frames black or unreadable"
            )

        return (
            not fails,
            "all checks passed"
            if not fails
            else "; ".join(fails)
        )


    def _info(
        self,
        msg,
        *args
    ):

        if self.log:
            self.log.info(
                msg,
                *args
            )


def ingest(
    cfg: dict,
    source: str,
    video_id: str | None = None,
    logger_factory=None,
    force: bool = False
) -> tuple[bool, str]:

    """Convenience entry point used by main.py."""

    from src.common.utils import get_logger

    vl.require_tools()

    with tempfile.TemporaryDirectory(
        prefix="cinecomic_dl_"
    ) as tmp:

        local = vl.acquire_source(
            source,
            tmp
        )

        sha = vl.sha256_file(local)

        vid = vl.make_video_id(
            sha,
            video_id
        )

        log = (
            logger_factory or get_logger
        )(
            vid,
            cfg["log_dir"]
        )

        log.info(
            "ingesting %s as %s",
            source,
            vid
        )

        stage = VideoIngestionStage(
            cfg,
            vid,
            local,
            sha,
            source,
            log
        )

        ok = stage.execute(
            force=force
        )

        return ok, vid
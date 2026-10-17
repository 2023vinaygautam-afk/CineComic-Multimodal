"""CineComic-MML entry point."""

from __future__ import annotations

import argparse
import sys

from src.common.utils import (
    get_logger,
    load_config,
    set_seed,
)
from src.ingestion.metadata import (
    IngestionError,
)
from src.ingestion.video_ingestion import (
    ingest,
)


STAGES = [
    "ingestion",
    "asr",
    "diarization",
    "phase4",
]


def main(argv=None) -> int:

    ap = argparse.ArgumentParser(
        description="CineComic-MML pipeline"
    )

    ap.add_argument(
        "--config",
        required=True,
        help="YAML config"
    )

    ap.add_argument(
        "--video",
        default=None,
        help="video file path or URL"
    )

    ap.add_argument(
        "--video-id",
        default=None,
        help="video ID"
    )

    ap.add_argument(
        "--stages",
        nargs="+",
        default=["ingestion"],
        choices=STAGES,
        help="stages to run, in order",
    )

    ap.add_argument(
        "--force",
        action="store_true",
        help="ignore cache"
    )

    args = ap.parse_args(argv)

    cfg = load_config(
        args.config
    )

    set_seed(
        cfg["seed"]
    )

    video_id = args.video_id

    if "ingestion" in args.stages:

        if not args.video:
            ap.error(
                "--video is required "
                "for the ingestion stage"
            )

        try:

            ok, video_id = ingest(
                cfg,
                args.video,
                video_id,
                force=args.force,
            )

        except IngestionError as exc:

            print(
                f"INGESTION FAILED: {exc}",
                file=sys.stderr,
            )

            return 2

        print(
            f"[ingestion] "
            f"video_id={video_id} "
            f"gate={'PASS' if ok else 'FAIL'}"
        )

        if not ok:
            return 1

    elif not video_id:

        ap.error(
            "--video-id is required "
            "when ingestion is not in --stages"
        )

    if "asr" in args.stages:

        from src.audio.whisper_asr import (
            AsrError,
            AsrStage,
        )

        log = get_logger(
            video_id,
            cfg["log_dir"]
        )

        try:

            ok = AsrStage(
                cfg,
                video_id,
                log
            ).execute(
                force=args.force
            )

        except AsrError as exc:

            print(
                f"ASR FAILED: {exc}",
                file=sys.stderr
            )

            return 2

        print(
            f"[asr] "
            f"video_id={video_id} "
            f"gate={'PASS' if ok else 'FAIL'}"
        )

        if not ok:
            return 1

    if "diarization" in args.stages:

        from src.audio.diarization import (
            DiarError,
            DiarStage,
        )

        log = get_logger(
            video_id,
            cfg["log_dir"]
        )

        try:

            ok = DiarStage(
                cfg,
                video_id,
                log
            ).execute(
                force=args.force
            )

        except DiarError as exc:

            print(
                f"DIARIZATION FAILED: {exc}",
                file=sys.stderr
            )

            return 2

        print(
            f"[diarization] "
            f"video_id={video_id} "
            f"gate={'PASS' if ok else 'FAIL'}"
        )

        if not ok:
            return 1

    if "phase4" in args.stages:

        from src.video.phase4 import Phase4Stage

        log = get_logger(
            video_id,
            cfg["log_dir"]
        )

        ok = Phase4Stage(
            cfg,
            video_id,
        ).execute(
            force=args.force
        )

        print(
            f"[phase4] "
            f"video_id={video_id} "
            f"gate={'PASS' if ok else 'FAIL'}"
        )

        if not ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
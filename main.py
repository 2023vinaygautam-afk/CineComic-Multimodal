"""CineComic-MML entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from src.common.utils import (
    get_logger,
    load_pipeline_config,
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
    "phase5",
    "phase6",
    "phase7",
    "phase8",
    "phase9",
    "phase10",
    "phase11",
    "phase12",
]


def _stage_config(base_config: str | Path, stage_name: str) -> Path:
    p = Path(base_config)
    stage_yaml = p.parent / f"{stage_name}.yaml"
    if stage_yaml.exists():
        return stage_yaml
    return p


def _normalize_stages(stages):
    """Expand the special 'all' stage while preserving pipeline order."""
    if not stages:
        return ["ingestion"]

    if "all" in stages:
        return list(STAGES)

    ordered = []
    for name in STAGES:
        if name in stages:
            ordered.append(name)
    return ordered


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
        choices=STAGES + ["all"],
        help="stages to run, in order; use 'all' for the full pipeline",
    )

    ap.add_argument(
        "--force",
        action="store_true",
        help="ignore cache"
    )

    args = ap.parse_args(argv)
    stages = _normalize_stages(args.stages)

    cfg = load_pipeline_config(
        args.config
    )

    set_seed(
        cfg["seed"]
    )

    video_id = args.video_id

    if "ingestion" in stages:

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

    if "asr" in stages:

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

    if "diarization" in stages:

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

    if "phase4" in stages:

        from src.video.phase4 import Phase4Stage

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

    if "phase5" in stages:

        from src.vision.phase5 import run_phase5

        cfg["force"] = args.force
        summary = run_phase5(
            video_id=video_id,
            config=cfg,
            video_path=Path(args.video) if args.video else None,
        )

        print(
            f"[phase5] "
            f"video_id={video_id} "
            f"tracks={summary.get('tracks', 0)}"
        )

    if "phase6" in stages:

        from src.vision.phase6 import run_phase6

        result = run_phase6(
            project_root=Path(__file__).resolve().parent,
            video_id=video_id,
            config_path=_stage_config(args.config, "phase6"),
        )

        print(
            f"[phase6] "
            f"video_id={video_id} "
            f"segments={result.get('num_segments', 0)}"
        )

    if "phase7" in stages:

        from src.speaker.phase7 import run_phase7

        result = run_phase7(
            config_path=_stage_config(args.config, "phase7"),
            video_id=video_id,
        )

        print(
            f"[phase7] "
            f"video_id={video_id} "
            f"match_rate={result.get('match_rate', 0.0)}"
        )

    if "phase8" in stages:

        from src.keyframes.phase8 import run_phase8

        result = run_phase8(
            str(_stage_config(args.config, "phase8")),
            video_id,
        )

        num_kf = result.get('selected_panels') or result.get('num_keyframes', 0)
        print(
            f"[phase8] "
            f"video_id={video_id} "
            f"keyframes={num_kf}"
        )

    if "phase9" in stages:

        from src.emotion.phase9 import Phase9Stage

        ok = Phase9Stage(cfg, video_id).execute(force=args.force)

        print(
            f"[phase9] "
            f"video_id={video_id} "
            f"gate={'PASS' if ok else 'FAIL'}"
        )

        if not ok:
            return 1

    if "phase10" in stages:

        from src.layout.phase10 import run_phase10

        result = run_phase10(
            video_id,
            _stage_config(args.config, "phase10"),
        )

        p_count = result.get('metrics', {}).get('panel_count', result.get('num_panels', 0))
        print(
            f"[phase10] "
            f"video_id={video_id} "
            f"panels={p_count}"
        )

    if "phase11" in stages:

        from src.comic.phase11 import run_phase11

        result = run_phase11(
            video_id,
            _stage_config(args.config, "phase11"),
        )

        p_styled = result.get('metrics', {}).get('successful_panels', result.get('num_panels', 0))
        print(
            f"[phase11] "
            f"video_id={video_id} "
            f"panels={p_styled}"
        )

    if "phase12" in stages:

        from src.comic.phase12 import run_phase12

        result = run_phase12(
            video_id,
            _stage_config(args.config, "phase12"),
        )

        pg_count = result.get('metrics', {}).get('page_count', result.get('num_pages', 0))
        print(
            f"[phase12] "
            f"video_id={video_id} "
            f"pages={pg_count}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
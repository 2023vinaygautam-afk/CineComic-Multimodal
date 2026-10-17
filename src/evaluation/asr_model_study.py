"""Model-size study: WER and real-time factor."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from src.audio.whisper_asr import run_asr
from src.common.utils import (
    artifact_path,
    load_config,
    set_seed,
)
from src.evaluation.asr_metrics import (
    evaluate_against_gt,
)


def main(argv=None) -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True
    )

    ap.add_argument(
        "--video-id",
        required=True
    )

    ap.add_argument(
        "--sizes",
        nargs="+",
        default=[
            "tiny",
            "base",
            "small",
            "medium",
        ],
    )

    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    set_seed(cfg["seed"])

    meta = json.loads(
        artifact_path(
            cfg,
            "metadata",
            args.video_id
        ).read_text(
            encoding="utf-8"
        )
    )

    gt_path = artifact_path(
        cfg,
        "subtitles",
        args.video_id,
        "gt.json"
    )

    gt = json.loads(
        gt_path.read_text(
            encoding="utf-8"
        )
    )

    rows = []

    for size in args.sizes:

        acfg = copy.deepcopy(
            cfg["asr"]
        )

        acfg["model_size"] = size

        (
            segs,
            dropped,
            _,
            runtime,
        ) = run_asr(
            acfg,
            meta["audio_path"],
            meta["audio_duration_s"]
        )

        m = evaluate_against_gt(
            segs,
            gt,
            acfg
        )

        rows.append({
            "model": size,
            "wer": m.get("wer"),
            "cer": m.get("cer"),
            "start_mae_s": m.get(
                "start_mae_s"
            ),
            "hallucination_rate": m.get(
                "hallucination_rate"
            ),
            "rtf": round(
                runtime
                / meta["audio_duration_s"],
                4
            ),
            "segments": len(segs),
            "dropped": len(dropped),
        })

        print(rows[-1])

    out = (
        Path(cfg["results_dir"])
        / "asr_model_study.csv"
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        out,
        "w",
        newline="",
        encoding="utf-8"
    ) as fh:

        w = csv.DictWriter(
            fh,
            fieldnames=list(rows[0])
        )

        w.writeheader()
        w.writerows(rows)

    print("wrote", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
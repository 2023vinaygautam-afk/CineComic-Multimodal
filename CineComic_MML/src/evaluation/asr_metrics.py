"""Phase 2 evaluation: WER, CER, start-time error, hallucination rate."""

from __future__ import annotations

import difflib
import re


def normalise_text(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""

    text = (
        text.lower()
        .replace("'", "")
        .replace("’", "")
    )

    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9 ]", " ", text)
    ).strip()


def wer_cer(
    reference: str,
    hypothesis: str
) -> dict:

    try:
        import jiwer
    except ImportError as exc:
        raise RuntimeError(
            "jiwer is not installed "
            "(pip install jiwer)"
        ) from exc

    ref = normalise_text(reference)
    hyp = normalise_text(hypothesis)

    if not ref:
        raise ValueError(
            "reference transcript is empty "
            "after normalisation"
        )

    if not hyp:
        return {
            "wer": 1.0,
            "cer": 1.0,
            "ref_words": len(ref.split()),
        }

    return {
        "wer": round(
            jiwer.wer(ref, hyp),
            4
        ),
        "cer": round(
            jiwer.cer(ref, hyp),
            4
        ),
        "ref_words": len(ref.split()),
    }


def start_time_mae(
    segments: list[dict],
    marks: list[dict],
    min_ratio: float = 0.5
) -> dict:

    errors = []
    unmatched = 0

    norm_segs = [
        normalise_text(s["text"])
        for s in segments
    ]

    for mark in marks:

        target = normalise_text(
            mark["text"]
        )

        best = None
        best_ratio = 0.0

        for seg, ns in zip(
            segments,
            norm_segs
        ):
            ratio = difflib.SequenceMatcher(
                None,
                target,
                ns
            ).ratio()

            if ratio > best_ratio:
                best = seg
                best_ratio = ratio

        if (
            best is None
            or best_ratio < min_ratio
        ):
            unmatched += 1
        else:
            errors.append(
                abs(
                    best["start"]
                    - mark["start"]
                )
            )

    mae = (
        round(
            sum(errors) / len(errors),
            4
        )
        if errors
        else None
    )

    return {
        "start_mae_s": mae,
        "marks_matched": len(errors),
        "marks_unmatched": unmatched,
    }


def hallucination_rate(
    segments: list[dict],
    non_speech: list[list[float]],
    min_overlap: float = 0.5
) -> dict:

    if not segments:
        return {
            "hallucination_rate": 0.0,
            "hallucinated": 0,
        }

    bad = 0

    for s in segments:

        length = max(
            s["end"] - s["start"],
            1e-6
        )

        overlap = sum(
            max(
                0.0,
                min(s["end"], b)
                - max(s["start"], a)
            )
            for a, b in non_speech
        )

        if overlap / length >= min_overlap:
            bad += 1

    return {
        "hallucination_rate": round(
            bad / len(segments),
            4
        ),
        "hallucinated": bad,
    }


def evaluate_against_gt(
    segments: list[dict],
    gt: dict,
    acfg: dict
) -> dict:

    metrics = {}

    if gt.get("reference_text"):
        hyp = " ".join(
            s["text"]
            for s in segments
        )

        metrics.update(
            wer_cer(
                gt["reference_text"],
                hyp
            )
        )

    if gt.get("timestamp_marks"):
        metrics.update(
            start_time_mae(
                segments,
                gt["timestamp_marks"],
                acfg["timestamp_match_min_ratio"],
            )
        )

    if "non_speech" in gt:
        metrics.update(
            hallucination_rate(
                segments,
                gt["non_speech"],
                acfg["nonspeech_overlap_ratio"],
            )
        )

    return metrics
    
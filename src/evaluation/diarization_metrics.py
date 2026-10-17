"""Phase 3 evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


SPECIAL = ("mixed", "unknown")


def _activity(
    turns: list[dict],
    names: list[str],
    n: int,
    step: float
) -> np.ndarray:

    idx = {
        s: i
        for i, s in enumerate(names)
    }

    mat = np.zeros(
        (len(names), n),
        dtype=bool
    )

    for t in turns:

        a = int(
            round(t["start"] / step)
        )

        b = int(
            round(t["end"] / step)
        )

        mat[
            idx[t["speaker"]],
            max(a, 0):max(b, 0)
        ] = True

    return mat


def compute_der(
    ref: list[dict],
    hyp: list[dict],
    collar: float = 0.25,
    step: float = 0.01
) -> dict:

    if not ref:
        raise ValueError(
            "reference has no speech turns"
        )

    end = max(
        t["end"]
        for t in ref + hyp
    )

    n = int(
        math.ceil(end / step)
    ) + 1

    ref_names = sorted(
        {t["speaker"] for t in ref}
    )

    hyp_names = sorted(
        {t["speaker"] for t in hyp}
    )

    R = _activity(
        ref,
        ref_names,
        n,
        step
    )

    H = _activity(
        hyp,
        hyp_names,
        n,
        step
    )

    keep = np.ones(
        n,
        dtype=bool
    )

    if collar > 0:

        half = collar / 2.0

        for t in ref:

            for b in (
                t["start"],
                t["end"]
            ):

                keep[
                    max(
                        int(
                            round(
                                (b - half) / step
                            )
                        ),
                        0
                    ):
                    max(
                        int(
                            round(
                                (b + half) / step
                            )
                        ),
                        0
                    )
                ] = False

    R = R[:, keep]
    H = H[:, keep]

    n_ref = R.sum(axis=0)
    n_hyp = H.sum(axis=0)

    total = float(
        n_ref.sum()
    )

    if total == 0:
        raise ValueError(
            "no reference speech left after applying the collar"
        )

    co = (
        R.astype(np.int64)
        @ H.astype(np.int64).T
    )

    correct = 0
    mapping = {}

    if co.size:

        rows, cols = linear_sum_assignment(
            -co
        )

        for r, c in zip(rows, cols):

            correct += int(
                co[r, c]
            )

            mapping[
                hyp_names[c]
            ] = ref_names[r]

    missed = float(
        np.maximum(
            n_ref - n_hyp,
            0
        ).sum()
    )

    false_alarm = float(
        np.maximum(
            n_hyp - n_ref,
            0
        ).sum()
    )

    confusion = (
        float(
            np.minimum(
                n_ref,
                n_hyp
            ).sum()
        )
        - correct
    )

    return {
        "der": round(
            (
                missed
                + false_alarm
                + confusion
            ) / total,
            4
        ),
        "der_missed": round(
            missed / total,
            4
        ),
        "der_false_alarm": round(
            false_alarm / total,
            4
        ),
        "der_confusion": round(
            confusion / total,
            4
        ),
        "ref_speech_s": round(
            total * step,
            2
        ),
        "speaker_mapping": mapping,
    }


def der_pyannote(
    ref: list[dict],
    hyp: list[dict],
    collar: float = 0.25
):

    try:

        from pyannote.core import (
            Annotation,
            Segment
        )

        from pyannote.metrics.diarization import (
            DiarizationErrorRate
        )

    except ImportError:

        return None

    def to_ann(turns):

        ann = Annotation()

        for t in turns:

            ann[
                Segment(
                    t["start"],
                    t["end"]
                )
            ] = t["speaker"]

        return ann

    return round(
        float(
            DiarizationErrorRate(
                collar=collar,
                skip_overlap=False
            )(
                to_ann(ref),
                to_ann(hyp)
            )
        ),
        4
    )


def segment_attribution(
    segment_speakers: list[dict],
    ref_labels: dict
) -> dict:

    pairs = [
        (
            s["speaker"],
            ref_labels[
                str(s["segment_id"])
            ]
        )
        for s in segment_speakers
        if str(s["segment_id"])
        in ref_labels
    ]

    if not pairs:

        return {
            "segment_accuracy": None,
            "accuracy_assigned": None,
            "coverage": None,
            "segments_labelled": 0,
        }

    assigned = [
        (h, r)
        for h, r in pairs
        if h not in SPECIAL
    ]

    hyp_names = sorted(
        {h for h, _ in assigned}
    )

    ref_names = sorted(
        {r for _, r in pairs}
    )

    correct = 0

    if hyp_names:

        co = np.zeros(
            (
                len(ref_names),
                len(hyp_names)
            ),
            dtype=np.int64
        )

        for h, r in assigned:

            co[
                ref_names.index(r),
                hyp_names.index(h)
            ] += 1

        rows, cols = linear_sum_assignment(
            -co
        )

        correct = int(
            co[
                rows,
                cols
            ].sum()
        )

    return {
        "segment_accuracy": round(
            correct / len(pairs),
            4
        ),
        "accuracy_assigned": (
            round(
                correct / len(assigned),
                4
            )
            if assigned
            else None
        ),
        "coverage": round(
            len(assigned) / len(pairs),
            4
        ),
        "segments_labelled": len(pairs),
    }
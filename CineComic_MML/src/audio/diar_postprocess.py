"""Phase 3 diarization post-processing."""

from __future__ import annotations

MIXED = "mixed"
UNKNOWN = "unknown"
SPECIAL = (MIXED, UNKNOWN)


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Return overlap duration between two intervals."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def smooth_turns(turns: list[dict], dcfg: dict) -> list[dict]:
    """Merge nearby same-speaker turns and remove very short turns."""

    by_spk: dict[str, list[dict]] = {}

    for t in turns:
        if t["end"] > t["start"]:
            by_spk.setdefault(t["speaker"], []).append(dict(t))

    merged: list[dict] = []

    for spk, items in by_spk.items():

        items.sort(key=lambda t: (t["start"], t["end"]))

        cur = items[0]

        for t in items[1:]:

            if t["start"] - cur["end"] < dcfg["merge_gap_s"]:
                cur["end"] = max(cur["end"], t["end"])
            else:
                merged.append(cur)
                cur = t

        merged.append(cur)

    kept = [
        t for t in merged
        if t["end"] - t["start"] >= dcfg["min_turn_s"]
    ]

    kept.sort(
        key=lambda t: (
            t["start"],
            t["end"],
            t["speaker"]
        )
    )

    return [
        {
            "start": round(t["start"], 3),
            "end": round(t["end"], 3),
            "speaker": t["speaker"],
        }
        for t in kept
    ]


def relabel_by_first_appearance(turns: list[dict]) -> list[dict]:
    """Rename speakers according to first appearance."""

    mapping: dict[str, str] = {}
    out = []

    for t in sorted(
        turns,
        key=lambda t: (
            t["start"],
            t["end"],
            t["speaker"]
        )
    ):

        if t["speaker"] not in mapping:
            mapping[t["speaker"]] = (
                f"SPEAKER_{len(mapping):02d}"
            )

        out.append(
            dict(
                t,
                speaker=mapping[t["speaker"]]
            )
        )

    return out


def attribute_segments(
    segments: list[dict],
    turns: list[dict],
    dcfg: dict
) -> list[dict]:
    """
    Assign one speaker to every Whisper segment.

    Rules:

    - no overlapping speaker -> unknown
    - second speaker >= mixed_second_ratio -> mixed
    - best speaker < min_overlap_ratio -> unknown
    - otherwise -> best speaker
    """

    out = []

    for seg in segments:

        dur = max(
            seg["end"] - seg["start"],
            1e-6
        )

        per: dict[str, float] = {}

        for t in turns:

            ov = overlap(
                seg["start"],
                seg["end"],
                t["start"],
                t["end"]
            )

            if ov > 0:
                per[t["speaker"]] = (
                    per.get(t["speaker"], 0.0) + ov
                )

        ranked = sorted(
            per.items(),
            key=lambda kv: (-kv[1], kv[0])
        )

        best, best_ov = (
            ranked[0]
            if ranked
            else (None, 0.0)
        )

        second_ratio = (
            ranked[1][1] / dur
            if len(ranked) > 1
            else 0.0
        )

        ratio = min(
            best_ov / dur,
            1.0
        )

        if best is None:

            label = UNKNOWN
            reason = "no_speaker_turn"

        elif second_ratio >= dcfg["mixed_second_ratio"]:

            label = MIXED
            reason = "overlapping_speakers"

        elif ratio < dcfg["min_overlap_ratio"]:

            label = UNKNOWN
            reason = "low_overlap"

        else:

            label = best
            reason = "ok"

        out.append(
            {
                "segment_id": seg["segment_id"],
                "speaker": label,
                "overlap_ratio": round(ratio, 4),
                "second_ratio": round(
                    min(second_ratio, 1.0),
                    4
                ),
                "best_speaker": best,
                "reason": reason,
            }
        )

    return out


def validate_diarization(
    turns: list[dict],
    segment_speakers: list[dict],
    segments: list[dict]
) -> list[str]:

    problems = []

    for i, t in enumerate(turns):

        if t["end"] <= t["start"]:
            problems.append(
                f"turn {i}: non-positive duration"
            )

    if len(segment_speakers) != len(segments):

        problems.append(
            f"{len(segment_speakers)} attributions "
            f"for {len(segments)} segments"
        )

    known = {
        t["speaker"]
        for t in turns
    }

    ids = [
        s["segment_id"]
        for s in segments
    ]

    for ss, sid in zip(
        segment_speakers,
        ids
    ):

        if ss["segment_id"] != sid:

            problems.append(
                f"segment {sid}: attribution out of order"
            )

        if (
            ss["speaker"] not in known
            and ss["speaker"] not in SPECIAL
        ):

            problems.append(
                f"segment {sid}: speaker "
                f"'{ss['speaker']}' not in turns"
            )

        if not 0.0 <= ss["overlap_ratio"] <= 1.0:

            problems.append(
                f"segment {sid}: overlap_ratio "
                f"outside [0,1]"
            )

    return problems
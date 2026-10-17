"""Turn raw Whisper output into clean, balloon-sized, time-ordered segments.

Pure functions (no model, no files) so they are easy to unit-test.
"""

from __future__ import annotations

import re


STRONG = (".", "!", "?", "…")
SOFT = (",", ";", ":", "-", "–", "—")
_TRAIL = "\"')]}"


def _norm(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9 ]", " ", text.lower())
    ).strip()


def _ends_with(word: dict, marks: tuple) -> bool:
    return word["w"].rstrip(_TRAIL).endswith(marks)


def _dur(seg: dict) -> float:
    return seg["end"] - seg["start"]


# -------------------------------------------------------------------------
# 1. Filter
# -------------------------------------------------------------------------

def filter_segments(
    raw: list[dict],
    acfg: dict
) -> tuple[list[dict], list[dict]]:
    """Split raw segments into (kept, dropped)."""

    phrases = [_norm(p) for p in acfg["hallucination_phrases"]]

    kept = []
    dropped = []

    for seg in raw:
        text = seg["text"].strip()
        norm = _norm(text)

        reason = None

        if not norm:
            reason = "empty"

        elif (
            seg.get("no_speech_prob", 0.0) > acfg["max_no_speech_prob"]
            and seg.get("avg_logprob", 0.0) < acfg["min_avg_logprob"]
        ):
            reason = "no_speech_low_confidence"

        else:
            for p in phrases:
                if (
                    p
                    and p in norm
                    and len(norm.split()) <= len(p.split()) + 3
                ):
                    reason = f"hallucination_phrase:{p}"
                    break

        if reason:
            dropped.append({
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": text,
                "reason": reason,
            })
        else:
            kept.append(seg)

    return kept, dropped


# -------------------------------------------------------------------------
# 2. Words
# -------------------------------------------------------------------------

def ensure_words(seg: dict) -> dict:
    """Estimate word timestamps when backend gives none."""

    if seg.get("words"):
        return seg

    tokens = seg["text"].split()

    step = _dur(seg) / max(len(tokens), 1)

    seg = dict(seg)

    seg["words"] = [
        {
            "w": t,
            "start": seg["start"] + i * step,
            "end": seg["start"] + (i + 1) * step,
            "p": None,
        }
        for i, t in enumerate(tokens)
    ]

    seg["words_estimated"] = True

    return seg


# -------------------------------------------------------------------------
# 3. Split
# -------------------------------------------------------------------------

def _sentence_split(
    words: list[dict],
    gap: float
) -> list[list[dict]]:

    chunks = []
    cur = []

    for i, w in enumerate(words):
        cur.append(w)

        last = i == len(words) - 1

        pause = (
            0.0
            if last
            else words[i + 1]["start"] - w["end"]
        )

        if (
            last
            or _ends_with(w, STRONG)
            or pause >= gap
        ):
            chunks.append(cur)
            cur = []

    return chunks


def _too_big(chunk: list[dict], acfg: dict) -> bool:
    return (
        len(chunk) > acfg["max_words"]
        or
        (chunk[-1]["end"] - chunk[0]["start"])
        > acfg["max_duration_s"]
    )


def _split_balanced(
    chunk: list[dict],
    acfg: dict
) -> list[list[dict]]:
    """Recursively cut oversized chunk near middle."""

    n = len(chunk)

    if n < 2 or not _too_big(chunk, acfg):
        return [chunk]

    lo = 2 if n >= 4 else 1
    hi = n - lo

    mid = n / 2

    soft = [
        i
        for i in range(lo, hi + 1)
        if _ends_with(chunk[i - 1], SOFT)
    ]

    if soft:
        cut = min(
            soft,
            key=lambda i: abs(i - mid)
        )
    else:
        lo_g = max(lo, n // 3)
        hi_g = min(hi, 2 * n // 3)

        rng = (
            range(lo_g, hi_g + 1)
            if lo_g <= hi_g
            else range(lo, hi + 1)
        )

        cut = max(
            rng,
            key=lambda i: (
                chunk[i]["start"] - chunk[i - 1]["end"],
                -abs(i - mid),
            ),
        )

    return (
        _split_balanced(chunk[:cut], acfg)
        +
        _split_balanced(chunk[cut:], acfg)
    )


def _make_segment(
    words: list[dict],
    parent: dict
) -> dict:

    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": " ".join(w["w"] for w in words),
        "words": words,
        "avg_logprob": parent.get("avg_logprob"),
        "no_speech_prob": parent.get("no_speech_prob"),
    }


def split_segment(
    seg: dict,
    acfg: dict
) -> list[dict]:

    seg = ensure_words(seg)

    words = [
        dict(w, w=w["w"].strip())
        for w in seg["words"]
        if w["w"].strip()
    ]

    if not words:
        return []

    out = []

    for sentence in _sentence_split(
        words,
        acfg["split_gap_s"]
    ):
        for piece in _split_balanced(
            sentence,
            acfg
        ):
            out.append(
                _make_segment(piece, seg)
            )

    return out


# -------------------------------------------------------------------------
# 4. Merge
# -------------------------------------------------------------------------

def _merge_two(a: dict, b: dict) -> dict:

    na = len(a["words"])
    nb = len(b["words"])

    def wavg(key):
        va = a.get(key)
        vb = b.get(key)

        if va is None or vb is None:
            return va if vb is None else vb

        return (
            (va * na + vb * nb)
            / (na + nb)
        )

    words = a["words"] + b["words"]

    return {
        "start": a["start"],
        "end": b["end"],
        "text": " ".join(w["w"] for w in words),
        "words": words,
        "avg_logprob": wavg("avg_logprob"),
        "no_speech_prob": wavg("no_speech_prob"),
    }


def merge_short(
    segs: list[dict],
    acfg: dict
) -> list[dict]:

    segs = list(segs)

    changed = True

    while changed:
        changed = False

        for i, s in enumerate(segs):

            if _dur(s) >= acfg["min_duration_s"]:
                continue

            options = []

            if i > 0:
                p = segs[i - 1]

                gap = s["start"] - p["end"]

                if (
                    gap <= acfg["merge_gap_s"]
                    and
                    len(p["words"]) + len(s["words"])
                    <= acfg["merge_max_words"]
                ):
                    options.append((gap, "prev"))

            if i + 1 < len(segs):
                n = segs[i + 1]

                gap = n["start"] - s["end"]

                if (
                    gap <= acfg["merge_gap_s"]
                    and
                    len(n["words"]) + len(s["words"])
                    <= acfg["merge_max_words"]
                ):
                    options.append((gap, "next"))

            if not options:
                continue

            side = min(options)[1]

            if side == "prev":
                segs[i - 1:i + 1] = [
                    _merge_two(segs[i - 1], s)
                ]
            else:
                segs[i:i + 2] = [
                    _merge_two(s, segs[i + 1])
                ]

            changed = True
            break

    return segs


# -------------------------------------------------------------------------
# 5. Timeline
# -------------------------------------------------------------------------

MIN_SEG_S = 0.05


def enforce_timeline(
    segs: list[dict],
    audio_duration: float | None = None
) -> list[dict]:

    segs = sorted(
        (dict(s) for s in segs),
        key=lambda s: (s["start"], s["end"])
    )

    out = []

    for s in segs:

        if audio_duration is not None:
            s["start"] = min(
                max(s["start"], 0.0),
                audio_duration
            )

            s["end"] = min(
                s["end"],
                audio_duration
            )

        if out and s["start"] < out[-1]["end"]:

            prev = out[-1]

            if s["start"] - prev["start"] >= MIN_SEG_S:
                prev["end"] = s["start"]
            else:
                s["start"] = prev["end"]

        if s["end"] - s["start"] < MIN_SEG_S:
            continue

        out.append(s)

    return out


# -------------------------------------------------------------------------
# 6. Full pipeline
# -------------------------------------------------------------------------

def postprocess(
    raw: list[dict],
    acfg: dict,
    audio_duration: float | None = None
) -> tuple[list[dict], list[dict]]:

    kept, dropped = filter_segments(
        raw,
        acfg
    )

    pieces = []

    for seg in sorted(
        kept,
        key=lambda s: s["start"]
    ):
        pieces.extend(
            split_segment(seg, acfg)
        )

    merged = merge_short(
        pieces,
        acfg
    )

    final = enforce_timeline(
        merged,
        audio_duration
    )

    result = []

    for i, s in enumerate(final, 1):

        result.append({
            "segment_id": i,
            "start": round(s["start"], 3),
            "end": round(s["end"], 3),
            "text": s["text"].strip(),
            "words": [
                {
                    "w": w["w"],
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                }
                for w in s["words"]
            ],
            "avg_logprob": (
                None
                if s["avg_logprob"] is None
                else round(s["avg_logprob"], 3)
            ),
            "no_speech_prob": (
                None
                if s["no_speech_prob"] is None
                else round(s["no_speech_prob"], 3)
            ),
        })

    return result, dropped


def validate_timeline(
    segs: list[dict]
) -> list[str]:

    problems = []

    for i, s in enumerate(segs):

        if s["segment_id"] != i + 1:
            problems.append(
                f"segment_id {s['segment_id']} "
                f"out of sequence at position {i + 1}"
            )

        if s["end"] <= s["start"]:
            problems.append(
                f"segment {s['segment_id']}: end <= start"
            )

        if i and s["start"] < segs[i - 1]["start"]:
            problems.append(
                f"segment {s['segment_id']}: "
                "not sorted by start"
            )

        if (
            i
            and s["start"]
            < segs[i - 1]["end"] - 1e-3
        ):
            problems.append(
                f"segment {s['segment_id']}: "
                "overlaps the previous segment"
            )

        if not s["text"].strip():
            problems.append(
                f"segment {s['segment_id']}: empty text"
            )

    return problems
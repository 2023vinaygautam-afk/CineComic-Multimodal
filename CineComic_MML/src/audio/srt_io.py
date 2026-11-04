"""Read and write SRT subtitle files."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import srt


def write_srt(
    segments: list[dict],
    path: str | Path
) -> None:

    subs = [
        srt.Subtitle(
            index=s["segment_id"],
            start=timedelta(seconds=s["start"]),
            end=timedelta(seconds=s["end"]),
            content=s["text"],
        )
        for s in segments
    ]

    Path(path).write_text(
        srt.compose(subs),
        encoding="utf-8"
    )


def read_srt(
    path: str | Path
) -> list[dict]:

    text = Path(path).read_text(
        encoding="utf-8"
    )

    return [
        {
            "segment_id": sub.index,
            "start": sub.start.total_seconds(),
            "end": sub.end.total_seconds(),
            "text": sub.content,
        }
        for sub in srt.parse(text)
    ]
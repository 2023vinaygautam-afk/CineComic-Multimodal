"""Read and write RTTM diarization files."""

from __future__ import annotations

from pathlib import Path


def write_rttm(
    turns: list[dict],
    video_id: str,
    path: str | Path
) -> None:

    lines = [
        (
            f"SPEAKER {video_id} 1 "
            f"{t['start']:.3f} "
            f"{t['end'] - t['start']:.3f} "
            f"<NA> <NA> "
            f"{t['speaker']} "
            f"<NA> <NA>"
        )
        for t in turns
    ]

    Path(path).write_text(
        "\n".join(lines)
        + ("\n" if lines else ""),
        encoding="utf-8"
    )


def read_rttm(
    path: str | Path
) -> list[dict]:

    turns = []

    for line in Path(path).read_text(
        encoding="utf-8"
    ).splitlines():

        parts = line.split()

        if len(parts) < 8:
            continue

        if parts[0] != "SPEAKER":
            continue

        start = float(parts[3])
        dur = float(parts[4])

        turns.append(
            {
                "start": start,
                "end": start + dur,
                "speaker": parts[7],
            }
        )

    return turns
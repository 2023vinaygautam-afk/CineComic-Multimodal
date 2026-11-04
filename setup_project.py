"""Create the CineComic_MML folder skeleton (cross-platform replacement for the
PowerShell mkdir / New-Item commands in the guide).

Usage: python setup_project.py
"""

from pathlib import Path


DATA = [
    "raw",
    "metadata",
    "audio",
    "shots",
    "frames",
    "subtitles",
    "diarization",
    "faces",
    "tracks",
    "embeddings",
    "alignment",
    "speaker",
    "keyframes",
    "emotions",
    "layout",
    "outputs",
]

SRC = [
    "common",
    "ingestion",
    "audio",
    "video",
    "vision",
    "speaker",
    "alignment",
    "keyframes",
    "emotion",
    "layout",
    "comic",
    "evaluation",
]

TOP = [
    "configs",
    "tests",
    "results",
    "logs",
    "notebooks",
    "models",
]


def main():
    root = Path(__file__).resolve().parent

    dirs = [root / "data" / d for d in DATA]
    dirs += [root / "src" / s for s in SRC]
    dirs += [root / t for t in TOP]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    (root / "src" / "__init__.py").touch()

    for s in SRC:
        (root / "src" / s / "__init__.py").touch()

    print(f"Created {len(dirs)} folders under {root}")


if __name__ == "__main__":
    main()
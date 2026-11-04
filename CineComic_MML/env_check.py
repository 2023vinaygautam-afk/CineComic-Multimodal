"""Phase 0 environment check:
imports, versions, CUDA, ffmpeg
-> results/environment_report.json.
"""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REQUIRED = [
    "yaml",
    "numpy",
    "cv2",
    "pytest"
]

OPTIONAL = [
    "torch",
    "whisper",
    "yt_dlp",
    "scenedetect",
    "insightface",
    "open_clip"
]


def version_of(name: str):

    try:

        mod = importlib.import_module(name)

        return getattr(
            mod,
            "__version__",
            "ok"
        )

    except Exception as exc:

        return (
            f"MISSING "
            f"({type(exc).__name__})"
        )


def tool_version(tool: str):

    if shutil.which(tool) is None:
        return None

    out = subprocess.run(
        [tool, "-version"],
        capture_output=True,
        text=True
    )

    return (
        out.stdout.splitlines()[0]
        if out.returncode == 0
        and out.stdout
        else None
    )


def main() -> int:

    report = {
        "python": platform.python_version(),
        "platform": platform.platform()
    }

    failed = []

    for name in REQUIRED:

        report[name] = version_of(name)

        if str(
            report[name]
        ).startswith("MISSING"):

            failed.append(name)

    for name in OPTIONAL:

        report[name] = version_of(name)

    for tool in (
        "ffmpeg",
        "ffprobe"
    ):

        report[tool] = tool_version(tool)

        if report[tool] is None:
            failed.append(tool)

    try:

        import torch

        report["cuda"] = bool(
            torch.cuda.is_available()
        )

    except Exception:
        report["cuda"] = False

    report["timestamp"] = (
        datetime.now()
        .isoformat(
            timespec="seconds"
        )
    )

    out = (
        Path("results")
        / "environment_report.json"
    )

    out.parent.mkdir(
        exist_ok=True
    )

    out.write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        json.dumps(
            report,
            indent=2
        )
    )

    if failed:

        print(
            "FAILED:",
            ", ".join(failed),
            file=sys.stderr
        )

        return 1

    print("Environment OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Per-frame quality statistics and relative blur / duplicate / dark filters."""

from __future__ import annotations

from collections import defaultdict

import cv2
import imagehash
import numpy as np
from PIL import Image


def frame_stats(crop_bgr: np.ndarray, border_thresh: float = 16.0) -> dict:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    ph = imagehash.phash(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))
    return {
        "sharp": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "phash": str(ph),
        "border_dark": has_black_bars(gray, border_thresh),
    }


def has_black_bars(gray: np.ndarray, thresh: float = 16.0) -> bool:
    return bool(gray[0].mean() < thresh or gray[-1].mean() < thresh
                or gray[:, 0].mean() < thresh or gray[:, -1].mean() < thresh)


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def apply_filters(frames: list[dict], q: dict) -> None:
    """Set rec['reject'] to None | 'dark' | 'blurry' | 'duplicate' in place."""
    for f in frames:
        f["reject"] = "dark" if f["brightness"] < q["min_brightness"] else None

    live = [f for f in frames if f["reject"] is None]
    if not live:
        return

    median = float(np.median([f["sharp"] for f in live]))
    by_shot = defaultdict(list)
    for f in live:
        by_shot[f["shot_id"]].append(f)

    for fs in by_shot.values():
        sharpest = max(fs, key=lambda f: f["sharp"])
        for f in fs:
            if f is not sharpest and f["sharp"] < q["blur_rel_thresh"] * median:
                f["reject"] = "blurry"

    for fs in by_shot.values():
        kept: list[dict] = []
        for f in sorted((x for x in fs if x["reject"] is None), key=lambda x: (-x["sharp"], x["t"])):
            if any(hamming_hex(f["phash"], k["phash"]) <= q["phash_hamming"] for k in kept):
                f["reject"] = "duplicate"
            else:
                kept.append(f)

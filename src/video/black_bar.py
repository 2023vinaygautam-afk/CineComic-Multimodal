"""Black-bar (letterbox/pillarbox) detection."""

from __future__ import annotations

import cv2
import numpy as np


def crop_from_gray_frames(grays: list[np.ndarray], thresh: float, min_keep_frac: float) -> list[int]:
    """Return the crop box [x0, y0, x1, y1] in original pixels."""
    if not grays:
        raise ValueError("grays must not be empty")

    H, W = grays[0].shape
    full = [0, 0, W, H]

    row_max = np.max([g.mean(axis=1) for g in grays], axis=0)
    rows = np.where(row_max > thresh)[0]
    if len(rows) == 0:
        return full

    y0, y1 = int(rows[0]), int(rows[-1]) + 1

    col_max = np.max([g[y0:y1].mean(axis=0) for g in grays], axis=0)
    cols = np.where(col_max > thresh)[0]
    if len(cols) == 0:
        return full

    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    if (x1 - x0) * (y1 - y0) < min_keep_frac * W * H:
        return full

    return [x0, y0, x1, y1]


def detect_black_bars(video_path, n_samples=30, thresh=16.0, min_keep_frac=0.5) -> list[int]:
    """Detect black bars across a video and return the crop box."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    idxs = (np.linspace(0, n - 1, n_samples + 2)[1:-1] if n > n_samples else np.arange(n)).astype(int)
    grays: list[np.ndarray] = []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    cap.release()

    if not grays:
        return [0, 0, W, H]

    return crop_from_gray_frames(grays, thresh, min_keep_frac)

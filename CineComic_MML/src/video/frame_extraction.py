"""Sampling plan and the single sequential decode pass that saves cropped frames."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.video.quality_filter import frame_stats

log = logging.getLogger("phase4")


def add_index_ranges(shots: list[dict], fps: float, n_frames: int) -> list[dict]:
    """Give every shot an inclusive frame range [i0, i1]; shots tile the video with no overlap."""
    for k, s in enumerate(shots):
        s["i0"] = min(int(round(s["start"] * fps)), n_frames - 1)
        nxt = int(round(shots[k + 1]["start"] * fps)) if k + 1 < len(shots) else n_frames
        s["i1"] = max(s["i0"], min(nxt - 1, n_frames - 1))
        s["start"], s["end"] = s["i0"] / fps, (s["i1"] + 1) / fps
    return shots


def assign_shot(seg: dict, shots: list[dict]) -> dict:
    """Shot with the largest time overlap with the segment (nearest shot if no overlap)."""
    ov = [max(0.0, min(seg["end"], s["end"]) - max(seg["start"], s["start"])) for s in shots]
    if max(ov) > 0:
        return shots[int(np.argmax(ov))]
    mid = (seg["start"] + seg["end"]) / 2
    return min(shots, key=lambda s: 0 if s["start"] <= mid < s["end"] else min(abs(mid - s["start"]), abs(mid - s["end"])))


def segment_window(seg: dict, shot: dict, pad: float) -> tuple[float, float]:
    lo, hi = max(shot["start"], seg["start"] - pad), min(shot["end"], seg["end"] + pad)
    if hi < lo:
        lo = hi = min(max((seg["start"] + seg["end"]) / 2, shot["start"]), shot["end"])
    return lo, hi


def build_sampling_plan(shots, segments, fps, cfg) -> dict[int, tuple[int, str]]:
    """Returns {frame_idx: (shot_id, frame_id)}. frame_ids are numbered by time inside each shot."""
    step = max(1, int(round(fps / cfg["fps"])))
    per_shot = {}

    for s in shots:
        i0, i1 = s["i0"], s["i1"]
        per_shot[s["shot_id"]] = set(range(i0, i1 + 1, step)) | {i0, (i0 + i1) // 2, i1}

    for seg in segments:
        s = assign_shot(seg, shots)
        lo_t, hi_t = segment_window(seg, s, cfg["window_pad_s"])
        lo = min(max(int(round(lo_t * fps)), s["i0"]), s["i1"])
        hi = min(max(int(round(hi_t * fps)), lo), s["i1"])
        have = [i for i in per_shot[s["shot_id"]] if lo <= i <= hi]
        if len(have) < cfg["min_per_segment"]:
            extra = np.linspace(lo, hi, cfg["min_per_segment"]).round().astype(int)
            per_shot[s["shot_id"]] |= {int(e) for e in extra}

    plan = {}
    for sid, idxs in per_shot.items():
        for n, i in enumerate(sorted(idxs)):
            plan[i] = (sid, f"s{sid:02d}_f{n:03d}")
    return plan


def extract_frames(video_path, plan, crop_box, frames_dir: Path, fps: float,
                   jpeg_quality=90, border_thresh=16.0) -> list[dict]:
    """One sequential pass (exact frames, no seeking). Saves cropped JPEGs and computes quality stats."""
    x0, y0, x1, y1 = crop_box
    cap = cv2.VideoCapture(str(video_path))
    last, idx, records = max(plan), 0, []

    while idx <= last:
        if not cap.grab():
            log.warning("decode ended early at frame %d (expected up to %d)", idx, last)
            break
        if idx in plan:
            ok, frame = cap.retrieve()
            if ok:
                sid, fid = plan[idx]
                crop = frame[y0:y1, x0:x1]
                path = frames_dir / f"shot_{sid:03d}" / f"{fid}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                rec = {"frame_id": fid, "shot_id": sid, "idx": idx, "t": idx / fps, "path": path.as_posix()}
                rec.update(frame_stats(crop, border_thresh))
                records.append(rec)
        idx += 1

    cap.release()
    return records

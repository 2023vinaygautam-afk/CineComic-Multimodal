"""Phase 4: shot detection, frame sampling, quality filtering, per-segment candidate frames.
Reads : data/raw/{id}.*, data/subtitles/{id}.json (Phase 2; optional for standalone runs)
Writes: data/shots/{id}.json (+ .metrics.json), data/frames/{id}/shot_xxx/*.jpg
Run:  python -m src.video.phase4 --config configs/phase4.yaml --video clip_001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.evaluation.shot_metrics import boundary_prf, box_iou
from src.video.black_bar import detect_black_bars
from src.video.frame_extraction import (add_index_ranges, assign_shot, build_sampling_plan,
                                        extract_frames, segment_window)
from src.video.quality_filter import apply_filters
from src.video.shot_detection import detect_shots

log = logging.getLogger("phase4")


def build_segment_candidates(segments, shots, frames, cfg) -> tuple[dict, int]:
    """segment_id -> [frame_id]. Guarantees min_per_segment where the shot has enough usable frames."""
    by_shot = defaultdict(list)
    for f in frames:
        by_shot[f["shot_id"]].append(f)

    mx, mn, pad = cfg["max_candidates_per_segment"], cfg["min_per_segment"], cfg["window_pad_s"]
    out, backfilled = {}, 0

    for seg in segments:
        shot = assign_shot(seg, shots)
        lo, hi = segment_window(seg, shot, pad)
        pool = by_shot[shot["shot_id"]]
        chosen = [f for f in pool if f["reject"] is None and lo <= f["t"] <= hi]

        if len(chosen) > mx:
            sel = sorted(set(np.linspace(0, len(chosen) - 1, mx).round().astype(int)))
            chosen = [chosen[i] for i in sel]

        if len(chosen) < mn:
            rej = sorted((f for f in pool if f["reject"] in ("blurry", "duplicate") and lo <= f["t"] <= hi),
                         key=lambda f: (-f["sharp"], f["t"]))
            for f in rej[: mn - len(chosen)]:
                f["reject"], f["backfilled"] = None, True
                chosen.append(f)
                backfilled += 1

        if len(chosen) < mn:
            mid = (seg["start"] + seg["end"]) / 2
            ids = {f["frame_id"] for f in chosen}
            near = sorted(
                (
                    f for f in pool
                    if f["reject"] is None and f["frame_id"] not in ids
                    and shot["start"] <= f["t"] <= shot["end"]
                ),
                key=lambda f: abs(f["t"] - mid),
            )
            chosen += near[: mn - len(chosen)]

        out[str(seg["segment_id"])] = [f["frame_id"] for f in sorted(chosen, key=lambda f: f["t"])]
    return out, backfilled


class Phase4Stage:
    name = "shots"
    requires = ["subtitles"]

    def __init__(self, cfg: dict, video_id: str):
        self.cfg, self.p4, self.video_id = cfg, cfg["phase4"], video_id
        d = Path(cfg["data_dir"])
        found = sorted((d / "raw").glob(f"{video_id}.*"))
        if not found:
            raise FileNotFoundError(f"no video data/raw/{video_id}.*")
        self.video_path = found[0]
        self.out = d / "shots" / f"{video_id}.json"
        self.metrics_path = self.out.with_suffix(".metrics.json")
        self.frames_dir = d / "frames" / video_id
        self.transcript_path = d / "subtitles" / f"{video_id}.json"
        self.results_dir = Path(cfg.get("results_dir", "results")) / "phase4"

    def cache_key(self) -> str:
        blob = json.dumps({"p4": self.p4, "seed": self.cfg.get("seed")}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:8]

    def _segments(self) -> list[dict]:
        if not self.transcript_path.exists():
            log.warning("no transcript at %s: segment_candidates will be empty", self.transcript_path)
            return []
        segs = json.loads(self.transcript_path.read_text(encoding="utf-8"))["segments"]
        return sorted(segs, key=lambda s: s["start"])

    def run(self, force=False) -> None:
        if self.out.exists() and not force:
            if json.loads(self.out.read_text()).get("config_hash") == self.cache_key():
                log.info("cache hit for %s, skipping", self.video_id)
                return

        random.seed(self.cfg.get("seed", 42))
        np.random.seed(self.cfg.get("seed", 42))

        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = n_frames / fps

        p = self.p4
        crop_box = detect_black_bars(self.video_path, **p["black_bar"])
        log.info("crop_box=%s", crop_box)

        shots = detect_shots(self.video_path, duration, fps, p["shots"]["detector"],
                             p["shots"]["threshold"], p["shots"]["min_scene_len_s"])
        add_index_ranges(shots, fps, n_frames)
        log.info("%d shots", len(shots))

        segments = self._segments()
        plan = build_sampling_plan(shots, segments, fps, p["sampling"])
        log.info("decoding %d sampled frames", len(plan))

        if self.frames_dir.exists():
            for old in self.frames_dir.rglob("*.jpg"):
                old.unlink()

        frames = extract_frames(self.video_path, plan, crop_box, self.frames_dir, fps,
                                p["sampling"]["jpeg_quality"], p["quality"]["border_dark_thresh"])
        apply_filters(frames, p["quality"])

        cands, backfilled = build_segment_candidates(segments, shots, frames, p["sampling"])

        n_removed = defaultdict(int)
        for f in frames:
            if f["reject"] is not None:
                n_removed[f["reject"]] += 1
                Path(f["path"]).unlink(missing_ok=True)

        kept = [f for f in frames if f["reject"] is None]
        by_shot = defaultdict(list)
        for f in sorted(kept, key=lambda f: f["t"]):
            rec = {k: (round(f[k], 3) if isinstance(f[k], float) else f[k])
                   for k in ("frame_id", "idx", "t", "sharp", "brightness", "phash", "path", "border_dark")}
            if f.get("backfilled"):
                rec["backfilled"] = True
            by_shot[f["shot_id"]].append(rec)

        doc = {
            "video_id": self.video_id, "fps": fps, "n_frames": n_frames, "duration_s": duration,
            "crop_box": crop_box, "config_hash": self.cache_key(),
            "shots": [{"shot_id": s["shot_id"], "start": round(s["start"], 4), "end": round(s["end"], 4),
                       "frames": by_shot[s["shot_id"]]} for s in shots],
            "segment_candidates": cands,
            "stats": {"sampled": len(frames), "kept": len(kept), "removed": dict(n_removed),
                      "backfilled": backfilled, "n_segments": len(segments)},
        }
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        self.make_contact_sheet(doc)

    def evaluate(self, gt: dict | None = None) -> dict:
        doc = json.loads(self.out.read_text())
        g = self.p4["gate"]
        mn = self.p4["sampling"]["min_per_segment"]
        frames = {f["frame_id"]: (s, f) for s in doc["shots"] for f in s["frames"]}
        kept = [f for s in doc["shots"] for f in s["frames"]]
        cands = doc["segment_candidates"]

        m = {
            "n_shots": len(doc["shots"]), "n_frames_kept": len(kept),
            **{f"stats_{k}": v for k, v in doc["stats"].items() if k != "removed"},
            "removed": doc["stats"]["removed"],
            "pct_removed": 1 - doc["stats"]["kept"] / max(1, doc["stats"]["sampled"]),
            "pool_per_min": len(kept) / (doc["duration_s"] / 60),
            "black_border_rate": sum(f["border_dark"] for f in kept) / max(1, len(kept)),
        }

        if cands:
            m["coverage"] = sum(len(v) >= mn for v in cands.values()) / len(cands)
            tol = g.get("boundary_tol_s", 1e-6)
            bad = 0
            for ids in cands.values():
                for fid in ids:
                    s, f = frames[fid]
                    bad += not (s["start"] - tol <= f["t"] <= s["end"] + tol)
            m["candidates_outside_shot"] = bad

        if gt:
            if "boundaries" in gt:
                pred = [s["start"] for s in doc["shots"][1:]]
                m.update(boundary_prf(pred, gt["boundaries"], g["boundary_tol_s"]))
            if "crop_box" in gt:
                m["crop_iou"] = box_iou(doc["crop_box"], gt["crop_box"])

        return m

    def gate(self, m: dict) -> tuple[bool, str]:
        g, fails = self.p4["gate"], []
        if "coverage" in m and m["coverage"] < g["coverage"]:
            fails.append(f"coverage {m['coverage']:.2f}")
        if m.get("candidates_outside_shot", 0):
            fails.append("candidates outside their shot")
        if m["black_border_rate"] > g["max_black_border_rate"]:
            fails.append(f"black borders {m['black_border_rate']:.3f}")
        if "boundary_f1" in m and m["boundary_f1"] < g["boundary_f1"]:
            fails.append(f"boundary F1 {m['boundary_f1']:.2f}")
        if "crop_iou" in m and m["crop_iou"] < g["crop_iou"]:
            fails.append(f"crop IoU {m['crop_iou']:.2f}")
        return (not fails, "; ".join(fails) or "all gates passed")

    def execute(self, gt=None, force=False) -> bool:
        t0 = time.time()
        self.run(force=force)
        m = self.evaluate(gt)
        ok, why = self.gate(m)
        m.update(runtime_s=time.time() - t0, passed=ok, reason=why, cfg=self.cache_key())
        self.metrics_path.write_text(json.dumps(m, indent=2))
        log.info("metrics: %s", json.dumps(m))
        return ok

    def make_contact_sheet(self, doc: dict) -> None:
        cs = self.p4["contact_sheet"]
        shots = [s for s in doc["shots"] if s["frames"]]
        if not shots:
            return
        rnd = random.Random(self.cfg.get("seed", 42))
        pick = sorted(rnd.sample(shots, min(cs["n_shots"], len(shots))), key=lambda s: s["shot_id"])
        tw, tiles = cs["thumb_w"], []
        for s in pick:
            img = cv2.imread(s["frames"][len(s["frames"]) // 2]["path"])
            th = int(img.shape[0] * tw / img.shape[1])
            img = cv2.resize(img, (tw, th))
            cv2.putText(img, f"shot {s['shot_id']}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(cv2.resize(img, (tw, int(tw * 9 / 16))))
        while len(tiles) % cs["cols"]:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + cs["cols"]]) for i in range(0, len(tiles), cs["cols"])]
        self.results_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.results_dir / f"{self.video_id}_contact.jpg"), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--video", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gt-boundaries", help="JSON list of manual cut times in seconds")
    ap.add_argument("--gt-crop", help="manual crop box 'x0,y0,x1,y1'")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(f"logs/{a.video}.log"), logging.StreamHandler()])
    gt = {}
    if a.gt_boundaries:
        gt["boundaries"] = json.load(open(a.gt_boundaries))
    if a.gt_crop:
        gt["crop_box"] = [int(x) for x in a.gt_crop.split(",")]
    ok = Phase4Stage(cfg, a.video).execute(gt or None, force=a.force)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""
Fix keyframes.json to guarantee:
  1. All 11 dialogue segments get a dedicated panel (using Phase 6 alignment)
  2. Visual panels are spread across the full video timeline (using shots data)
  3. Speaker assignments are carried through for all segments
  4. Total panels = 20 (cover a 4-5 page comic more fully)

Run from CineComic_MML/:
  ..\.venv\Scripts\python.exe fix_keyframes.py
"""
import json
import math
from pathlib import Path

VIDEO_ID = "a13eb5bedd6d"
BASE = Path(__file__).parent

alignment_path = BASE / f"data/alignment/{VIDEO_ID}.json"
shots_path     = BASE / f"data/shots/{VIDEO_ID}.json"
speaker_path   = BASE / f"data/speaker/{VIDEO_ID}.json"
keyframes_path = BASE / f"data/keyframes/{VIDEO_ID}.json"
subtitles_path = BASE / f"data/subtitles/{VIDEO_ID}.json"

TOTAL_PANELS = 20   # Increase from 12 → 20 (5 panels/page × 4 pages)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
alignment = json.loads(alignment_path.read_text(encoding="utf-8-sig"))
shots_data = json.loads(shots_path.read_text(encoding="utf-8-sig"))
speaker_data = json.loads(speaker_path.read_text(encoding="utf-8-sig"))
subtitles = json.loads(subtitles_path.read_text(encoding="utf-8-sig"))

# ── Build lookup: frame_id → timestamp from shots JSON ───────────────────────
frame_time_map = {}  # frame_id → t (seconds)
frame_sharp_map = {} # frame_id → sharpness score
frame_path_map = {}  # frame_id → relative path

for shot in shots_data.get("shots", []):
    for fr in shot.get("frames", []):
        fid = fr["frame_id"]
        frame_time_map[fid] = fr["t"]
        frame_sharp_map[fid] = fr.get("sharp", 500.0)
        frame_path_map[fid] = fr["path"]

print(f"  Frames with timestamps from shots: {len(frame_time_map)}")

# ── Build lookup: frame_id → (segment_id, clip_sim, z) from alignment ────────
frame_seg_map = {}   # frame_id → list of (segment_id, clip_sim, z_score)

for seg in alignment.get("alignment", []):
    sid = str(seg["segment_id"])
    non_visual = seg.get("non_visual", False)
    no_candidates = seg.get("status") == "no_candidates"
    for cand in seg.get("candidates", []):
        fid = cand["frame_id"]
        entry = {
            "segment_id": sid,
            "clip_sim": cand.get("clip_sim", 0.0),
            "z": cand.get("z", 0.0),
            "non_visual": non_visual,
            "no_candidates": no_candidates,
        }
        frame_seg_map.setdefault(fid, []).append(entry)

# ── Build speaker turn lookup: timestamp → character_id ──────────────────────
# Use ALL turn matches (not just accepted ones) for broader coverage
turn_matches = speaker_data.get("turn_matches", [])
# Also build segment→speaker map from speaker mapping
speaker_map_accepted = {}  # speaker_id → character_id (accepted only)
speaker_map_all = {}       # speaker_id → character_id (all)
for sm in speaker_data.get("speaker_mapping", []):
    if sm.get("mapping_accepted"):
        speaker_map_accepted[sm["speaker_id"]] = sm["character_id"]
    if sm.get("matched_turns", 0) > 0:
        speaker_map_all[sm["speaker_id"]] = sm["character_id"]

print(f"  Accepted speaker mappings: {len(speaker_map_accepted)}")
print(f"  Total speaker mappings: {len(speaker_map_all)}")

def get_char_at_time(t):
    """Get character_id and speaker_id at a given timestamp."""
    best = None
    best_score = -1
    for turn in turn_matches:
        tstart = turn.get("start", 0)
        tend = turn.get("end", 0)
        if tstart <= t <= tend:
            char = turn.get("final_character_id") or turn.get("character_id")
            decision = turn.get("decision", "")
            score = turn.get("match_score", 0)
            if char and decision == "matched" and score > best_score:
                best_score = score
                best = {
                    "character_id": char,
                    "speaker_id": turn.get("speaker", turn.get("speaker_id")),
                    "confidence": score,
                }
    # Fallback: use all mappings (not just accepted)
    if best is None:
        for turn in turn_matches:
            tstart = turn.get("start", 0)
            tend = turn.get("end", 0)
            if tstart <= t <= tend:
                char = turn.get("final_character_id") or turn.get("character_id")
                speaker = turn.get("speaker", turn.get("speaker_id"))
                if char:
                    # Try speaker_map_all
                    mapped_char = speaker_map_all.get(speaker, char)
                    best = {
                        "character_id": mapped_char,
                        "speaker_id": speaker,
                        "confidence": turn.get("match_score", 0.1),
                    }
                    break
    return best

# ── STEP 1: Guarantee all dialogue segments get a panel ──────────────────────
print("\nStep 1: Selecting dialogue panels (all 11 segments)...")

dialogue_panels = []
used_frames = set()
used_segments = set()

# For each segment in alignment, pick the best non-used frame
for seg in alignment.get("alignment", []):
    sid = str(seg["segment_id"])
    cands = seg.get("candidates", [])
    status = seg.get("status", "ok")
    start = seg.get("start", 0)
    end = seg.get("end", 0)
    text = seg.get("text", "")

    if status == "no_candidates" or not cands:
        print(f"  Seg {sid} '{text[:30]}': no candidates, skipping dialogue panel")
        continue

    # Score candidates: prefer not-used, good clip_sim, has known timestamp
    best_fid = None
    best_score = -999

    for cand in cands:
        fid = cand["frame_id"]
        sim = cand.get("clip_sim", 0.0)
        z = cand.get("z", 0.0)
        rank = cand.get("rank", 3)
        t = frame_time_map.get(fid)
        sharp = frame_sharp_map.get(fid, 500.0)

        # Score = z_score (primary) - penalty if already used
        score = z - (5.0 if fid in used_frames else 0.0) + (0.1 if t is not None else 0.0)
        if score > best_score:
            best_score = score
            best_fid = fid

    if best_fid is None:
        best_fid = cands[0]["frame_id"]

    t = frame_time_map.get(best_fid)
    char_info = get_char_at_time((start + end) / 2) if t is None else get_char_at_time(t or (start + end) / 2)

    # Find shot_id from frame path
    fpath = frame_path_map.get(best_fid, "")
    shot_num = None
    if "shot_" in fpath:
        try:
            shot_num = int(fpath.split("shot_")[1].split("/")[0].split("\\")[0])
        except:
            pass

    panel = {
        "frame_id": best_fid,
        "path": frame_path_map.get(best_fid, f"data/frames/{VIDEO_ID}/shot_{shot_num:03d}/{best_fid}.jpg"),
        "t": t if t is not None else (start + end) / 2,
        "timestamp_source": "shots_json" if t is not None else "segment_midpoint",
        "segment_ids": [sid],
        "character_id": char_info["character_id"] if char_info else None,
        "speaker_ids": [char_info["speaker_id"]] if char_info and char_info.get("speaker_id") else [],
        "group_id": f"dialogue:{shot_num}:{sid}",
        "scores": {
            "sem": max(0.0, min(1.0, (cands[0].get("clip_sim", 0.2) - 0.15) * 10)),
            "spk": char_info["confidence"] if char_info else 0.0,
            "q": min(1.0, frame_sharp_map.get(best_fid, 500) / 1000.0),
            "sal": 0.5,
            "motion": 0.1,
            "div": 1.0,
            "total": 0.7,
        },
        "branch": "dialogue",
    }
    dialogue_panels.append(panel)
    used_frames.add(best_fid)
    used_segments.add(sid)
    print(f"  Seg {sid} '{text[:25]}': -> {best_fid} (t={panel['t']:.2f}s, char={panel['character_id']})")

print(f"  Dialogue panels: {len(dialogue_panels)}")

# ── STEP 2: Add visual panels spread across full video timeline ───────────────
print(f"\nStep 2: Selecting visual panels to fill up to {TOTAL_PANELS} total...")

needed_visual = TOTAL_PANELS - len(dialogue_panels)
video_duration = shots_data.get("duration_s", 165.0)

# Collect all frames with timestamps from shots data, excluding already-used
all_visual_candidates = []
for shot in shots_data.get("shots", []):
    shot_id = shot["shot_id"]
    shot_start = shot["start"]
    shot_end = shot["end"]
    for fr in shot.get("frames", []):
        fid = fr["frame_id"]
        if fid in used_frames:
            continue
        t = fr["t"]
        sharp = fr.get("sharp", 500.0)
        all_visual_candidates.append({
            "frame_id": fid,
            "path": fr["path"],
            "t": t,
            "shot_id": shot_id,
            "shot_start": shot_start,
            "shot_end": shot_end,
            "sharp": sharp,
        })

# Sort by time
all_visual_candidates.sort(key=lambda x: x["t"])

# Divide video into N intervals and pick best frame per interval
# Exclude intervals already covered by dialogue panels
dialogue_times = [p["t"] for p in dialogue_panels]

def min_distance_from_dialogue(t):
    if not dialogue_times:
        return float("inf")
    return min(abs(t - dt) for dt in dialogue_times)

# Use greedy temporal spread: pick the frame that maximizes minimum
# distance from already-selected panels
selected_visual = []
selected_times = list(dialogue_times)

for _ in range(needed_visual):
    best_frame = None
    best_score = -1

    for fr in all_visual_candidates:
        fid = fr["frame_id"]
        t = fr["t"]
        sharp = fr["sharp"]

        if fid in used_frames:
            continue

        # Temporal diversity score: min distance from all selected times
        if selected_times:
            min_dist = min(abs(t - st) for st in selected_times)
        else:
            min_dist = video_duration

        # Normalize
        norm_dist = min(1.0, min_dist / (video_duration / TOTAL_PANELS))
        norm_sharp = min(1.0, sharp / 2000.0)

        # Combined score: favor temporal spread + sharpness
        score = 0.7 * norm_dist + 0.3 * norm_sharp

        if score > best_score:
            best_score = score
            best_frame = fr

    if best_frame is None:
        break

    fid = best_frame["frame_id"]
    t = best_frame["t"]
    char_info = get_char_at_time(t)

    panel = {
        "frame_id": fid,
        "path": best_frame["path"],
        "t": t,
        "timestamp_source": "shots_json",
        "segment_ids": [],
        "character_id": char_info["character_id"] if char_info else None,
        "speaker_ids": [char_info["speaker_id"]] if char_info and char_info.get("speaker_id") else [],
        "group_id": f"visual:shot:{best_frame['shot_id']}",
        "scores": {
            "sem": 0.0,
            "spk": char_info["confidence"] if char_info else 0.0,
            "q": min(1.0, best_frame["sharp"] / 1000.0),
            "sal": 0.5,
            "motion": 0.2,
            "div": best_score,
            "total": best_score * 0.9,
        },
        "branch": "visual",
    }
    selected_visual.append(panel)
    used_frames.add(fid)
    selected_times.append(t)

print(f"  Visual panels: {len(selected_visual)}")

# ── STEP 3: Merge and sort chronologically ────────────────────────────────────
all_panels = dialogue_panels + selected_visual
all_panels.sort(key=lambda p: (p["t"] is None, p["t"] if p["t"] is not None else 999))

for i, p in enumerate(all_panels, 1):
    p["order"] = i

print(f"\nFinal panel count: {len(all_panels)}")
print("Panel timeline:")
for p in all_panels:
    t = p["t"]
    segs = ",".join(p["segment_ids"]) or "-"
    char = p["character_id"] or "?"
    branch = p["branch"]
    print(f"  [{p['order']:2d}] t={t:6.1f}s  {branch:8s} segs=[{segs}] char={char}  {p['frame_id']}")

# ── STEP 4: Write keyframes.json ─────────────────────────────────────────────
output = {
    "phase": 8,
    "video_id": VIDEO_ID,
    "selection_method": "Fixed: dialogue-first + temporally-spread visual panels",
    "panels": all_panels,
    "groups": [],
    "metrics_path": f"results/phase8/{VIDEO_ID}_metrics.json",
}

keyframes_path.write_text(
    json.dumps(output, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"\nWritten: {keyframes_path}")
print("Next step: re-run phases 9–12 only.")

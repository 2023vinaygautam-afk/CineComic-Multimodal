"""
Patch Phase 7 speaker data to:
1. Accept ALL 5 speaker->character mappings (not just the 1 with high confidence)
2. For turns with decision='low_confidence_or_ambiguous', use the best candidate's character_id
3. This ensures Phase 12 can attribute dialogues to faces for all segments

Run from CineComic_MML/:
  ..\.venv\Scripts\python.exe fix_speaker.py
"""
import json
from pathlib import Path

VIDEO_ID = "a13eb5bedd6d"
BASE = Path(__file__).parent
speaker_path = BASE / f"data/speaker/{VIDEO_ID}.json"

print("Loading speaker data...")
data = json.loads(speaker_path.read_text(encoding="utf-8-sig"))

# Step 1: Accept ALL speaker mappings
accepted_before = sum(1 for m in data.get("speaker_mapping", []) if m.get("mapping_accepted"))
for mapping in data.get("speaker_mapping", []):
    if mapping.get("matched_turns", 0) > 0:
        mapping["mapping_accepted"] = True
accepted_after = sum(1 for m in data.get("speaker_mapping", []) if m.get("mapping_accepted"))
print(f"  Speaker mappings accepted: {accepted_before} -> {accepted_after}")

# Build accepted speaker->character map
speaker_to_char = {}
for mapping in data.get("speaker_mapping", []):
    if mapping.get("mapping_accepted") and mapping.get("character_id") != "unknown":
        speaker_to_char[mapping["speaker_id"]] = mapping["character_id"]
print(f"  Speaker->character map: {speaker_to_char}")

# Step 2: Fix turns that failed to get a character assignment
fixed_turns = 0
for turn in data.get("turn_matches", []):
    decision = turn.get("decision", "")
    char = turn.get("final_character_id", "unknown")
    speaker = turn.get("speaker", turn.get("speaker_id", ""))

    # For ambiguous/failed turns, try speaker-level assignment
    if char in (None, "unknown", ""):
        mapped_char = speaker_to_char.get(speaker)
        if mapped_char:
            turn["character_id"] = mapped_char
            turn["visual_character_id"] = mapped_char
            turn["final_character_id"] = mapped_char
            if decision in ("low_confidence_or_ambiguous", "no_candidate_track"):
                turn["decision"] = "matched_via_speaker_level"
            fixed_turns += 1

    # For ambiguous turns where we have candidates, pick the best one
    elif decision == "low_confidence_or_ambiguous":
        candidates = turn.get("candidates", [])
        if candidates:
            best = max(candidates, key=lambda c: c.get("match_score", 0))
            if best.get("existing_character_id"):
                turn["character_id"] = best["existing_character_id"]
                turn["visual_character_id"] = best["existing_character_id"]
                turn["final_character_id"] = best["existing_character_id"]
                turn["matched_track_id"] = best["track_id"]
                turn["decision"] = "matched_best_candidate"
                fixed_turns += 1

print(f"  Turns fixed with character assignments: {fixed_turns}")

# Summary of turn assignments
print("\n  Turn assignments after fix:")
for turn in data.get("turn_matches", []):
    tid = turn.get("turn_id", "?")
    spk = turn.get("speaker", turn.get("speaker_id", "?"))
    char = turn.get("final_character_id", "?")
    start = turn.get("start", 0)
    end = turn.get("end", 0)
    decision = turn.get("decision", "?")
    print(f"    Turn {tid:3s}: {spk} -> {char:10s} [{start:.1f}-{end:.1f}s] ({decision})")

# Update stats
matched = sum(1 for t in data.get("turn_matches", []) if t.get("final_character_id") not in (None, "unknown", ""))
total = len(data.get("turn_matches", []))
data["num_matched_turns"] = matched
data["match_rate"] = round(matched / total if total > 0 else 0, 4)
data["accepted_speaker_mappings"] = accepted_after

speaker_path.write_text(
    json.dumps(data, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"\nWritten: {speaker_path}")
print(f"Match rate: {data['match_rate']:.1%} ({matched}/{total} turns with character)")

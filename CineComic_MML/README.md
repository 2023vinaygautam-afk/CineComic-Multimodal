# CineComic-MML

Multimodal video-to-comic system, built one stage at a time. This repository currently
contains **Phase 0 (foundation) and Phase 1 (video ingestion and validation)**.

## Setup

1. Install Python 3.10/3.11+ and FFmpeg
   (`ffmpeg -version` and `ffprobe -version` must work).

2. `python -m venv .venv` and activate it, then:

   `pip install -r requirements.txt`

3. Create the folder skeleton:

   `python setup_project.py`

4. Run:

   `python env_check.py`

   This writes:

   `results/environment_report.json`

## Run a stage or the full pipeline

The runner automatically merges the stage-specific YAML fragments when you use `--stages all`, so a single config file such as `configs/phase2.yaml` is enough for the full pipeline.

```bash
python main.py --config configs/phase1.yaml --video path/to/clip.mp4 --video-id clip_001 --stages ingestion
python main.py --config configs/phase2.yaml --video path/to/clip.mp4 --video-id clip_001 --stages all
```
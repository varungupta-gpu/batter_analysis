# Batter Trigger Analysis

This repository contains the trigger-focused batting workflow only.

## What is included

- Renumbered trigger scripts inside `trigger_detection/`
- All prompt and knowledge markdown files inside `trigger_detection/md/`
- One consolidated manifest CSV: `trigger_detection/data/segment_manifest.csv`
- Batsman keypoint CSV files for the 43 segments used in the LLM workflow inside `trigger_detection/data/batsman_keypoints/`
- `segment.xlsx`

Generated LLM outputs are intentionally not committed.

## Folder layout

- `trigger_detection/01_fetch_batsman_keypoints.py`
  Trigger pose jobs / fetch segment-level batsman keypoint generation inputs.
- `trigger_detection/02_download_batsman_keypoints.py`
  Download generated batsman keypoint CSV files from GCS.
- `trigger_detection/03_detect_trigger_window.py`
  Smooth keypoints and detect the trigger window.
- `trigger_detection/04_feature_extraction.py`
  Compute original trigger features frame by frame.
- `trigger_detection/05_compute_trigger_metrics.py`
  Compute trigger metrics from extracted features.
- `trigger_detection/06_phase_feature_calculations.py`
  Compute the added stance / movement / body-organisation features.
- `trigger_detection/07_llm_feature_payload.py`
  Build the structured feature payload and rolling statistics for segment-level LLM analysis.
- `trigger_detection/08_interactive_llm_segment_analysis.py`
  Interactive segment-level LLM analysis runner.
- `trigger_detection/09_interactive_llm_player_analysis.py`
  Interactive player-level LLM analysis runner.
- `trigger_detection/md/`
  Prompt files and knowledge-source markdowns.
- `trigger_detection/data/segment_manifest.csv`
  Single manifest containing video id, video name, ball, segment id, release frame, trigger metadata, and local batsman keypoint CSV path.
- `trigger_detection/data/batsman_keypoints/`
  Local batsman keypoint CSV files referenced by the manifest.

## Flow

### 1. Fetch / trigger batsman keypoint jobs

Use `01_fetch_batsman_keypoints.py` when you want to trigger the upstream pose/keypoint generation jobs for selected video and segment pairs.

### 2. Download batsman keypoints

Use `02_download_batsman_keypoints.py` to download the generated batsman keypoint CSV files locally.

This repository already includes the batsman keypoint CSVs required for the curated trigger workflow.

### 3. Trigger detection

`03_detect_trigger_window.py`, `04_feature_extraction.py`, `05_compute_trigger_metrics.py`, and `06_phase_feature_calculations.py` are the core trigger-analysis modules.

These scripts handle:

- smoothing keypoints
- detecting trigger windows
- computing original trigger features
- computing extended body-organisation / phase features

### 4. Segment-level LLM analysis

Run:

```powershell
python trigger_detection\08_interactive_llm_segment_analysis.py
```

This script:

- reads `trigger_detection/data/segment_manifest.csv`
- lets you choose a segment interactively
- smooths the batsman keypoints
- computes feature payloads
- builds the segment prompt using the markdown files
- optionally calls Gemini and saves JSON output inside `trigger_detection_outputs/llm_segment_analysis/segment_manifest/`

### 5. Player-level LLM analysis

Run:

```powershell
python trigger_detection\09_interactive_llm_player_analysis.py
```

This script:

- reads the generated segment-level outputs
- groups players using the current project rule:
  videos starting with `f` -> `player1`
  all others -> `player2`
- builds a player-level JSON input
- optionally calls Gemini and saves player-level JSON output inside `trigger_detection_outputs/llm_player_analysis/`

## Prompt files

All markdown files inside `trigger_detection/md/` are included in this repository.

The current segment and player LLM scripts use these prompt files directly from that folder.

## Notes

- `trigger_detection_outputs/` is gitignored because it contains generated outputs.
- The repository contains one consolidated manifest CSV instead of many intermediate CSV files.
- The player-level script expects segment-level LLM outputs to be created first.

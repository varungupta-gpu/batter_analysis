# Batter Trigger Analysis

This repository contains the trigger-focused batting workflow only: batsman keypoint fetching/downloading, trigger detection, segment-level LLM analysis, and player-level LLM analysis.

Generated LLM outputs are intentionally not committed.

## What is included

- Renumbered trigger scripts inside `trigger_detection/`
- All prompt and knowledge markdown files inside `trigger_detection/md/`
- One consolidated manifest CSV: `trigger_detection/data/segment_manifest.csv`
- Batsman keypoint CSV files for the 43 curated segments inside `trigger_detection/data/batsman_keypoints/`
- `segment.xlsx`

## Major files

### Entry-point scripts

- `trigger_detection/01_fetch_batsman_keypoints.py`
  Use this when you want to trigger or fetch upstream batsman-keypoint generation jobs for selected videos and segments.

- `trigger_detection/02_download_batsman_keypoints.py`
  Use this to download batsman keypoint CSV files from GCS after upstream jobs have completed.

- `trigger_detection/08_interactive_llm_segment_analysis.py`
  Main interactive runner for segment-level analysis. This is the script most users will run first inside this repo.

- `trigger_detection/09_interactive_llm_player_analysis.py`
  Main interactive runner for player-level analysis. This should be run only after segment-level LLM outputs exist.

### Core trigger-analysis modules

- `trigger_detection/03_detect_trigger_window.py`
  Core trigger detector. It smooths batsman keypoints, defines the stance/trigger window, and prepares the frame range used for analysis.

- `trigger_detection/04_feature_extraction.py`
  Computes the original trigger features frame by frame, such as foot progression, ankle displacement, stride width, and knee movement.

- `trigger_detection/05_compute_trigger_metrics.py`
  Converts extracted features into trigger metrics used for trigger scoring and downstream interpretation.

- `trigger_detection/06_phase_feature_calculations.py`
  Computes the extended body-organisation and biomechanics-style features such as hip direction, rotation, trunk flexion, and knee angles.

- `trigger_detection/07_llm_feature_payload.py`
  Builds the structured feature package used by the segment LLM step. It combines original trigger features, extended phase features, rolling statistics, and summary values into one LLM-ready payload.

### Shared helpers

- `trigger_detection/phase_pipeline_common.py`
  Shared utility layer for loading and normalising keypoints/release data, inferring standard columns, and handling common geometry helpers.

- `trigger_detection/gcp_fetch_utils.py`
  Shared Google Cloud / API helper functions used by the fetch and download scripts.

### Data and prompts

- `trigger_detection/data/segment_manifest.csv`
  The single source CSV used by the trigger workflow in this repo. It contains video id, video name, ball, segment id, release frame, trigger metadata, and the local batsman keypoint CSV path.

- `trigger_detection/data/batsman_keypoints/`
  Contains the local batsman keypoint CSV files referenced by the manifest.

- `trigger_detection/md/`
  Contains all prompt and knowledge markdown files used by the LLM analysis scripts.

## Prompt files

All markdown files inside `trigger_detection/md/` are included.

Important ones are:

- `MD1_Segment_Level_Stance_Trigger_Analysis_Prompt (1).md`
  Main segment-level analysis prompt.

- `MD2_Keypoints_and_Feature_Calculation.md`
  Feature-definition and calculation reference used in the segment workflow.

- `MD3_Stance_and_Trigger_Classification.md`
  Stance and trigger classification knowledge source.

- `MD4_Trigger_Corrections_and_Biomechanical_Concerns.md`
  Correction and biomechanical-concern guidance.

- `Player_Level_Combined_Stance_Trigger_Prompt (1).md`
  Main player-level aggregation prompt.

## Workflow

### 1. Fetch or download batsman keypoints

If batsman keypoints do not already exist:

- run `01_fetch_batsman_keypoints.py` to trigger/fetch upstream jobs
- run `02_download_batsman_keypoints.py` to download the keypoint CSV files

For the curated set in this repository, the required batsman keypoint CSV files are already included.

### 2. Run segment-level LLM analysis

Run:

```powershell
python trigger_detection\08_interactive_llm_segment_analysis.py
```

This script:

- reads `trigger_detection/data/segment_manifest.csv`
- lets you choose a segment interactively
- smooths batsman keypoints
- computes trigger and phase features
- builds the segment prompt using the markdown files
- optionally calls Gemini
- saves generated outputs under `trigger_detection_outputs/llm_segment_analysis/segment_manifest/`

### 3. Run player-level LLM analysis

Run:

```powershell
python trigger_detection\09_interactive_llm_player_analysis.py
```

This script:

- reads the generated segment-level LLM outputs
- groups players using the current project rule:
  videos starting with `f` -> `player1`
  all others -> `player2`
- builds a player-level JSON input
- optionally calls Gemini
- saves generated outputs under `trigger_detection_outputs/llm_player_analysis/`

## Notes

- `trigger_detection_outputs/` is gitignored because it contains generated outputs.
- This repository uses one consolidated manifest CSV instead of many intermediate CSV files.
- Player-level analysis depends on segment-level LLM outputs being created first.

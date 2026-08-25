# Batter Trigger Analysis

This repository contains the trigger-analysis part of the batter workflow. It detects trigger movement from smoothed batsman keypoints, then uses Gemini for segment-level and player-level batting analysis.

Generated LLM outputs and generated videos are not committed. The repo keeps the runnable code, prompt markdown files, curated manifest CSV, batsman keypoint CSVs, and `segment.xlsx`.

## Setup

```powershell
cd "C:\Users\raksh\OneDrive\Desktop\batter_analysis"
pip install -r requirements.txt
```

## Folder Structure

- `trigger_detection/trigger_core/`
  Trigger detection and frame-by-frame feature calculation logic.

- `trigger_detection/segment_llm_analysis/`
  Segment-level LLM runner and feature-payload builder.

- `trigger_detection/player_llm_analysis/`
  Player-level LLM runner that combines completed segment outputs.

- `trigger_detection/keypoint_download/`
  Optional scripts for triggering/downloading batsman keypoints from upstream services.

- `trigger_detection/common/`
  Shared helpers for JSON writing, Gemini API calls, GCS utilities, and keypoint normalization.

- `trigger_detection/config.py`
  Central project paths and output locations.

- `trigger_detection/md/`
  Prompt and knowledge markdown files used by the LLM scripts.

- `trigger_detection/data/`
  Curated input data, including `segment_manifest.csv` and batsman keypoint CSV files.

## Full Flow

The expected order is:

1. Detect trigger frames.
2. Run segment-level LLM analysis.
3. Run player-level LLM analysis.

## Step 1: Detect Trigger Frames

Run this first:

```powershell
cd "C:\Users\raksh\OneDrive\Desktop\batter_analysis"
python trigger_detection\trigger_core\detect_trigger_window.py
```

This script reads:

- `trigger_detection/data/segment_manifest.csv`
- `trigger_detection/data/batsman_keypoints/`

It applies the smoothed-keypoint trigger logic. The detector compares movement before load-up with movement from load-up to release. If enough important movement factors cross their thresholds during the load-up-to-release window, the row is marked as an actual trigger.

It writes trigger result files to:

```text
trigger_detection_outputs/
```

Important outputs:

- `trigger_detection_outputs/smoothed_trigger_results.csv`
  Full trigger-detection output with detailed movement metrics.

- `trigger_detection_outputs/smoothed_trigger_summary.csv`
  Shorter trigger summary with video name, ball, segment id, trigger frames, and trigger true/false.

- `trigger_detection_outputs/smoothed_trigger_feature_metrics.json`
  Feature and metric frames used by the trigger detector.

## Step 2: Run Segment-Level LLM Analysis

After trigger frames exist, run:

```powershell
cd "C:\Users\raksh\OneDrive\Desktop\batter_analysis"
python trigger_detection\segment_llm_analysis\interactive_segment_analysis.py
```

This script asks which processed segment/video to analyze. It then builds the full segment input using:

- smoothed batsman keypoints
- trigger frames from the manifest/result fields
- original trigger features
- extended phase/biomechanics features
- rolling statistics and summary statistics
- segment-level prompt markdown files

Prompt files used:

- `trigger_detection/md/MD1_Segment_Level_Stance_Trigger_Analysis_Prompt (1).md`
- `trigger_detection/md/MD2_Keypoints_and_Feature_Calculation.md`
- `trigger_detection/md/MD3_Stance_and_Trigger_Classification.md`
- `trigger_detection/md/MD4_Trigger_Corrections_and_Biomechanical_Concerns.md`

If the selected row has no trigger detected, the script still passes fallback trigger frames from:

```text
release_frame - 20 to release_frame - 12
```

This lets the LLM judge whether that fallback movement is actually a trigger or not.

Segment outputs are saved under:

```text
trigger_detection_outputs/llm_segment_analysis/segment_manifest/
```

Typical saved files per segment:

- `segment_metadata.json`
- `smoothed_keypoints.json`
- `all_features.json`
- `feature_statistics.json`
- `llm_prompt.txt`
- `llm_analysis.json`

## Step 3: Run Player-Level LLM Analysis

Run this only after segment-level `llm_analysis.json` files exist:

```powershell
cd "C:\Users\raksh\OneDrive\Desktop\batter_analysis"
python trigger_detection\player_llm_analysis\interactive_player_analysis.py
```

This script asks which player to analyze and combines the already-generated segment-level JSON outputs into one player-level input.

Current player grouping rule:

- video names starting with `f` are treated as `player1`
- all other video names are treated as `player2`

Player-level prompt files used:

- `trigger_detection/md/MD3_Stance_and_Trigger_Classification.md`
- `trigger_detection/md/MD4_Trigger_Corrections_and_Biomechanical_Concerns.md`
- `trigger_detection/md/Player_Level_Combined_Stance_Trigger_Prompt (1).md`

Player outputs are saved under:

```text
trigger_detection_outputs/llm_player_analysis/
```

Typical saved files per player:

- `player_input.json`
- `player_llm_request.json`
- `player_llm_analysis.json`

## Main Inputs

- `trigger_detection/data/segment_manifest.csv`
  Main CSV for video name, ball number, segment id, release frame, trigger prediction fields, and batsman keypoint path.

- `trigger_detection/data/batsman_keypoints/`
  Batsman keypoint CSVs used for trigger detection and LLM feature generation.

- `trigger_detection/md/`
  All prompt and reference markdown files used by segment/player analysis.

## Notes

- `trigger_detection_outputs/` is ignored by git because it contains generated run outputs.

- Gemini API keys are not stored in the repo. The segment and player scripts ask for the key interactively if `GEMINI_API_KEY` is not already set.

- The code is split by responsibility: trigger detection, segment LLM analysis, player LLM analysis, shared helpers, and config are separate.

- Shared Gemini request code lives in `trigger_detection/common/llm_client.py`.

- Shared JSON cleanup/writing code lives in `trigger_detection/common/json_utils.py`.

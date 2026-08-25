# Batter Trigger Analysis

This repo contains the curated batting trigger workflow: batsman keypoint download, trigger detection, segment-level LLM analysis, and player-level LLM analysis.

Generated videos and LLM outputs are not committed. The repo keeps only the code, prompt markdown, manifest CSV, selected batsman keypoint CSVs, and `segment.xlsx`.

## Install

```powershell
cd "C:\Users\raksh\OneDrive\Desktop\batter_analysis"
pip install -r requirements.txt
```

## Folder Structure

- `trigger_detection/trigger_core/`
  Contains only trigger detection and feature-calculation logic.

- `trigger_detection/segment_llm_analysis/`
  Contains the interactive segment-level LLM runner and the feature-payload builder used by that runner.

- `trigger_detection/player_llm_analysis/`
  Contains the interactive player-level LLM runner. It reads completed segment-level JSON outputs and combines them into one batter report.

- `trigger_detection/keypoint_download/`
  Contains helper entry points for triggering/downloading batsman keypoints from upstream services.

- `trigger_detection/common/`
  Contains reusable non-workflow helpers such as Gemini API calls, JSON serialization, GCS helpers, and keypoint normalization helpers.

- `trigger_detection/config.py`
  Central path/config module. Keep project paths here, not inside `common`.

- `trigger_detection/md/`
  Contains all markdown prompt/knowledge files used by the LLM analysis.

- `trigger_detection/data/segment_manifest.csv`
  The single curated CSV used by this repo. It includes video name, ball, segment id, release frame, trigger prediction fields, and local batsman keypoint paths.

- `trigger_detection/data/batsman_keypoints/`
  Local batsman keypoint CSV files used by the manifest rows.

## Main File Flow

1. `trigger_detection/keypoint_download/fetch_batsman_keypoints.py`
   Optional upstream job trigger for batsman pose/keypoint generation.

2. `trigger_detection/keypoint_download/download_batsman_keypoints.py`
   Optional GCS downloader for generated `batter_keypoints.csv` files.

3. `trigger_detection/trigger_core/detect_trigger_window.py`
   Runs trigger detection on smoothed batsman keypoints. It compares pre-load-up motion against load-up-to-release motion and outputs trigger frames/metrics.

4. `trigger_detection/segment_llm_analysis/feature_payload.py`
   Builds the LLM-ready feature payload. It combines original trigger features, extended phase/biomechanics features, rolling stats, summary stats, and trigger metric frames.

5. `trigger_detection/segment_llm_analysis/interactive_segment_analysis.py`
   Interactive segment-level analysis. It asks which processed segment to analyze, prepares JSON inputs, optionally asks for a Gemini API key, and saves the segment report.

6. `trigger_detection/player_llm_analysis/interactive_player_analysis.py`
   Interactive player-level analysis. It asks which player to analyze, groups existing segment reports using the current rule, prepares JSON input, optionally asks for a Gemini API key, and saves the player report.

## Run Segment-Level Analysis

```powershell
cd "C:\Users\raksh\OneDrive\Desktop\batter_analysis"
python trigger_detection\segment_llm_analysis\interactive_segment_analysis.py
```

This uses:

- `trigger_detection/data/segment_manifest.csv`
- `trigger_detection/data/batsman_keypoints/`
- `trigger_detection/md/MD1_Segment_Level_Stance_Trigger_Analysis_Prompt (1).md`
- `trigger_detection/md/MD2_Keypoints_and_Feature_Calculation.md`
- `trigger_detection/md/MD3_Stance_and_Trigger_Classification.md`
- `trigger_detection/md/MD4_Trigger_Corrections_and_Biomechanical_Concerns.md`

Outputs are written to:

```text
trigger_detection_outputs/llm_segment_analysis/segment_manifest/
```

If no trigger exists in the source row, the segment runner passes `release_frame - 20` through `release_frame - 12` as fallback trigger frames so the LLM can still judge whether that movement is a real trigger.

## Run Player-Level Analysis

Run this after segment-level `llm_analysis.json` files exist:

```powershell
cd "C:\Users\raksh\OneDrive\Desktop\batter_analysis"
python trigger_detection\player_llm_analysis\interactive_player_analysis.py
```

This uses:

- existing segment JSON outputs from `trigger_detection_outputs/llm_segment_analysis/segment_manifest/`
- `trigger_detection/md/MD3_Stance_and_Trigger_Classification.md`
- `trigger_detection/md/MD4_Trigger_Corrections_and_Biomechanical_Concerns.md`
- `trigger_detection/md/Player_Level_Combined_Stance_Trigger_Prompt (1).md`

Current player grouping rule:

- video names starting with `f` become `player1`
- all other video names become `player2`

Outputs are written to:

```text
trigger_detection_outputs/llm_player_analysis/
```

## Design Notes

- Single Responsibility: trigger detection, segment LLM analysis, player LLM analysis, config, JSON helpers, and API calls are separated into different modules.

- DRY: shared Gemini request logic lives in `trigger_detection/common/llm_client.py`, and JSON cleanup/writing lives in `trigger_detection/common/json_utils.py`.

- Encapsulation: each workflow folder imports stable helpers instead of loading numbered scripts dynamically.

- Config location: project paths are centralized in `trigger_detection/config.py` as requested.

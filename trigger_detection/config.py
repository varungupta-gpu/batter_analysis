"""Project configuration and filesystem paths for trigger analysis."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIGGER_ROOT = REPO_ROOT / "trigger_detection"
DATA_DIR = TRIGGER_ROOT / "data"
MD_DIR = TRIGGER_ROOT / "md"

SEGMENT_MANIFEST_PATH = DATA_DIR / "segment_manifest.csv"
BATSMAN_KEYPOINTS_DIR = DATA_DIR / "batsman_keypoints"

ANNOTATED_VIDEO_DIR = REPO_ROOT / "annotated_videos_output"
OUTPUT_ROOT = REPO_ROOT / "trigger_detection_outputs"
SEGMENT_LLM_OUTPUT_ROOT = OUTPUT_ROOT / "llm_segment_analysis"
PLAYER_LLM_OUTPUT_ROOT = OUTPUT_ROOT / "llm_player_analysis"

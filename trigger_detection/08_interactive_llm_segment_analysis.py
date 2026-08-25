#!/usr/bin/env python3
"""Interactive segment-level LLM batting analysis runner."""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
TRIGGER_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "trigger_detection_outputs" / "llm_segment_analysis"
MD_DIR = TRIGGER_DIR / "md"
ANNOTATED_DIR = REPO_ROOT / "annotated_videos_output"
RESULT_SOURCE_PATH = TRIGGER_DIR / "data" / "segment_manifest.csv"

API_KEY = os.getenv("GEMINI_API_KEY", "")
API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MODEL = "gemini-3.7-flash"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _load_local_module(module_name: str, filename: str):
    module_path = TRIGGER_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


td01 = _load_local_module("td01_llm_runner", "03_detect_trigger_window.py")
llm_payload = _load_local_module("td07_llm_runner", "07_llm_feature_payload.py")


def _json_ready(value):
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if pd.isna(value):
        return None
    return value


def get_result_source() -> Path:
    if not RESULT_SOURCE_PATH.exists():
        raise FileNotFoundError(f"Required result source not found: {RESULT_SOURCE_PATH}")
    sample_df = pd.read_csv(RESULT_SOURCE_PATH, nrows=1)
    required = {"video_name", "ball", "segment_id", "trigger_detected"}
    if not required.issubset(set(sample_df.columns)):
        raise ValueError(f"Result source is missing required columns: {RESULT_SOURCE_PATH}")
    return RESULT_SOURCE_PATH


def load_processed_rows(source_path: Path) -> pd.DataFrame:
    results_df = pd.read_csv(source_path)
    if "status" in results_df.columns:
        results_df = results_df[results_df["status"].astype(str).str.lower() == "processed"].copy()
    results_df["ball"] = pd.to_numeric(results_df["ball"], errors="coerce")
    return results_df


def choose_processed_segment(results_df: pd.DataFrame) -> pd.Series:
    search = input("\nEnter part of video name to filter, or press Enter to show all processed rows: ").strip().lower()
    filtered = results_df.copy()
    if search:
        filtered = filtered[filtered["video_name"].astype(str).str.lower().str.contains(search, na=False)].copy()
    if filtered.empty:
        raise ValueError("No processed rows matched that filter.")

    display_df = filtered[["video_name", "ball", "segment_id", "trigger_detected"]].reset_index(drop=False)
    print("\nMatching processed segments:")
    for _, row in display_df.iterrows():
        print(
            f"  {int(row['index'])}: {row['video_name']} | ball {int(row['ball'])} | "
            f"segment {row['segment_id']} | trigger_detected={row['trigger_detected']}"
        )

    while True:
        raw = input("\nEnter the row index to analyze: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if idx in filtered.index:
                return filtered.loc[idx]
        print("Please enter a valid row index from the list above.")


def annotated_video_path(video_name: str, ball: int) -> Path:
    stem = video_name.replace(".mp4", "")
    prefix = stem.rsplit("_", 1)[0]
    return ANNOTATED_DIR / f"{prefix}_ball{int(ball)}.mp4"


def read_video_props(video_path: Path) -> tuple[float, Optional[int]]:
    if not video_path.exists():
        return 30.0, None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 30.0, None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return (fps if fps > 0 else 30.0), (frame_count if frame_count > 0 else None)


def parse_trigger_frames(row: pd.Series, release_frame: int) -> Dict[str, object]:
    candidate_fields = ["predicted_trigger_frames", "expected_trigger_frames", "trigger_frames"]
    parsed_frames: List[int] = []
    for field in candidate_fields:
        if field in row.index and pd.notna(row[field]) and str(row[field]).strip():
            try:
                value = json.loads(str(row[field]))
                if isinstance(value, list):
                    parsed_frames = [int(frame) for frame in value]
                    break
            except json.JSONDecodeError:
                continue

    trigger_detected = str(row.get("trigger_detected", "")).lower() == "true"
    if trigger_detected and parsed_frames:
        return {
            "detected": True,
            "frames": parsed_frames,
            "start_frame": parsed_frames[0],
            "end_frame": parsed_frames[-1],
            "source": "predicted_trigger",
        }

    fallback_frames = list(range(release_frame - 20, release_frame - 11))
    return {
        "detected": False,
        "frames": fallback_frames,
        "start_frame": fallback_frames[0],
        "end_frame": fallback_frames[-1],
        "source": "fallback_release_minus20_to_minus12",
    }


def determine_stance_frames(row: pd.Series, trigger_start_frame: int, release_frame: int) -> Dict[str, int]:
    start_candidates = ["pre_loadup_start_frame", "baseline_start_frame"]
    end_candidates = ["pre_loadup_end_frame", "baseline_end_frame"]

    stance_start = None
    stance_end = None
    for field in start_candidates:
        if field in row.index and pd.notna(row[field]):
            stance_start = int(float(row[field]))
            break
    for field in end_candidates:
        if field in row.index and pd.notna(row[field]):
            stance_end = int(float(row[field]))
            break

    if stance_start is None or stance_end is None or stance_end < stance_start:
        stance_end = max(0, trigger_start_frame - 1)
        stance_start = max(0, stance_end - 9)

    if stance_end >= release_frame:
        stance_end = max(0, min(release_frame - 1, trigger_start_frame - 1))
    if stance_start > stance_end:
        stance_start = max(0, stance_end - 9)

    return {"start_frame": stance_start, "end_frame": stance_end}


def load_prompt_docs() -> Dict[str, str]:
    docs = {}
    filenames = {
        "md1_prompt": "MD1_Segment_Level_Stance_Trigger_Analysis_Prompt (1).md",
        "md2_feature_rules": "MD2_Keypoints_and_Feature_Calculation.md",
        "md3_classification_rules": "MD3_Stance_and_Trigger_Classification.md",
        "md4_corrections_and_concerns": "MD4_Trigger_Corrections_and_Biomechanical_Concerns.md",
    }
    for key, filename in filenames.items():
        path = MD_DIR / filename
        docs[key] = path.read_text(encoding="utf-8")
    return docs


def build_prompt(
    docs: Dict[str, str],
    metadata: Dict[str, object],
    smoothed_keypoints: List[Dict[str, object]],
    all_features: List[Dict[str, object]],
    feature_statistics: Dict[str, object],
) -> str:
    return f"""{docs['md1_prompt']}

---

## Segment Metadata

{json.dumps(_json_ready(metadata), indent=2)}

## Smoothed Keypoints JSON

{json.dumps(_json_ready(smoothed_keypoints), indent=2)}

## All Features JSON

{json.dumps(_json_ready(all_features), indent=2)}

## Feature Statistics JSON

{json.dumps(_json_ready(feature_statistics), indent=2)}

## MD2 Feature Calculation Rules

{docs['md2_feature_rules']}

## MD3 Stance and Trigger Classification Rules

{docs['md3_classification_rules']}

## MD4 Trigger Corrections and Biomechanical Concerns

{docs['md4_corrections_and_concerns']}

---

## Response Requirements

- Return valid JSON only.
- Do not use markdown fences.
- Do not add commentary before or after the JSON.
- Ensure every required field is present.
- If trigger_detected_in_source is false, the supplied fallback trigger frames must still be evaluated normally.
- The `stance.basic_analysis` field must always be present and non-empty.
- The `corrections` array must always contain at least one item.
- The `potential_injury_risks` array must always contain at least one item.
- If no strong correction is needed, provide a basic style-preserving coaching cue instead of leaving the array empty.
- If no strong injury concern is visible, provide one cautious "no strong visible concern identified" entry instead of leaving the array empty.
"""


def call_llm(prompt: str, api_key: str, model: str = MODEL, max_attempts: int = 6) -> str:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.15,
            "maxOutputTokens": 12000,
            "responseMimeType": "application/json",
        },
    }
    last_error: Optional[str] = None
    url = API_URL_TEMPLATE.format(model=model)

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                f"{url}?key={api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            response_data = response.json()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            response = None
            response_data = {}
        else:
            if response.status_code == 200:
                candidate = (response_data.get("candidates") or [{}])[0]
                content = candidate.get("content") or {}
                parts = content.get("parts") or []
                texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
                joined = "\n".join(text.strip() for text in texts if text and text.strip()).strip()
                finish_reason = candidate.get("finishReason")
                if joined:
                    return joined
                last_error = f"Empty content with finishReason={finish_reason}"
            else:
                error = response_data.get("error", {})
                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{error.get('status', 'UNKNOWN')} - {error.get('message', 'No message')}"
                )
                if response.status_code not in {429, 500, 503}:
                    break

        if attempt < max_attempts:
            time.sleep(min(2 ** (attempt - 1), 20))

    raise RuntimeError(f"LLM call failed after {max_attempts} attempts. Last error: {last_error}")


def main() -> int:
    print("=" * 70)
    print("INTERACTIVE LLM SEGMENT ANALYSIS")
    print("=" * 70)
    source_path = get_result_source()
    print(f"\nUsing result source: {source_path.name}")
    results_df = load_processed_rows(source_path)
    selected_row = choose_processed_segment(results_df)

    video_name = str(selected_row["video_name"])
    ball = int(float(selected_row["ball"]))
    segment_id = str(selected_row["segment_id"])
    release_frame = int(round(float(selected_row["release_frame"])))
    batter_path = (REPO_ROOT / str(selected_row["batter_keypoints_path"])).resolve()
    video_path = annotated_video_path(video_name, ball)
    fps, total_frames = read_video_props(video_path)

    trigger_info = parse_trigger_frames(selected_row, release_frame)
    stance_info = determine_stance_frames(selected_row, int(trigger_info["start_frame"]), release_frame)
    analysis_start_frame = min(stance_info["start_frame"], int(trigger_info["start_frame"]))
    analysis_end_frame = release_frame
    rolling_window = 10 if fps >= 45 else 5

    smoothed_df = td01.smooth_raw_keypoint_file(batter_path, fps=fps, total_frames=total_frames)
    all_feature_df = llm_payload.compute_all_feature_dataframe(
        keypoints_df=smoothed_df,
        start_frame=analysis_start_frame,
        end_frame=analysis_end_frame,
        stance_start_frame=stance_info["start_frame"],
        stance_end_frame=stance_info["end_frame"],
    )
    trigger_feature_df = td01.mod03.compute_trigger_feature_dataframe(
        bowler_keypoints_df=smoothed_df,
        start_frame=analysis_start_frame,
        end_frame=analysis_end_frame,
        stance_start_frame=stance_info["start_frame"],
        stance_end_frame=stance_info["end_frame"],
    )
    feature_statistics = llm_payload.compute_feature_statistics(
        all_feature_df=all_feature_df,
        trigger_feature_df=trigger_feature_df,
        fps=fps,
        baseline_end_frame=stance_info["end_frame"],
        rolling_window=rolling_window,
    )

    metadata = {
        "source_results_csv": str(source_path),
        "video_name": video_name,
        "ball": ball,
        "segment_id": segment_id,
        "batter_handedness": None,
        "fps": fps,
        "annotated_video_path": str(video_path) if video_path.exists() else None,
        "batter_keypoints_path": str(batter_path),
        "release_frame": release_frame,
        "stable_stance_frames": stance_info,
        "trigger": trigger_info,
        "trigger_detected_in_source": str(selected_row.get("trigger_detected", "")).lower() == "true",
        "trigger_source_reason": "fallback applied because source row had no trigger detected"
        if trigger_info["source"] != "predicted_trigger"
        else "using source trigger frames",
        "analysis_frame_window": {
            "start_frame": analysis_start_frame,
            "end_frame": analysis_end_frame,
        },
    }

    smoothed_records = _json_ready(smoothed_df.to_dict(orient="records"))
    feature_records = _json_ready(all_feature_df.to_dict(orient="records"))
    docs = load_prompt_docs()
    prompt = build_prompt(docs, metadata, smoothed_records, feature_records, feature_statistics)

    output_dir = OUTPUT_DIR / source_path.stem / f"{video_name.replace('.mp4', '')}_ball{ball}"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "segment_metadata.json").write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")
    (output_dir / "smoothed_keypoints.json").write_text(json.dumps(smoothed_records, indent=2), encoding="utf-8")
    (output_dir / "all_features.json").write_text(json.dumps(feature_records, indent=2), encoding="utf-8")
    (output_dir / "feature_statistics.json").write_text(json.dumps(_json_ready(feature_statistics), indent=2), encoding="utf-8")
    (output_dir / "llm_prompt.txt").write_text(prompt, encoding="utf-8")

    api_key = API_KEY
    if not api_key:
        api_key = input("\nPaste Gemini API key now, or press Enter to stop after preparing files: ").strip()

    if not api_key:
        print("\nPrepared analysis bundle only. No LLM call was made.")
        print(f"Files saved in: {output_dir}")
        return 0

    print(f"\nCalling LLM with model {MODEL}...")
    response_text = call_llm(prompt, api_key, model=MODEL)
    (output_dir / "llm_analysis_raw.txt").write_text(response_text, encoding="utf-8")

    cleaned = response_text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        (output_dir / "llm_analysis.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        print("\nLLM analysis complete.")
        print(f"Saved JSON: {output_dir / 'llm_analysis.json'}")
    except json.JSONDecodeError:
        print("\nLLM responded, but output was not valid JSON.")
        print(f"Saved raw response: {output_dir / 'llm_analysis_raw.txt'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


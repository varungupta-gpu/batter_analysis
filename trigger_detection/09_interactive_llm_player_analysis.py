#!/usr/bin/env python3
"""Interactive player-level LLM analysis from existing segment-level outputs."""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
TRIGGER_DIR = Path(__file__).resolve().parent
MD_DIR = TRIGGER_DIR / "md"
SEGMENT_SOURCE_DIR = (
    REPO_ROOT
    / "trigger_detection_outputs"
    / "llm_segment_analysis"
    / "segment_manifest"
)
OUTPUT_ROOT = REPO_ROOT / "trigger_detection_outputs" / "llm_player_analysis"
MODEL = "gemini-3.7-flash"
API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
API_KEY = os.getenv("GEMINI_API_KEY", "")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _collect_md_bundle() -> str:
    md_files = [
        MD_DIR / "MD3_Stance_and_Trigger_Classification.md",
        MD_DIR / "MD4_Trigger_Corrections_and_Biomechanical_Concerns.md",
        MD_DIR / "Player_Level_Combined_Stance_Trigger_Prompt (1).md",
    ]
    missing = [str(path) for path in md_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required prompt/knowledge files: {missing}")
    blocks = []
    for path in md_files:
        blocks.append(f"--- {path.name} ---\n{_read_text(path).strip()}")
    return "\n\n".join(blocks)


def _load_segment_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _player_name_for_video(video_name: str) -> str:
    return "player1" if video_name.lower().startswith("f") else "player2"


def _find_segment_output_dirs() -> List[Path]:
    if not SEGMENT_SOURCE_DIR.exists():
        raise FileNotFoundError(f"Segment source directory not found: {SEGMENT_SOURCE_DIR}")
    return sorted(
        path
        for path in SEGMENT_SOURCE_DIR.iterdir()
        if path.is_dir() and (path / "llm_analysis.json").exists()
    )


def _build_player_packets() -> Dict[str, List[Dict[str, Any]]]:
    packets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for segment_dir in _find_segment_output_dirs():
        analysis_path = segment_dir / "llm_analysis.json"
        analysis = _load_segment_json(analysis_path)
        segment_id = analysis.get("segment_id", "")
        metadata_path = segment_dir / "segment_metadata.json"
        metadata = _load_segment_json(metadata_path) if metadata_path.exists() else {}
        video_name = metadata.get("video_name") or segment_dir.name.rsplit("_ball", 1)[0]
        ball = metadata.get("ball")
        if ball is None:
            ball = segment_dir.name.rsplit("_ball", 1)[-1]
        player_name = _player_name_for_video(str(video_name))
        packets[player_name].append(
            {
                "video_name": video_name,
                "ball": int(ball),
                "segment_id": segment_id,
                "source_folder": str(segment_dir),
                "segment_analysis": analysis,
                "segment_metadata": metadata,
            }
        )
    return packets


def _build_player_input(player_name: str, packets: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "player_name": player_name,
        "segments_reviewed": len(packets),
        "segments": packets,
    }


def _build_prompt(player_input: Dict[str, Any], md_bundle: str) -> str:
    return (
        f"{md_bundle}\n\n"
        f"PLAYER INPUT JSON:\n{json.dumps(player_input, indent=2, ensure_ascii=False)}\n\n"
        "Return exactly one valid JSON object that follows the player-level schema from the prompt."
    )


def _build_llm_request_json(prompt: str) -> Dict[str, Any]:
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.95,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
        },
    }


def choose_player(packets_by_player: Dict[str, List[Dict[str, Any]]]) -> str:
    print("\nAvailable players:")
    names = sorted(packets_by_player.keys())
    for idx, name in enumerate(names):
        print(f"  {idx}: {name} ({len(packets_by_player[name])} segments)")

    while True:
        raw = input("\nEnter the player index to analyze: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(names):
                return names[idx]
        print("Please enter a valid player index from the list above.")


def _call_gemini(prompt: str, api_key: str) -> str:
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    url = API_URL_TEMPLATE.format(model=MODEL)
    payload = _build_llm_request_json(prompt)
    last_error = None
    for attempt in range(1, 7):
        print(f"\nCalling Gemini model {MODEL} (attempt {attempt}/6)...", flush=True)
        response = requests.post(url, params={"key": api_key}, json=payload, timeout=600)
        if response.ok:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
                if text.strip():
                    print("Gemini response received.", flush=True)
                    return text
            raise RuntimeError(f"Unexpected Gemini response payload: {data}")
        last_error = f"HTTP {response.status_code}: {response.text}"
        if response.status_code in {429, 503} and attempt < 6:
            print(f"Gemini returned a temporary error. Retrying soon...", flush=True)
            time.sleep(min(8 * attempt, 30))
            continue
        raise RuntimeError(last_error)
    raise RuntimeError(last_error or "Gemini call failed")


def main() -> int:
    md_bundle = _collect_md_bundle()
    packets_by_player = _build_player_packets()
    if not packets_by_player:
        raise SystemExit("No segment-level llm_analysis.json files were found.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Using segment source: {SEGMENT_SOURCE_DIR}")
    player_name = choose_player(packets_by_player)
    packets = packets_by_player[player_name]
    player_dir = OUTPUT_ROOT / player_name
    player_dir.mkdir(parents=True, exist_ok=True)
    player_input = _build_player_input(player_name, packets)
    prompt = _build_prompt(player_input, md_bundle)
    llm_request = _build_llm_request_json(prompt)
    (player_dir / "player_input.json").write_text(
        json.dumps(player_input, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (player_dir / "player_llm_request.json").write_text(
        json.dumps(llm_request, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nPrepared prompt for {player_name} with {len(packets)} segment outputs.")

    api_key = API_KEY
    if api_key:
        entered = input(
            "\nPress Enter to use the existing Gemini API key, or paste a new key to override it: "
        ).strip()
        if entered:
            api_key = entered
    else:
        api_key = input("\nPaste Gemini API key now, or press Enter to stop after preparing files: ").strip()
    if not api_key:
        print(f"Saved player input JSON: {player_dir / 'player_input.json'}")
        print(f"Saved LLM request JSON: {player_dir / 'player_llm_request.json'}")
        print("No API key provided, so no LLM request was made.")
        return 0

    print("\nStarting player-level LLM analysis...", flush=True)
    try:
        response_text = _call_gemini(prompt, api_key)
    except RuntimeError as exc:
        print(f"\nLLM call failed: {exc}")
        print(f"Saved player input JSON: {player_dir / 'player_input.json'}")
        print(f"Saved LLM request JSON: {player_dir / 'player_llm_request.json'}")
        print("Please rerun and paste a valid Gemini API key if needed.")
        return 1
    (player_dir / "player_llm_analysis_raw.txt").write_text(response_text, encoding="utf-8")
    try:
        parsed = json.loads(response_text)
        (player_dir / "player_llm_analysis.json").write_text(
            json.dumps(parsed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"saved {player_dir / 'player_llm_analysis.json'}")
    except json.JSONDecodeError:
        (player_dir / "player_llm_analysis_error.txt").write_text(
            "Gemini did not return valid JSON.", encoding="utf-8"
        )
        print(f"saved raw response {player_dir / 'player_llm_analysis_raw.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


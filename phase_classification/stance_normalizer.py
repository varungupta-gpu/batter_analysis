from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


LEFT_RIGHT_PAIRS = [
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]

STANCE_PAIRS = [
    ("left_wrist", "right_wrist", 3.0),
    ("left_ankle", "right_ankle", 3.0),
    ("left_knee", "right_knee", 2.0),
    ("left_elbow", "right_elbow", 1.5),
    ("left_shoulder", "right_shoulder", 1.0),
    ("left_hip", "right_hip", 1.0),
]


def _valid_coordinate(value: object) -> float | None:
    try:
        coordinate = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(coordinate) or coordinate <= 0:
        return None
    return coordinate


def _pair_offset(row: dict, left_name: str, right_name: str) -> float | None:
    left_x = _valid_coordinate(row.get(f"{left_name}_x"))
    right_x = _valid_coordinate(row.get(f"{right_name}_x"))
    if left_x is None or right_x is None:
        return None
    offset = left_x - right_x
    if abs(offset) < 2.0:
        return None
    return offset


def _depth_offset(row: dict, left_name: str, right_name: str) -> float | None:
    left_y = _valid_coordinate(row.get(f"{left_name}_y"))
    right_y = _valid_coordinate(row.get(f"{right_name}_y"))
    if left_y is None or right_y is None:
        return None
    offset = left_y - right_y
    if abs(offset) < 0.5:
        return None
    return offset


def _front_depth_score(row: dict) -> tuple[float, int]:
    score = 0.0
    votes = 0
    for left_name, right_name, weight in STANCE_PAIRS:
        offset = _depth_offset(row, left_name, right_name)
        if offset is None:
            continue
        score += (1.0 if offset > 0 else -1.0) * weight
        votes += 1
    return score, votes


def _horizontal_side_score(row: dict) -> tuple[float, int]:
    score = 0.0
    votes = 0
    for left_name, right_name, weight in STANCE_PAIRS:
        offset = _pair_offset(row, left_name, right_name)
        if offset is None:
            continue
        score += (1.0 if offset > 0 else -1.0) * weight
        votes += 1
    return score, votes


def detect_stance_from_shoulders(rows: list[dict]) -> str:
    """Infer stance from the first reliable detected pose.

    In the front-on/bowler-end camera, the front hand and leg sit lower in the
    image. Right-handed batters present the left side in front; left-handed
    batters present the right side. Horizontal ordering is only a fallback.
    """
    depth_fallback_score = 0.0
    depth_fallback_votes = 0
    horizontal_fallback_score = 0.0
    horizontal_fallback_votes = 0

    for row in rows:
        depth_score, depth_votes = _front_depth_score(row)
        if depth_votes >= 2:
            return "right" if depth_score >= 0 else "left"
        depth_fallback_score += depth_score
        depth_fallback_votes += depth_votes

        horizontal_score, horizontal_votes = _horizontal_side_score(row)
        horizontal_fallback_score += horizontal_score
        horizontal_fallback_votes += horizontal_votes

    if depth_fallback_votes:
        return "right" if depth_fallback_score >= 0 else "left"
    if horizontal_fallback_votes:
        return "right" if horizontal_fallback_score >= 0 else "left"
    return "right"


def normalize_rows_to_right_handed(rows: list[dict]) -> tuple[list[dict], str]:
    """Return pose rows normalized to the pipeline's right-handed convention."""
    normalized = [dict(row) for row in rows]
    detected_stance = detect_stance_from_shoulders(normalized)
    if detected_stance == "right":
        return normalized, detected_stance

    for row in normalized:
        for left_name, right_name in LEFT_RIGHT_PAIRS:
            for axis in ("x", "y"):
                left_key = f"{left_name}_{axis}"
                right_key = f"{right_name}_{axis}"
                row[left_key], row[right_key] = row.get(right_key, ""), row.get(left_key, "")
    return normalized, detected_stance


def normalize_keypoint_csv_to_right_handed(path: Path) -> Path:
    """Normalize a named-keypoint CSV in place, preserving its schema."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or not rows:
        return path

    normalized, detected_stance = normalize_rows_to_right_handed(rows)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)
    print(f"[STANCE] Detected {detected_stance}-handed batsman; saved right-handed normalized keypoints: {path}")
    return path


def validate_right_handed_keypoint_csv(path: Path, min_detected_frames: int = 1) -> str:
    """Validate that a keypoint CSV is non-empty and normalized right-handed."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    if not rows:
        raise ValueError(f"No keypoint rows found: {path}")

    detected_rows = [
        row for row in rows
        if any(
            _valid_coordinate(row.get(f"{name}_x")) is not None
            and _valid_coordinate(row.get(f"{name}_y")) is not None
            for name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle", "left_shoulder", "right_shoulder")
        )
    ]
    if len(detected_rows) < min_detected_frames:
        raise ValueError(
            f"Only {len(detected_rows)} detected pose frame(s) found in {path}; "
            f"need at least {min_detected_frames}."
        )

    detected_stance = detect_stance_from_shoulders(detected_rows)
    if detected_stance != "right":
        raise ValueError(
            f"Keypoints are not normalized to right-handed convention: {path} "
            f"(detected {detected_stance})."
        )
    return detected_stance

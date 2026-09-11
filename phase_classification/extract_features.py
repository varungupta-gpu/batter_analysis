from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


# FPS NORMALIZATION: All speed features are normalized to 30 FPS
# This ensures consistent feature scales across videos with different frame rates
# Input videos can be 25, 30, 60 FPS etc. but features are always scaled to 30 FPS
NORMALIZED_FPS = 30.0

# DISTANCE/LENGTH FEATURES TO EXCLUDE
# These features depend on camera distance/scale and should not be used for classification
DISTANCE_FEATURES_TO_EXCLUDE = [
    "left_arm_length",
    "right_arm_length",
    "left_forearm_length",
    "right_forearm_length",
    "left_shoulder_wrist_dist",
    "right_shoulder_wrist_dist",
    "bat_arm_extension",
    "eye_separation",
    "ear_separation",
]


# All 17 COCO keypoints (includes eyes, ears, elbows)
COCO_POINTS = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# For backward compatibility, map old names to COCO indices if needed
POINTS = COCO_POINTS


def _xy(row: dict, name: str) -> np.ndarray:
    x = row.get(f"{name}_x")
    y = row.get(f"{name}_y")
    if x in (None, "") or y in (None, ""):
        return np.array([np.nan, np.nan], dtype=float)
    return np.array([float(x), float(y)], dtype=float)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    # Euclidean distance between two 2D points
    if np.isnan(a).any() or np.isnan(b).any():
        return 0.0
    return float(np.linalg.norm(a - b))


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    # Angle from point a to point b (in degrees)
    if np.isnan(a).any() or np.isnan(b).any():
        return 0.0
    return float(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # Angle at joint b between rays ba and bc (e.g., elbow angle)
    if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
        return 0.0
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos_value = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(math.degrees(math.acos(cos_value)))


def _speed(current: np.ndarray, previous: np.ndarray | None, fps: float, normalize_to_fps: float = NORMALIZED_FPS) -> float:
    # Frame-to-frame velocity (normalized to standard FPS for consistency)
    # Always normalizes to 30 FPS for training/testing consistency regardless of input video FPS
    if previous is None:
        return 0.0
    return _dist(current, previous) * normalize_to_fps


def extract_features(keypoints_csv: Path, output_csv: Path | None = None) -> Path:
    # Extract per-frame features (speeds, angles, positions) from keypoints
    # All speed features are normalized to 30 FPS for consistency across different video frame rates
    output_csv = output_csv or Path("data/features") / keypoints_csv.name.replace("_keypoints.csv", "_features.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with keypoints_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return output_csv

    # Read original FPS but normalize all speeds to 30 FPS
    original_fps = float(rows[0].get("fps") or 25.0)
    print(f"[INFO] Original video FPS: {original_fps:.1f} â†’ Normalizing all speed features to {NORMALIZED_FPS:.1f} FPS")

    previous: dict[str, np.ndarray] | None = None
    feature_rows = []

    for row in rows:
        fps = float(row.get("fps") or 25.0)  # Read FPS from row
        pts = {name: _xy(row, name) for name in COCO_POINTS}
        prev = previous or {}

        # === ARM FEATURES (Priority) ===
        # Arm speeds (normalized to 30 FPS for consistency)
        left_wrist_speed = _speed(pts["left_wrist"], prev.get("left_wrist"), fps)
        right_wrist_speed = _speed(pts["right_wrist"], prev.get("right_wrist"), fps)
        left_elbow_speed = _speed(pts["left_elbow"], prev.get("left_elbow"), fps)
        right_elbow_speed = _speed(pts["right_elbow"], prev.get("right_elbow"), fps)

        # Elbow angles relative to shoulder
        left_elbow_angle = _angle(pts["left_shoulder"], pts["left_elbow"])
        right_elbow_angle = _angle(pts["right_shoulder"], pts["right_elbow"])

        # Joint angles (bend at each joint)
        left_elbow_joint_angle = _joint_angle(pts["left_shoulder"], pts["left_elbow"], pts["left_wrist"])
        right_elbow_joint_angle = _joint_angle(pts["right_shoulder"], pts["right_elbow"], pts["right_wrist"])
        left_wrist_joint_angle = _joint_angle(pts["left_elbow"], pts["left_wrist"], pts["left_wrist"])  # Proxy
        right_wrist_joint_angle = _joint_angle(pts["right_elbow"], pts["right_wrist"], pts["right_wrist"])  # Proxy

        # Arm dimensions
        left_arm_length = _dist(pts["left_shoulder"], pts["left_wrist"])
        right_arm_length = _dist(pts["right_shoulder"], pts["right_wrist"])
        left_forearm_length = _dist(pts["left_elbow"], pts["left_wrist"])
        right_forearm_length = _dist(pts["right_elbow"], pts["right_wrist"])

        # Shoulder-wrist distance (arm extension)
        left_shoulder_wrist_dist = _dist(pts["left_shoulder"], pts["left_wrist"])
        right_shoulder_wrist_dist = _dist(pts["right_shoulder"], pts["right_wrist"])

        # Wrist height above shoulder
        left_wrist_height_rel = pts["left_shoulder"][1] - pts["left_wrist"][1]
        right_wrist_height_rel = pts["right_shoulder"][1] - pts["right_wrist"][1]

        # Batting arm features (right arm for right-handed batsman)
        bat_arm_wrist_speed = right_wrist_speed
        bat_arm_elbow_speed = right_elbow_speed
        bat_arm_elbow_angle = right_elbow_joint_angle
        bat_arm_extension = right_arm_length

        # === BODY ROTATION & CORE FEATURES ===
        # Shoulder/hip motion speeds
        left_shoulder_speed = _speed(pts["left_shoulder"], prev.get("left_shoulder"), fps)
        right_shoulder_speed = _speed(pts["right_shoulder"], prev.get("right_shoulder"), fps)
        left_hip_speed = _speed(pts["left_hip"], prev.get("left_hip"), fps)
        right_hip_speed = _speed(pts["right_hip"], prev.get("right_hip"), fps)

        # Shoulder and hip line angles (rotation around vertical axis)
        shoulder_line_angle = _angle(pts["left_shoulder"], pts["right_shoulder"])
        hip_line_angle = _angle(pts["left_hip"], pts["right_hip"])

        # Trunk positioning
        shoulders_mid = np.nanmean([pts["left_shoulder"], pts["right_shoulder"]], axis=0)
        hips_mid = np.nanmean([pts["left_hip"], pts["right_hip"]], axis=0)
        body_center = np.nanmean([shoulders_mid, hips_mid], axis=0)
        trunk_lean = _angle(hips_mid, shoulders_mid)

        prev_body_center = None
        if previous:
            prev_shoulders_mid = np.nanmean([previous["left_shoulder"], previous["right_shoulder"]], axis=0)
            prev_hips_mid = np.nanmean([previous["left_hip"], previous["right_hip"]], axis=0)
            prev_body_center = np.nanmean([prev_shoulders_mid, prev_hips_mid], axis=0)

        # === LEG FEATURES ===
        # Leg motion speeds
        left_knee_speed = _speed(pts["left_knee"], prev.get("left_knee"), fps)
        right_knee_speed = _speed(pts["right_knee"], prev.get("right_knee"), fps)
        left_ankle_speed = _speed(pts["left_ankle"], prev.get("left_ankle"), fps)
        right_ankle_speed = _speed(pts["right_ankle"], prev.get("right_ankle"), fps)

        # Knee bend angles
        left_knee_angle = _joint_angle(pts["left_hip"], pts["left_knee"], pts["left_ankle"])
        right_knee_angle = _joint_angle(pts["right_hip"], pts["right_knee"], pts["right_ankle"])

        # === HEAD & EYE FEATURES ===
        head_speed = _speed(pts["nose"], prev.get("nose"), fps)
        left_eye_speed = _speed(pts["left_eye"], prev.get("left_eye"), fps)
        right_eye_speed = _speed(pts["right_eye"], prev.get("right_eye"), fps)
        eye_separation = _dist(pts["left_eye"], pts["right_eye"])

        # === EAR FEATURES (Balance & Head Orientation) ===
        left_ear_speed = _speed(pts["left_ear"], prev.get("left_ear"), fps)
        right_ear_speed = _speed(pts["right_ear"], prev.get("right_ear"), fps)
        ear_separation = _dist(pts["left_ear"], pts["right_ear"])

        # Head orientation (based on eyes and ears)
        head_orientation_angle = _angle(pts["left_eye"], pts["right_eye"])

        previous_shoulder_angle = shoulder_line_angle
        previous_hip_angle = hip_line_angle
        if previous:
            previous_shoulder_angle = _angle(previous["left_shoulder"], previous["right_shoulder"])
            previous_hip_angle = _angle(previous["left_hip"], previous["right_hip"])

        # === COMPOSITE FEATURES ===
        wrist_height_change = 0.0
        if previous:
            wrist_height_change = float(
                np.nanmean([previous["left_wrist"][1], previous["right_wrist"][1]])
                - np.nanmean([pts["left_wrist"][1], pts["right_wrist"][1]])
            )

        # Arm activity score (all arm speeds)
        arm_activity_score = float(np.mean([
            left_wrist_speed, right_wrist_speed,
            left_elbow_speed, right_elbow_speed,
            left_shoulder_speed, right_shoulder_speed
        ]))

        # Full body activity score
        pose_activity_score = float(np.mean([
            left_wrist_speed, right_wrist_speed,
            left_elbow_speed, right_elbow_speed,
            left_shoulder_speed, right_shoulder_speed,
            left_hip_speed, right_hip_speed,
            left_knee_speed, right_knee_speed,
            left_ankle_speed, right_ankle_speed
        ]))

        feature_rows.append({
            # === METADATA ===
            "video_id": row["video_id"],
            "frame_no": row["frame_no"],

            # === ARM FEATURES (Priority) ===
            "left_wrist_speed": left_wrist_speed,
            "right_wrist_speed": right_wrist_speed,
            "wrist_speed_max": max(left_wrist_speed, right_wrist_speed),
            "left_elbow_speed": left_elbow_speed,
            "right_elbow_speed": right_elbow_speed,
            "elbow_speed_max": max(left_elbow_speed, right_elbow_speed),
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "left_elbow_joint_angle": left_elbow_joint_angle,
            "right_elbow_joint_angle": right_elbow_joint_angle,
            # EXCLUDED: left_arm_length, right_arm_length, left_forearm_length, right_forearm_length
            # EXCLUDED: left_shoulder_wrist_dist, right_shoulder_wrist_dist
            "left_wrist_height_rel": left_wrist_height_rel,
            "right_wrist_height_rel": right_wrist_height_rel,
            "bat_arm_wrist_speed": bat_arm_wrist_speed,
            "bat_arm_elbow_speed": bat_arm_elbow_speed,
            "bat_arm_elbow_angle": bat_arm_elbow_angle,
            # EXCLUDED: bat_arm_extension
            "arm_activity_score": arm_activity_score,

            # === BODY ROTATION & CORE FEATURES ===
            "shoulder_line_angle": shoulder_line_angle,
            "hip_line_angle": hip_line_angle,
            "hip_shoulder_separation": abs(shoulder_line_angle - hip_line_angle),
            "trunk_lean": trunk_lean,
            "shoulder_rotation_change": shoulder_line_angle - previous_shoulder_angle,
            "left_shoulder_speed": left_shoulder_speed,
            "right_shoulder_speed": right_shoulder_speed,
            "shoulder_speed_max": max(left_shoulder_speed, right_shoulder_speed),
            "body_center_x": 0.0 if np.isnan(body_center[0]) else float(body_center[0]),
            "body_center_y": 0.0 if np.isnan(body_center[1]) else float(body_center[1]),
            "body_center_speed": _speed(body_center, prev_body_center, fps),
            "upper_body_rotation": shoulder_line_angle,
            "lower_body_rotation": hip_line_angle,

            # === LEG FEATURES ===
            "left_knee_speed": left_knee_speed,
            "right_knee_speed": right_knee_speed,
            "knee_speed_max": max(left_knee_speed, right_knee_speed),
            "left_ankle_speed": left_ankle_speed,
            "right_ankle_speed": right_ankle_speed,
            "ankle_speed_max": max(left_ankle_speed, right_ankle_speed),
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            "front_knee_angle": right_knee_angle,
            "back_knee_angle": left_knee_angle,
            "left_hip_speed": left_hip_speed,
            "right_hip_speed": right_hip_speed,
            "hip_speed_max": max(left_hip_speed, right_hip_speed),

            # === HEAD & EYE FEATURES ===
            "head_movement": head_speed,
            "head_speed": head_speed,
            "left_eye_speed": left_eye_speed,
            "right_eye_speed": right_eye_speed,
            # EXCLUDED: eye_separation
            "head_orientation_angle": head_orientation_angle,

            # === EAR FEATURES ===
            "left_ear_speed": left_ear_speed,
            "right_ear_speed": right_ear_speed,
            # EXCLUDED: ear_separation

            # === COMPOSITE FEATURES ===
            "wrist_height_change": wrist_height_change,
            "pose_activity_score": pose_activity_score,
            "impact_like_frame_flag": int(max(left_wrist_speed, right_wrist_speed) > 600),
        })
        previous = pts

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()) if feature_rows else ["video_id", "frame_no"])
        writer.writeheader()
        writer.writerows(feature_rows)
    print(f"Saved features to: {output_csv}")
    return output_csv

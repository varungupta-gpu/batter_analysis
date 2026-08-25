import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


JOINT_ALIASES: Dict[str, List[str]] = {
    "nose": ["nose"],
    "neck": ["neck", "mid_shoulder", "base_neck"],
    "left_shoulder": ["left_shoulder", "l_shoulder"],
    "right_shoulder": ["right_shoulder", "r_shoulder"],
    "left_elbow": ["left_elbow", "l_elbow"],
    "right_elbow": ["right_elbow", "r_elbow"],
    "left_wrist": ["left_wrist", "l_wrist"],
    "right_wrist": ["right_wrist", "r_wrist"],
    "left_hip": ["left_hip", "l_hip"],
    "right_hip": ["right_hip", "r_hip"],
    "left_knee": ["left_knee", "l_knee"],
    "right_knee": ["right_knee", "r_knee"],
    "left_ankle": ["left_ankle", "l_ankle"],
    "right_ankle": ["right_ankle", "r_ankle"],
    "left_heel": ["left_heel", "l_heel", "left_backfoot", "l_heel_back"],
    "right_heel": ["right_heel", "r_heel", "right_backfoot", "r_heel_back"],
    "left_toe": [
        "left_toe",
        "left_foot_index",
        "l_toe",
        "l_foot_index",
        "left_big_toe",
    ],
    "right_toe": [
        "right_toe",
        "right_foot_index",
        "r_toe",
        "r_foot_index",
        "right_big_toe",
    ],
}


RAW_LEFT_RIGHT_SWAP_PAIRS = [
    ("Left elbow angle", "Right elbow angle"),
    ("Left shoulder angle", "Right shoulder angle"),
    ("Left hip angle", "Right hip angle"),
    ("Left knee angle", "Right knee angle"),
    ("Left ankle angle", "Right ankle angle"),
]


PHASE_TO_ID = {
    "no_phase": 0,
    "bbc": 1,
    "ffc": 2,
}

ID_TO_PHASE = {value: key for key, value in PHASE_TO_ID.items()}


def load_json_or_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix.lower() == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("data", "results", "items", "rows"):
                if isinstance(payload.get(key), list):
                    return pd.DataFrame(payload[key])
            return pd.DataFrame([payload])
    return pd.read_csv(file_path)


def _snake_token(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    return "_".join(part for part in cleaned.split("_") if part)


def _column_lookup(columns: Iterable[str]) -> Dict[str, str]:
    return {_snake_token(column): column for column in columns}


def _find_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lookup = _column_lookup(columns)
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _frame_column(df: pd.DataFrame) -> str:
    column = _find_existing_column(df.columns, ["frame", "frame_id", "frame_idx", "frame_number"])
    if column:
        return column
    return df.columns[0]


def _pivot_long_keypoints(df: pd.DataFrame) -> pd.DataFrame:
    keypoint_column = _find_existing_column(df.columns, ["keypoint", "joint", "landmark", "name"])
    x_column = _find_existing_column(df.columns, ["x", "x_px", "pixel_x", "x_coordinate"])
    y_column = _find_existing_column(df.columns, ["y", "y_px", "pixel_y", "y_coordinate"])
    if not keypoint_column or not x_column or not y_column:
        return df

    frame_column = _frame_column(df)
    wide = (
        df[[frame_column, keypoint_column, x_column, y_column]]
        .assign(_joint=lambda x: x[keypoint_column].map(_snake_token))
        .pivot_table(index=frame_column, columns="_joint", values=[x_column, y_column], aggfunc="first")
    )
    wide.columns = [
        f"{joint}_{'x' if value_type == x_column else 'y'}"
        for value_type, joint in wide.columns.to_flat_index()
    ]
    wide = wide.reset_index().rename(columns={frame_column: "frame"})
    return wide


def normalize_keypoint_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = _pivot_long_keypoints(raw_df.copy())
    frame_column = _frame_column(df)
    if frame_column != "frame":
        df = df.rename(columns={frame_column: "frame"})

    output = pd.DataFrame({"frame": pd.to_numeric(df["frame"], errors="coerce")})
    for joint_name, aliases in JOINT_ALIASES.items():
        x_candidates = [f"{alias}_x" for alias in aliases] + [f"x_{alias}" for alias in aliases]
        y_candidates = [f"{alias}_y" for alias in aliases] + [f"y_{alias}" for alias in aliases]
        x_column = _find_existing_column(df.columns, x_candidates)
        y_column = _find_existing_column(df.columns, y_candidates)
        if x_column:
            output[f"{joint_name}_x"] = pd.to_numeric(df[x_column], errors="coerce")
        if y_column:
            output[f"{joint_name}_y"] = pd.to_numeric(df[y_column], errors="coerce")

    output = output.sort_values("frame").drop_duplicates("frame").reset_index(drop=True)
    return output


def load_keypoints_csv(path: str) -> pd.DataFrame:
    raw_df = load_json_or_csv(path)
    normalized = normalize_keypoint_dataframe(raw_df)
    if normalized.empty:
        raise ValueError(f"No usable keypoints found in {path}")
    return normalized


def load_release_info(path: str) -> Dict[str, float]:
    release_df = load_json_or_csv(path)
    if release_df.empty:
        raise ValueError(f"No release-point data found in {path}")
    row = release_df.iloc[0].to_dict()
    normalized = {_snake_token(key): value for key, value in row.items()}
    release_info = {}
    for key in ("release_frame", "frame", "frame_id", "frame_idx"):
        if key in normalized and pd.notna(normalized[key]):
            release_info["release_frame"] = float(normalized[key])
            break
    for key in ("release_point_x", "x", "release_x", "ball_release_x"):
        if key in normalized and pd.notna(normalized[key]):
            release_info["release_point_x"] = float(normalized[key])
            break
    for key in ("release_point_y", "y", "release_y", "ball_release_y"):
        if key in normalized and pd.notna(normalized[key]):
            release_info["release_point_y"] = float(normalized[key])
            break
    return release_info


def build_release_info_from_frame(release_frame: float) -> Dict[str, float]:
    return {"release_frame": float(release_frame)}


def _point(row: pd.Series, prefix: str) -> np.ndarray:
    return np.array([row.get(f"{prefix}_x", np.nan), row.get(f"{prefix}_y", np.nan)], dtype=float)


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0], dtype=float)


def _safe_norm(vector: np.ndarray) -> float:
    if np.isnan(vector).any():
        return float("nan")
    return float(np.linalg.norm(vector))


def _angle_three_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
        return float("nan")
    ba = a - b
    bc = c - b
    denom = _safe_norm(ba) * _safe_norm(bc)
    if denom == 0 or math.isnan(denom):
        return float("nan")
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    if np.isnan(v1).any() or np.isnan(v2).any():
        return float("nan")
    denom = _safe_norm(v1) * _safe_norm(v2)
    if denom == 0 or math.isnan(denom):
        return float("nan")
    cosine = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _orientation_degrees(start: np.ndarray, end: np.ndarray) -> float:
    if np.isnan(start).any() or np.isnan(end).any():
        return float("nan")
    vector = end - start
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def _line_tilt_degrees(start: np.ndarray, end: np.ndarray) -> float:
    angle = _orientation_degrees(start, end)
    if math.isnan(angle):
        return angle
    while angle <= -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    if np.isnan(a).any() or np.isnan(b).any():
        return float("nan")
    return float(np.linalg.norm(a - b))


def infer_bowling_hand(keypoints_df: pd.DataFrame, release_info: Dict[str, float]) -> str:
    release_frame = release_info.get("release_frame")
    release_x = release_info.get("release_point_x")
    release_y = release_info.get("release_point_y")
    if release_frame is not None:
        target_index = (keypoints_df["frame"] - release_frame).abs().idxmin()
        row = keypoints_df.loc[target_index]
    else:
        row = keypoints_df.iloc[len(keypoints_df) // 2]

    left_wrist = _point(row, "left_wrist")
    right_wrist = _point(row, "right_wrist")

    if release_x is not None and release_y is not None:
        release_point = np.array([release_x, release_y], dtype=float)
        left_distance = _distance(left_wrist, release_point)
        right_distance = _distance(right_wrist, release_point)
        if math.isnan(left_distance):
            return "right"
        if math.isnan(right_distance):
            return "left"
        return "left" if left_distance < right_distance else "right"

    left_shoulder = _point(row, "left_shoulder")
    right_shoulder = _point(row, "right_shoulder")
    left_extension = _distance(left_wrist, left_shoulder)
    right_extension = _distance(right_wrist, right_shoulder)
    if math.isnan(left_extension):
        return "right"
    if math.isnan(right_extension):
        return "left"
    return "left" if left_extension > right_extension else "right"


def _foot_orientation(row: pd.Series, side: str) -> float:
    heel = _point(row, f"{side}_heel")
    toe = _point(row, f"{side}_toe")
    if np.isnan(heel).any() or np.isnan(toe).any():
        ankle = _point(row, f"{side}_ankle")
        hip = _point(row, f"{side}_hip")
        if np.isnan(ankle).any() or np.isnan(hip).any():
            return float("nan")
        return _orientation_degrees(hip, ankle)
    return _orientation_degrees(heel, toe)


def _compute_single_frame_features(row: pd.Series, bowling_hand: str) -> Dict[str, float]:
    left_side = "left"
    right_side = "right"
    bowling_side = "left" if bowling_hand == "left" else "right"
    non_bowling_side = "right" if bowling_side == "left" else "left"
    front_side = non_bowling_side
    back_side = bowling_side

    left_shoulder = _point(row, "left_shoulder")
    right_shoulder = _point(row, "right_shoulder")
    left_elbow = _point(row, "left_elbow")
    right_elbow = _point(row, "right_elbow")
    left_wrist = _point(row, "left_wrist")
    right_wrist = _point(row, "right_wrist")
    left_hip = _point(row, "left_hip")
    right_hip = _point(row, "right_hip")
    left_knee = _point(row, "left_knee")
    right_knee = _point(row, "right_knee")
    left_ankle = _point(row, "left_ankle")
    right_ankle = _point(row, "right_ankle")
    nose = _point(row, "nose")
    neck = _point(row, "neck")

    mid_shoulder = _midpoint(left_shoulder, right_shoulder)
    mid_hip = _midpoint(left_hip, right_hip)
    trunk_vector = mid_shoulder - mid_hip
    head_vector = nose - neck

    bowling_shoulder = _point(row, f"{bowling_side}_shoulder")
    bowling_elbow = _point(row, f"{bowling_side}_elbow")
    bowling_wrist = _point(row, f"{bowling_side}_wrist")
    non_bowling_shoulder = _point(row, f"{non_bowling_side}_shoulder")
    non_bowling_elbow = _point(row, f"{non_bowling_side}_elbow")
    non_bowling_wrist = _point(row, f"{non_bowling_side}_wrist")

    features = {
        "frame": float(row["frame"]),
        "Left elbow angle": _angle_three_points(left_shoulder, left_elbow, left_wrist),
        "Right elbow angle": _angle_three_points(right_shoulder, right_elbow, right_wrist),
        "Left shoulder angle": _angle_three_points(left_elbow, left_shoulder, left_hip),
        "Right shoulder angle": _angle_three_points(right_elbow, right_shoulder, right_hip),
        "Bowling-arm elevation angle": _angle_between(
            bowling_wrist - bowling_shoulder, np.array([0.0, -1.0], dtype=float)
        ),
        "Bowling-arm elbow extension angle": _angle_three_points(
            bowling_shoulder, bowling_elbow, bowling_wrist
        ),
        "Non-bowling-arm elevation angle": _angle_between(
            non_bowling_wrist - non_bowling_shoulder, np.array([0.0, -1.0], dtype=float)
        ),
        "Shoulder-line tilt angle": _line_tilt_degrees(left_shoulder, right_shoulder),
        "Hip-line tilt angle": _line_tilt_degrees(left_hip, right_hip),
        "Hip-shoulder separation angle": abs(
            _line_tilt_degrees(left_shoulder, right_shoulder)
            - _line_tilt_degrees(left_hip, right_hip)
        ),
        "Trunk lateral-flexion angle": _angle_between(trunk_vector, np.array([0.0, -1.0], dtype=float)),
        "Trunk rotation angle": _angle_between(
            right_shoulder - left_shoulder, right_hip - left_hip
        ),
        "Neck/head tilt angle": _angle_between(head_vector, np.array([0.0, -1.0], dtype=float)),
        "Left hip angle": _angle_three_points(left_shoulder, left_hip, left_knee),
        "Right hip angle": _angle_three_points(right_shoulder, right_hip, right_knee),
        "Left knee angle": _angle_three_points(left_hip, left_knee, left_ankle),
        "Right knee angle": _angle_three_points(right_hip, right_knee, right_ankle),
        "Left ankle angle": _angle_between(left_knee - left_ankle, np.array([1.0, 0.0], dtype=float)),
        "Right ankle angle": _angle_between(right_knee - right_ankle, np.array([1.0, 0.0], dtype=float)),
        "Front-leg knee angle": _angle_three_points(
            _point(row, f"{front_side}_hip"),
            _point(row, f"{front_side}_knee"),
            _point(row, f"{front_side}_ankle"),
        ),
        "Back-leg knee angle": _angle_three_points(
            _point(row, f"{back_side}_hip"),
            _point(row, f"{back_side}_knee"),
            _point(row, f"{back_side}_ankle"),
        ),
        "Front-foot orientation angle": _foot_orientation(row, front_side),
        "Back-foot orientation angle": _foot_orientation(row, back_side),
        "Pelvis rotation angle": _orientation_degrees(left_hip, right_hip),
        "Shoulder-pelvis alignment angle": abs(
            _orientation_degrees(left_shoulder, right_shoulder)
            - _orientation_degrees(left_hip, right_hip)
        ),
        "Head-trunk alignment angle": _angle_between(head_vector, trunk_vector),
        "Front-knee-hip alignment angle": _angle_between(
            _point(row, f"{front_side}_knee") - _point(row, f"{front_side}_hip"),
            np.array([0.0, 1.0], dtype=float),
        ),
        "Back-knee-hip alignment angle": _angle_between(
            _point(row, f"{back_side}_knee") - _point(row, f"{back_side}_hip"),
            np.array([0.0, 1.0], dtype=float),
        ),
        "Bowling-arm-to-trunk angle": _angle_between(bowling_wrist - bowling_shoulder, trunk_vector),
        "Non-bowling-arm-to-trunk angle": _angle_between(
            non_bowling_wrist - non_bowling_shoulder, trunk_vector
        ),
        "Inter-thigh angle": _angle_between(left_knee - left_hip, right_knee - right_hip),
        "Inter-shank angle": _angle_between(left_ankle - left_knee, right_ankle - right_knee),
        "Arm-to-arm separation angle": _angle_between(
            left_wrist - left_shoulder, right_wrist - right_shoulder
        ),
        "bowling_hand": bowling_hand,
    }

    if bowling_hand == "left":
        for left_feature, right_feature in RAW_LEFT_RIGHT_SWAP_PAIRS:
            features[left_feature], features[right_feature] = features[right_feature], features[left_feature]

    return features


def compute_features_dataframe(keypoints_df: pd.DataFrame, release_info: Dict[str, float]) -> pd.DataFrame:
    bowling_hand = infer_bowling_hand(keypoints_df, release_info)
    feature_rows = [
        _compute_single_frame_features(row, bowling_hand)
        for _, row in keypoints_df.iterrows()
    ]
    feature_df = pd.DataFrame(feature_rows)
    numeric_columns = [column for column in feature_df.columns if column not in {"bowling_hand"}]
    feature_df[numeric_columns] = feature_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    feature_df[numeric_columns] = feature_df[numeric_columns].interpolate(limit_direction="both")
    feature_df[numeric_columns] = feature_df[numeric_columns].ffill().bfill()
    return feature_df


def temporal_expand(feature_df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    expanded = feature_df[["frame", "bowling_hand"]].copy()
    base_columns = [column for column in feature_df.columns if column not in {"frame", "bowling_hand"}]
    for offset in range(-window, window + 1):
        shifted = feature_df[base_columns].shift(-offset)
        suffix = f"t{offset:+d}"
        shifted.columns = [f"{column}_{suffix}" for column in base_columns]
        expanded = pd.concat([expanded, shifted], axis=1)
    expanded = expanded.ffill().bfill()
    return expanded


def attach_binary_phase_labels(
    feature_df: pd.DataFrame,
    bbc_start: int,
    bbc_end: int,
    ffc_start: int,
    ffc_end: int,
) -> pd.DataFrame:
    labeled = feature_df.copy()
    labeled["phase"] = "no_phase"
    labeled.loc[(labeled["frame"] >= bbc_start) & (labeled["frame"] <= bbc_end), "phase"] = "bbc"
    labeled.loc[(labeled["frame"] >= ffc_start) & (labeled["frame"] <= ffc_end), "phase"] = "ffc"
    labeled["phase_id"] = labeled["phase"].map(PHASE_TO_ID)
    return labeled


def smooth_phase_sequence(phases: List[str], gap_threshold: int = 2) -> List[str]:
    if not phases:
        return phases

    smoothed = list(phases)

    start = 0
    while start < len(smoothed):
        end = start
        while end + 1 < len(smoothed) and smoothed[end + 1] == smoothed[start]:
            end += 1

        run_length = end - start + 1
        prev_phase = smoothed[start - 1] if start > 0 else None
        next_phase = smoothed[end + 1] if end + 1 < len(smoothed) else None
        if (
            run_length <= gap_threshold
            and prev_phase is not None
            and next_phase is not None
            and prev_phase == next_phase
        ):
            for idx in range(start, end + 1):
                smoothed[idx] = prev_phase
        start = end + 1

    return smoothed


def postprocess_original_phase_output(
    frame_phase_df: pd.DataFrame,
    release_frame: Optional[float] = None,
) -> pd.DataFrame:
    output_df = frame_phase_df.copy().sort_values("frame").reset_index(drop=True)
    smoothed = smooth_phase_sequence(output_df["predicted_phase"].tolist(), gap_threshold=2)
    output_df["smoothed_model_phase"] = smoothed
    output_df["original_phase"] = "load_up"

    frames = pd.to_numeric(output_df["frame"], errors="coerce")
    valid_frame_mask = frames.notna()
    if not valid_frame_mask.any():
        return output_df

    min_frame = int(frames[valid_frame_mask].min())
    max_frame = int(frames[valid_frame_mask].max())
    release_int = int(round(release_frame)) if release_frame is not None and not pd.isna(release_frame) else None
    ffc_end_frame = min(max_frame, release_int + 2) if release_int is not None else None

    def _idx_for_strongest_probability(phase_name: str, candidate_mask: pd.Series) -> Optional[int]:
        prob_column = f"prob_{phase_name}"
        if prob_column not in output_df.columns:
            return None
        candidate_rows = output_df.loc[candidate_mask & output_df[prob_column].notna()]
        if candidate_rows.empty:
            return None
        return int(candidate_rows[prob_column].idxmax())

    ffc_window_mask = valid_frame_mask.copy()
    if release_int is not None:
        ffc_window_mask &= (frames >= release_int - 12) & (frames <= release_int + 2)

    ffc_idx = next(
        (idx for idx, phase in enumerate(smoothed) if phase == "ffc" and bool(ffc_window_mask.iloc[idx])),
        None,
    )
    if ffc_idx is None:
        ffc_idx = _idx_for_strongest_probability("ffc", ffc_window_mask)
    if ffc_idx is None and release_int is not None:
        ffc_idx = int((frames - release_int).abs().idxmin())

    ffc_start_frame = int(frames.iloc[ffc_idx]) if ffc_idx is not None and pd.notna(frames.iloc[ffc_idx]) else None
    if ffc_start_frame is None:
        return output_df

    before_ffc_mask = valid_frame_mask & (frames < ffc_start_frame)
    bbc_idx = next(
        (idx for idx, phase in enumerate(smoothed) if phase == "bbc" and bool(before_ffc_mask.iloc[idx])),
        None,
    )
    if bbc_idx is None:
        bbc_idx = _idx_for_strongest_probability("bbc", before_ffc_mask)
    if bbc_idx is None:
        previous_frames = output_df.loc[before_ffc_mask, "frame"]
        if not previous_frames.empty:
            bbc_idx = int(previous_frames.index.max())

    bbc_start_frame = int(frames.iloc[bbc_idx]) if bbc_idx is not None and pd.notna(frames.iloc[bbc_idx]) else None
    if bbc_start_frame is None:
        bbc_start_frame = ffc_start_frame

    if bbc_start_frame > ffc_start_frame:
        bbc_start_frame = ffc_start_frame

    output_df["original_phase"] = "load_up"
    output_df.loc[
        (frames >= bbc_start_frame) & (frames < ffc_start_frame),
        "original_phase",
    ] = "bbc"
    output_df.loc[frames >= ffc_start_frame, "original_phase"] = "ffc"

    if ffc_end_frame is not None:
        output_df.loc[frames > ffc_end_frame, "original_phase"] = "follow_through"
    else:
        output_df.loc[frames > ffc_start_frame + 20, "original_phase"] = "follow_through"

    return output_df


def summarize_event_ranges(frame_phase_df: pd.DataFrame) -> Dict[str, Optional[int]]:
    summary: Dict[str, Optional[int]] = {
        "bbc_start": None,
        "bbc_end": None,
        "ffc_start": None,
        "ffc_end": None,
    }
    for phase_name, start_key, end_key in (
        ("bbc", "bbc_start", "bbc_end"),
        ("ffc", "ffc_start", "ffc_end"),
    ):
        phase_frames = frame_phase_df.loc[frame_phase_df["predicted_phase"] == phase_name, "frame"]
        if not phase_frames.empty:
            summary[start_key] = int(phase_frames.min())
            summary[end_key] = int(phase_frames.max())
    return summary

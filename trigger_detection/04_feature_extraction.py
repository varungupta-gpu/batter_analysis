from typing import Dict

import numpy as np
import pandas as pd


def _segment_window(df: pd.DataFrame, start_frame: int, end_frame: int) -> pd.DataFrame:
    segment = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)].copy()
    if segment.empty:
        raise ValueError("No keypoint rows found in the requested frame range.")
    return segment


def _stance_window(df: pd.DataFrame, stance_start_frame: int, stance_end_frame: int) -> pd.DataFrame:
    stance = df[(df["frame"] >= stance_start_frame) & (df["frame"] <= stance_end_frame)].copy()
    if stance.empty:
        raise ValueError("No stance rows found in the requested frame range.")
    return stance


def _stance_stride_geometry(stance: pd.DataFrame) -> Dict[str, float]:
    stance_front_x = stance["left_ankle_x"].mean()
    stance_front_y = stance["left_ankle_y"].mean()
    stance_back_x = stance["right_ankle_x"].mean()
    stance_back_y = stance["right_ankle_y"].mean()

    stride_vec_x = stance_front_x - stance_back_x
    stride_vec_y = stance_front_y - stance_back_y
    stride_width = float(np.sqrt(stride_vec_x**2 + stride_vec_y**2))
    if stride_width == 0 or np.isnan(stride_width):
        raise ValueError("Stride width is zero or invalid in the stance baseline.")

    return {
        "stance_front_x": float(stance_front_x),
        "stance_front_y": float(stance_front_y),
        "stance_back_x": float(stance_back_x),
        "stance_back_y": float(stance_back_y),
        "stride_unit_x": float(stride_vec_x / stride_width),
        "stride_unit_y": float(stride_vec_y / stride_width),
        "stride_width": stride_width,
    }


def compute_front_foot_progression(df, start_frame, end_frame, stance_start_frame, stance_end_frame):
    segment = _segment_window(df, start_frame, end_frame)
    stance = _stance_window(df, stance_start_frame, stance_end_frame)
    geometry = _stance_stride_geometry(stance)

    dx = segment["left_ankle_x"] - geometry["stance_front_x"]
    dy = segment["left_ankle_y"] - geometry["stance_front_y"]
    return (dx * geometry["stride_unit_x"] + dy * geometry["stride_unit_y"]) / geometry["stride_width"]


def compute_back_foot_progression(df, start_frame, end_frame, stance_start_frame, stance_end_frame):
    segment = _segment_window(df, start_frame, end_frame)
    stance = _stance_window(df, stance_start_frame, stance_end_frame)
    geometry = _stance_stride_geometry(stance)

    dx = segment["right_ankle_x"] - geometry["stance_back_x"]
    dy = segment["right_ankle_y"] - geometry["stance_back_y"]
    return (dx * geometry["stride_unit_x"] + dy * geometry["stride_unit_y"]) / geometry["stride_width"]


def compute_stride_width(df, start_frame, end_frame):
    segment = _segment_window(df, start_frame, end_frame)
    return np.sqrt(
        (segment["left_ankle_x"] - segment["right_ankle_x"]) ** 2
        + (segment["left_ankle_y"] - segment["right_ankle_y"]) ** 2
    )


def compute_front_ankle_displacement(df, start_frame, end_frame, stance_start_frame, stance_end_frame):
    segment = _segment_window(df, start_frame, end_frame)
    stance = _stance_window(df, stance_start_frame, stance_end_frame)
    stance_front_x = stance["left_ankle_x"].mean()
    stance_front_y = stance["left_ankle_y"].mean()
    return np.sqrt(
        (segment["left_ankle_x"] - stance_front_x) ** 2
        + (segment["left_ankle_y"] - stance_front_y) ** 2
    )


def compute_back_ankle_displacement(df, start_frame, end_frame, stance_start_frame, stance_end_frame):
    segment = _segment_window(df, start_frame, end_frame)
    stance = _stance_window(df, stance_start_frame, stance_end_frame)
    stance_back_x = stance["right_ankle_x"].mean()
    stance_back_y = stance["right_ankle_y"].mean()
    return np.sqrt(
        (segment["right_ankle_x"] - stance_back_x) ** 2
        + (segment["right_ankle_y"] - stance_back_y) ** 2
    )


def compute_front_knee_displacement(df, start_frame, end_frame, stance_start_frame, stance_end_frame):
    segment = _segment_window(df, start_frame, end_frame)
    stance = _stance_window(df, stance_start_frame, stance_end_frame)
    stance_knee_x = stance["left_knee_x"].mean()
    stance_knee_y = stance["left_knee_y"].mean()
    return np.sqrt(
        (segment["left_knee_x"] - stance_knee_x) ** 2
        + (segment["left_knee_y"] - stance_knee_y) ** 2
    )


def compute_back_knee_displacement(df, start_frame, end_frame, stance_start_frame, stance_end_frame):
    segment = _segment_window(df, start_frame, end_frame)
    stance = _stance_window(df, stance_start_frame, stance_end_frame)
    stance_knee_x = stance["right_knee_x"].mean()
    stance_knee_y = stance["right_knee_y"].mean()
    return np.sqrt(
        (segment["right_knee_x"] - stance_knee_x) ** 2
        + (segment["right_knee_y"] - stance_knee_y) ** 2
    )


def compute_knee_to_knee_distance(df, start_frame, end_frame):
    segment = _segment_window(df, start_frame, end_frame)
    return np.sqrt(
        (segment["left_knee_x"] - segment["right_knee_x"]) ** 2
        + (segment["left_knee_y"] - segment["right_knee_y"]) ** 2
    )


def compute_trigger_feature_dataframe(
    bowler_keypoints_df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    stance_start_frame: int,
    stance_end_frame: int,
) -> pd.DataFrame:
    segment = _segment_window(bowler_keypoints_df, start_frame, end_frame)[["frame"]].copy()

    segment["front_foot_progression"] = compute_front_foot_progression(
        bowler_keypoints_df, start_frame, end_frame, stance_start_frame, stance_end_frame
    ).values
    segment["back_foot_progression"] = compute_back_foot_progression(
        bowler_keypoints_df, start_frame, end_frame, stance_start_frame, stance_end_frame
    ).values
    segment["stride_width"] = compute_stride_width(
        bowler_keypoints_df, start_frame, end_frame
    ).values
    segment["front_ankle_displacement"] = compute_front_ankle_displacement(
        bowler_keypoints_df, start_frame, end_frame, stance_start_frame, stance_end_frame
    ).values
    segment["back_ankle_displacement"] = compute_back_ankle_displacement(
        bowler_keypoints_df, start_frame, end_frame, stance_start_frame, stance_end_frame
    ).values
    segment["front_knee_displacement"] = compute_front_knee_displacement(
        bowler_keypoints_df, start_frame, end_frame, stance_start_frame, stance_end_frame
    ).values
    segment["back_knee_displacement"] = compute_back_knee_displacement(
        bowler_keypoints_df, start_frame, end_frame, stance_start_frame, stance_end_frame
    ).values
    segment["knee_to_knee_distance"] = compute_knee_to_knee_distance(
        bowler_keypoints_df, start_frame, end_frame
    ).values

    numeric_columns = [column for column in segment.columns if column != "frame"]
    segment[numeric_columns] = segment[numeric_columns].apply(pd.to_numeric, errors="coerce")
    segment[numeric_columns] = segment[numeric_columns].interpolate(limit_direction="both")
    segment[numeric_columns] = segment[numeric_columns].ffill().bfill()
    return segment

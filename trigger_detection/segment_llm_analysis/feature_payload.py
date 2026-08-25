import json
from typing import Dict, List

import numpy as np
import pandas as pd

from trigger_detection.common.json_utils import json_ready
from trigger_detection.trigger_core import compute_trigger_metrics as trigger_metrics
from trigger_detection.trigger_core import feature_extraction as trigger_features
from trigger_detection.trigger_core import phase_feature_calculations as phase_features


ALL_FEATURE_COLUMNS = [
    "front_foot_progression",
    "back_foot_progression",
    "front_ankle_displacement",
    "back_ankle_displacement",
    "stride_width",
    "front_knee_displacement",
    "back_knee_displacement",
    "knee_to_knee_distance",
    "hip_direction",
    "shoulder_line_progression_angle",
    "stride_line_progression_angle",
    "hip_shoulder_alignment",
    "front_foot_ankle_knee_line",
    "back_foot_ankle_knee_line",
    "weighted_com",
    "trunk_lateral_flexion",
    "upper_body_rotation",
    "lower_body_rotation",
    "front_knee_angle",
    "back_knee_angle",
]


def _segment_window(df: pd.DataFrame, start_frame: int, end_frame: int) -> pd.DataFrame:
    segment = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)].copy()
    if segment.empty:
        raise ValueError("No keypoint rows found in the requested frame range.")
    return segment


def _safe_mad(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    median = float(clean.median())
    return float((clean - median).abs().median())


def _series_summary(series: pd.Series) -> Dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "range": float("nan"),
            "mad": float("nan"),
            "start_value": float("nan"),
            "end_value": float("nan"),
            "delta": float("nan"),
        }
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "median": float(clean.median()),
        "q05": float(clean.quantile(0.05)),
        "q25": float(clean.quantile(0.25)),
        "q75": float(clean.quantile(0.75)),
        "q95": float(clean.quantile(0.95)),
        "range": float(clean.max() - clean.min()),
        "mad": _safe_mad(clean),
        "start_value": float(clean.iloc[0]),
        "end_value": float(clean.iloc[-1]),
        "delta": float(clean.iloc[-1] - clean.iloc[0]),
    }


def compute_all_feature_dataframe(
    keypoints_df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    stance_start_frame: int,
    stance_end_frame: int,
) -> pd.DataFrame:
    segment = _segment_window(keypoints_df, start_frame, end_frame)[["frame"]].copy()

    trigger_feature_df = trigger_features.compute_trigger_feature_dataframe(
        bowler_keypoints_df=keypoints_df,
        start_frame=start_frame,
        end_frame=end_frame,
        stance_start_frame=stance_start_frame,
        stance_end_frame=stance_end_frame,
    )
    for column in trigger_feature_df.columns:
        if column != "frame":
            segment[column] = trigger_feature_df[column].values

    segment["hip_direction"] = phase_features.compute_hip_direction(
        keypoints_df, start_frame, end_frame
    )
    segment["shoulder_line_progression_angle"] = phase_features.compute_shoulder_line_progression_angle(
        keypoints_df, start_frame, end_frame
    )
    segment["stride_line_progression_angle"] = phase_features.compute_stride_line_progression_angle(
        keypoints_df, start_frame, end_frame
    )
    segment["hip_shoulder_alignment"] = phase_features.compute_hip_shoulder_alignment(
        keypoints_df, start_frame, end_frame
    )
    segment["front_foot_ankle_knee_line"] = phase_features.compute_front_foot_ankle_knee_line(
        keypoints_df, start_frame, end_frame
    )
    segment["back_foot_ankle_knee_line"] = phase_features.compute_back_foot_ankle_knee_line(
        keypoints_df, start_frame, end_frame
    )
    segment["weighted_com"] = phase_features.compute_weighted_com(
        keypoints_df, start_frame, end_frame, stance_start_frame, stance_end_frame
    )
    segment["trunk_lateral_flexion"] = phase_features.compute_trunk_lateral_flexion(
        keypoints_df, start_frame, end_frame, stance_start_frame, stance_end_frame
    )
    segment["upper_body_rotation"] = phase_features.compute_upper_body_rotation(
        keypoints_df, start_frame, end_frame, stance_start_frame, stance_end_frame
    )
    segment["lower_body_rotation"] = phase_features.compute_lower_body_rotation(
        keypoints_df, start_frame, end_frame, stance_start_frame, stance_end_frame
    )
    segment["front_knee_angle"] = phase_features.compute_front_knee_angle(
        keypoints_df, start_frame, end_frame
    )
    segment["back_knee_angle"] = phase_features.compute_back_knee_angle(
        keypoints_df, start_frame, end_frame
    )

    numeric_columns = [column for column in segment.columns if column != "frame"]
    segment[numeric_columns] = segment[numeric_columns].apply(pd.to_numeric, errors="coerce")
    segment[numeric_columns] = segment[numeric_columns].interpolate(limit_direction="both")
    segment[numeric_columns] = segment[numeric_columns].ffill().bfill()
    return segment


def compute_feature_statistics(
    all_feature_df: pd.DataFrame,
    trigger_feature_df: pd.DataFrame,
    fps: float,
    baseline_end_frame: int,
    rolling_window: int,
) -> Dict[str, object]:
    summaries = {}
    rolling_stats = {}
    for column in ALL_FEATURE_COLUMNS:
        if column not in all_feature_df.columns:
            continue
        summaries[column] = _series_summary(all_feature_df[column])
        series = pd.to_numeric(all_feature_df[column], errors="coerce")
        rolling_stats[column] = {
            "rolling_mean": json_ready(series.rolling(window=rolling_window, min_periods=1).mean().tolist()),
            "rolling_std": json_ready(series.rolling(window=rolling_window, min_periods=2).std().fillna(0.0).tolist()),
            "rolling_variance": json_ready(series.rolling(window=rolling_window, min_periods=2).var().fillna(0.0).tolist()),
            "rolling_range": json_ready(
                (
                    series.rolling(window=rolling_window, min_periods=2).max()
                    - series.rolling(window=rolling_window, min_periods=2).min()
                ).fillna(0.0).tolist()
            ),
            "rolling_mad": json_ready(
                series.rolling(window=rolling_window, min_periods=2).apply(_safe_mad, raw=False).fillna(0.0).tolist()
            ),
        }

    trigger_metrics_df = trigger_metrics.compute_trigger_metrics(
        feature_df=trigger_feature_df,
        fps=fps,
        baseline_end_frame=baseline_end_frame,
        min_active_features=4,
    )

    return json_ready(
        {
            "rolling_window": rolling_window,
            "feature_columns": ALL_FEATURE_COLUMNS,
            "per_feature_summary": summaries,
            "per_feature_rolling_stats": rolling_stats,
            "trigger_metric_frames_for_original_features": trigger_metrics_df.to_dict(orient="records"),
        }
    )


def dataframe_records_json(df: pd.DataFrame) -> str:
    cleaned = df.replace([np.inf, -np.inf], np.nan)
    records = cleaned.where(pd.notna(cleaned), None).to_dict(orient="records")
    return json.dumps(records)

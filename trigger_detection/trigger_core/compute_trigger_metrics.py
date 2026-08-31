from typing import Dict, List

import numpy as np
import pandas as pd


BASE_FEATURE_COLUMNS = [
    "front_foot_progression",
    "back_foot_progression",
    "front_ankle_displacement",
    "back_ankle_displacement",
    "stride_width",
    "front_knee_displacement",
    "back_knee_displacement",
    "knee_to_knee_distance",
]

FEATURE_WEIGHTS = {
    "front_foot_progression": 2.4,
    "back_foot_progression": 2.2,
    "front_ankle_displacement": 2.8,
    "back_ankle_displacement": 2.6,
    "stride_width": 2.0,
    "front_knee_displacement": 2.4,
    "back_knee_displacement": 2.2,
    "knee_to_knee_distance": 2.4,
}


def _mad(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    median = float(clean.median())
    return float((clean - median).abs().median())


def _baseline_stats(feature_df: pd.DataFrame, baseline_end_frame: int) -> Dict[str, Dict[str, float]]:
    baseline_df = feature_df[feature_df["frame"] <= baseline_end_frame]
    if baseline_df.empty:
        raise ValueError("Baseline frame range is empty.")

    stats: Dict[str, Dict[str, float]] = {}
    for column in BASE_FEATURE_COLUMNS:
        series = pd.to_numeric(baseline_df[column], errors="coerce")
        stats[column] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0) if series.notna().any() else np.nan),
            "var": float(series.var(ddof=0) if series.notna().any() else np.nan),
            "range": float(series.max() - series.min()) if series.notna().any() else np.nan,
            "mad": _mad(series),
        }
    return stats


def _safe_denominator(value: float) -> float:
    if pd.isna(value) or value == 0:
        return 1e-6
    return float(abs(value))


def compute_trigger_metrics(
    feature_df: pd.DataFrame,
    fps: float,
    baseline_end_frame: int,
    min_active_features: int = 4,
) -> pd.DataFrame:
    metrics_df = feature_df.copy().sort_values("frame").reset_index(drop=True)
    baseline_stats = _baseline_stats(metrics_df, baseline_end_frame)
    rolling_window = 10 if fps >= 45 else 5

    active_flag_columns: List[str] = []
    confidence_columns: List[str] = []
    computed_columns: Dict[str, pd.Series] = {}

    for feature_name in BASE_FEATURE_COLUMNS:
        weight = FEATURE_WEIGHTS.get(feature_name, 1.0)
        baseline_mean = baseline_stats[feature_name]["mean"]
        baseline_std = _safe_denominator(baseline_stats[feature_name]["std"])
        baseline_var = _safe_denominator(baseline_stats[feature_name]["var"])
        baseline_range = _safe_denominator(baseline_stats[feature_name]["range"])
        baseline_mad = _safe_denominator(baseline_stats[feature_name]["mad"])

        baseline_diff = metrics_df[feature_name] - baseline_mean
        frame_diff = metrics_df[feature_name].diff().fillna(0.0)
        rolling_variance = metrics_df[feature_name].rolling(window=rolling_window, min_periods=2).var().fillna(0.0)
        rolling_std = metrics_df[feature_name].rolling(window=rolling_window, min_periods=2).std().fillna(0.0)
        rolling_range = (
            metrics_df[feature_name].rolling(window=rolling_window, min_periods=2).max()
            - metrics_df[feature_name].rolling(window=rolling_window, min_periods=2).min()
        ).fillna(0.0)
        mad = metrics_df[feature_name].rolling(window=rolling_window, min_periods=2).apply(_mad, raw=False).fillna(0.0)

        baseline_change_strength = (baseline_diff.abs() / baseline_std) * weight
        frame_diff_strength = (frame_diff.abs() / baseline_std) * weight
        rolling_variance_strength = (rolling_variance.abs() / baseline_var) * weight
        rolling_std_strength = (rolling_std.abs() / baseline_std) * weight
        rolling_range_strength = (rolling_range.abs() / baseline_range) * weight
        mad_strength = (mad.abs() / baseline_mad) * weight

        movement_flag = (
            (
                (baseline_change_strength >= 3.0)
                | (frame_diff_strength >= 2.4)
                | (rolling_variance_strength >= 3.0)
                | (rolling_std_strength >= 3.0)
                | (rolling_range_strength >= 3.0)
                | (mad_strength >= 3.0)
            )
        ).astype(int)

        computed_columns[f"{feature_name}_baseline_diff"] = baseline_diff
        computed_columns[f"{feature_name}_frame_diff"] = frame_diff
        computed_columns[f"{feature_name}_rolling_variance"] = rolling_variance
        computed_columns[f"{feature_name}_rolling_std"] = rolling_std
        computed_columns[f"{feature_name}_rolling_range"] = rolling_range
        computed_columns[f"{feature_name}_mad"] = mad
        computed_columns[f"{feature_name}_baseline_change_strength"] = baseline_change_strength
        computed_columns[f"{feature_name}_frame_diff_strength"] = frame_diff_strength
        computed_columns[f"{feature_name}_rolling_variance_strength"] = rolling_variance_strength
        computed_columns[f"{feature_name}_rolling_std_strength"] = rolling_std_strength
        computed_columns[f"{feature_name}_rolling_range_strength"] = rolling_range_strength
        computed_columns[f"{feature_name}_mad_strength"] = mad_strength
        computed_columns[f"{feature_name}_movement_flag"] = movement_flag

        active_flag_columns.append(f"{feature_name}_movement_flag")
        confidence_column = f"{feature_name}_confidence"
        computed_columns[confidence_column] = (
            baseline_change_strength
            + frame_diff_strength
            + rolling_variance_strength
            + rolling_std_strength
            + rolling_range_strength
            + mad_strength
        ) / 6.0
        confidence_columns.append(confidence_column)

    metrics_df = pd.concat([metrics_df, pd.DataFrame(computed_columns, index=metrics_df.index)], axis=1)
    metrics_df["active_feature_count"] = metrics_df[active_flag_columns].sum(axis=1)
    metrics_df["window_confidence"] = metrics_df[confidence_columns].mean(axis=1)
    metrics_df["movement_flag"] = (
        (metrics_df["active_feature_count"] >= min_active_features)
        & (metrics_df["window_confidence"] >= 2.5)
    ).astype(int)

    aggregate_names = [
        "baseline_change_strength",
        "frame_diff_strength",
        "rolling_variance_strength",
        "rolling_std_strength",
        "rolling_range_strength",
        "mad_strength",
    ]
    for aggregate_name in aggregate_names:
        matching_columns = [f"{feature_name}_{aggregate_name}" for feature_name in BASE_FEATURE_COLUMNS]
        metrics_df[aggregate_name] = metrics_df[matching_columns].mean(axis=1)

    return metrics_df

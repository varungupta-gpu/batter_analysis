import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from trigger_detection.common.phase_pipeline_common import (
    build_release_info_from_frame,
    load_json_or_csv,
    load_keypoints_csv,
    normalize_keypoint_dataframe,
)
from trigger_detection.config import OUTPUT_ROOT, REPO_ROOT, SEGMENT_MANIFEST_PATH
from trigger_detection.trigger_core import compute_trigger_metrics as trigger_metrics
from trigger_detection.trigger_core import feature_extraction as trigger_features


OVERLAP_MANIFEST = SEGMENT_MANIFEST_PATH
OUTPUT_DIR = OUTPUT_ROOT
DETAILED_OUTPUT_CSV = OUTPUT_DIR / "smoothed_trigger_results.csv"
SUMMARY_OUTPUT_CSV = OUTPUT_DIR / "smoothed_trigger_summary.csv"
FEATURE_METRICS_JSON = OUTPUT_DIR / "smoothed_trigger_feature_metrics.json"

# Conditions requested for the final smoothed-keypoint trigger check.
SMOOTH_WINDOW = 5
FPS_VALUE = 30.0
LOADUP_TO_RELEASE_FRAMES = 27
PRE_LOADUP_WINDOW_FRAMES = 27
MIN_PRE_LOADUP_FRAMES = 1
MIN_ACTIVE_FEATURES = 4
MIN_ANKLE_ACTIVE_FEATURES = 3.5
MIN_WINDOW_CONFIDENCE = 2.8
MIN_FACTOR_MOVEMENT_RATIO = 1.4
MIN_FACTOR_ACTIVE_FRACTION = 0.65
MIN_CROSSED_FACTORS = 4
KEYPOINT_FILTER_CONF_THRESH = 0.1
MAX_GAP_INTERP = 5


def _resolve_window_lengths(fps: float) -> tuple[int, int]:
    if fps >= 45:
        return 8, 10
    return 4, 5


def _select_baseline_end_frame(
    loadup_start: int,
    release_frame: int,
    max_trigger_frames: int,
) -> int:
    baseline_span = max(3, max_trigger_frames)
    baseline_end = min(release_frame, loadup_start + baseline_span - 1)
    if baseline_end <= loadup_start:
        baseline_end = min(release_frame, loadup_start + 2)
    return baseline_end


def _window_score(window_df: pd.DataFrame, metric_columns: List[str]) -> float:
    if window_df.empty:
        return float("-inf")

    score = 0.0
    for column in metric_columns:
        if column not in window_df.columns:
            continue
        series = pd.to_numeric(window_df[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if not series.empty:
            score += float(series.mean())

    active_features = int(
        pd.to_numeric(
            window_df.get("active_feature_count", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0).mean()
    )
    score += active_features * 0.75
    ankle_active_features = int(
        pd.to_numeric(
            window_df.get("ankle_active_feature_count", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0).mean()
    )
    score += ankle_active_features * 1.5
    return score


class OneEuro:
    def __init__(self, freq: float, mincutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        self.freq = max(float(freq), 1e-6)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x_prev: Optional[float] = None
        self._dx_prev = 0.0

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / self.freq
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self._x_prev is None:
            self._x_prev = float(x)
            return float(x)
        dx = (float(x) - self._x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * float(x) + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


def _keypoint_names_from_raw_columns(columns: List[str]) -> List[str]:
    names = []
    for column in columns:
        if column.endswith("_conf"):
            names.append(column[:-5])
    return sorted(set(names))


def postprocess_track(
    g: pd.DataFrame,
    fps: float,
    timestamps: Optional[List[float]] = None,
    max_gap_interp: Optional[int] = MAX_GAP_INTERP,
    limit_area: Optional[str] = "inside",
    total_frames: Optional[int] = None,
) -> pd.DataFrame:
    g = g.sort_values("frame").reset_index(drop=True)
    frames = pd.to_numeric(g["frame"], errors="coerce").dropna().astype(int).to_numpy()
    if len(frames) == 0:
        return g.copy()

    min_frame = int(frames.min())
    max_frame = int(total_frames) if total_frames is not None else int(frames.max())
    full = np.arange(min_frame, max_frame + 1)
    g = g.set_index("frame").reindex(full)
    g.index.name = "frame"

    if "tracker_id" in g.columns and g["tracker_id"].dropna().any():
        tid = int(float(g["tracker_id"].dropna().iloc[0]))
    else:
        tid = 1
    g["tracker_id"] = tid

    if timestamps is not None:
        g["timestamp_sec"] = [
            round(float(timestamps[int(f) - 1]), 3) if 0 <= (int(f) - 1) < len(timestamps) else round((int(f) - 1) / fps, 3)
            for f in g.index
        ]
    elif "timestamp_sec" in g.columns:
        g["timestamp_sec"] = pd.to_numeric(g["timestamp_sec"], errors="coerce")
    elif "timestamp" in g.columns:
        g["timestamp_sec"] = pd.to_numeric(g["timestamp"], errors="coerce")
    else:
        g["timestamp_sec"] = [round((int(f) - 1) / fps, 3) for f in g.index]
    g["timestamp"] = g["timestamp_sec"]

    for name in _keypoint_names_from_raw_columns(list(g.columns)):
        cx, cy, cc = f"{name}_x", f"{name}_y", f"{name}_conf"
        if cx not in g.columns or cy not in g.columns or cc not in g.columns:
            continue

        low = pd.to_numeric(g[cc], errors="coerce").fillna(0.0) < KEYPOINT_FILTER_CONF_THRESH
        g.loc[low, [cx, cy]] = np.nan

        g[cx] = pd.to_numeric(g[cx], errors="coerce").interpolate(limit=max_gap_interp, limit_area=limit_area)
        g[cy] = pd.to_numeric(g[cy], errors="coerce").interpolate(limit=max_gap_interp, limit_area=limit_area)
        g[cc] = pd.to_numeric(g[cc], errors="coerce").fillna(0.0)

        fx, fy = OneEuro(fps), OneEuro(fps)
        xs = g[cx].to_numpy(copy=True)
        ys = g[cy].to_numpy(copy=True)
        for i in range(len(xs)):
            if not np.isnan(xs[i]):
                xs[i] = fx(float(xs[i]))
            if not np.isnan(ys[i]):
                ys[i] = fy(float(ys[i]))
        g[cx], g[cy] = xs, ys

    return g.reset_index()


def smooth_keypoint_dataframe(df: pd.DataFrame, window: int) -> pd.DataFrame:
    # Old smoothing retained here for reference only:
    # smoothed = df.copy()
    # coord_columns = [col for col in smoothed.columns if col.endswith("_x") or col.endswith("_y")]
    # for column in coord_columns:
    #     values = pd.to_numeric(smoothed[column], errors="coerce")
    #     values = values.mask(values <= 0)
    #     smoothed[column] = values.rolling(window=window, min_periods=1, center=True).mean()
    # return smoothed
    return df.copy()


def smooth_raw_keypoint_file(path: Path, fps: float, total_frames: Optional[int] = None) -> pd.DataFrame:
    raw_df = load_json_or_csv(str(path))
    timestamp_values: Optional[List[float]] = None
    if "timestamp_sec" in raw_df.columns:
        timestamp_values = pd.to_numeric(raw_df["timestamp_sec"], errors="coerce").dropna().tolist()
    elif "timestamp" in raw_df.columns:
        timestamp_values = pd.to_numeric(raw_df["timestamp"], errors="coerce").dropna().tolist()
    processed_df = postprocess_track(
        g=raw_df,
        fps=fps,
        timestamps=timestamp_values,
        total_frames=total_frames,
    )
    return normalize_keypoint_dataframe(processed_df)


def _movement_amplitude(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.quantile(0.95) - clean.quantile(0.05))


def _minimum_factor_change(feature_name: str, stride_width: float) -> float:
    if feature_name in {"front_foot_progression", "back_foot_progression"}:
        return 0.04
    return max(1.0, 0.04 * stride_width)


def _compare_preload_and_loadup_movement(
    feature_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    baseline_start_frame: int,
    baseline_end_frame: int,
    loadup_start_frame: int,
    release_frame: int,
) -> Dict[str, object]:
    baseline_df = feature_df[
        (feature_df["frame"] >= baseline_start_frame)
        & (feature_df["frame"] <= baseline_end_frame)
    ]
    loadup_df = feature_df[
        (feature_df["frame"] >= loadup_start_frame)
        & (feature_df["frame"] <= release_frame)
    ]
    loadup_metrics_df = metrics_df[
        (metrics_df["frame"] >= loadup_start_frame)
        & (metrics_df["frame"] <= release_frame)
    ]

    stride_width_series = pd.to_numeric(baseline_df.get("stride_width"), errors="coerce")
    stride_width = float(stride_width_series.median()) if stride_width_series.notna().any() else 1.0
    if not np.isfinite(stride_width) or stride_width <= 0:
        stride_width = 1.0

    pre_movement: Dict[str, float] = {}
    loadup_movement: Dict[str, float] = {}
    movement_ratios: Dict[str, float] = {}
    minimum_changes: Dict[str, float] = {}
    threshold_crossing_frames: Dict[str, List[int]] = {}
    crossed_factors: List[str] = []

    for feature_name in trigger_metrics.BASE_FEATURE_COLUMNS:
        pre_amplitude = _movement_amplitude(baseline_df[feature_name])
        loadup_amplitude = _movement_amplitude(loadup_df[feature_name])
        minimum_change = _minimum_factor_change(feature_name, stride_width)
        comparison_floor = max(minimum_change * 0.25, 1e-6)
        movement_ratio = loadup_amplitude / max(pre_amplitude, comparison_floor)

        flag_column = f"{feature_name}_movement_flag"
        crossing_frames = [
            int(frame)
            for frame in loadup_metrics_df.loc[loadup_metrics_df[flag_column] == 1, "frame"].tolist()
        ]

        pre_movement[feature_name] = pre_amplitude
        loadup_movement[feature_name] = loadup_amplitude
        movement_ratios[feature_name] = movement_ratio
        minimum_changes[feature_name] = minimum_change
        threshold_crossing_frames[feature_name] = crossing_frames

        if (
            loadup_amplitude >= minimum_change
            and movement_ratio >= MIN_FACTOR_MOVEMENT_RATIO
            and crossing_frames
        ):
            crossed_factors.append(feature_name)

    return {
        "pre_loadup_factor_movement": pre_movement,
        "loadup_to_release_factor_movement": loadup_movement,
        "factor_movement_ratios": movement_ratios,
        "minimum_factor_changes": minimum_changes,
        "factor_crossing_frames": threshold_crossing_frames,
        "factors_crossed": crossed_factors,
    }


def detect_trigger_on_smoothed_keypoints(
    keypoints_df: pd.DataFrame,
    release_frame: float,
    fps_value: float = FPS_VALUE,
    required_trigger_frames: Optional[int] = None,
) -> Dict[str, object]:
    release_value = build_release_info_from_frame(release_frame)["release_frame"]
    if required_trigger_frames is not None:
        min_trigger_frames = required_trigger_frames
        max_trigger_frames = required_trigger_frames
    else:
        min_trigger_frames, max_trigger_frames = _resolve_window_lengths(fps_value)
    release_frame_int = int(round(release_value))
    loadup_start_frame = max(0, release_frame_int - LOADUP_TO_RELEASE_FRAMES)
    if release_frame_int <= loadup_start_frame:
        raise ValueError("Release frame must be after loadup start.")

    available_preload_df = keypoints_df[
        (keypoints_df["frame"] >= loadup_start_frame - PRE_LOADUP_WINDOW_FRAMES)
        & (keypoints_df["frame"] < loadup_start_frame)
    ]
    if len(available_preload_df) >= MIN_PRE_LOADUP_FRAMES:
        baseline_source = "pre_loadup"
        baseline_start_frame = int(available_preload_df["frame"].min())
        baseline_end_frame = int(available_preload_df["frame"].max())
    else:
        baseline_source = "loadup_fallback"
        baseline_start_frame = loadup_start_frame
        baseline_end_frame = _select_baseline_end_frame(
            loadup_start_frame,
            release_frame_int,
            max_trigger_frames,
        )

    feature_start_frame = min(baseline_start_frame, loadup_start_frame)
    feature_df = trigger_features.compute_trigger_feature_dataframe(
        bowler_keypoints_df=keypoints_df,
        start_frame=feature_start_frame,
        end_frame=release_frame_int,
        stance_start_frame=baseline_start_frame,
        stance_end_frame=baseline_end_frame,
    )
    metrics_df = trigger_metrics.compute_trigger_metrics(
        feature_df=feature_df,
        fps=fps_value,
        baseline_end_frame=baseline_end_frame,
        min_active_features=MIN_ACTIVE_FEATURES,
    )

    comparison = _compare_preload_and_loadup_movement(
        feature_df=feature_df,
        metrics_df=metrics_df,
        baseline_start_frame=baseline_start_frame,
        baseline_end_frame=baseline_end_frame,
        loadup_start_frame=loadup_start_frame,
        release_frame=release_frame_int,
    )
    globally_crossed_factors = set(comparison["factors_crossed"])

    candidate_mask = (
        (metrics_df["frame"] >= max(loadup_start_frame, baseline_end_frame + 1))
        & (metrics_df["frame"] <= release_frame_int)
        & (metrics_df["movement_flag"] == 1)
    )
    candidate_frames = metrics_df.loc[candidate_mask, "frame"].tolist()

    best_window: Optional[pd.DataFrame] = None
    best_window_crossed_factors: List[str] = []
    best_score = float("-inf")
    metric_columns = [
        "window_confidence",
        "baseline_change_strength",
        "frame_diff_strength",
        "rolling_variance_strength",
        "rolling_std_strength",
        "rolling_range_strength",
        "mad_strength",
    ]

    for start_idx in range(len(candidate_frames)):
        for length in range(min_trigger_frames, max_trigger_frames + 1):
            if start_idx + length > len(candidate_frames):
                continue
            frame_block = candidate_frames[start_idx : start_idx + length]
            if frame_block[-1] - frame_block[0] != length - 1:
                continue

            window_df = metrics_df[metrics_df["frame"].isin(frame_block)].copy()
            trigger_confidence = float(
                pd.to_numeric(window_df["window_confidence"], errors="coerce").mean()
            )
            mean_ankle_active = float(
                pd.to_numeric(
                    window_df.get("ankle_active_feature_count", pd.Series(dtype=float)),
                    errors="coerce",
                ).fillna(0).mean()
            )
            mean_active = float(
                pd.to_numeric(
                    window_df.get("active_feature_count", pd.Series(dtype=float)),
                    errors="coerce",
                ).fillna(0).mean()
            )

            window_crossed_factors = []
            for feature_name in trigger_metrics.BASE_FEATURE_COLUMNS:
                if feature_name not in globally_crossed_factors:
                    continue
                active_fraction = float(
                    pd.to_numeric(
                        window_df[f"{feature_name}_movement_flag"],
                        errors="coerce",
                    ).fillna(0).mean()
                )
                if active_fraction >= MIN_FACTOR_ACTIVE_FRACTION:
                    window_crossed_factors.append(feature_name)

            if (
                trigger_confidence < MIN_WINDOW_CONFIDENCE
                or mean_ankle_active < MIN_ANKLE_ACTIVE_FEATURES
                or mean_active < MIN_ACTIVE_FEATURES
                or len(window_crossed_factors) < MIN_CROSSED_FACTORS
            ):
                continue

            score = _window_score(window_df, metric_columns)
            if score > best_score:
                best_score = score
                best_window = window_df
                best_window_crossed_factors = window_crossed_factors

    trigger_detected = best_window is not None and not best_window.empty
    if trigger_detected:
        trigger_start = int(best_window["frame"].min())
        trigger_end = int(best_window["frame"].max())
        trigger_confidence = float(
            pd.to_numeric(best_window["window_confidence"], errors="coerce").mean()
        )
        expected_trigger_frames = [int(frame) for frame in best_window["frame"].tolist()]
        trigger_decision_reason = "actual_trigger"
    else:
        trigger_start = None
        trigger_end = None
        trigger_confidence = 0.0
        expected_trigger_frames = []
        if len(globally_crossed_factors) < MIN_CROSSED_FACTORS:
            trigger_decision_reason = "insufficient_factor_change_vs_preload"
        else:
            trigger_decision_reason = "no_strict_4_to_5_frame_window"

    return {
        "loadup_start_frame": loadup_start_frame,
        "pre_loadup_start_frame": (
            int(available_preload_df["frame"].min()) if not available_preload_df.empty else None
        ),
        "pre_loadup_end_frame": (
            int(available_preload_df["frame"].max()) if not available_preload_df.empty else None
        ),
        "pre_loadup_frame_count": int(len(available_preload_df)),
        "pre_loadup_comparison_available": baseline_source == "pre_loadup",
        "baseline_source": baseline_source,
        "baseline_start_frame": baseline_start_frame,
        "baseline_end_frame": baseline_end_frame,
        "release_frame": release_frame_int,
        "trigger_detected": trigger_detected,
        "trigger_start_frame": trigger_start,
        "trigger_end_frame": trigger_end,
        "trigger_confidence": trigger_confidence,
        "trigger_decision_reason": trigger_decision_reason,
        "expected_trigger_frames": expected_trigger_frames,
        "factors_evaluated": list(trigger_metrics.BASE_FEATURE_COLUMNS),
        "factors_crossed": comparison["factors_crossed"],
        "crossed_factor_count": len(comparison["factors_crossed"]),
        "trigger_window_factors_crossed": best_window_crossed_factors if trigger_detected else [],
        "trigger_window_crossed_factor_count": (
            len(best_window_crossed_factors) if trigger_detected else 0
        ),
        "factor_crossing_frames": comparison["factor_crossing_frames"],
        "pre_loadup_factor_movement": comparison["pre_loadup_factor_movement"],
        "loadup_to_release_factor_movement": comparison["loadup_to_release_factor_movement"],
        "factor_movement_ratios": comparison["factor_movement_ratios"],
        "minimum_factor_changes": comparison["minimum_factor_changes"],
        "feature_frame_count": int(len(feature_df)),
        "analysis_frame_count": int(
            len(
                feature_df[
                    (feature_df["frame"] >= loadup_start_frame)
                    & (feature_df["frame"] <= release_frame_int)
                ]
            )
        ),
        "metric_frame_count": int(len(metrics_df)),
        "feature_frames_json": _records_json(feature_df),
        "metric_frames_json": _records_json(metrics_df),
    }


def _json_value(value: object) -> str:
    return json.dumps(value)


def _records_json(df: pd.DataFrame) -> str:
    cleaned_df = df.replace([np.inf, -np.inf], np.nan)
    records = cleaned_df.where(pd.notna(cleaned_df), None).to_dict(orient="records")
    return json.dumps(records)


def _write_csv_with_locked_fallback(df: pd.DataFrame, target_path: Path) -> Path:
    try:
        df.to_csv(target_path, index=False)
        return target_path
    except PermissionError:
        fallback_path = target_path.with_name(f"{target_path.stem}_latest{target_path.suffix}")
        df.to_csv(fallback_path, index=False)
        return fallback_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.read_csv(OVERLAP_MANIFEST)
    rows: List[Dict[str, object]] = []

    for _, row in manifest_df.iterrows():
        segment_id = str(row["segment_id"]).strip()
        video_name = str(row["video_name"]).strip()
        release_frame = float(row["release_frame"])
        batter_path = (REPO_ROOT / str(row["batter_keypoints_path"])).resolve()

        result: Dict[str, object] = {
            "video_id": row["video_id"],
            "ball": row["ball"],
            "segment_id": segment_id,
            "video_name": video_name,
            "release_frame": int(round(release_frame)),
            "smooth_window": SMOOTH_WINDOW,
            "min_active_features": MIN_ACTIVE_FEATURES,
            "min_ankle_active_features": MIN_ANKLE_ACTIVE_FEATURES,
            "min_window_confidence": MIN_WINDOW_CONFIDENCE,
            "min_factor_movement_ratio": MIN_FACTOR_MOVEMENT_RATIO,
            "min_factor_active_fraction": MIN_FACTOR_ACTIVE_FRACTION,
            "min_crossed_factors": MIN_CROSSED_FACTORS,
            "frames_unchanged": False,
            "original_frame_count": "",
            "smoothed_frame_count": "",
            "keypoints_source": "batter_keypoints_release_based_loadup",
            "batter_keypoints_path": str(batter_path),
            "loadup_start_frame": "",
            "pre_loadup_start_frame": "",
            "pre_loadup_end_frame": "",
            "pre_loadup_frame_count": "",
            "pre_loadup_comparison_available": False,
            "baseline_source": "",
            "baseline_start_frame": "",
            "baseline_end_frame": "",
            "trigger_detected": False,
            "trigger_start_frame": "",
            "trigger_end_frame": "",
            "trigger_confidence": "",
            "trigger_decision_reason": "",
            "expected_trigger_frames": "",
            "factors_evaluated": "",
            "factors_crossed": "",
            "crossed_factor_count": "",
            "trigger_window_factors_crossed": "",
            "trigger_window_crossed_factor_count": "",
            "factor_crossing_frames": "",
            "pre_loadup_factor_movement": "",
            "loadup_to_release_factor_movement": "",
            "factor_movement_ratios": "",
            "minimum_factor_changes": "",
            "feature_frame_count": "",
            "analysis_frame_count": "",
            "metric_frame_count": "",
            "feature_frames_json": "",
            "metric_frames_json": "",
            "status": "",
            "error": "",
        }

        try:
            original_df = load_keypoints_csv(str(batter_path))
            smoothed_df = smooth_keypoint_dataframe(original_df, window=SMOOTH_WINDOW)
            frames_unchanged = original_df["frame"].equals(smoothed_df["frame"])
            trigger_result = detect_trigger_on_smoothed_keypoints(
                keypoints_df=smoothed_df,
                release_frame=release_frame,
            )

            result.update(
                {
                    "frames_unchanged": bool(frames_unchanged),
                    "original_frame_count": int(len(original_df)),
                    "smoothed_frame_count": int(len(smoothed_df)),
                    "loadup_start_frame": trigger_result["loadup_start_frame"],
                    "pre_loadup_start_frame": trigger_result["pre_loadup_start_frame"],
                    "pre_loadup_end_frame": trigger_result["pre_loadup_end_frame"],
                    "pre_loadup_frame_count": trigger_result["pre_loadup_frame_count"],
                    "pre_loadup_comparison_available": trigger_result[
                        "pre_loadup_comparison_available"
                    ],
                    "baseline_source": trigger_result["baseline_source"],
                    "baseline_start_frame": trigger_result["baseline_start_frame"],
                    "baseline_end_frame": trigger_result["baseline_end_frame"],
                    "trigger_detected": bool(trigger_result["trigger_detected"]),
                    "trigger_start_frame": trigger_result["trigger_start_frame"],
                    "trigger_end_frame": trigger_result["trigger_end_frame"],
                    "trigger_confidence": trigger_result["trigger_confidence"],
                    "trigger_decision_reason": trigger_result["trigger_decision_reason"],
                    "expected_trigger_frames": _json_value(
                        trigger_result["expected_trigger_frames"]
                    ),
                    "factors_evaluated": _json_value(trigger_result["factors_evaluated"]),
                    "factors_crossed": _json_value(trigger_result["factors_crossed"]),
                    "crossed_factor_count": trigger_result["crossed_factor_count"],
                    "trigger_window_factors_crossed": _json_value(
                        trigger_result["trigger_window_factors_crossed"]
                    ),
                    "trigger_window_crossed_factor_count": trigger_result[
                        "trigger_window_crossed_factor_count"
                    ],
                    "factor_crossing_frames": _json_value(
                        trigger_result["factor_crossing_frames"]
                    ),
                    "pre_loadup_factor_movement": _json_value(
                        trigger_result["pre_loadup_factor_movement"]
                    ),
                    "loadup_to_release_factor_movement": _json_value(
                        trigger_result["loadup_to_release_factor_movement"]
                    ),
                    "factor_movement_ratios": _json_value(
                        trigger_result["factor_movement_ratios"]
                    ),
                    "minimum_factor_changes": _json_value(
                        trigger_result["minimum_factor_changes"]
                    ),
                    "feature_frame_count": trigger_result["feature_frame_count"],
                    "analysis_frame_count": trigger_result["analysis_frame_count"],
                    "metric_frame_count": trigger_result["metric_frame_count"],
                    "feature_frames_json": trigger_result["feature_frames_json"],
                    "metric_frames_json": trigger_result["metric_frames_json"],
                    "status": "processed",
                }
            )
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)

        rows.append(result)

    detailed_df = pd.DataFrame(rows)
    detailed_output_path = _write_csv_with_locked_fallback(detailed_df, DETAILED_OUTPUT_CSV)

    feature_metrics_payload = []
    for row in rows:
        feature_metrics_payload.append(
            {
                "video_name": row.get("video_name"),
                "ball": row.get("ball"),
                "segment_id": row.get("segment_id"),
                "status": row.get("status"),
                "feature_frames": json.loads(row["feature_frames_json"]) if row.get("feature_frames_json") else [],
                "metric_frames": json.loads(row["metric_frames_json"]) if row.get("metric_frames_json") else [],
            }
        )
    FEATURE_METRICS_JSON.write_text(json.dumps(feature_metrics_payload, indent=2), encoding="utf-8")

    summary_df = detailed_df[
        ["video_name", "ball", "segment_id", "expected_trigger_frames", "trigger_detected"]
    ].rename(columns={"expected_trigger_frames": "trigger_frames"})
    summary_output_path = _write_csv_with_locked_fallback(summary_df, SUMMARY_OUTPUT_CSV)

    processed_count = int(detailed_df["status"].eq("processed").sum())
    true_count = int(detailed_df["trigger_detected"].eq(True).sum())
    print(f"processed={processed_count}")
    print(f"total={len(detailed_df)}")
    print(f"trigger_true={true_count}")
    print(f"trigger_false={len(detailed_df) - true_count}")
    print(detailed_output_path.resolve())
    print(summary_output_path.resolve())


if __name__ == "__main__":
    main()

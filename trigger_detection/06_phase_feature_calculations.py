"""Standalone stance and preparation phase feature calculations."""

import numpy as np
import pandas as pd


# ============================================================================
# STANCE PHASE FEATURES (7 features)
# ============================================================================

def compute_hip_direction(df, start_frame, end_frame):
    """Hip direction (degrees) measured with respect to vertical."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    dx = segment["right_hip_x"] - segment["left_hip_x"]
    dy = segment["right_hip_y"] - segment["left_hip_y"]

    hip_direction = np.degrees(np.arctan2(dx, -dy))

    return hip_direction


def compute_shoulder_line_progression_angle(df, start_frame, end_frame):
    """Shoulder line progression angle relative to vertical (degrees)."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    shoulder_x = segment["right_shoulder_x"] - segment["left_shoulder_x"]
    shoulder_y = segment["right_shoulder_y"] - segment["left_shoulder_y"]

    angle = np.degrees(np.arctan2(shoulder_x, -shoulder_y))

    return angle


def compute_stride_line_progression_angle(df, start_frame, end_frame):
    """Stride line progression angle relative to vertical (degrees)."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    stride_vec_x = segment["right_ankle_x"] - segment["left_ankle_x"]
    stride_vec_y = segment["right_ankle_y"] - segment["left_ankle_y"]

    stride_angle = np.degrees(np.arctan2(stride_vec_x, -stride_vec_y))

    return stride_angle


def compute_hip_shoulder_alignment(df, start_frame, end_frame):
    """Hip-Shoulder alignment (degrees) - angle between hip and shoulder lines."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    hip_vec_x = segment["right_hip_x"] - segment["left_hip_x"]
    hip_vec_y = segment["right_hip_y"] - segment["left_hip_y"]

    shoulder_vec_x = segment["right_shoulder_x"] - segment["left_shoulder_x"]
    shoulder_vec_y = segment["right_shoulder_y"] - segment["left_shoulder_y"]

    dot_product = hip_vec_x * shoulder_vec_x + hip_vec_y * shoulder_vec_y

    hip_mag = np.sqrt(hip_vec_x**2 + hip_vec_y**2)
    shoulder_mag = np.sqrt(shoulder_vec_x**2 + shoulder_vec_y**2)

    cosine = dot_product / (hip_mag * shoulder_mag + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)

    alignment = np.degrees(np.arccos(cosine))

    return alignment


def compute_front_foot_ankle_knee_line(df, start_frame, end_frame):
    """Front foot ankle-knee line orientation (degrees)."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    front_foot_ankle_knee_line = np.degrees(np.arctan2(
        segment["left_ankle_y"] - segment["left_knee_y"],
        segment["left_ankle_x"] - segment["left_knee_x"]
    ))

    return front_foot_ankle_knee_line


def compute_back_foot_ankle_knee_line(df, start_frame, end_frame):
    """Back foot ankle-knee line orientation (degrees)."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    back_foot_ankle_knee_line = np.degrees(np.arctan2(
        segment["right_ankle_y"] - segment["right_knee_y"],
        segment["right_ankle_x"] - segment["right_knee_x"]
    ))

    return back_foot_ankle_knee_line


def compute_stride_width(df, start_frame, end_frame):
    """Stride width (pixels) - distance between ankles."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    stride_width = np.sqrt(
        (segment["left_ankle_x"] - segment["right_ankle_x"]) ** 2 +
        (segment["left_ankle_y"] - segment["right_ankle_y"]) ** 2
    )

    return stride_width


# ============================================================================
# PREPARATION PHASE FEATURES (12 features)
# ============================================================================

def compute_front_foot_progression(df, start_frame, end_frame,
                                   stance_start_frame, stance_end_frame):
    """Front foot signed progression normalized by stance stride width."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    stance = df[(df["frame"] >= stance_start_frame) &
                (df["frame"] <= stance_end_frame)]

    stance_front_x = stance["left_ankle_x"].mean()
    stance_front_y = stance["left_ankle_y"].mean()
    stance_back_x = stance["right_ankle_x"].mean()
    stance_back_y = stance["right_ankle_y"].mean()

    stride_vec_x = stance_front_x - stance_back_x
    stride_vec_y = stance_front_y - stance_back_y

    stride_width = np.sqrt(stride_vec_x**2 + stride_vec_y**2)

    stride_unit_x = stride_vec_x / stride_width
    stride_unit_y = stride_vec_y / stride_width

    dx = segment["left_ankle_x"] - stance_front_x
    dy = segment["left_ankle_y"] - stance_front_y

    signed_progression = (dx * stride_unit_x + dy * stride_unit_y) / stride_width

    return signed_progression


def compute_back_foot_progression(df, start_frame, end_frame,
                                  stance_start_frame, stance_end_frame):
    """Back foot signed progression normalized by stance stride width."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    stance = df[(df["frame"] >= stance_start_frame) &
                (df["frame"] <= stance_end_frame)]

    stance_front_x = stance["left_ankle_x"].mean()
    stance_front_y = stance["left_ankle_y"].mean()
    stance_back_x = stance["right_ankle_x"].mean()
    stance_back_y = stance["right_ankle_y"].mean()

    stride_vec_x = stance_front_x - stance_back_x
    stride_vec_y = stance_front_y - stance_back_y

    stride_width = np.sqrt(stride_vec_x**2 + stride_vec_y**2)

    stride_unit_x = stride_vec_x / stride_width
    stride_unit_y = stride_vec_y / stride_width

    dx = segment["right_ankle_x"] - stance_back_x
    dy = segment["right_ankle_y"] - stance_back_y

    signed_progression = (dx * stride_unit_x + dy * stride_unit_y) / stride_width

    return signed_progression


def compute_weighted_com(df, start_frame, end_frame,
                         stance_start_frame, stance_end_frame):
    """Weighted center of mass shift from stance."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    stance = df[(df["frame"] >= stance_start_frame) &
                (df["frame"] <= stance_end_frame)]

    shoulder_x = (segment["left_shoulder_x"] + segment["right_shoulder_x"]) / 2
    shoulder_y = (segment["left_shoulder_y"] + segment["right_shoulder_y"]) / 2

    hip_x = (segment["left_hip_x"] + segment["right_hip_x"]) / 2
    hip_y = (segment["left_hip_y"] + segment["right_hip_y"]) / 2

    knee_x = (segment["left_knee_x"] + segment["right_knee_x"]) / 2
    knee_y = (segment["left_knee_y"] + segment["right_knee_y"]) / 2

    com_x = 0.25 * shoulder_x + 0.45 * hip_x + 0.30 * knee_x
    com_y = 0.25 * shoulder_y + 0.45 * hip_y + 0.30 * knee_y

    stance_shoulder_x = ((stance["left_shoulder_x"] + stance["right_shoulder_x"]) / 2).mean()
    stance_shoulder_y = ((stance["left_shoulder_y"] + stance["right_shoulder_y"]) / 2).mean()

    stance_hip_x = ((stance["left_hip_x"] + stance["right_hip_x"]) / 2).mean()
    stance_hip_y = ((stance["left_hip_y"] + stance["right_hip_y"]) / 2).mean()

    stance_knee_x = ((stance["left_knee_x"] + stance["right_knee_x"]) / 2).mean()
    stance_knee_y = ((stance["left_knee_y"] + stance["right_knee_y"]) / 2).mean()

    stance_com_x = 0.25 * stance_shoulder_x + 0.45 * stance_hip_x + 0.30 * stance_knee_x
    stance_com_y = 0.25 * stance_shoulder_y + 0.45 * stance_hip_y + 0.30 * stance_knee_y

    weighted_com = np.sqrt((com_x - stance_com_x)**2 +
                           (com_y - stance_com_y)**2)

    return weighted_com


def compute_trunk_lateral_flexion(df, start_frame, end_frame,
                                  stance_start_frame, stance_end_frame):
    """Trunk lateral flexion change relative to stance (degrees)."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    shoulder_x = (segment["left_shoulder_x"] + segment["right_shoulder_x"]) / 2
    shoulder_y = (segment["left_shoulder_y"] + segment["right_shoulder_y"]) / 2

    hip_x = (segment["left_hip_x"] + segment["right_hip_x"]) / 2
    hip_y = (segment["left_hip_y"] + segment["right_hip_y"]) / 2

    trunk_x = shoulder_x - hip_x
    trunk_y = shoulder_y - hip_y

    current_trunk_angle = np.degrees(np.arctan2(trunk_x, -trunk_y))

    stance = df[(df["frame"] >= stance_start_frame) &
                (df["frame"] <= stance_end_frame)]

    stance_shoulder_x = ((stance["left_shoulder_x"] + stance["right_shoulder_x"]) / 2).mean()
    stance_shoulder_y = ((stance["left_shoulder_y"] + stance["right_shoulder_y"]) / 2).mean()
    stance_hip_x = ((stance["left_hip_x"] + stance["right_hip_x"]) / 2).mean()
    stance_hip_y = ((stance["left_hip_y"] + stance["right_hip_y"]) / 2).mean()

    stance_trunk_x = stance_shoulder_x - stance_hip_x
    stance_trunk_y = stance_shoulder_y - stance_hip_y
    stance_trunk_angle = np.degrees(np.arctan2(stance_trunk_x, -stance_trunk_y))

    flexion_change = current_trunk_angle - stance_trunk_angle

    return flexion_change


def compute_upper_body_rotation(df, start_frame, end_frame,
                                stance_start_frame, stance_end_frame):
    """Upper body rotation relative to stance (degrees)."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    stance = df[(df["frame"] >= stance_start_frame) &
                (df["frame"] <= stance_end_frame)]

    shoulder_x = (segment["left_shoulder_x"] + segment["right_shoulder_x"]) / 2
    shoulder_y = (segment["left_shoulder_y"] + segment["right_shoulder_y"]) / 2

    elbow_x = (segment["left_elbow_x"] + segment["right_elbow_x"]) / 2
    elbow_y = (segment["left_elbow_y"] + segment["right_elbow_y"]) / 2

    current_angle = np.degrees(np.arctan2(
        elbow_y - shoulder_y,
        elbow_x - shoulder_x
    ))

    stance_shoulder_x = ((stance["left_shoulder_x"] + stance["right_shoulder_x"]) / 2).mean()
    stance_shoulder_y = ((stance["left_shoulder_y"] + stance["right_shoulder_y"]) / 2).mean()

    stance_elbow_x = ((stance["left_elbow_x"] + stance["right_elbow_x"]) / 2).mean()
    stance_elbow_y = ((stance["left_elbow_y"] + stance["right_elbow_y"]) / 2).mean()

    stance_angle = np.degrees(np.arctan2(
        stance_elbow_y - stance_shoulder_y,
        stance_elbow_x - stance_shoulder_x
    ))

    upper_body_rotation = current_angle - stance_angle

    return upper_body_rotation


def compute_lower_body_rotation(df, start_frame, end_frame,
                                stance_start_frame, stance_end_frame):
    """Lower body rotation relative to stance (degrees)."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    stance = df[(df["frame"] >= stance_start_frame) &
                (df["frame"] <= stance_end_frame)]

    hip_x = (segment["left_hip_x"] + segment["right_hip_x"]) / 2
    hip_y = (segment["left_hip_y"] + segment["right_hip_y"]) / 2

    knee_x = (segment["left_knee_x"] + segment["right_knee_x"]) / 2
    knee_y = (segment["left_knee_y"] + segment["right_knee_y"]) / 2

    current_angle = np.degrees(np.arctan2(
        knee_y - hip_y,
        knee_x - hip_x
    ))

    stance_hip_x = ((stance["left_hip_x"] + stance["right_hip_x"]) / 2).mean()
    stance_hip_y = ((stance["left_hip_y"] + stance["right_hip_y"]) / 2).mean()

    stance_knee_x = ((stance["left_knee_x"] + stance["right_knee_x"]) / 2).mean()
    stance_knee_y = ((stance["left_knee_y"] + stance["right_knee_y"]) / 2).mean()

    stance_angle = np.degrees(np.arctan2(
        stance_knee_y - stance_hip_y,
        stance_knee_x - stance_hip_x
    ))

    lower_body_rotation = current_angle - stance_angle

    return lower_body_rotation


def compute_front_knee_angle(df, start_frame, end_frame):
    """Front knee angle (degrees)."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    hip = np.column_stack((segment["left_hip_x"], segment["left_hip_y"]))
    knee = np.column_stack((segment["left_knee_x"], segment["left_knee_y"]))
    ankle = np.column_stack((segment["left_ankle_x"], segment["left_ankle_y"]))

    v1 = hip - knee
    v2 = ankle - knee

    cosine = np.sum(v1 * v2, axis=1) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    )
    cosine = np.clip(cosine, -1.0, 1.0)

    front_knee_angle = np.degrees(np.arccos(cosine))

    return front_knee_angle


def compute_back_knee_angle(df, start_frame, end_frame):
    """Back knee angle (degrees)."""
    segment = df[(df["frame"] >= start_frame) &
                 (df["frame"] <= end_frame)].copy()

    hip = np.column_stack((segment["right_hip_x"], segment["right_hip_y"]))
    knee = np.column_stack((segment["right_knee_x"], segment["right_knee_y"]))
    ankle = np.column_stack((segment["right_ankle_x"], segment["right_ankle_y"]))

    v1 = hip - knee
    v2 = ankle - knee

    cosine = np.sum(v1 * v2, axis=1) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    )
    cosine = np.clip(cosine, -1.0, 1.0)

    back_knee_angle = np.degrees(np.arccos(cosine))

    return back_knee_angle


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

# df = pd.read_csv('keypoints.csv')
#
# # STANCE FEATURES
# stance_hip_direction = compute_hip_direction(df, start_frame=0, end_frame=9)
# stance_shoulder_angle = compute_shoulder_line_progression_angle(df, start_frame=0, end_frame=9)
# stance_stride_angle = compute_stride_line_progression_angle(df, start_frame=0, end_frame=9)
# stance_hip_shoulder = compute_hip_shoulder_alignment(df, start_frame=0, end_frame=9)
# stance_front_ankle_knee = compute_front_foot_ankle_knee_line(df, start_frame=0, end_frame=9)
# stance_back_ankle_knee = compute_back_foot_ankle_knee_line(df, start_frame=0, end_frame=9)
# stance_stride_w = compute_stride_width(df, start_frame=0, end_frame=9)
#
# # PREPARATION FEATURES
# prep_front_prog = compute_front_foot_progression(df, start_frame=10, end_frame=50,
#                                                  stance_start_frame=0, stance_end_frame=9)
# prep_back_prog = compute_back_foot_progression(df, start_frame=10, end_frame=50,
#                                                stance_start_frame=0, stance_end_frame=9)
# prep_com = compute_weighted_com(df, start_frame=10, end_frame=50,
#                                 stance_start_frame=0, stance_end_frame=9)
# prep_trunk_flex = compute_trunk_lateral_flexion(df, start_frame=10, end_frame=50,
#                                                 stance_start_frame=0, stance_end_frame=9)
# prep_upper_rot = compute_upper_body_rotation(df, start_frame=10, end_frame=50,
#                                              stance_start_frame=0, stance_end_frame=9)
# prep_lower_rot = compute_lower_body_rotation(df, start_frame=10, end_frame=50,
#                                              stance_start_frame=0, stance_end_frame=9)
# prep_front_knee = compute_front_knee_angle(df, start_frame=10, end_frame=50)
# prep_back_knee = compute_back_knee_angle(df, start_frame=10, end_frame=50)

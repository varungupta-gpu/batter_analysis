# Front-On Batting Stance and Trigger Classification

## Overview

This document explains how the calculated COCO-17 features are used to identify:

- whether a trigger occurred;
- the type of trigger; and
- the batter's stance type.

Batter handedness must be known first:

- Right-handed batter: left leg is the front leg and right leg is the back leg.
- Left-handed batter: right leg is the front leg and left leg is the back leg.

---

# Part A: Trigger Detection

## 1. Detecting a Trigger

### Features Used

- `front_foot_progression`
- `back_foot_progression`
- `front_ankle_displacement`
- `back_ankle_displacement`
- `front_knee_displacement`
- `back_knee_displacement`
- `stride_width`
- `knee_to_knee_distance`

### Calculation

Compare each feature during the load-up period with its value during the stable stance.

A trigger is detected when:

- ankle movement is greater than the normal stance movement;
- the movement continues across multiple frames;
- more than one related feature changes; and
- ankle movement is supported by movement of the corresponding knee.

The first frame with sustained movement is the **trigger start**. The trigger ends when the feet settle or the preparatory movement changes into the next batting action.

If these conditions are not met, classify the result as **No Trigger**.

---

# Part B: Trigger-Type Classification

## 2. First-Moving Foot

Compare the first frame in which each ankle displacement becomes greater than its stable-stance movement.

- Front ankle moves first: front-foot-led movement.
- Back ankle moves first: back-foot-led movement.
- Both move at nearly the same time: simultaneous movement.

---

## 3. Back-and-Across Trigger

### Features Used

- `back_foot_progression`
- `back_ankle_displacement`
- `front_ankle_displacement`
- `back_knee_displacement`
- First-moving foot

### Classification

Classify as **Back-and-Across** when:

- the back foot moves first;
- back-ankle displacement is greater than front-ankle displacement;
- the back foot shows clear progression from its stance position; and
- the back knee moves with the back ankle.

In simple terms:

> **The back foot moves first and moves more than the front foot.**

From a front-on 2D view, the across component is more visible than the backward component. If depth movement is unclear, report **Back-Foot-Led / Back-and-Across**.

---

## 4. Forward Press

### Features Used

- `front_foot_progression`
- `front_ankle_displacement`
- `back_ankle_displacement`
- `front_knee_displacement`
- First-moving foot

### Classification

Classify as **Forward Press** when:

- the front foot moves first;
- front-ankle displacement is greater than back-ankle displacement;
- the front foot shows clear progression from its stance position; and
- the front knee moves with the front ankle.

In simple terms:

> **The front foot moves first and moves more than the back foot.**

A small front-foot tap or lift is included when it is part of the same front-foot-led movement.

---

## 5. Advance Down the Pitch

### Features Used

- `front_ankle_displacement`
- `back_ankle_displacement`
- `front_knee_displacement`
- `back_knee_displacement`
- `stride_width`
- `knee_to_knee_distance`
- `weighted_com`
- Apparent body-size change

### Classification

Classify as **Advance Down the Pitch** when:

- both ankles show large displacement;
- both feet begin moving together or nearly together;
- both knees move with the ankles;
- the estimated body centre moves with the feet; and
- movement is substantially larger than a normal press.

An increase in bounding-box size, shoulder width, hip width or torso length can support movement toward the camera.

In simple terms:

> **Both feet and the body show a large movement toward the bowler.**

Depth movement cannot be measured accurately from normal front-on 2D keypoints.

---

## 6. No Trigger

Classify as **No Trigger** when:

- both ankles remain close to their stance positions;
- knee movement remains small;
- stride width and knee distance remain stable; and
- no coordinated movement continues across multiple frames.

---

## 7. Trigger Type Uncertain

Report **Trigger Detected — Type Uncertain** when a valid movement is detected but:

- the first-moving foot cannot be identified;
- both feet move similarly but not enough for an advance;
- ankle and knee movement do not agree; or
- important keypoints are missing or unreliable.

---

# Part C: Stance Classification

Use the stable frames before the trigger. Calculate each stance feature across these frames and use the median value.

The final stance contains four independent characteristics:

> **Open/Closed-Neutral + Wide/Normal/Narrow + Upright/Normal/Crouched + Front/Central/Back Loaded**

---

## 8. Open or Closed/Neutral

### Features Used

- `shoulder_line_progression_angle`
- `hip_direction`
- `stride_line_progression_angle`
- `hip_shoulder_alignment`
- Visible shoulder and hip separation

### Calculation

Calculate the shoulder, hip and ankle-line directions. Also measure visible shoulder and hip widths and check whether the paired keypoints overlap.

### Classification

**Open:**

- shoulders and hips are clearly separated;
- chest is more visible to the camera; and
- shoulder, hip and ankle lines indicate a more front-facing position.

**Closed/Neutral:**

- shoulders or hips overlap more;
- visible shoulder and hip widths are smaller; and
- there is insufficient evidence for an open stance.

Closed and neutral are combined because COCO-17 does not provide toe and heel directions.

---

## 9. Wide, Normal or Narrow

### Features Used

- `stride_width`
- Visible shoulder width

### Calculation

```text
base_width_ratio = stride_width / shoulder_width
```

### Classification

- Smaller ratio: **Narrow**
- Middle range: **Normal**
- Larger ratio: **Wide**

The numerical boundaries must be selected from manually labelled examples. If an ankle is hidden or overlapping, report **Width Uncertain**.

---

## 10. Upright, Normal or Crouched

### Features Used

- `front_knee_angle`
- `back_knee_angle`
- Average knee angle
- Hip angle
- Trunk angle
- Hip-centre height

### Calculation

```text
average_knee_angle = (front_knee_angle + back_knee_angle) / 2
```

The hip angle is calculated using **shoulder → hip → knee**. The trunk angle is calculated from the hip centre to the shoulder centre relative to vertical.

### Classification

**Upright:**

- knee angles are closer to `180°`;
- hip bending is low;
- hip centre is higher; and
- trunk is closer to vertical.

**Normal:**

- knee bending and hip height are between the upright and crouched ranges.

**Crouched:**

- both knee angles are smaller;
- hip bending is greater;
- hip centre is lower; and
- the body appears more compressed.

One bent knee alone should not classify the stance as crouched.

---

## 11. Front-Foot, Central or Back-Foot Loaded

### Features Used

- `front_knee_angle`
- `back_knee_angle`

### Calculation

```text
knee_angle_difference = front_knee_angle - back_knee_angle
```

### Classification

- `front_knee_angle < back_knee_angle`: **Front-Foot Loaded**
- both angles are approximately equal: **Centrally Loaded**
- `back_knee_angle < front_knee_angle`: **Back-Foot Loaded**

A small angle difference should be treated as central loading. This is only an estimate from knee flexion and does not measure actual force or weight distribution.

---

## Threshold Rule

Do not invent universal numerical thresholds. Movement, width and posture boundaries should be calibrated using manually labelled examples from the same camera setup.

If the available features do not clearly support one category, return **Uncertain** instead of forcing a classification.

# Front-On Trigger Feature Calculations Using COCO-17 Keypoints

## Overview

This document explains how the eight features used for front-on trigger analysis are calculated from COCO-17 keypoints.

The stable stance position is used as the starting reference for measuring movement during the trigger.

For a conventional right-handed batter:

- Left ankle and left knee are treated as the front side.
- Right ankle and right knee are treated as the back side.

For a conventional left-handed batter, this mapping is reversed.

---

## 1. `front_foot_progression`

This feature measures how far the front ankle moves along the original stance line.

### Keypoints Used

- Front ankle
- Back ankle

### Calculation

First, calculate the average front- and back-ankle positions during the stance.

```text
stance_vector = stance_front_ankle - stance_back_ankle
stance_width = magnitude(stance_vector)
```

Convert the stance vector into a unit vector. Then project the front ankle's movement from its stance position onto this direction:

```text
front_foot_progression =
((current_front_ankle - stance_front_ankle) · stance_unit_vector)
/ stance_width
```

### Meaning

- Positive value: movement along the back-to-front direction of the original stance line.
- Negative value: movement in the opposite direction.
- Value near zero: little movement along the stance line.

The value is divided by the original stance width so that it is less dependent on image size.

---

## 2. `back_foot_progression`

This feature measures how far the back ankle moves along the original stance line.

### Keypoints Used

- Front ankle
- Back ankle

### Calculation

Use the same stance unit vector and original stance width calculated for `front_foot_progression`.

```text
back_foot_progression =
((current_back_ankle - stance_back_ankle) · stance_unit_vector)
/ stance_width
```

### Meaning

- Positive value: movement along the back-to-front direction of the original stance line.
- Negative value: movement in the opposite direction.
- Value near zero: little movement along the stance line.

Comparing front- and back-foot progression shows which foot moves more along the original stance direction.

---

## 3. `stride_width`

This feature measures the distance between the front and back ankles in each frame.

### Keypoints Used

- Front ankle
- Back ankle

### Calculation

```text
stride_width = sqrt(
    (front_ankle_x - back_ankle_x)²
    + (front_ankle_y - back_ankle_y)²
)
```

### Meaning

- Increasing value: the visible distance between the ankles is becoming wider.
- Decreasing value: the visible distance between the ankles is becoming narrower.
- Stable value: the visible base width is not changing significantly.

This is a 2D ankle-to-ankle distance in the image. It is not the actual physical distance between the feet.

---

## 4. `front_ankle_displacement`

This feature measures the total movement of the front ankle from its stance position.

### Keypoints Used

- Front ankle

### Calculation

```text
front_ankle_displacement = sqrt(
    (current_front_ankle_x - stance_front_ankle_x)²
    + (current_front_ankle_y - stance_front_ankle_y)²
)
```

### Meaning

- Larger value: the front ankle has moved farther from its original stance position.
- Smaller value: the front ankle remains close to its original stance position.

Unlike `front_foot_progression`, this is an unsigned distance and includes movement in every visible 2D direction.

---

## 5. `back_ankle_displacement`

This feature measures the total movement of the back ankle from its stance position.

### Keypoints Used

- Back ankle

### Calculation

```text
back_ankle_displacement = sqrt(
    (current_back_ankle_x - stance_back_ankle_x)²
    + (current_back_ankle_y - stance_back_ankle_y)²
)
```

### Meaning

- Larger value: the back ankle has moved farther from its original stance position.
- Smaller value: the back ankle remains close to its original stance position.

Comparing front- and back-ankle displacement shows which foot moves more during the trigger.

---

## 6. `front_knee_displacement`

This feature measures the total movement of the front knee from its stance position.

### Keypoints Used

- Front knee

### Calculation

```text
front_knee_displacement = sqrt(
    (current_front_knee_x - stance_front_knee_x)²
    + (current_front_knee_y - stance_front_knee_y)²
)
```

### Meaning

- Larger value: the front knee has moved farther from its stance position.
- Smaller value: the front knee remains close to its stance position.

This feature can be compared with `front_ankle_displacement` to check whether the front knee moves with the front foot.

---

## 7. `back_knee_displacement`

This feature measures the total movement of the back knee from its stance position.

### Keypoints Used

- Back knee

### Calculation

```text
back_knee_displacement = sqrt(
    (current_back_knee_x - stance_back_knee_x)²
    + (current_back_knee_y - stance_back_knee_y)²
)
```

### Meaning

- Larger value: the back knee has moved farther from its stance position.
- Smaller value: the back knee remains close to its stance position.

This feature can be compared with `back_ankle_displacement` to check whether the back knee moves with the back foot.

---

## 8. `knee_to_knee_distance`

This feature measures the distance between the front and back knees in each frame.

### Keypoints Used

- Front knee
- Back knee

### Calculation

```text
knee_to_knee_distance = sqrt(
    (front_knee_x - back_knee_x)²
    + (front_knee_y - back_knee_y)²
)
```

### Meaning

- Increasing value: the visible distance between the knees is increasing.
- Decreasing value: the visible distance between the knees is decreasing.
- Stable value: the visible knee separation is not changing significantly.

This feature should be considered together with `stride_width`, front-knee displacement and back-knee displacement.

---

## 9. `hip_direction`

This feature measures the direction of the hip line relative to the vertical image axis.

### Keypoints Used

- Left hip
- Right hip

### Calculation

```text
dx = right_hip_x - left_hip_x
dy = right_hip_y - left_hip_y

hip_direction = atan2(dx, -dy) × (180 / π)
```

### Meaning

- `0°`: the hip line is vertical in the image.
- Positive or negative values: the hip line is rotated to either side of the vertical axis.
- A changing value shows a change in visible hip orientation.

---

## 10. `shoulder_line_progression_angle`

This feature measures the direction of the shoulder line relative to the vertical image axis.

### Keypoints Used

- Left shoulder
- Right shoulder

### Calculation

```text
dx = right_shoulder_x - left_shoulder_x
dy = right_shoulder_y - left_shoulder_y

shoulder_line_progression_angle = atan2(dx, -dy) × (180 / π)
```

### Meaning

- `0°`: the shoulder line is vertical in the image.
- Positive or negative values show the visible direction of the shoulder line.
- Changes across frames indicate that the shoulder orientation is changing.

---

## 11. `stride_line_progression_angle`

This feature measures the direction of the line between the ankles relative to the vertical image axis.

### Keypoints Used

- Left ankle
- Right ankle

### Calculation

```text
dx = right_ankle_x - left_ankle_x
dy = right_ankle_y - left_ankle_y

stride_line_progression_angle = atan2(dx, -dy) × (180 / π)
```

### Meaning

- `0°`: the ankle line is vertical in the image.
- Positive or negative values show the visible direction of the stride line.
- A changing value indicates that the ankle-line orientation is changing.

---

## 12. `hip_shoulder_alignment`

This feature measures the angle between the hip line and the shoulder line.

### Keypoints Used

- Left hip
- Right hip
- Left shoulder
- Right shoulder

### Calculation

```text
hip_vector = right_hip - left_hip
shoulder_vector = right_shoulder - left_shoulder

cosine = dot(hip_vector, shoulder_vector)
         / (magnitude(hip_vector) × magnitude(shoulder_vector))

hip_shoulder_alignment = arccos(cosine) × (180 / π)
```

### Meaning

- Smaller angle: hip and shoulder lines are more parallel.
- Larger angle: greater visible separation between the hip and shoulder lines.

This is a projected 2D alignment measurement, not true three-dimensional torso rotation.

---

## 13. `front_foot_ankle_knee_line`

This feature measures the direction of the front ankle-to-knee line relative to the horizontal image axis.

### Keypoints Used

- Front knee
- Front ankle

### Calculation

```text
front_foot_ankle_knee_line = atan2(
    front_ankle_y - front_knee_y,
    front_ankle_x - front_knee_x
) × (180 / π)
```

### Meaning

- The value describes the visible direction of the front lower leg.
- A changing value indicates a change in the front knee–ankle alignment.

---

## 14. `back_foot_ankle_knee_line`

This feature measures the direction of the back ankle-to-knee line relative to the horizontal image axis.

### Keypoints Used

- Back knee
- Back ankle

### Calculation

```text
back_foot_ankle_knee_line = atan2(
    back_ankle_y - back_knee_y,
    back_ankle_x - back_knee_x
) × (180 / π)
```

### Meaning

- The value describes the visible direction of the back lower leg.
- A changing value indicates a change in the back knee–ankle alignment.

---

## 15. `weighted_com`

This feature estimates body-centre movement from the stance position.

### Keypoints Used

- Left and right shoulders
- Left and right hips
- Left and right knees

### Calculation

First, calculate the shoulder, hip and knee centres:

```text
shoulder_centre = (left_shoulder + right_shoulder) / 2
hip_centre = (left_hip + right_hip) / 2
knee_centre = (left_knee + right_knee) / 2
```

Calculate the weighted body centre:

```text
weighted_centre =
0.25 × shoulder_centre
+ 0.45 × hip_centre
+ 0.30 × knee_centre
```

Then calculate its distance from the stance weighted centre:

```text
weighted_com = distance(current_weighted_centre, stance_weighted_centre)
```

### Meaning

- Larger value: the estimated body centre has moved farther from its stance position.
- Smaller value: the estimated body centre remains close to its stance position.

This is only a weighted keypoint estimate and is not the batter's true centre of mass.

---

## 16. `trunk_lateral_flexion`

This feature measures the change in sideways trunk angle from the stance position.

### Keypoints Used

- Left shoulder
- Right shoulder
- Left hip
- Right hip

### Calculation

```text
shoulder_centre = (left_shoulder + right_shoulder) / 2
hip_centre = (left_hip + right_hip) / 2

trunk_x = shoulder_centre_x - hip_centre_x
trunk_y = shoulder_centre_y - hip_centre_y

current_trunk_angle = atan2(trunk_x, -trunk_y) × (180 / π)

trunk_lateral_flexion = current_trunk_angle - stance_trunk_angle
```

### Meaning

- Value near zero: trunk angle is close to the stance position.
- Positive or negative values: the trunk has leaned toward either side relative to the stance.

---

## 17. `upper_body_rotation`

This feature estimates upper-body orientation change using the shoulder centre and elbow centre.

### Keypoints Used

- Left shoulder
- Right shoulder
- Left elbow
- Right elbow

### Calculation

```text
shoulder_centre = (left_shoulder + right_shoulder) / 2
elbow_centre = (left_elbow + right_elbow) / 2

current_angle = atan2(
    elbow_centre_y - shoulder_centre_y,
    elbow_centre_x - shoulder_centre_x
) × (180 / π)

upper_body_rotation = current_angle - stance_angle
```

### Meaning

- Value near zero: upper-body orientation is close to the stance position.
- Larger positive or negative values: greater visible change from the stance orientation.

This is an upper-body orientation proxy based on shoulder and elbow centres, not a direct measurement of spinal rotation.

---

## 18. `lower_body_rotation`

This feature estimates lower-body orientation change using the hip centre and knee centre.

### Keypoints Used

- Left hip
- Right hip
- Left knee
- Right knee

### Calculation

```text
hip_centre = (left_hip + right_hip) / 2
knee_centre = (left_knee + right_knee) / 2

current_angle = atan2(
    knee_centre_y - hip_centre_y,
    knee_centre_x - hip_centre_x
) × (180 / π)

lower_body_rotation = current_angle - stance_angle
```

### Meaning

- Value near zero: lower-body orientation is close to the stance position.
- Larger positive or negative values: greater visible change from the stance orientation.

This is a lower-body orientation proxy and not a direct measurement of pelvic rotation.

---

## 19. `front_knee_angle`

This feature measures the joint angle at the front knee.

### Keypoints Used

- Front hip
- Front knee
- Front ankle

### Calculation

Create two vectors from the front knee:

```text
vector_1 = front_hip - front_knee
vector_2 = front_ankle - front_knee

cosine = dot(vector_1, vector_2)
         / (magnitude(vector_1) × magnitude(vector_2))

front_knee_angle = arccos(cosine) × (180 / π)
```

### Meaning

- Angle closer to `180°`: the front leg is straighter.
- Smaller angle: the front knee is more flexed.

---

## 20. `back_knee_angle`

This feature measures the joint angle at the back knee.

### Keypoints Used

- Back hip
- Back knee
- Back ankle

### Calculation

Create two vectors from the back knee:

```text
vector_1 = back_hip - back_knee
vector_2 = back_ankle - back_knee

cosine = dot(vector_1, vector_2)
         / (magnitude(vector_1) × magnitude(vector_2))

back_knee_angle = arccos(cosine) × (180 / π)
```

### Meaning

- Angle closer to `180°`: the back leg is straighter.
- Smaller angle: the back knee is more flexed.

---

## Front-On View Limitation

All eight features are calculated from 2D image coordinates. Movement toward or away from the camera cannot be measured accurately. Therefore, these values represent visible movement in the front-on image rather than exact three-dimensional movement.

# Trigger Corrections and Potential Injury Risks

## Overview

Use this document only after the trigger type, trigger-start frame and trigger-end frame have been identified.

A back-and-across trigger, forward press or advance is not automatically correct or incorrect. The system should preserve the batter's natural trigger type and suggest a correction only when a clear problem is visible.

Compare the trigger with:

- the stable stance immediately before it;
- the batter's normal trigger pattern across repeated deliveries; and
- the final position normally reached before ball release.

The error and detection sections below help the LLM understand the problem. The final output should contain only:

1. **Correction**
2. **Potential injury risk**

---

## 1. Inconsistent Trigger Timing

### Error

The trigger begins much earlier or later than usual, takes longer or shorter than normal, or remains incomplete at ball release.

### How to Identify It

- Compare trigger-start, trigger-end and ball-release frames.
- Convert frame differences into time using video FPS.
- Compare the duration and completion time across repeated deliveries.
- Do not label one early or late trigger as a problem unless it differs clearly from the batter's normal range.

### Correction

Use the same repeatable bowler cue to begin the trigger. Practise completing the movement before ball release so the feet and body are settled without rushing.

### Potential Injury Risk

Timing variation alone has no clear injury risk. However, a late and rushed trigger may indirectly produce an unstable landing, sudden joint loading or poor body control.

---

## 2. Inconsistent First-Moving Foot

### Error

The foot that initiates the trigger changes unexpectedly across similar deliveries even though the trigger type is supposed to remain the same.

### How to Identify It

- Find the first significant movement of both ankles.
- Confirm ankle movement using the corresponding knee.
- Compare the detected sequence with the batter's normal trigger sequence.
- Do not treat nearly simultaneous movement as a definite change in sequence.

### Correction

Rehearse the batter's chosen trigger slowly and consistently so the intended foot initiates the movement each time. The goal is repeatability, not changing the trigger style.

### Potential Injury Risk

There is no direct injury risk from which foot moves first. Repeated unexpected changes may indicate poor coordination and can contribute to an unstable final position.

---

## 3. Excessive Front- or Back-Foot Movement

### Error

The front or back foot travels much farther than the batter's usual range or finishes in an unfamiliar position.

### How to Identify It

- Use `front_ankle_displacement` and `back_ankle_displacement`.
- Use `front_foot_progression` and `back_foot_progression` to determine movement along the stance line.
- Compare movement with the original `stride_width`.
- Compare the final ankle positions with the batter's usual trigger-ending positions.

### Correction

Reduce unnecessary foot travel while maintaining the same trigger type. Aim to place the moving foot in the batter's normal final position and finish with the body supported by both legs.

### Potential Injury Risk

Large movement alone is not an injury risk. Repeated excessive movement may increase loss of balance or uncontrolled loading at the ankle, knee or hip, particularly when the foot lands away from the body centre.

---

## 4. Inconsistent Foot-Movement Direction

### Error

A foot repeatedly changes direction, moves along an unfamiliar path or makes unnecessary sideways corrections during the trigger.

### How to Identify It

- Compare each ankle path with the original stance line.
- Check whether the movement is along or across the line.
- Check whether the path progresses consistently across consecutive frames.
- Compare the direction with the batter's usual trigger path.

### Correction

Use one smooth and repeatable foot path. Practise the movement slowly first, then increase speed while preserving the same direction and final foot position.

### Potential Injury Risk

Direction change alone does not establish an injury risk. Concern increases when the foot changes direction while the knee remains fixed, producing visible twisting, knee collapse or an uncontrolled landing.

---

## 5. Poor Ankle–Knee Coordination

### Error

The ankle moves without related movement of the corresponding knee, or the ankle and knee travel in visibly conflicting directions.

### How to Identify It

- Compare `front_ankle_displacement` with `front_knee_displacement`.
- Compare `back_ankle_displacement` with `back_knee_displacement`.
- Compare the start frames and movement directions of each ankle–knee pair.
- Treat an isolated ankle jump as possible pose-estimation error before calling it a technique problem.

### Correction

Move the foot and knee as one coordinated leg action. The knee should generally follow the direction of the foot instead of remaining fixed or collapsing as the foot moves.

### Potential Injury Risk

Repeated poor ankle–knee coordination may indicate reduced lower-limb control and may increase stress around the knee or ankle. This should be reported as a potential concern, not as an injury diagnosis.

---

## 6. Body Centre Moving Outside the Support Base

### Error

The hip centre moves too far toward one side, passes outside the area between the feet or continues drifting after the feet settle.

### How to Identify It

- Calculate the centre of the left and right hips.
- Compare its horizontal position with both ankles.
- Use `weighted_com` as supporting evidence.
- Compare the final hip-centre position with the batter's normal range.

### Correction

Keep the hips supported between the feet while completing the trigger. Reduce excessive sideways movement and finish with the body centre controlled over the final base.

### Potential Injury Risk

Repeated movement outside the support base may increase balance loss and produce uneven loading of the knees, ankles, hips or trunk during the next movement.

---

## 7. Head Falling Outside the Base

### Error

The head moves excessively toward the front-foot or back-foot side and finishes outside the batter's normal support area.

### How to Identify It

- Use the nose as the head-position estimate.
- Compare the nose with the positions of both ankles.
- Measure horizontal and vertical movement from trigger start to trigger end.
- Check that the movement continues across several frames and is not a single keypoint jump.

### Correction

Keep the head controlled while the feet move underneath the body. The head does not need to remain perfectly still, but it should finish within the batter's normal balanced position.

### Potential Injury Risk

Head movement is mainly a balance and visual-stability concern. If it repeatedly occurs with torso collapse or loss of balance, it may contribute to uncontrolled loading of the trunk and lower limbs.

---

## 8. Head and Body Moving in Opposite Directions

### Error

The hips and feet move strongly toward one side while the head and shoulders fall toward the opposite side.

### How to Identify It

- Compare movement directions of the nose, shoulder centre, hip centre and ankles.
- Check whether the opposite movement is larger than the batter's usual counter-movement.
- Confirm that the pattern continues across multiple frames.

### Correction

Coordinate the head, trunk and lower body so they move toward a controlled final position. Avoid allowing the head and torso to pull strongly against the direction of the lower body.

### Potential Injury Risk

Repeated large counter-movement may increase balance loss and uncontrolled loading through the trunk, hips, knees and ankles.

---

## 9. Excessive Shoulder Rotation

### Error

The shoulders open or close suddenly, or rotate much farther than the batter's normal trigger pattern.

### How to Identify It

- Track `shoulder_line_progression_angle` from trigger start to trigger end.
- Compare the change with `hip_direction` and `hip_shoulder_alignment`.
- Check whether the rotation is gradual and coordinated or sudden and isolated.

### Correction

Keep the shoulders controlled while the feet initiate the trigger. Allow only the amount of shoulder movement normally required by the batter's technique.

### Potential Injury Risk

Repeated uncontrolled shoulder rotation may increase stress through the upper back, neck or shoulder region, particularly when it is combined with sudden trunk twisting or pain.

---

## 10. Front Shoulder Dropping Excessively

### Error

The front shoulder continues dropping during the trigger and finishes lower than the batter's normal range.

### How to Identify It

- Compare the vertical positions of both shoulders.
- Track the shoulder-line tilt from trigger start to trigger end.
- Check whether the nose and torso also move toward the dropping shoulder.
- Compare with the batter's normal shoulder tilt.

### Correction

Maintain the batter's normal shoulder level during the trigger. Avoid allowing the front shoulder to keep falling as the feet move.

### Potential Injury Risk

Repeated excessive shoulder drop may contribute to side bending and uneven loading of the neck, shoulder and trunk. A stable shoulder tilt that is normal for the batter should not be flagged.

---

## 11. Torso Falling Sideways

### Error

Sideways trunk lean increases throughout the trigger and the torso finishes outside the batter's usual range.

### How to Identify It

- Use `trunk_lateral_flexion`.
- Track the line between the hip centre and shoulder centre.
- Confirm the direction using the nose and shoulder-centre movement.
- Compare the final lean with the stable stance and repeated deliveries.

### Correction

Keep the trunk supported over the hips as the feet move. Reduce unnecessary sideways lean while preserving the batter's natural stance and trigger style.

### Potential Injury Risk

Repeated excessive lateral flexion may increase uneven stress on the lower back and side of the trunk, especially when the movement is sudden or associated with pain.

---

## 12. Excessive Upper-Body Movement

### Error

The head, shoulders and torso move much more than required during a trigger that should mainly prepare the lower body.

### How to Identify It

- Track the nose, shoulder centre and hip centre.
- Compare upper-body displacement with ankle displacement.
- Use `upper_body_rotation` and `trunk_lateral_flexion` as supporting features.
- Compare the movement with the batter's normal trigger.

### Correction

Let the feet initiate the preparatory movement while keeping the head and upper body controlled. Remove extra upper-body movement that does not help reach the final trigger position.

### Potential Injury Risk

Repeated uncontrolled upper-body movement may increase balance loss and unnecessary loading of the neck, shoulders or back. It does not identify a specific injury.

---

## 13. Loss of Hip–Shoulder Coordination

### Error

The hips and shoulders rotate at unrelated times or the difference between their orientations changes suddenly.

### How to Identify It

- Compare `hip_direction` with `shoulder_line_progression_angle`.
- Track `hip_shoulder_alignment` across the trigger.
- Use `upper_body_rotation` and `lower_body_rotation` to compare timing.
- Determine whether the change is outside the batter's normal movement pattern.

### Correction

Maintain coordinated movement between the hips and shoulders. Avoid sudden twisting of one body segment while the other segment remains fixed.

### Potential Injury Risk

Repeated uncontrolled twisting may increase stress through the trunk and lower back. Two-dimensional keypoints cannot measure actual spinal forces.

---

## 14. Knee Collapsing Inward or Outward

### Error

The front or back knee moves inward or outward beyond the batter's normal range while the foot moves or settles.

### How to Identify It

- Track the hip, knee and ankle of the affected leg.
- Use `front_foot_ankle_knee_line` or `back_foot_ankle_knee_line`.
- Compare knee displacement with ankle displacement.
- Confirm that the visible collapse continues across multiple frames.

### Correction

Keep the knee tracking in the same general direction as the foot throughout the trigger. Control the leg as the foot moves and settles instead of allowing the knee to fall inward or outward.

### Potential Injury Risk

Repeated visible knee collapse may indicate poor lower-limb control and may increase stress around the knee. This is only a potential concern from a front-on 2D view and not an injury diagnosis.

---

## 15. Excessive or Asymmetric Knee Flexion

### Error

One knee bends or straightens much more than the batter's usual pattern, creating an uncontrolled difference between the legs.

### How to Identify It

- Compare `front_knee_angle` with `back_knee_angle`.
- Measure the change in each angle from trigger start to trigger end.
- Compare the final angles with the batter's normal trigger-ending position.
- Do not flag asymmetry that is a stable part of the batter's chosen technique.

### Correction

Control the amount of knee bend in both legs and finish with the lower body supporting the batter's normal balanced position. Correct only the unexpected asymmetry, not the batter's natural loading preference.

### Potential Injury Risk

Repeated uncontrolled asymmetry may place more load on one knee, hip or ankle. A naturally asymmetric but stable trigger should not be labelled risky.

---

## 16. Excessive Stance-Width Change

### Error

The trigger finishes much wider or narrower than the batter's usual final base.

### How to Identify It

- Compare `stride_width` before and after the trigger.
- Compare `knee_to_knee_distance` before and after the trigger.
- Check whether the hips and head remain supported by the new base.
- Compare the result with repeated deliveries.

### Correction

Finish with a base width that preserves balance and allows movement in either direction. Reduce excessive widening or narrowing while keeping the same trigger style.

### Potential Injury Risk

Stance width alone is not an injury risk. Concern increases when the changed width repeatedly causes knee collapse, uneven loading or loss of balance.

---

## 17. Jerky or Uncontrolled Trigger Movement

### Error

The trigger repeatedly stops, restarts, reverses direction or contains sudden movements that are not coordinated across connected joints.

### How to Identify It

- Examine ankle, knee and hip paths across consecutive frames.
- Check for repeated changes in movement direction.
- Compare total path length with final displacement.
- Exclude missing or low-confidence keypoints before reporting the problem.

### Correction

Perform the trigger as one smooth and continuous preparatory action. Practise the movement slowly and progressively increase speed without adding stops or extra adjustments.

### Potential Injury Risk

Repeated jerky movement may cause uncontrolled joint loading, particularly when combined with poor alignment, instability or pain.

---

## 18. Unstable Final Trigger Position

### Error

After the trigger should be complete, the feet, knees, hips, head or torso continue adjusting instead of reaching a controlled position.

### How to Identify It

After the trigger-end frame, check whether:

- ankle positions continue changing;
- knee angles continue changing;
- the hip centre or head continues drifting;
- the shoulders continue rotating;
- trunk lean continues increasing; or
- stance width continues changing.

### Correction

Complete the trigger early enough to settle the feet, head and body before responding to the ball. Aim to reach the batter's usual final position without additional adjustments.

### Potential Injury Risk

An unstable final position may increase balance loss and uncontrolled loading during the next batting movement, particularly if the batter must react quickly from that position.

---

## Decision Rules

Generate a correction only when:

1. The error is visible across multiple frames.
2. The required keypoints are reliable.
3. Connected body parts support the observation.
4. The movement is outside the batter's normal range or produces a visibly unstable position.

Do not generate an injury warning from one unusual frame. Do not treat the trigger type itself as an injury risk.

The system cannot diagnose an injury or calculate an injury probability from COCO-17 keypoints. It may only identify a body region that could experience increased stress if the visible problem is repeated.

If no meaningful error is detected, use:

```text
Correction: No correction required; the trigger remained controlled and consistent.

Potential injury risk: No clear injury-related concern detected from the visible trigger movement.
```

If a meaningful error is detected, use:

```text
Correction: <state the body part, the detected problem and one practical action cue>

Potential injury risk: <state the potentially affected body area and why repeated movement may increase stress>
```

Do not include the internal **Error** or **How to Identify It** sections in the final output.

# Creep toward an item (ElRobot)

Close the last few centimetres to a salami slice or cheese stick in small, checked steps using
both cameras, instead of one long move on a single estimate. Use it whenever the arm is on the
**real robot** (the station reports `plant: remote-real`), where you are in the cloud and the arm
is across a network link; it also works in the simulator. Read
[elrobot-tool-usage](../elrobot-tool-usage/SKILL.md) first for frames, gripper semantics and
the approach planner. Creeping ends at a **hover** pose; the descent and grasp are the
[pizza-assembly](pizza-assembly.md) pick procedure.

## Why small steps

You decide in the cloud and the arm obeys across a link: every command is a trajectory the robot
executes before you see the result, and you cannot interrupt it. So a command is only as large as
you would be willing to execute blind. The helper sizes it from how sure you are; a missed
estimate then costs a few millimetres, not a swing into a bin wall.

## Procedure

1. **Plan the approach once.** `python -m cap_harness.elrobot.approach X Y GRASP_Z --hover 0.09`
   with the overhead estimate. Keep the first `clear: true` result's tilt and its
   `hover_pose.quaternion_wxyz`; creeping holds that orientation, so a step never needs a wrist
   turn (a wrist turn is where tiny Cartesian steps turn into huge joint swings).
2. **Scout.** One ordinary `cap_move_to_pose` (`max_steps: 800`) to that approach's hover
   orientation about 4-6 cm beside and ~5 cm above the hover point, on the side the arm reaches
   (the item may sit near the inner edge of the reach ring; if `ik_unreachable`, try another side).
   From Home the wrist does not face the table, so wrist localization only becomes useful after this.
3. **Look.** Localize the item on **both** cameras, the wrist one *pinned to the same item*:
   - `cap_localize_object("salami center", camera_name="overhead")` gives X, Y;
   - `cap_localize_object("salami at X,Y", camera_name="wrist")` with that X, Y. Each camera's own
     "center" or "left" can be a different slice; `at X,Y` asks for the one nearest that base-frame point.
   Note each result's position and `segmentation.score`. A camera that finds nothing is simply left out.
4. **Ask for the step.** Pass the fingertip (`cap_get_robot_state`) and the estimates:

   ```bash
   .venv-system2/bin/python -m cap_harness.elrobot.creep --tip X Y Z \
       --est overhead X Y Z SCORE --est wrist X Y Z SCORE \
       --stop-z 0.09 --quat W X Y Z --profile real
   ```

   **`--profile real` on the real robot** (omit it in the simulator). It judges agreement at the
   real cameras' measured noise: overhead and wrist estimates of the same topping differ by a
   median 27 mm there, so the simulator's 15 mm "agree" threshold would shrink every step.

   `--quat` is the approach's `hover_pose.quaternion_wxyz` from step 1. The printed JSON has
   `action`, `confidence`, `notes`, `step_m` and a ready `target_pose`.
5. **Act on `action`:**
   - `move`: `cap_move_to_pose(target_pose, max_steps=<its max_steps>)`. Use the returned images
     as the next look; **do not issue a second move without a fresh localization in between.**
   - `hold`: do not move. Call `observe`, wait for the fresh images, localize again, ask again.
     Hold means nothing was found or confidence is near zero.
   - `done`: the fingertip is over the item at hover height. Verify in the wrist image that the
     item is between the open jaws' line, then continue with the pick (ask the planner again with
     the *refined* X, Y; do not reuse the old ones).
6. Repeat from 3.

## How big, and why

| Situation | Step |
| --- | --- |
| Both cameras agree (< 15 mm), far (> 10 cm) | up to 4 cm |
| Mid range (4-10 cm) | up to 2.5 cm |
| Close (< 4 cm) | up to 1.2 cm |
| Only one camera sees it | confidence capped at 0.6: half steps |
| Cameras disagree by 15-50 mm | step shrinks linearly |
| Cameras > 50 mm apart | likely two different items: wrist dropped, overhead followed, steps ~3-10 mm |
| Segmentation score low | smaller; near zero holds |

It also never covers more than 60% of what is left, moves sideways at the current height until
the fingertip is over the item, and never goes below `--stop-z`. The overhead camera is the
global reference; the wrist camera takes over as the fingertip nears.

## What the real robot's estimates are (measured on the demonstrations)

- On the real robot `cap_localize_object` uses the real-scene segmenter: it knows the toppings
  **by tray** (`salami`, `cheese`, `olives`) and returns the topping inside its tray, not one slice.
  Expect about **25-30 mm** of error (cheese: overhead 29 mm, wrist 21 mm), so creeping gets you
  over the right part of the tray, not onto a particular slice. Choose the slice from the wrist image.
- The wrist camera is the sharper of the two once the fingertip is over the tray; its score
  tracks its error. The overhead score is fixed and carries no information.
- `salami` is the weaker case: its tray position is a by-eye landmark, ~50 mm from where
  operators grab. Trust the wrist view over it as you get close.
- The arm over the tray hides it from the overhead camera; a `no ... pixels inside its tray`
  failure there is occlusion, not absence (see AGENTS.md on occluded views).

## Guards you must respect

- **`planned trajectory exceeds max_steps` on a creep step is a warning, not an inconvenience.**
  A few millimetres should plan in well under a hundred ticks. If it does not, the IK jumped to
  another arm configuration (a wrist roll swing of several radians for a tiny move). Do **not**
  raise `max_steps` to get it through. Halve the step, or scout again from a different side.
- A tracking error on a creep step means contact. Back up one step along your own path and look.
- A localization that jumps more than ~3 cm between two looks while the arm barely moved: treat
  the new one as unconfirmed, ask for a step with it, and expect `hold`/tiny steps until two
  consecutive looks agree.
- The creep ends at hover. Never descend into a bin from here without the approach planner's
  `clear: true` for the refined X, Y.
- Depth on the real robot is synthesized from the table plane (the cameras report none). The
  position of an item's top is a few centimetres short; use the item heights in the task context
  for grasp depth, as the pizza recipe already says.

## Report

Say once what the stage is and the decisive evidence (fingertip within 8 mm of the item's X, Y
at hover height, from two agreeing looks). Do not narrate each step.

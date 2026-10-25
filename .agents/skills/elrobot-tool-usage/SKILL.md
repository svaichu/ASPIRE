---
name: elrobot-tool-usage
description: Operate the single-arm ElRobot in the MuJoCo pizza-table simulation through native CAP MCP tools. Covers frames, camera conventions, gripper semantics, reach, clearance planning, and verification. Use for any task on the ElRobot station (embodiment `elrobot`).
---

# ElRobot tool usage

Follow [AGENTS.md](../../../AGENTS.md), including its **ElRobot station** section. This
skill replaces [cap-tool-usage](../cap-tool-usage/SKILL.md) on this station: that one
describes a two-arm YAM with cuRobo, which does not exist here. Use only the native
`mcp__cap__*` tools. The station is a simulation, but treat it as hardware: the same
rules about verification and recovery apply.

## The robot

One 7-joint arm named `arm` (`primary` resolves to it) with a scissor-jaw gripper. The
station is **sim-only** and **has no collision-aware planner**: motion is mink IK followed
by joint-space interpolation, and nothing between the waypoints is checked against the
bins, plate, table or the arm itself. You own clearance. See
[Clearance](#clearance-you-are-the-collision-checker).

| Tool | Use |
| --- | --- |
| `cap_program_go_home` | Opens the gripper, then joint-space move to the folded Home (the real robot's rest pose, tucked beside the base). Once per fresh session; on request. |
| `cap_move_to_pose` | `target_pose` = `{position:[x,y,z], frame:"robot_base", quaternion_wxyz:[w,x,y,z]}` (or `rpy_deg`, exactly one). Omit `strategy`: mink is the only IK and is the default. **Pass `max_steps: 800`** (see [Speed](#speed)). |
| `cap_move_to_joints` | Seven radian targets. For recovery or a known-safe fold; it is **not** IK. |
| `cap_set_gripper` | `position` 0 closed … 1 open. Returns only after the jaws stop moving. |
| `cap_localize_object` | `query` + optional `camera_name`. Colour segmentation + depth (see [Perception](#perception)). |
| `cap_get_robot_state` | Joint state, fingertip pose, gripper reading. |
| `observe` / `get_job` | Fresh images and state; every action result already carries both cameras. |

`cap_move_synchronized` exists but is meaningless with one arm; do not use it.
Tool results carry terminal state and images; do not insert waits.

## Frames, units, poses

- Everything is in `robot_base`: origin on the table surface under the arm's base,
  metres, **+z up**, the arm reaches toward **+y**, **+x is image-left** in the overhead
  camera (that view is from beyond the plate looking back at the robot: image-top is
  toward the robot, image-bottom away from it).
- `cap_move_to_pose` drives the **fingertip point** (the `grasp_site`, centre of the
  pads at the tip end). Its z axis is the approach direction out of the fingers, its x
  axis is the jaw-closing axis. Poses are checked against 10 mm / 10° on arrival.
- Quaternions are wxyz. **Do not write them by hand.** Generate them (pure geometry, no
  robot access, allowed from the shell):

  ```bash
  .venv-system2/bin/python -m cap_harness.elrobot.poses X Y Z --tilt 40 [--jaw-azimuth 90]
  ```

  `--tilt` is the approach's angle from straight down; it leans **away from the base**
  (`--lean D` turns that direction). The jaw axis defaults to the horizontal tangent
  across the line from the base; `--jaw-azimuth D` instead closes along the base-frame
  direction `D` (0 = +x, 90 = +y). The printed JSON is a ready `target_pose`.

## Speed

The arm moves at the real robot's speeds: per-joint limits of 0.8–1.3 rad/s, taken from the
99th percentile of the real teleoperated demonstrations. A large move takes seconds, which is
many 20 ms ticks, and `cap_move_to_pose` / `cap_move_to_joints` refuse a plan longer than
`max_steps` (default 120 = 2.4 s) with "planned trajectory exceeds max_steps". That is not a
failed pose: **pass `max_steps: 800`** on every motion and the same pose works. Do not ask
for faster; the limit is deliberate (a pinched item does not survive a faster swing).

## Reach

This wrist cannot hold the gripper pointing straight down over most of the table, and
a fixed jaw yaw makes things worse. What was measured:

- **Tilt 40° is reachable everywhere** that matters (bins, plate, hover heights up to
  ~0.11 m). Tilt 30° fails for a few far positions at hover height; 20° fails for many.
  Top-down (0°) is mostly unreachable above ~0.06 m. Start at 40° and fall back.
- **The arm's reach is a ring**: about 0.13 m to 0.33 m from the base, not the whole table.
  The pizza is 36 cm across and its far side is out of reach; only x ≈ 0.14–0.30 of it can be
  placed on (which is also where the real operator put toppings). Aim inside that region.
- A `move_to_pose` whose IK cannot converge fails **before any motion** with
  `ik_unreachable` and the residual. That is information, not a fault: change tilt,
  lean, or height; do not retry the identical pose.
- A motion that ends "N rad from its final waypoint … did not track" **was blocked**
  (contact or a joint limit). Do not push farther, loosen tolerances, or repeat it; back
  out to the hover pose and re-plan.

## Clearance: you are the collision checker

Before any descent into a bin, tray, or onto the plate, ask the approach planner. It
searches tilt/lean combinations for IK-reachable approaches whose joint-space path
(start → hover → grasp) stays clear of the table, bins and plate, using the same
geometry the simulator uses:

```bash
.venv-system2/bin/python -m cap_harness.elrobot.approach X Y GRASP_Z --hover 0.07 [--jaw-azimuth D]
```

It prints up to a few approaches, best first, each with `hover_pose` and `grasp_pose`
ready to pass to `cap_move_to_pose` (move to the hover pose, then the grasp pose, from
the **same** approach so the descent is vertical along the tool axis). `clear: false`
with `blocked_by` means no approach found that avoids that obstacle; do not descend,
pick another item or approach order. What it does **not** know: loose items (an open
jaw reaches **28 mm** to each side of the fingertip and can nudge a neighbour), what the
gripper is holding, and dynamics. "Clear" is necessary, not sufficient — look at the
returned images after every descent.

Rules that follow:

1. Hover first, descend vertically along the tool axis, retreat the same way.
2. Do not call Home from inside a bin or with the arm low over the plate: Home is a
   straight joint-space line with no clearance check. Retreat to a hover pose first.
3. Carrying an item is not covered by the planner. It checks the approach and the descent,
   not the swing between them, and a move interpolates in **joint** space, so the gripper
   can tip away from level mid-swing. A cylinder (a cheese stick) pinched at its middle
   can twist out; a flat slice does not. Check the gripper reading at the destination's
   hover pose before descending (it must still be in band), and split a long carry into a few
   short `move_to_pose` steps at carry height with the same orientation (same `quaternion_wxyz`).
4. Items shift when touched. Re-localize before every pick; never reuse coordinates
   from before a contact.

## Gripper

- `cap_set_gripper` 1.0 opens to a **52 mm** gap between the pads; 0.0 closes them.
  Commanded force is capped (~3.5 N per jaw); intermediate values hold, but you do not
  need them: close with 0.0 and let the jaws stall on the object.
- The pads are flat plates, not the tapered blades in the renders. The grip is a
  pinch between two parallel faces. Wide, flat-sided items hold; rounding and corners
  slip.
- **The reading is evidence, not proof.** After `cap_set_gripper(0.0)` read
  `gripper_positions.arm` (also in `cap_get_robot_state`). Empty ≈ **0.00**. A 29 mm
  salami slice ≈ **0.55–0.60** (0.7+ means a shallow corner grip — release and retry).
  A 14 mm cheese stick ≈ **0.17–0.20**. A reading in band can still be a grip that
  will not survive the lift: **read it again after lifting**. Held keeps its value; an
  item that slips lets the jaws keep closing and the reading drops (to ~0.00 for a
  stick). Then look at the camera: the item must be rising with the fingers.
- Do not open the gripper over the bin or table to "test" a grip.

## Perception

`cap_localize_object` is colour segmentation over the overhead RGB-D camera, not
SAM3. Only the vocabulary below is understood; anything else is refused with the list.

| Query contains | Finds | Notes |
| --- | --- | --- |
| `salami` (`pepperoni`) | red slices | One instance each, in the bin. |
| `cheese` (`yellow`, `shred`) | yellow sticks | One instance each, in the tray. |
| `plate` (`pizza`, `dough`) | the cream disc | One region; includes anything on it. |

- Only the **first** instance is returned, so the query selects it. Add one qualifier
  word: `center`/`middle` (nearest the middle of the remaining ones), `left`, `right`,
  `top`, `bottom` (extremes **as the overhead image shows them**: image-left = +x),
  `smallest`. Bare `salami` returns the largest, least-occluded one.
- Items already on the plate are excluded from `salami`/`cheese` queries so "center"
  keeps meaning "the middle of the bin". Add `on the plate` to find those instead.
- The result pose `z` is the **top surface** of the item. Use the item heights in the
  task context for grasp depth, not the localized z.
- Positions are accurate to a few millimetres for isolated items. A cluster that merges
  into one region (two touching slices) is a sign to re-look, not to trust the centre.
- `camera_name="wrist"` works but only sees what the gripper points at; plan with
  the overhead camera. The arm in view hides what is under it: **retract to Home before
  counting or judging placement**.

## Evidence rules specific to this station

- `ok: true` on a move means the pose was reached, not that anything was grasped.
- Success of a pick: item rises with the fingers (image), reading unchanged after the
  lift, and the bin's instance count drops once the arm is clear.
- Success of a placement: after releasing and retracting to Home, the overhead image
  shows the item on the plate at the intended spot, not touching its neighbours.
- A stick or slice that leaves the table is gone for good. Report it; do not search.

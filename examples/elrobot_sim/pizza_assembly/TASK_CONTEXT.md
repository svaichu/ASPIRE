---
task: "Assemble the pizza: move the requested salami slices and cheese sticks onto the plate."
---

# Pizza assembly (ElRobot simulation)

## Goal

Move the requested number of salami slices and cheese sticks from their containers
onto the round pizza plate, one item at a time, leave them spread out on the plate,
and finish with the gripper open and the arm at Home. Default request when none is
given: **3 salami slices and 4 cheese sticks**. At most 9 of each exist.

Procedure and recovery: [pizza-assembly recipe](../../../.agents/skills/manipulation-tasks/pizza-assembly.md).
Robot and tool facts: [elrobot-tool-usage](../../../.agents/skills/elrobot-tool-usage/SKILL.md).

## Scene (robot base frame, metres; +x is image-left in the overhead view)

The layout follows the real station: it was recovered from 30 real teleoperated demonstrations
(`pizza-robot/real/`), not invented. The real pizza is 36 cm across with the toppings in trays
beside it, the robot at the far edge, and the camera looking back at the robot.

| Thing | Where | Size | Notes |
| --- | --- | --- | --- |
| Pizza | centre (0.26, 0.235), top z 0.010 | radius 0.178 (36 cm) | Cream disc. **Only x ≈ 0.14–0.30 of it is reachable**; place there. |
| Salami bin | centre (−0.04, 0.18), interior ±0.098 | walls 30 mm | Right of the pizza in the overhead image. |
| Salami slices ×9 | x ∈ {−0.10, −0.04, 0.02}, y ∈ {0.12, 0.18, 0.24} | ≈ 29 mm across flats, 41 mm across corners, 16 mm thick | Centre z 0.016, top 0.024. Red. 60 mm pitch. |
| Cheese tray | centre (−0.255, 0.14), interior ±0.088 | walls 30 mm | Right of the salami bin. |
| Cheese sticks ×9 | x ∈ {−0.30, −0.255, −0.21}, y ∈ {0.10, 0.14, 0.18} | 14 mm diameter, 38 mm long, lying along x | Centre z 0.023, top 0.030. Yellow. |
| Robot | base at the origin; Home is the real robot's folded rest pose | — | Clear of the pizza and containers. |

Cameras: `overhead` (calibrated RGB-D, 640×480, fitted to the real camera's pose: high on the far
side of the table, tilted back toward the robot; image-left is +x, image-top is toward the
robot) and `wrist` (on the gripper). The overhead view is the one to count with; the hand hides
what it hovers over.

## What is and is not guaranteed

- The positions above are the initial layout. **Re-localize before every pick**; items
  shift when touched and the layout changes as items are removed.
- Slices were laid on a 60 mm pitch and sticks 40 mm apart across their length because
  the open jaws reach 28 mm either side of the fingertip. That spacing, not the arm,
  is what lets them be picked one at a time.
- Moves run at the real robot's speeds (0.8–1.3 rad/s per joint); pass `max_steps: 800`.
- Carrying is the fragile step, sticks more than slices (see the tool skill).
- There is no collision-aware planner: clearance comes from the approach planner and
  from looking. See the tool skill.
- No ground truth is exposed. Judge from images, the gripper reading and localization.

## Completion evidence

All of these, from the final overhead image with the arm at Home:

1. The requested number of red slices and yellow sticks lie on the plate (query
   `salami on the plate`, `cheese on the plate`, and look).
2. They are spread: none overlaps or is stacked on another, none hangs over the rim.
3. The bins hold the remainder; nothing requested is on the table or floor.
4. The gripper reads open (≥ 0.95) and Home was verified.

If fewer than requested were placed, the task is not complete: report the shortfall and
what blocked it.

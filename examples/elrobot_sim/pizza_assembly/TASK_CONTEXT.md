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

| Thing | Where | Size | Notes |
| --- | --- | --- | --- |
| Plate | centre (0, 0.21), top z 0.010 | radius 0.11 | Cream disc. |
| Salami bin | centre (0.22, 0.16), interior ±0.098 | walls 30 mm | Left of the plate in the overhead image. |
| Salami slices ×9 | x ∈ {0.16, 0.22, 0.28}, y ∈ {0.10, 0.16, 0.22} | ≈ 29 mm across flats, 41 mm across corners, 16 mm thick | Centre z 0.016, top 0.024. Red. 60 mm pitch. |
| Cheese tray | centre (−0.22, 0.16), interior ±0.088 | walls 30 mm | Right of the plate in the overhead image. |
| Cheese sticks ×9 | x ∈ {−0.265, −0.22, −0.175}, y ∈ {0.12, 0.16, 0.20} | 14 mm diameter, 38 mm long, lying along x | Centre z 0.023, top 0.030. Yellow. |
| Robot | base at the origin, Home folded beside it at about (0.06, 0.08, 0.09) | — | Clear of the plate and containers. |

Cameras: `overhead` (calibrated RGB-D, 640×480, looks back at the robot — image-left is
+x, image-top is toward the robot) and `wrist` (on the gripper). The overhead view is
the one to count with; the hand hides what it hovers over.

## What is and is not guaranteed

- The positions above are the initial layout. **Re-localize before every pick**; items
  shift when touched and the layout changes as items are removed.
- Slices were laid on a 60 mm pitch and sticks 40 mm apart across their length because
  the open jaws reach 28 mm either side of the fingertip. That spacing, not the arm,
  is what lets them be picked one at a time.
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

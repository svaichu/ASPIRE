# Pizza assembly (ElRobot)

Move salami slices and cheese sticks from their containers onto the pizza plate with
the single ElRobot arm, using the native CAP tools. Read
[elrobot-tool-usage](../elrobot-tool-usage/SKILL.md) first for frames, gripper
semantics, perception and the approach planner, and the
[task context](../../../examples/elrobot_sim/pizza_assembly/TASK_CONTEXT.md) for the
scene, item dimensions and completion predicate. This recipe is the same procedure as
the runnable reference, `pizza-robot/scripts/reference_episode.py`, which places
3 slices and 4 sticks in about a minute of simulated work.

## Plan once, then loop

State: counts wanted and placed per ingredient, the plate slots already used (each with
its measured position), and what failed. For a multi-stage request announce the first
stage once, then run the loop with tool calls and no commentary.

1. **Home** once at startup (`cap_program_go_home`); report `home_verified`.
2. **Place slots on the plate** before picking. Spread items so a release never lands on
   or drags a placed one: slices on a ring of radius ≈ 55 mm around the plate centre
   (0, 0.21), 120° apart — `(0, 0.265)`, `(−0.048, 0.182)`, `(0.048, 0.182)` — and
   cheese sticks inside the ring at `(0, 0.21)`, `(0, 0.235)`, `(0, 0.185)`,
   `(0.03, 0.21)`. If the operator asks for more, add slots ≥ 35 mm from every
   neighbour and 25 mm inside the plate edge (radius 110 mm); never stack.
3. **Order**: all slices first, then cheese. Within an ingredient take the middle of
   the remaining items first (`salami center`), then whichever qualifier names an item
   with room around it; edge items next to a wall are last.
4. **Per item — pick**, **transit**, **place**, **verify** (below). One item at a time.
5. **Finish** at Home. The overhead view must be unobstructed to count, and the task
   contract requires Home at the end. Then report counts from that image.

## Pick

1. `cap_localize_object` for the next item (overhead camera). Keep its x, y only.
2. Ask the planner for a clear approach to that x, y at the item's **centre height**:

   | Item | grasp z | hover | jaw axis |
   | --- | --- | --- | --- |
   | salami slice | 0.016 | 0.07 | default (tangent) |
   | cheese stick | 0.023 | 0.07 | `--jaw-azimuth 90` (sticks lie along x; close across them) |

   `python -m cap_harness.elrobot.approach X Y Z --hover 0.07 [--jaw-azimuth 90]`.
   It tries tilt 40° first, then 30°, 50°, 20°, with several lean directions. Use the
   `hover_pose` and `grasp_pose` of the first `clear: true` result. No clear result →
   skip this item for now (it may clear once its neighbours are gone); if nothing is
   clear, report which items and the blocking obstacle.
3. `cap_set_gripper(1.0)`; `cap_move_to_pose(hover_pose)`; `cap_move_to_pose(grasp_pose)`.
   Look at the returned image: the open jaws should straddle the item with no neighbour
   between them. A tracking error here means contact — retreat to hover, do not push.
4. `cap_set_gripper(0.0)`. Read the gripper: slice ≈ 0.55–0.60, stick ≈ 0.17–0.20. Empty
   (≈ 0.00) or a slice above 0.7: open, return to hover, take the planner's next
   approach (a different tilt gives different pad contact), or jitter x/y by ≤ 3 mm
   after re-localizing.
5. Lift: `cap_move_to_pose(hover_pose)` of the same approach, then confirm: the item is
   in the fingers in the image and the reading is unchanged. If the reading dropped or
   the item stayed in the bin, the grip failed; open, re-localize, retry.
6. Three failed picks of one item: leave it, take another, report it at the end.

## Transit and place

1. From the lifted hover pose go straight to the plate slot's hover: ask the planner
   for `X Y 0.035 --hover 0.11` at the slot (same jaw axis as the pick), move to its
   `hover_pose` then its `grasp_pose` (the release pose, 35 mm above the table, about
   25 mm above the plate surface).
2. `cap_set_gripper(1.0)`, then `cap_move_to_pose(hover_pose)` to withdraw.
3. Never open while still over a bin: a dropped item lands in the wrong place or off the
   table.

## Verify

After each placement go Home (or at least clear the view) and read the overhead image:
the new item sits at its slot, inside the plate, not overlapping neighbours; the bin
holds one fewer. A localization of `salami on the plate` / `cheese on the plate` returns
placed items for position checks. An item off the table is lost: record it, continue
with what remains, and say so in the final report. Do not claim a count you did not see.

## Recovery

- **IK unreachable / planner blocked**: change item, approach order, or hover height
  (lower hover reaches further than higher). Not a reason to disable any check.
- **Dropped in the bin or on the plate**: re-localize and treat it as a new item. Fixing
  a misplaced item on the plate by picking it up again (centre height = plate top + half
  its thickness) is plausible but not validated; try once, never drag it with a closed
  gripper.
- **Item knocked onto the bare table** (not off it): not validated on this station;
  one attempt with the planner at z = half the item thickness is reasonable, else
  report it.
- Keep the plan's parent stage in mind: recovery moves are not progress. Resume the
  loop at the item that was interrupted.

## Completion

Done when the requested count of each ingredient is on the plate in the final
unobstructed overhead view, nothing requested is unaccounted for, the gripper is open,
and the arm is at Home. Report: counts placed, any items left or lost, and the retries
needed. If a count could not be reached, say which item and the evidence.

---
name: manipulation-tasks
description: Perform task-specific manipulation with native CAP tools. Real YAM - solving or turning a physical Rubik's Cube, bottle-cap removal and pouring, pill insertion and bottle closure, inverted cup nesting. ElRobot simulation - pizza assembly (salami slices and cheese sticks onto the plate).
---

# Manipulation tasks

The recipes below are for the real bimanual YAM, except **pizza assembly**, which is for the
single-arm ElRobot simulation (embodiment `elrobot`). Pick the one matching the station
reported by `get_capability_graph`.

Follow [AGENTS.md](../../../AGENTS.md) and the loaded CAP operational skills.
Read the guide for the requested task:

- [Rubik's Cube](rubiks-cube.md): reconstruct the cube, execute supported layer turns, reorient, and place the solved cube. Includes the reusable solve cycle.

Read the [cube task context](../../../examples/yam_real/rubiks_cube/TASK_CONTEXT.md)
for the requested outcome and the recipe for a new scene.

Other task recipes (read only the one matching the operator's task):

- [Unscrew and pour](../../../examples/yam_real/unscrew_bottle/TASK_CONTEXT.md): stabilize the table-supported bottle, turn and regrasp the cap, verify detachment, and pour onto the table.
- [Pill insertion and bottle closure](../../../examples/yam_real/pill_in_bottle/TASK_CONTEXT.md): acquire the pill, verify entry, seat the cap, and distinguish threaded seating from unverified final tightness.
- [Inverted cup nesting](../../../examples/yam_real/cup_stacking/TASK_CONTEXT.md): count all cups, orient the opening downward, establish insertion before release, and recover without disturbing the existing stack.
- [Pizza assembly (ElRobot sim)](pizza-assembly.md): move salami slices and cheese sticks from their containers to the plate with a clearance-checked approach, verify each grip by lifting, and finish at Home. Read [elrobot-tool-usage](../elrobot-tool-usage/SKILL.md) and the [task context](../../../examples/elrobot_sim/pizza_assembly/TASK_CONTEXT.md) with it.

# Pizza station agent operating guide

These instructions are for the coding agent operating the ElRobot pizza-table CAP MCP
session. `scripts/start_pizza_agent.sh` includes this file in the agent's initial
instructions; Claude Code reads it from the repository. The operator uses the
[pizza-robot runbook](pizza-robot.md) to start the simulation.

Follow the repository's [AGENTS.md](../AGENTS.md), including its **ElRobot station**
section. Before the first robot command read
[.agents/skills/elrobot-tool-usage/SKILL.md](../.agents/skills/elrobot-tool-usage/SKILL.md);
for the pizza task also read
[manipulation-tasks](../.agents/skills/manipulation-tasks/SKILL.md), its
[pizza-assembly](../.agents/skills/manipulation-tasks/pizza-assembly.md) recipe and the
[task context](../examples/elrobot_sim/pizza_assembly/TASK_CONTEXT.md). Read each once and
reuse it through the session. You do not need the YAM skills or the cube guide.

Discover the station through MCP (`get_capability_graph`). The tools are
`cap_program_go_home`, `cap_move_to_pose`, `cap_move_to_joints`, `cap_set_gripper`,
`cap_localize_object`, `cap_get_robot_state`, `observe`, `get_job`. Use `cap_` tools,
never code of your own, to move the arm.

Follow the launcher's startup mode. For a motion-enabled fresh session call
`cap_program_go_home` with `{}` exactly once before the task, wait for the terminal result
and fresh views, and report `READY` with `home_verified`, the joint residual and the
gripper reading. If it fails or the gripper is not open, stop and report. For
observation-only startup observe and report without moving. When the operator says "go
home" call `cap_program_go_home` again; Home opens the gripper, so place anything held
first.

A bare request such as "make the pizza" means the default task: **3 salami slices and 4
cheese sticks** on the plate. Ask only if a requested count exceeds the 9 available of an
ingredient.

Every action result already carries both camera images and the terminal state; use them.
Do not insert waits or poll. The overhead camera is mirrored relative to the robot: its
image-left is the base frame's +x. Retract to Home before counting what is on the plate.

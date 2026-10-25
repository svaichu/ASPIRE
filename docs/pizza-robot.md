# ElRobot pizza station: operator runbook

A single-arm ElRobot in a MuJoCo pizza-making scene, driven by the same CAP MCP session,
tools and agent loop as the real YAM station. Simulation only; there are no arm services,
CAN bus or cameras to set up. What the agent knows is in
[elrobot-tool-usage](../.agents/skills/elrobot-tool-usage/SKILL.md) and the
[pizza-assembly recipe](../.agents/skills/manipulation-tasks/pizza-assembly.md).

## Install

Run from the repository root.

```bash
scripts/bootstrap_system2.sh                    # .venv-system2 (mujoco, mink, viser, mcp, ...)
sudo apt-get install -y libegl1 libopengl0 libusb-1.0-0   # headless GL + RealSense import
```

Offscreen rendering needs EGL. `MUJOCO_GL=egl` is chosen automatically when `DISPLAY` is
unset; set `MUJOCO_GL` yourself to override. The SAM3 and cuRobo providers are **not**
used: segmentation is the built-in colour segmenter and motion is mink IK with
joint-space interpolation.

## Start

```bash
scripts/start_pizza_system2.sh --allow-motion   # MCP at http://127.0.0.1:8222/mcp/
curl --noproxy '*' -s http://127.0.0.1:8222/health
.venv-system2/bin/python pizza-robot/scripts/check_scene.py /tmp/pizza   # optional: layout + camera PNGs
```

Without `--allow-motion` the server is observation-only and rejects every actuating call
before the simulator is touched. `--scene-dir DIR` points at another copy of
`pizza-robot/scene/` (`elrobot_table.xml` + `home.json`); `ASPIRE_ELROBOT_SCENE` does the
same. A fresh server is a fresh scene: restart it to reset the items.

## Attach an agent

- **Claude Code**: open the repository. `.mcp.json` registers the endpoint as `cap`; the
  tools appear as `mcp__cap__*`. Say "Read docs/pizza-agent-guide.md. Make the pizza."
- **Codex**: `scripts/start_pizza_agent.sh --allow-motion "Make the pizza."` (add `exec`
  for non-interactive). Drop the flag for an observation-only start.

Each motion-enabled fresh agent homes once at startup.

## Validate

```bash
.venv-system2/bin/python pizza-robot/scripts/reference_episode.py     # scripted recipe: 3 slices + 4 sticks, ~1 min
.venv-system2/bin/python -m pytest tests/test_elrobot.py              # plant, IK, adapter, perception, grasping, full episode
```

The reference episode scores itself against simulator truth (which an agent never sees)
and exits non-zero unless every requested item is on the plate.

## What the scene is, and why it looks like this

`pizza-robot/scene/elrobot_table.xml` descends from the hand-built scene in
`robotdesigner`, tuned until the gripper could actually do the task. The constraints
that shaped it are worth knowing before editing it:

- **Jaws.** Scissor jaws that pass through each other: fingertips are 52 mm apart at
  `gripper = 0 rad` (open) and meet at 2.2 rad (closed), the opposite of the jaw-body
  separation. The harness maps 1.0 = open, 0.0 = closed.
- **Collision shape.** MuJoCo collides meshes by convex hull, which turns the hook-shaped
  jaws into slanted wedges that squeeze objects out like a seed. The mesh is kept only for
  rendering; each jaw collides as a flat 4 mm pad (`jaw_01_pad`, `jaw_02_pad`) with
  friction 3.
- **Force.** The gripper actuator is force-capped (±0.04 N·m, ≈ 3.5 N per jaw). Uncapped,
  the soft constraint coupling the jaws to the servo transmits ~2 kN and crushes through
  anything.
- **Contacts.** Contact force in MuJoCo scales with the *lighter* body's mass, so light
  food items need a smaller timestep (0.5 ms) and `solref 0.002` to resist a pad. Flat
  cylinders and meshes fall through box floors, so slices are unions of three rotated
  boxes.
- **Arm.** The arm bodies carry `gravcomp`, with stiff servos (`kp` 150), because the
  original soft gains sagged 3 cm under load. Joint limits are the actuator ranges.
- **Spacing and walls.** Open jaws reach 28 mm either side of the fingertip and the wrist
  base is 8 cm wide, so slices sit on a 60 mm pitch in a 20 cm bin, sticks 40 mm apart
  across their length in an 18 cm tray, and both walls are 30 mm high. Tighter or taller,
  and no approach exists.
- **Items.** 9 salami slices (≈ 29 mm, 16 mm thick) and 9 cheese sticks (14 × 38 mm). The
  original 150 decorative + 50 thin shreds could not be pinched.

If you change the scene, re-run `check_scene.py`, the reference episode and
`tests/test_elrobot.py`; the tests encode these dimensions.

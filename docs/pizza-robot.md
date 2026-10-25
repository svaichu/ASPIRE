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
.venv-system2/bin/python pizza-robot/scripts/reference_episode.py     # scripted recipe: 3 slices + 4 sticks (sticks are the unreliable half), ~2 min
.venv-system2/bin/python -m pytest tests/test_elrobot.py              # plant, IK, adapter, perception, grasping, full episode
```

The reference episode scores itself against simulator truth (which an agent never sees)
and exits non-zero unless every requested item is on the plate.

## Grounded in the real data

The scene's layout, camera, Home pose and joint speeds come from the 30 real demonstrations in
`ApoorvaBajaj/pizza-topping-camera-cleaned`, not from guesses. The pipeline:

```bash
# 1. load with robotdataset (finetune-groot env has it) -> small arrays in pizza-robot/real/
/workspace/finetune-groot/.venv/bin/python pizza-robot/real/export_real.py \
    --data-dir /workspace/finetune-groot/outputs/pizza_hf/cleaned
# 2. validate the joint map, fit the real camera, recover the layout -> pizza-robot/real/layout.json
.venv-system2/bin/python pizza-robot/real/analyze.py
# 3. write the layout into the scene (idempotent)
.venv-system2/bin/python pizza-robot/scripts/apply_real_layout.py
# 4. replay real demos in the sim and measure tracking
.venv-system2/bin/python pizza-robot/scripts/replay_real.py --frames /tmp/replay
```

What it established, and how sure it is:

- **Joint map.** Recorded `position_norm` maps linearly onto the sim's joint limits
  (`cap_harness.elrobot.real`). The calibrated spans match the URDF limit spans to 2% on six
  joints; the fingertip never goes below the table over 12.8k steps; a replay of six demos
  tracks to 0.004–0.006 rad RMS. Joints 2-4 and 1 are determined by the data; wrist joints 5-7
  are an assumption position data cannot test.
- **Camera.** A pinhole camera fitted to fingertip positions (forward kinematics) against the
  gripper's pixels in the real overhead view: high on the far side, tilted ~33° back toward
  the robot, 71° vertical field of view; median reprojection error 20 px of 640. Mirroring
  joint 1 fits 2.7x worse, which is how its direction was settled.
- **Pizza and trays.** Landmarks read off the median real overhead image, unprojected through
  that camera: pizza centre (0.248, 0.235), radius 0.178 m (the sim nudges x to 0.26 to keep
  clear of the salami bin; the fit's error is larger than that).
- **Not copied.** The real trays are small (about 5 x 9 cm) and the olives tray has no counterpart
  here. This sim's containers must be 18-20 cm wide for the open jaws to clear their walls, so
  they cannot sit where the real trays do. Their positions came from a search over 56 x 110
  candidate pairs for ones that keep every one of the 18 items reachable, leave the pizza and
  each other clear, and stay close to the real arrangement. The camera was pulled back 1.45x
  along its own axis so the larger layout is in view.
- **Speeds.** Per-joint limits are the demos' 99th-percentile joint speeds (0.8-1.3 rad/s, floored
  at 0.8 for the barely-moving wrist rolls). That is slower than the interpolating planner's
  old 2.5 rad/s, and it makes long moves exceed the API's default `max_steps`; pass 800.

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
- **Planner tolerance.** `approach.py` calls a path blocked past 3.5 mm of overlap. At 0.5 mm it
  refused approaches that grip fine (the pad's lower corner dips ~3 mm into the tray bed at
  the standard tilt, a soft contact) and steered cheese toward ones that grip worse.
- **Items.** 9 salami slices (≈ 29 mm, 16 mm thick) and 9 cheese sticks (14 × 38 mm). The
  original 150 decorative + 50 thin shreds could not be pinched.

**Known limit.** Carrying a cheese stick across the table is unreliable: it can twist out of the
pads when the wrist swings (the interpolator moves in joint space), and two of the nine stick
approaches survive the swing. Slices carry reliably, and every stick is grasped and lifted
(`tests/test_elrobot.py`). The reference episode asks for 3 slices and 2 sticks for that reason.

If you change the scene, re-run `check_scene.py`, the reference episode and
`tests/test_elrobot.py`; the tests encode these dimensions.

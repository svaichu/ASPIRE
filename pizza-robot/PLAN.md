# Pizza-robot: plan of action

Goal: run ASPIRE's System 2 (CAP MCP agent harness) against **ElRobot**
(norma-core's 7+1 DOF printed arm) in the **robotdesigner** pizza-making MuJoCo
scene, instead of the bimanual YAM arm System 2 currently targets.

## Inventory (as of 2026-10-25)

| Piece | Where | State |
| --- | --- | --- |
| ASPIRE System 2 (CAP harness, MCP, agent loop) | `/workspace/ASPIRE` | Bootstrapped (`.venv-system2`, `.venv-sam3`, `.venv-curobo`), built only for the bimanual YAM embodiment today |
| ElRobot hardware + URDF | `/workspace/norma-core/hardware/elrobot` | 7+1 DOF, Feetech ST3215 bus servos, `simulation/elrobot_follower.urdf` |
| Pizza scene (MuJoCo) | `/workspace/robotdesigner/pizza_robot/elrobot_table.xml` + `run_elrobot_table.py` | Standalone, hand-built, viewer-only. Not wired to any gym env, MCP, or the `robotdesigner` package. See its own `pizza_robot/.agent/memory.md` for scene/physics notes. |
| `robotdesigner` package | `/workspace/robotdesigner` | Generates **abstract** N-DOF arms (`build_arm`) and registers them as Gymnasium/MJX envs. Does not load arbitrary URDFs like ElRobot's; not directly reusable for this arm. Useful only as a reference for the Gym/MJX wrapping pattern, and as the home for a real ElRobot `MjSpec` builder if we choose to generate (rather than hand-edit) the scene going forward. |

ASPIRE is **not** a generic multi-embodiment framework with a plug-in seam.
Each embodiment is its own package (`yam_real/`, `robosuite/`, `libero/`)
with a hand-written `adapter.py` + `codec.py` (+ `kinematics.py`/`sim.py`/
`station.py` for a from-scratch MuJoCo plant like YAM's). Supporting ElRobot
means adding a new package of the same shape, not configuring an existing one.
Confirmed reusable, embodiment-agnostic layers: `contracts.py` (keyed by
arbitrary `arm` name, not hardcoded to `left`/`right`), `api.py` (`CapApi`),
`registry.py` (`ToolRegistry` + per-embodiment allowlist constants),
`agent/mcp.py`, `agent/session.py`, SAM3 segmentation provider.

## Target architecture

New package `src/cap_harness/elrobot/` mirroring `yam_real/`'s shape, sized to
one arm instead of two:

- `codec.py` — `ARMS = ("arm",)`, `ARM_DOF = 7`, joint-name/quaternion
  conversions. Model after `yam_real/codec.py`; much simpler with a single arm
  (no `joint7`-style gripper packing across two sides).
- `kinematics.py` — mink FK/IK over the pizza-table MJCF, modeled on
  `yam_real/kinematics.py`'s `YamKinematics`. Single `ARM_QSLICE`, and the
  jaw mimic noted in the scene's memory file (`<equality><joint>`, not the
  URDF's native mimic tag, since MuJoCo ignores that tag) has to be respected
  by the IK model the same way `sim.py` already respects it for YAM's fingers.
- `sim.py` — MuJoCo station analogous to `yam_real/sim.py`'s `SimYamStation`:
  owns `elrobot_table.xml`, steps physics, exposes
  `get_observations`/`execute_action_batch`/`set_gripper`/`go_home`/
  `read_camera`/`control_contract`/`close` per the `YamStation`-shaped
  `Protocol` in `yam_real/station.py` (write an equivalent `ElRobotStation`
  Protocol rather than reusing the YAM one, since the contract is otherwise
  identical but belongs to this embodiment).
- `adapter.py` — `ElRobotAdapter`, the protocol-facing wrapper
  (`YamRealAdapter` equivalent). Same motion-interlock pattern
  (`allow_physical_motion`, pre-RPC refusal, `command_rpc_count`) — that
  safety pattern is embodiment-independent and should carry over verbatim in
  spirit, not reinvented.
- `registry.py` — `ELROBOT_PUBLIC_TOOL_NAMES` (probably empty or near-empty
  at first; no embodiment-specific extensions needed until a pizza-specific
  tool proves necessary) and an `ElRobotTaskRegistry` if task metadata is
  wanted (compare `yam_real/registry.py`).
- No real-hardware `station.py` yet (see **Real hardware** below) — sim-only
  to start, i.e. only `build_sim_station`-equivalent, no `build_real_station`.

Cameras: the scene already declares `overhead` and `wrist` MuJoCo cameras.
`sim.py`'s `read_camera` needs an offscreen `mujoco.Renderer`-based capture
(the scene memory file already flags the default 640x480 cap and the
`<visual><global offwidth offheight/>` fix needed for larger frames — apply
that up front since SAM3 segmentation will want full-resolution frames).

Kinematics/planning: start **mink-only**, no cuRobo integration. cuRobo's
robot config is hardcoded per-embodiment today (`providers/curobo/service.py`
builds YAM/Franka configs specifically); adding ElRobot there is a real task
but a separable, later one. Early milestones use straight-line/mink-solved
joint targets the way YAM's pattern does for descent/contact segments, and
simply skip the "collision-aware trajectory" tool tier until cuRobo support
exists. SAM3 segmentation is embodiment-agnostic and usable immediately once
cameras are wired.

## Milestones

**Status (2026-10-25): milestones 1-6 done; 7-8 still deferred.** An agent can run the
whole pizza task through MCP. How to run, what is in the scene and why it looks the way it
does: [docs/pizza-robot.md](../docs/pizza-robot.md). What the agent is told:
[elrobot-tool-usage](../.agents/skills/elrobot-tool-usage/SKILL.md),
[pizza-assembly](../.agents/skills/manipulation-tasks/pizza-assembly.md), and the ElRobot
section of [AGENTS.md](../AGENTS.md).

Where the build departed from the plan below, and why:

- **Scene.** The hand-built scene could not be manipulated by this gripper as authored, so it
  was reworked (copy under `pizza-robot/scene/`, original untouched): jaw collision replaced
  with flat pads (MuJoCo collides meshes by convex hull), gripper force capped, arm
  gravity-compensated with stiff servos, timestep 0.5 ms with stiffer contacts, slices on a
  60 mm pitch in a wider bin, cheese shreds replaced by 9 chunky sticks, walls lowered to 30 mm.
  Each of those was found by a failed grasp; the reasons are in docs/pizza-robot.md.
- **Home** is a folded pose beside the base (the plan's raised arch hid the plate from the
  overhead camera).
- **Package** `src/cap_harness/elrobot/` has `config`, `sim`, `kinematics`, `ik`, `codec`,
  `adapter`, `registry`, `station`, plus three modules the plan did not list:
  `segmentation` (colour segmenter standing in for SAM3), `poses` (grasp-pose geometry) and
  `approach` (a clearance-checked approach search standing in for cuRobo). The live wrapper
  and control worker are reused from `yam_real`; they were already embodiment-neutral.
  Shared code touched: `contracts.JOINT_DIMENSIONS`, `registry` allowlist and prefix,
  `agent/mcp.py` direct-tool set, and the Home program (now loops over the arms present).
- **Server** `python -m cap_harness.agent.elrobot_server` (`scripts/start_pizza_system2.sh`).
  No Viser page yet: the visualizer is YAM-URDF specific.
- **Validation.** `tests/test_elrobot.py` (plant, IK, interlock, perception, two physical
  pick-and-place regressions, and a 3 slices + 2 sticks episode),
  `pizza-robot/scripts/reference_episode.py`, and `pizza-robot/scripts/mcp_smoke.py` (one
  slice over the real MCP interface).

**Grounded in the real dataset (`ApoorvaBajaj/pizza-topping-camera-cleaned`).** Loaded with
`robotdataset`; the layout, camera, Home pose and joint speeds now come from the 30 real
demonstrations (`pizza-robot/real/`, see docs/pizza-robot.md): the pizza is 36 cm across and
moved to its real position, the camera stand sits at the fitted pose, Home is the real rest
pose, speeds are the demos' p99, and a replay of real demos tracks to ~5 mrad RMS. No objects
were added or changed; the two containers moved (they cannot sit where the small real trays do).

Open items: a viewer for the sim (Viser), operator-facing success scoring outside the agent's
reach, recovery from items knocked onto the bare table (unvalidated), a trial with a model
agent in the loop (everything above was exercised with scripted clients), carrying cheese sticks
across the table (they twist out of the pads on long joint-space swings), and milestones 7-8.
The questions at the bottom remain open, except that the camera question is answered:
`overhead` + `wrist` were enough.

## Milestones

1. **Scene hardening (no CAP yet).** Fix `read_camera`-readiness: add
   `<visual><global offwidth offheight/>` to `elrobot_table.xml`, confirm
   offscreen rendering of `overhead`/`wrist` at a SAM3-friendly resolution.
   Set a sane folded home pose (the scene memory file notes this is still
   open) and bake it in as `home_joints`. Decide the final qpos layout once
   gripper mimic + any task objects with free joints are finalized, since
   (per the scene's own lesson) joint lookup must go by name, never index.
   **Done** (see the status above for what changed since): 1280x960 offscreen, wrist camera moved above the
   gripper base, home pose baked into `scene/home.json`.

2. **`elrobot` package skeleton + sim-only station.** Write
   `codec.py`, `kinematics.py`, `sim.py`, `adapter.py`, `registry.py`,
   mirroring `yam_real`'s split. Unit tests mirroring
   `tests/test_yam_real_codec.py`, `tests/test_yam_kinematics_*`,
   `tests/test_yam_real_adapter.py`, `tests/test_yam_real_interlock.py` for
   the single-arm case. No MCP yet — exercise the adapter directly in tests,
   same as the YAM suite does.

3. **Station builder + `agent/server.py` wiring.** Add an
   `elrobot.station.build_sim_station()` (no real variant yet) and a
   `build_elrobot_runtime()` alongside `build_yam_runtime()` in
   `agent/server.py` (or a sibling entry point if keeping `server.py` YAM-only
   is cleaner — decide once the adapter exists). Add a CLI entry point
   (`scripts/start_pizza_system2.sh`?) that starts MCP + Viser against the
   pizza-table sim station, no arm services, no SocketCAN, no cameras beyond
   MuJoCo's own.

4. **Fresh-agent smoke test.** Reuse `scripts/start_system2_agent.sh` (or a
   pizza-specific variant) against the new MCP endpoint. Confirm
   `cap_program_go_home`, `move_to_pose`, `move_to_joints`, open/close gripper
   all work through MCP on the single arm. This is the first point ElRobot
   is actually driven by a CAP-tool-using agent.

5. **Task content.** Add `examples/elrobot_sim/pizza_assembly/TASK_CONTEXT.md`
   (model on `examples/yam_real/*/TASK_CONTEXT.md`) describing the actual
   goal (pick salami discs from `bin_salami`, cheese shreds from
   `bin_cheese`, place on `pizza_plate`) and completion evidence. Add a task
   recipe skill under `.agents/skills/manipulation-tasks/pizza-assembly.md`
   and link it from `manipulation-tasks/SKILL.md`, following the existing
   per-task-recipe pattern (read only the matching recipe, not all of them).
   Update this repo's `AGENTS.md` only if the shared execution rules need a
   pizza-specific carve-out — expect they mostly don't, since Home/interlock/
   recovery rules are embodiment-agnostic.

6. **Validation.** Run several scripted and agent-driven episodes; check the
   scene's own documented limits (gripper only interacts with the 50 real
   cheese shreds, not the 150 decorative ones; no joint-limit enforcement
   beyond actuator `ctrlrange`) don't silently break task success claims.

7. **(Deferred) cuRobo support for ElRobot.** Add an ElRobot robot config to
   `providers/curobo/service.py` (URDF + collision spheres) once mink-only
   motion proves insufficient for the pizza task's clutter (bins, plate,
   shreds). Only pursue if milestone 6 shows collision-aware planning is
   actually needed.

8. **(Deferred) Real hardware.** ElRobot's follower is Feetech ST3215 bus
   servos over a Waveshare/SeeedStudio USB-serial driver board, **not**
   CAN/DaMiao/Portal RPC like YAM. `yam_real/transport.py`
   (`FollowerArmClient`) and the arm-service CLI (`yam-servers`) are
   CAN-specific and don't apply. A real `ElRobotStation` needs its own
   transport over the bus-servo protocol — a materially different, separate
   effort from anything above, and out of scope until the sim path works
   end-to-end.

## Open decisions (ask before committing)

- Keep `elrobot` sim-only indefinitely, or is real-hardware teleoperation
  eventually in scope (changes how much to generalize `codec.py`/`adapter.py`
  now vs. later)?
- Does the pizza task want a second arm/gripper station camera analog to
  YAM's wrist camera, or are `overhead` + `wrist` sufficient for SAM3
  grasp-point segmentation?
- Should `elrobot`'s `MjSpec` eventually be generated through `robotdesigner`
  (programmatic, parametric) instead of the hand-authored
  `elrobot_table.xml`, now that the scene's one-off lessons are documented?
  That would make `robotdesigner` genuinely load-bearing rather than just a
  naming/architectural reference.

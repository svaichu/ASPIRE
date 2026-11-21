"""Typed, frame-explicit adapter for the GR-3 humanoid (robotenv / MuJoCo) pick tasks.

The simulator runs in-process (``gr3-humanoid`` supplies ``gr3_scene.build_env``). Only the
right arm (7 joints) and its two-finger gripper are exposed; the waist, head, left arm and
legs are held at their reset pose. Generated programs never see native state.

Frame conventions
-----------------
``BASE_FRAME`` is the pelvis frame. The end-effector (tool) frame follows the usual gripper
convention: origin at the fingertip centre, +z along the approach direction (out of the palm,
towards the fingertips), +x across the finger opening. MuJoCo's ``right_end_effector_link`` has
the hand pointing along its -z axis, so the tool frame is that link rotated 180 degrees about x.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    ExecutionResult,
    Observation,
    Pose,
    RobotAction,
    RobotPlanningContext,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import matrix_to_pose
from cap_harness.gr3.registry import Gr3TaskMetadata, Gr3TaskRegistry

ARM_JOINT_NAMES = tuple(
    f"right_{name}_joint"
    for name in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow_pitch",
        "wrist_yaw",
        "wrist_pitch",
        "wrist_roll",
    )
)
FINGER_JOINT_NAMES = ("right_gripper_finger_1_joint", "right_gripper_finger_2_joint")
EE_LINK = "right_end_effector_link"
FINGER_TRAVEL_M = 0.035
TRAJECTORY_TOLERANCE_RAD = 0.01
TRAJECTORY_START_TOLERANCE_RAD = 0.02
TRAJECTORY_MAX_STEPS_PER_WAYPOINT = 120
GRASP_HOLD_RADIUS_M = 0.08
BIAS_GAIN = 0.03
WAIST_STIFFNESS_SCALE = 8.0
BIAS_LIMIT_RAD = 0.2
# Tool frame = site frame rotated 180 degrees about x (approach axis +z points to the fingertips).
TOOL_FROM_SITE = np.diag([1.0, -1.0, -1.0])


def _transform(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    result[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return result


class Gr3Adapter:
    """Normalize the GR-3 robotenv scene without exposing native state to generated code."""

    gripper_holds_intermediate = True
    BASE_FRAME = "gr3_base"

    def __init__(
        self,
        *,
        registry: Gr3TaskRegistry | None = None,
        camera_height: int = 256,
        camera_width: int = 256,
        horizon: int = 600,
        run_observer: Any | None = None,
        mount_x: float | None = None,
        mount_z: float | None = None,
    ) -> None:
        if camera_height <= 0 or camera_width <= 0 or horizon <= 0:
            raise ValueError("camera dimensions and horizon must be positive")
        self.registry = registry or Gr3TaskRegistry()
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self.horizon = int(horizon)
        self._run_observer = run_observer
        self._mount = {k: v for k, v in (("mount_x", mount_x), ("mount_z", mount_z)) if v is not None}
        self.controller = "JOINT_POSITION_ABS"
        self._env: Any = None
        self._td: Any = None
        self._metadata: Gr3TaskMetadata | None = None
        self._task_context: TaskContext | None = None
        self._elapsed_steps = 0
        self._closed = False
        self._protocol_evaluator: Any | None = None
        self._joint_targets: dict[str, np.ndarray] = {}
        self._gripper_targets: dict[str, float] = {}
        self._rest_z: float | None = None
        self.control_frequency = 20.0
        self.control_period_s = 1.0 / self.control_frequency
        self._renderer: Any = None
        self._render_scratch: Any = None
        self.success_observed_step: int | None = None

    # ------------------------------------------------------------------ properties

    @property
    def embodiment(self) -> str:
        return "gr3"

    @property
    def current_time_s(self) -> float:
        return self._elapsed_steps * self.control_period_s

    @property
    def metadata(self) -> Gr3TaskMetadata:
        if self._metadata is None:
            raise RuntimeError("reset() must be called before task metadata is available")
        return self._metadata

    @property
    def arms(self) -> tuple[str, ...]:
        return ("primary",)

    @property
    def native_env(self) -> Any:
        """Runtime-only native environment; never registered with generated programs."""
        return self._require_env()

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._arm_lo.copy(), self._arm_hi.copy()

    def bind_protocol_evaluator(self, evaluator: Any) -> None:
        if self._protocol_evaluator is not None:
            raise RuntimeError("a protocol evaluator is already bound")
        self._protocol_evaluator = evaluator

    def _require_env(self) -> Any:
        if self._env is None:
            raise RuntimeError("reset() must be called before using the GR-3 adapter")
        return self._env

    # ------------------------------------------------------------------ construction

    def reset(self, task_ref: str | tuple[str, int] | Gr3TaskMetadata, seed: int) -> Observation:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        import gr3_scene  # supplied by the gr3-humanoid environment

        import mujoco

        metadata = self.registry.resolve(task_ref)
        self._close_environment()
        self._env, self._robot = gr3_scene.build_env(
            control_mode="joint_pos_abs", seed=seed, **self._mount
        )
        self._closed = False
        self._metadata = metadata
        self._td = self._env.reset()
        self._seed = seed
        sim = self._env._sim
        model = sim.model
        self._sim = sim
        self._names = [joint.name for joint in self._robot.joints]
        self._arm_idx = [self._names.index(n) for n in ARM_JOINT_NAMES]
        self._finger_idx = [self._names.index(n) for n in FINGER_JOINT_NAMES]
        self._all_lo = np.array([j.range[0] for j in self._robot.joints], dtype=np.float64)
        self._all_hi = np.array([j.range[1] for j in self._robot.joints], dtype=np.float64)
        self._arm_lo = self._all_lo[self._arm_idx]
        self._arm_hi = self._all_hi[self._arm_idx]
        iface = sim.iface
        self._qadr = np.asarray(iface.qpos_adr, dtype=int)
        self._vadr = np.asarray(iface.qvel_adr, dtype=int)
        self._site = int(iface.ee_site_id)
        self._cube = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"rs_{metadata.target}_main"
        )
        self._mujoco = mujoco
        self._stiffen_waist(model, mujoco)
        self.control_period_s = float(sim.control_dt)
        self.control_frequency = 1.0 / self.control_period_s
        self._elapsed_steps = 0
        self.success_observed_step = None
        self._protocol_evaluator = None
        self._rest_z = float(sim.data[0].xpos[self._cube][2])
        full = self._full_qpos()
        self._desired_full = full.copy()
        self._bias = np.zeros(len(full))
        self._bias_limit = np.full(len(full), 0.0)
        self._bias_limit[self._arm_idx] = BIAS_LIMIT_RAD
        state = self.get_robot_state()
        self._joint_targets = {"primary": np.array(state.joint_positions["primary"], copy=True)}
        self._gripper_targets = {"primary": float(state.gripper_positions["primary"])}
        self._task_context = TaskContext(
            suite=metadata.suite_name,
            task_id=metadata.task_id,
            task_name=metadata.task_name,
            language=metadata.language,
            family=metadata.family,
            metadata={"task_ref": metadata.task_ref},
        )
        observation = self.get_observation()
        if self._run_observer is not None:
            self._run_observer.on_reset(observation, metadata.to_manifest_record())
        return observation

    def _stiffen_waist(self, model: Any, mujoco: Any) -> None:
        """The stock waist position gains let the torso sag under the extended arm (about 0.06 rad
        of pitch), which moves the shoulder 3 cm or more relative to the base frame after IK has
        been solved. Scale those gains up so the base frame the planner assumes is the base frame
        the arm actually has."""
        for index, name in enumerate(self._names):
            if not name.startswith("waist"):
                continue
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ext_" + name)
            for act in range(model.nu):
                if int(model.actuator_trnid[act, 0]) == joint:
                    scale = WAIST_STIFFNESS_SCALE
                    model.actuator_gainprm[act, 0] *= scale
                    model.actuator_biasprm[act, 1] *= scale
                    model.actuator_biasprm[act, 2] *= scale
                    model.actuator_forcerange[act] *= scale

    # ------------------------------------------------------------------ native state

    def _data(self) -> Any:
        return self._sim.data[0]

    def _full_qpos(self) -> np.ndarray:
        return np.asarray(self._data().qpos[self._qadr], dtype=np.float64)

    def _base(self) -> tuple[np.ndarray, np.ndarray]:
        pos, rot = self._sim.base_frame()
        return np.asarray(pos, dtype=np.float64), np.asarray(rot, dtype=np.float64)

    def base_from_world(self) -> np.ndarray:
        pos, rot = self._base()
        # ``rot`` maps base coordinates to world: p_world = pos + rot @ p_base.
        return np.linalg.inv(_transform(rot, pos))

    def tool_in_base(self) -> np.ndarray:
        data = self._data()
        site_world = _transform(data.site_xmat[self._site].reshape(3, 3), data.site_xpos[self._site])
        tool_world = site_world.copy()
        tool_world[:3, :3] = tool_world[:3, :3] @ TOOL_FROM_SITE
        return self.base_from_world() @ tool_world

    def get_planning_context(self) -> RobotPlanningContext:
        return RobotPlanningContext(
            embodiment=self.embodiment,
            model="gr3",
            joint_names={"primary": ARM_JOINT_NAMES},
            base_transforms={"primary": np.eye(4, dtype=np.float64)},
            end_effector_links={"primary": EE_LINK},
        )

    def _measured_gripper(self) -> float:
        qpos = self._full_qpos()[self._finger_idx]
        return float(np.clip(1.0 + float(np.mean(qpos)) / FINGER_TRAVEL_M, 0.0, 1.0))

    def get_robot_state(self) -> RobotState:
        data = self._data()
        joints = np.asarray(data.qpos[self._qadr][self._arm_idx], dtype=np.float64)
        velocities = np.asarray(data.qvel[self._vadr][self._arm_idx], dtype=np.float64)
        pose = matrix_to_pose(self.tool_in_base(), frame=self.BASE_FRAME)
        return RobotState(
            joint_positions={"primary": joints},
            joint_velocities={"primary": velocities},
            end_effector_poses={"primary": pose},
            gripper_positions={"primary": self._measured_gripper()},
            base_frame=self.BASE_FRAME,
            joint_names={"primary": ARM_JOINT_NAMES},
            timestamp_s=self.current_time_s,
            embodiment=self.embodiment,
        )

    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise RuntimeError("reset() must be called before reading task context")
        return self._task_context

    # ------------------------------------------------------------------ cameras

    def _prepare_render(self, name: str) -> Any:
        """Copy live state into the render scratch data and load camera ``name``'s scene."""
        mujoco = self._mujoco
        model = self._sim.model
        if self._renderer is None:
            g = model.vis.global_
            g.offwidth = max(int(g.offwidth), self.camera_width)
            g.offheight = max(int(g.offheight), self.camera_height)
            # Software GL (Mesa) dominates step time; shadows, reflections and MSAA are not
            # needed for localization and cost roughly half of each render.
            model.vis.quality.offsamples = 0
            self._renderer = mujoco.Renderer(model, self.camera_height, self.camera_width)
            self._render_scratch = mujoco.MjData(model)
        live = self._data()
        scratch = self._render_scratch
        scratch.qpos[:] = live.qpos
        scratch.qvel[:] = live.qvel
        scratch.act[:] = live.act
        scratch.ctrl[:] = live.ctrl
        scratch.mocap_pos[:] = live.mocap_pos
        scratch.mocap_quat[:] = live.mocap_quat
        mujoco.mj_forward(model, scratch)
        renderer = self._renderer
        renderer.update_scene(scratch, camera=name)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        renderer.disable_depth_rendering()
        return renderer

    def render_rgb(self, name: str) -> np.ndarray:
        """RGB only, for live monitoring; programs use :meth:`get_observation`."""
        return np.ascontiguousarray(self._prepare_render(name).render())

    def _camera_observation(self, name: str) -> CameraObservation:
        mujoco = self._mujoco
        model = self._sim.model
        renderer = self._prepare_render(name)
        scratch = self._render_scratch
        rgb = np.ascontiguousarray(renderer.render())
        renderer.enable_depth_rendering()
        depth = np.ascontiguousarray(renderer.render(), dtype=np.float64)
        renderer.disable_depth_rendering()
        depth[~np.isfinite(depth)] = 0.0
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        fovy = float(model.cam_fovy[camera_id])
        focal = 0.5 * rgb.shape[0] / np.tan(np.deg2rad(fovy) / 2.0)
        intrinsics = np.array(
            [[focal, 0.0, 0.5 * rgb.shape[1]], [0.0, focal, 0.5 * rgb.shape[0]], [0.0, 0.0, 1.0]]
        )
        world_from_cam = _transform(
            scratch.cam_xmat[camera_id].reshape(3, 3), scratch.cam_xpos[camera_id]
        )
        image_from_opengl = np.diag((1.0, -1.0, -1.0, 1.0))
        base_from_camera = self.base_from_world() @ world_from_cam @ image_from_opengl
        return CameraObservation(
            rgb=rgb,
            depth_m=depth,
            intrinsics=intrinsics,
            frame=f"camera/{name}",
            camera_pose=matrix_to_pose(base_from_camera, frame=self.BASE_FRAME),
            timestamp_s=self.current_time_s,
        )

    def get_observation(self) -> Observation:
        return Observation(
            cameras={name: self._camera_observation(name) for name in self.metadata.camera_names},
            robot_state=self.get_robot_state(),
            task_context=self.get_task_context(),
            timestamp_s=self.current_time_s,
        )

    # ------------------------------------------------------------------ stepping

    def _encode(self, action: RobotAction) -> np.ndarray:
        command = action.arms["primary"]
        self._joint_targets["primary"] = np.array(command.target, copy=True)
        if command.gripper_position is not None:
            self._gripper_targets["primary"] = float(command.gripper_position)
        desired = self._desired_full.copy()
        desired[self._arm_idx] = np.clip(
            self._joint_targets["primary"], self._arm_lo, self._arm_hi
        )
        desired[self._finger_idx] = (self._gripper_targets["primary"] - 1.0) * FINGER_TRAVEL_M
        # The arm's position actuators are soft, so gravity leaves a steady-state offset. An integral term on the commanded
        # targets of the arm joints removes the first without changing the joint-space
        # contract: programs still ask for, and measure, the same joint angles.
        measured = self._full_qpos()
        error = desired - measured
        error[self._finger_idx] = 0.0
        self._bias = np.clip(self._bias + BIAS_GAIN * error, -self._bias_limit, self._bias_limit)
        target = np.clip(desired + self._bias, self._all_lo, self._all_hi)
        target[self._finger_idx] = desired[self._finger_idx]
        span = np.where(self._all_hi > self._all_lo, self._all_hi - self._all_lo, 1.0)
        return np.clip(2.0 * (target - self._all_lo) / span - 1.0, -1.0, 1.0)

    def native_step(self, action: RobotAction) -> StepResult:
        import torch

        self._require_env()
        if action.embodiment != self.embodiment:
            raise ValueError(
                f"action embodiment {action.embodiment!r} does not match {self.embodiment!r}"
            )
        unknown = set(action.arms) - set(self.arms)
        if unknown:
            return StepResult(
                ok=False,
                error=ApiError(ErrorCode.NOT_FOUND, f"unknown GR-3 arms: {sorted(unknown)}"),
            )
        if self._run_observer is not None:
            self._run_observer.before_step(action)
        try:
            native = self._encode(action)
            self._td["action"] = torch.tensor(native[None], dtype=torch.float32)
            self._td = self._env.step(self._td)["next"]
            self._elapsed_steps += 1
            success = self.check_success()
            if success and self.success_observed_step is None:
                self.success_observed_step = self._elapsed_steps
            truncated = self._elapsed_steps >= self.horizon
            result = StepResult(
                ok=True,
                observation=self.get_observation(),
                terminated=False,
                truncated=bool(truncated and not success),
                reward=1.0 if success else 0.0,
                diagnostics={},
            )
        # Convert native failures into the public typed error result.
        except Exception as exc:
            result = StepResult(
                ok=False,
                error=ApiError(
                    ErrorCode.ADAPTER_FAILED,
                    f"GR-3 native step failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                ),
            )
        if self._protocol_evaluator is not None:
            self._protocol_evaluator.after_step(action, result)
        if self._run_observer is not None:
            self._run_observer.after_step(action, result)
        return result

    def step(self, action: RobotAction) -> StepResult:
        runtime = self.native_step(action)
        return StepResult(
            ok=runtime.ok,
            observation=runtime.observation,
            terminated=runtime.terminated,
            truncated=runtime.truncated,
            error=runtime.public_view().error,
            diagnostics={
                "control_period_s": self.control_period_s,
                "step_index": self._elapsed_steps,
            },
        )

    # ------------------------------------------------------------------ execution

    @staticmethod
    def _terminal_error(terminated: bool, truncated: bool) -> ApiError:
        del terminated
        if truncated:
            return ApiError(
                ErrorCode.TRUNCATED,
                "episode was truncated before trajectory execution completed",
            )
        return ApiError(
            ErrorCode.TERMINATED, "episode terminated before trajectory execution completed"
        )

    @staticmethod
    def _stale_trajectory_result(
        trajectory: Trajectory, state: RobotState
    ) -> ExecutionResult | None:
        expected = trajectory.expected_start
        joint_error = float(
            np.max(np.abs(expected.joint_positions["primary"] - state.joint_positions["primary"]))
        )
        gripper_error = abs(
            expected.gripper_positions["primary"] - state.gripper_positions["primary"]
        )
        if (
            tuple(expected.joint_names["primary"]) != tuple(state.joint_names["primary"])
            or joint_error > TRAJECTORY_START_TOLERANCE_RAD
            or gripper_error > max(TRAJECTORY_START_TOLERANCE_RAD, 0.1)
        ):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.STALE_PLAN,
                    "trajectory start no longer matches measured robot state",
                    details={
                        "joint_error_rad": {"primary": joint_error},
                        "gripper_error": {"primary": gripper_error},
                        "tolerance": TRAJECTORY_START_TOLERANCE_RAD,
                    },
                ),
                final_errors={"primary": joint_error},
            )
        return None

    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult:
        if isinstance(trajectory, SynchronizedTrajectory) or not isinstance(trajectory, Trajectory):
            raise ValueError("GR-3 executes single-arm Trajectory objects only")
        if trajectory.embodiment != self.embodiment:
            raise ValueError(
                f"trajectory embodiment {trajectory.embodiment!r} does not match {self.embodiment!r}"
            )
        if not np.isclose(trajectory.dt_s, self.control_period_s, atol=1e-9, rtol=0.0):
            raise ValueError("trajectory dt_s must equal the GR-3 control period")
        if trajectory.arm != "primary":
            raise ValueError("trajectory contains an unavailable arm")
        state = self.get_robot_state()
        stale = self._stale_trajectory_result(trajectory, state)
        if stale is not None:
            return stale
        if tuple(trajectory.joint_names) != tuple(state.joint_names["primary"]):
            raise ValueError("trajectory joint_names do not match the primary arm")
        positions = trajectory.joint_positions
        grippers = trajectory.gripper_positions
        final: Observation | None = None
        steps = 0

        def command(index: int) -> RobotAction:
            return RobotAction(
                {
                    "primary": ArmCommand(
                        mode="joint_position",
                        target=positions[index],
                        gripper_position=None if grippers is None else float(grippers[index]),
                        embodiment=self.embodiment,
                    )
                }
            )

        def error_of(obs: Observation) -> float:
            return float(
                np.max(np.abs(obs.robot_state.joint_positions["primary"] - positions[-1]))
            )

        def terminal(result: StepResult, err: float) -> ExecutionResult:
            return ExecutionResult(
                ok=False,
                steps_executed=steps,
                final_observation=final,
                terminated=result.terminated,
                truncated=result.truncated,
                error=self._terminal_error(result.terminated, result.truncated),
                final_errors={"primary": err},
                diagnostics={"joint_error_rad": {"primary": err}},
            )

        for index in range(len(positions)):
            result = self.step(command(index))
            if not result.ok or result.observation is None:
                return ExecutionResult(
                    ok=False, steps_executed=steps, final_observation=final, error=result.error
                )
            steps += 1
            final = result.observation
            if result.terminated or result.truncated:
                return terminal(result, error_of(final))
        err = error_of(final)
        for _ in range(TRAJECTORY_MAX_STEPS_PER_WAYPOINT - 1):
            if err <= TRAJECTORY_TOLERANCE_RAD:
                break
            result = self.step(command(len(positions) - 1))
            if not result.ok or result.observation is None:
                return ExecutionResult(
                    ok=False, steps_executed=steps, final_observation=final, error=result.error
                )
            steps += 1
            final = result.observation
            err = error_of(final)
            if result.terminated or result.truncated:
                return terminal(result, err)
        if err > TRAJECTORY_TOLERANCE_RAD:
            return ExecutionResult(
                ok=False,
                steps_executed=steps,
                final_observation=final,
                error=ApiError(
                    ErrorCode.TIMEOUT,
                    "final trajectory waypoint did not converge",
                    details={"joint_error_rad": {"primary": err}},
                ),
                final_errors={"primary": err},
            )
        return ExecutionResult(
            ok=True,
            steps_executed=steps,
            final_observation=final,
            final_errors={"primary": err},
            diagnostics={"joint_error_rad": {"primary": err}},
        )

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        if set(positions) != {"primary"}:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.INVALID_REQUEST, "positions must name the primary arm"),
            )
        return self.set_gripper(positions["primary"], arm="primary")

    def set_gripper(self, position: float, *, arm: str = "primary") -> ExecutionResult:
        if arm != "primary":
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.NOT_FOUND, f"unknown arm {arm!r}"),
            )
        numeric = float(position)
        if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.INVALID_REQUEST, "gripper position must be in [0, 1]"),
            )
        self._gripper_targets["primary"] = numeric
        result: StepResult | None = None
        final: Observation | None = None
        index = 0
        for index in range(30):
            result = self.step(
                RobotAction(
                    {
                        "primary": ArmCommand(
                            "joint_position",
                            self._joint_targets["primary"],
                            numeric,
                            embodiment=self.embodiment,
                        )
                    }
                )
            )
            if not result.ok or result.observation is None:
                break
            final = result.observation
            if result.terminated or result.truncated:
                break
        assert result is not None
        terminal = result.terminated or result.truncated
        return ExecutionResult(
            ok=result.ok and not terminal,
            steps_executed=index + 1 if result.ok else index,
            final_observation=final,
            terminated=result.terminated,
            truncated=result.truncated,
            error=result.error
            or (self._terminal_error(result.terminated, result.truncated) if terminal else None),
        )

    # ------------------------------------------------------------------ success & metadata

    def cube_position_world(self) -> np.ndarray:
        return np.asarray(self._data().xpos[self._cube], dtype=np.float64).copy()

    def check_success(self) -> bool:
        """Target cube lifted by the task height and still held at the fingertips."""
        cube = self.cube_position_world()
        lifted = cube[2] - float(self._rest_z) >= self.metadata.lift_height_m
        tip = np.asarray(self._data().site_xpos[self._site], dtype=np.float64)
        held = float(np.linalg.norm(cube - tip)) <= GRASP_HOLD_RADIUS_M
        return bool(lifted and held)

    def compute_reward(self) -> float | None:
        return 1.0 if self.success_observed_step is not None else 0.0

    def get_task_metadata(self) -> Mapping[str, object]:
        return MappingProxyType(self.metadata.to_manifest_record())

    def get_controller_metadata(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "arm_names": self.arms,
                "base_frame": self.BASE_FRAME,
                "camera_names": self.metadata.camera_names,
                "control_frequency_hz": self.control_frequency,
                "control_period_s": self.control_period_s,
                "controller": self.controller,
                "end_effector_frame": "fingertip centre; +z approach (towards fingertips), +x across the fingers",
                "gripper_semantics": {"normalized": "0=closed, 1=open", "finger_travel_m": FINGER_TRAVEL_M},
                "joint_lower_limits": tuple(float(v) for v in self._arm_lo),
                "joint_upper_limits": tuple(float(v) for v in self._arm_hi),
                "controllable_gripper_arms": self.arms,
                "supported_action_modes": ("joint_position",),
            }
        )

    # ------------------------------------------------------------------ planner & lifecycle

    def planner(self) -> Any:
        from cap_harness.gr3.planning import Gr3MujocoPlanner

        return Gr3MujocoPlanner(self)

    def _close_environment(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close(raise_if_closed=False)
        self._env = None

    def close(self) -> None:
        if self._closed:
            return
        self._close_environment()
        self._closed = True


__all__ = ["ARM_JOINT_NAMES", "Gr3Adapter"]

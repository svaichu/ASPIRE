"""``EnvironmentAdapter`` for the single-arm ElRobot pizza station.

Deliberately thin, like :mod:`cap_harness.yam_real.adapter`: normalize, delegate,
never actuate except through the station's command methods. The interlock is the
same pattern and has the same limits: construction and reads always work; anything
that can move the arm is refused unless ``allow_physical_motion`` was set, and
refused *before* the station is touched, so an unauthorized call leaves no trace.
``command_count`` counts commands actually submitted, never attempted.

The station is a simulation, so "physical" motion here means moving the simulated
arm. The gate is kept because the same agent code later drives hardware, and
because an observation-only session is a useful thing to be able to start.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from cap_harness.contracts import (
    ExecutionResult,
    Observation,
    RobotAction,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.elrobot import codec
from cap_harness.elrobot.registry import ElRobotTaskMetadata, ElRobotTaskRegistry
from cap_harness.errors import ApiError, ErrorCode

ARMS = codec.ARMS
ARM_DOF = codec.ARM_DOF

#: Held at the final waypoint after a trajectory so the servo can close its error.
TRAJECTORY_SETTLE_S = 0.5
#: Following-error budget for a trajectory to count as executed. The plant tracks
#: to ~0.002 rad, so this sits far above servo error and far below a missed motion
#: -- it exists to catch an arm blocked by contact, which stops short and would
#: otherwise report success.
TRAJECTORY_JOINT_TOLERANCE_RAD = 0.05
MINIMUM_HOMING_S = 1.0
MAXIMUM_HOMING_S = 12.0
HOME_DIAGNOSTIC_TOLERANCE_RAD = 0.03
#: A normalized gripper reading at or above this counts as open.
GRIPPER_OPEN_THRESHOLD = 0.95


class ElRobotAdapter:
    embodiment = codec.EMBODIMENT
    registry_prefix = "elrobot"

    def __init__(
        self,
        env: Any,
        *,
        registry: ElRobotTaskRegistry | None = None,
        task_context: TaskContext | None = None,
        allow_physical_motion: bool = False,
    ) -> None:
        self._env = env
        self._registry = registry or ElRobotTaskRegistry()
        self._task_context = task_context
        self._allow_physical_motion = bool(allow_physical_motion)
        self._command_count = 0
        self._closed = False
        self.control_frequency = env.config.control_frequency_hz
        self.control_period_s = 1.0 / self.control_frequency

    # -- identity ----------------------------------------------------------

    @property
    def native_env(self) -> Any:
        return self._env

    @property
    def arms(self) -> tuple[str, ...]:
        return ARMS

    @property
    def dof(self) -> int:
        return ARM_DOF

    @property
    def command_rpc_count(self) -> int:
        """Commands actually submitted; named to match the YAM adapter's evidence field."""
        return self._command_count

    @property
    def physical_motion_authorized(self) -> bool:
        return self._allow_physical_motion

    def resolve_arm(self, arm: str) -> str:
        """The one arm answers to its own name and to the shared ``primary`` role."""
        return "arm" if str(arm) in {"arm", "primary"} else str(arm)

    # -- the interlock -----------------------------------------------------

    def _motion_denied(self) -> ApiError | None:
        if self._allow_physical_motion:
            return None
        return ApiError(
            code=ErrorCode.SAFETY_INTERLOCK,
            message=(
                "motion is not authorized for this ElRobot adapter; "
                "start the server with --allow-motion"
            ),
            recoverable=False,
            details={"command_count": self._command_count},
        )

    def _plant_failure(self, exc: Exception, operation: str) -> ApiError:
        return ApiError(
            code=ErrorCode.EXECUTION_FAILED,
            message=f"{operation} rejected by the station: {exc}",
            details={"command_count": self._command_count},
        )

    def _refusal(self, result: Mapping[str, Any], operation: str) -> ApiError | None:
        """Typed failure for a plant that REPORTS a refusal instead of raising."""
        if bool(result.get("success", True)):
            return None
        reason = str(result.get("reason", "")).strip() or "no reason given"
        return ApiError(
            code=ErrorCode.EXECUTION_FAILED,
            message=f"{operation} refused by the station: {reason}",
            details={"command_count": self._command_count, "reason": reason},
        )

    def _denied_step(self, error: ApiError) -> StepResult:
        return StepResult(ok=False, error=error)

    def _denied_execution(self, error: ApiError) -> ExecutionResult:
        return ExecutionResult(ok=False, steps_executed=0, error=error)

    # -- lifecycle ---------------------------------------------------------

    def reset(self, task_ref: object = None, seed: int = 0) -> Observation:
        """Restore the authored scene and home the arm. The seed is provenance only.

        Resetting the scene teleports objects, so it is gated like motion: an
        observation-only session observes and returns without touching anything.
        """
        if isinstance(task_ref, TaskContext):
            self._task_context = task_ref
        elif task_ref is not None:
            self._task_context = self._task_context_for(self._registry.resolve(task_ref), seed)
        if self._motion_denied() is None:
            self._env.reset()
            self._command_count += 1
        return self.get_observation()

    def _task_context_for(self, metadata: ElRobotTaskMetadata, seed: int) -> TaskContext:
        return TaskContext(
            suite=metadata.suite_name,
            task_id=metadata.task_id,
            task_name=metadata.task_name,
            language=metadata.language,
            family=metadata.family,
            metadata={
                "scene_reset": metadata.scene_reset,
                "seed": int(seed),
                "station": self._env.config.station,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._env.close()

    # -- observation -------------------------------------------------------

    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise RuntimeError("no task context set; pass one to the adapter or to reset()")
        return self._task_context

    def get_observation(self) -> Observation:
        return codec.observation(self._env, task_context=self._task_context)

    def get_robot_state(self) -> RobotState:
        return codec.robot_state(self._env)

    def check_success(self) -> bool:
        """No success predicate is exposed to the agent; the scene is not privileged.

        False means "cannot tell", not "failed". Task completion is judged from
        camera evidence, exactly as on hardware.
        """
        return False

    # -- actuation ---------------------------------------------------------

    def _hold(self) -> tuple[np.ndarray, float]:
        state = codec.robot_state(self._env)
        held = np.clip(
            state.joint_positions["arm"],
            self._env.config.joint_limits_lower,
            self._env.config.joint_limits_upper,
        )
        return held, float(state.gripper_positions["arm"])

    def step(self, action: RobotAction) -> StepResult:
        denied = self._motion_denied()
        if denied is not None:
            return self._denied_step(denied)
        joints, grippers = codec.arm_command_targets(action)
        unknown = set(joints) - set(ARMS)
        if unknown:
            return self._denied_step(
                ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"action names unknown ElRobot arms: {sorted(unknown)}",
                    details={"known_arms": list(ARMS)},
                )
            )
        held, held_grip = self._hold()
        target = joints.get("arm", held)
        grip = held_grip if grippers.get("arm") is None else float(grippers["arm"])
        try:
            result = self._env.execute_waypoints(
                [0.0, self.control_period_s], np.vstack([held, target]), [held_grip, grip]
            )
        except Exception as exc:
            return self._denied_step(self._plant_failure(exc, "step"))
        self._command_count += 1
        return StepResult(
            ok=bool(result.get("success", True)),
            error=self._refusal(result, "step"),
            observation=self.get_observation(),
            diagnostics={"command_count": self._command_count},
        )

    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult:
        """Stream a joint trajectory, hold the last waypoint, and check it was reached."""
        denied = self._motion_denied()
        if denied is not None:
            return self._denied_execution(denied)
        if isinstance(trajectory, Trajectory):
            rows = np.asarray(trajectory.joint_positions, dtype=np.float64)
            grip_column = trajectory.gripper_positions
        else:
            rows = np.asarray(trajectory.joint_positions["arm"], dtype=np.float64)
            grip_column = (trajectory.gripper_positions or {}).get("arm")
        _, held_grip = self._hold()
        grippers = (
            np.full(len(rows), held_grip)
            if grip_column is None
            else np.asarray(grip_column, dtype=np.float64).reshape(-1)
        )
        # Planned waypoints exclude the start pose, so the first one is a tick ahead of it.
        times = (np.arange(len(rows)) + 1) * float(trajectory.dt_s)
        if len(rows) == 1:  # a one-waypoint trajectory still needs a duration to stream
            times, rows, grippers = (
                np.array([0.0, self.control_period_s]),
                np.vstack([rows, rows]),
                np.repeat(grippers, 2),
            )
        try:
            result = self._env.execute_waypoints(
                times, rows, grippers, settle_s=TRAJECTORY_SETTLE_S
            )
        except Exception as exc:
            return self._denied_execution(self._plant_failure(exc, "trajectory"))
        self._command_count += 1
        residual = float(np.max(np.abs(self._env.joint_positions() - rows[-1])))
        ok = bool(result.get("success", True))
        error = self._refusal(result, "execute_trajectory")
        if ok and residual > TRAJECTORY_JOINT_TOLERANCE_RAD:
            ok = False
            error = ApiError(
                code=ErrorCode.EXECUTION_FAILED,
                message=(
                    f"trajectory ended {residual:.3f} rad from its final waypoint "
                    f"(tolerance {TRAJECTORY_JOINT_TOLERANCE_RAD:.3f} rad); the arm did not "
                    "track the commanded motion -- suspect contact or a joint limit"
                ),
                details={"joint_residual_rad": {"arm": residual}},
            )
        return ExecutionResult(
            ok=ok,
            steps_executed=int(result.get("command_count", len(rows))),
            final_observation=self.get_observation(),
            error=error,
            diagnostics={
                "joint_residual_rad": {"arm": residual},
                "command_count": self._command_count,
            },
        )

    def set_gripper(self, position: float, *, arm: str = "arm") -> ExecutionResult:
        denied = self._motion_denied()
        if denied is not None:
            return self._denied_execution(denied)
        try:
            result = self._env.set_gripper(float(position))
        except Exception as exc:
            return self._denied_execution(self._plant_failure(exc, "set_gripper"))
        self._command_count += 1
        return ExecutionResult(
            ok=bool(result.get("success", True)),
            error=self._refusal(result, "set_gripper"),
            steps_executed=1,
            final_observation=self.get_observation(),
            diagnostics={
                "commanded_gripper": float(position),
                "command_count": self._command_count,
            },
        )

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        denied = self._motion_denied()
        if denied is not None:
            return self._denied_execution(denied)
        steps = 0
        for arm, position in positions.items():
            result = self.set_gripper(float(position), arm=str(arm))
            steps += int(result.steps_executed)
            if not result.ok:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps,
                    final_observation=result.final_observation,
                    error=result.error,
                )
        return ExecutionResult(
            ok=True, steps_executed=steps, final_observation=self.get_observation()
        )

    def _homing_duration_s(self) -> float:
        config = self._env.config
        travel = np.abs(self._env.joint_positions() - config.home_joints)
        seconds = float(np.max(travel / config.joint_speed_limits_rad_s))
        return float(min(MAXIMUM_HOMING_S, max(MINIMUM_HOMING_S, seconds)))

    def go_home(self, *, arm: str = "arm", duration: float | None = None) -> ExecutionResult:
        """Joint-space move to the profile's home, scaled to distance, gripper opened.

        The path is a straight line in joint space with no collision checking, so
        it can sweep through anything standing between the arm and home. Clear
        held objects first -- Home releases nothing deliberately, it just opens.
        """
        denied = self._motion_denied()
        if denied is not None:
            return self._denied_execution(denied)
        seconds = self._homing_duration_s() if duration is None else float(duration)
        try:
            result = self._env.go_home(duration_s=seconds)
        except Exception as exc:
            return self._denied_execution(self._plant_failure(exc, "go_home"))
        self._command_count += 1
        observation = self.get_observation()
        residual = float(
            np.max(
                np.abs(
                    observation.robot_state.joint_positions["arm"] - self._env.config.home_joints
                )
            )
        )
        return ExecutionResult(
            ok=bool(result.get("success", True)),
            error=self._refusal(result, "go_home"),
            steps_executed=int(result.get("command_count", 1)),
            final_observation=observation,
            diagnostics={
                "joint_residual_rad": {"arm": residual},
                "home_tolerance_rad": HOME_DIAGNOSTIC_TOLERANCE_RAD,
                "home_verified": residual <= HOME_DIAGNOSTIC_TOLERANCE_RAD,
                "grippers_open": {
                    "arm": float(observation.robot_state.gripper_positions["arm"])
                    >= GRIPPER_OPEN_THRESHOLD
                },
                "duration_s": seconds,
                "command_count": self._command_count,
            },
        )

    # -- metadata ----------------------------------------------------------

    def get_task_metadata(self) -> dict[str, Any]:
        context = self._task_context
        return {
            "embodiment": self.embodiment,
            "station": self._env.config.station,
            "arms": list(ARMS),
            "dof": ARM_DOF,
            "cameras": list(self._env.camera_aliases),
            "language": "" if context is None else context.language,
        }

    def get_controller_metadata(self) -> dict[str, Any]:
        contract = self._env.control_contract()
        config = self._env.config
        return {
            "control_frequency_hz": self.control_frequency,
            "control_period_s": self.control_period_s,
            "arm_names": list(ARMS),
            "control_contract": contract,
            "control_contract_version": contract.get("version"),
            # CapApi turns these into the per-tick step of its interpolating planner
            # (bound / control_frequency), so they carry the per-joint SPEED limit, rad/s --
            # joint position limits would make it a 3 rad/s planner. Position limits are
            # enforced by the plant and reported below.
            "action_lower_bounds": (-config.joint_speed_limits_rad_s).tolist(),
            "action_upper_bounds": config.joint_speed_limits_rad_s.tolist(),
            "joint_speed_limits_rad_s": config.joint_speed_limits_rad_s.tolist(),
            "joint_limits_lower": config.joint_limits_lower.tolist(),
            "joint_limits_upper": config.joint_limits_upper.tolist(),
            "home_joints": config.home_joints.tolist(),
            "physical_motion_authorized": self._allow_physical_motion,
            "command_count": self._command_count,
        }


__all__ = [
    "GRIPPER_OPEN_THRESHOLD",
    "TRAJECTORY_JOINT_TOLERANCE_RAD",
    "TRAJECTORY_SETTLE_S",
    "ElRobotAdapter",
]

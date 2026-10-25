"""ElRobot's own mink IK behind the harness ``IKProvider`` seam.

The shared PyRoki provider solves a seven-joint Panda, so ElRobot gets its solver
the way YAM does: in process, over the scene's own model. It answers "is this pose
reachable", not "is this path collision-free" -- there is no cuRobo here, so
nothing checks the path between waypoints against the bins and plate. Callers
that need clearance must create it with their own waypoints.
"""

from __future__ import annotations

import threading

import numpy as np

from cap_harness.contracts import IKResult, Pose, RobotState
from cap_harness.elrobot.codec import ARM_DOF, EMBODIMENT
from cap_harness.errors import ApiError, ErrorCode

#: Matches the default execution tolerance in ``CapApi.move_to_pose``: a solution
#: the executor would reject on arrival must not be handed out beforehand.
DEFAULT_POSITION_TOLERANCE_M = 0.01
#: A little under the executor's 10 degree orientation bound, for the same reason.
DEFAULT_ORIENTATION_TOLERANCE_RAD = float(np.deg2rad(8.0))


class ElRobotIKProvider:
    def __init__(
        self,
        kinematics: object,
        *,
        position_tolerance_m: float = DEFAULT_POSITION_TOLERANCE_M,
        orientation_tolerance_rad: float = DEFAULT_ORIENTATION_TOLERANCE_RAD,
    ) -> None:
        self._kin = kinematics
        self._position_tolerance_m = float(position_tolerance_m)
        self._orientation_tolerance_rad = float(orientation_tolerance_rad)
        self._lock = threading.Lock()

    def _fail(self, code: ErrorCode, message: str, **details: object) -> IKResult:
        return IKResult(
            ok=False,
            arm="arm",
            embodiment=EMBODIMENT,
            error=ApiError(code=code, message=message, details=details),
        )

    def solve_ik(
        self, target_pose: Pose, robot_state: RobotState, *, arm: str = "primary"
    ) -> IKResult:
        if arm not in {"arm", "primary"}:
            return self._fail(ErrorCode.NOT_FOUND, f"ElRobot has one arm, 'arm'; got {arm!r}")
        if target_pose.frame != robot_state.base_frame:
            return self._fail(
                ErrorCode.INVALID_REQUEST,
                f"target pose is in frame {target_pose.frame!r}; "
                f"this station solves in {robot_state.base_frame!r}",
            )
        seed = np.asarray(robot_state.joint_positions["arm"], dtype=np.float64).reshape(ARM_DOF)
        try:
            with self._lock:
                joints, metres, radians = self._kin.inverse_kinematics(
                    target_pose.position, target_pose.quaternion_wxyz, seed
                )
        except Exception as exc:
            return self._fail(
                ErrorCode.IK_FAILED,
                f"mink IK failed: {exc}",
                exception_type=type(exc).__name__,
            )
        # mink returns its best effort whether or not it converged, so the
        # solution is judged by its own forward kinematics before it is a success.
        if not np.all(np.isfinite(joints)) or not (
            metres <= self._position_tolerance_m and radians <= self._orientation_tolerance_rad
        ):
            return self._fail(
                ErrorCode.IK_UNREACHABLE,
                f"mink did not converge: best configuration is {metres:.3f} m and "
                f"{np.rad2deg(radians):.1f} deg from the target",
                position_error_m=float(metres),
                orientation_error_rad=float(radians),
            )
        return IKResult(
            ok=True,
            joint_positions=joints,
            arm="arm",
            embodiment=EMBODIMENT,
            diagnostics={
                "position_error_m": float(metres),
                "orientation_error_rad": float(radians),
            },
        )


__all__ = [
    "DEFAULT_ORIENTATION_TOLERANCE_RAD",
    "DEFAULT_POSITION_TOLERANCE_M",
    "ElRobotIKProvider",
]

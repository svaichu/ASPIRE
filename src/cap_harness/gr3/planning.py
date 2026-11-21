"""In-process IK and collision-aware motion planning for the GR-3 right arm.

Like the BEHAVIOR planner, this runs on the simulator's own robot model and world geometry:
no external planning service is involved. IK is damped least squares on a scratch ``MjData``;
collisions are MuJoCo contacts between robot geoms and (robot | table | cube) geoms.

Contacts that exist at the start configuration of a call (for example the squatted legs
overlapping table legs, or fingers resting at the cube) are ignored for that call, and finger
contact with a cube is always allowed, so a grasp can be approached and a held cube carried.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from cap_harness.contracts import (
    IKResult,
    PlanResult,
    Pose,
    RobotState,
    SynchronizedPlanResult,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import matrix_to_pose
from cap_harness.gr3.adapter import ARM_JOINT_NAMES, TOOL_FROM_SITE

PLANNER_NAME = "gr3_mujoco"
MAX_JOINT_SPEED_RAD_S = 1.0
MIN_DURATION_S = 0.5
EDGE_RESOLUTION_RAD = 0.04
RRT_STEP_RAD = 0.2
RRT_ITERATIONS = 2500
IK_POSITION_TOL_M = 1.5e-3
IK_ROTATION_TOL_RAD = 0.03
IK_RESTARTS = 24
JOINT_MARGIN_RAD = 0.02


def _rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Rotation vector taking ``current`` onto ``target`` (both 3x3, world frame)."""
    delta = target @ current.T
    cos = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array(
        [delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]]
    )
    norm = np.linalg.norm(axis)
    if norm < 1e-9:  # angle ~ pi: pick any axis from the symmetric part
        diag = np.diag(delta)
        axis = np.eye(3)[int(np.argmax(diag))]
        return axis * angle
    return axis / norm * angle


class Gr3MujocoPlanner:
    """IK, joint-space planning and integrated pose planning for one GR-3 arm."""

    def __init__(self, adapter: Any, *, seed: int = 0) -> None:
        self._adapter = adapter
        self._rng = np.random.default_rng(seed)
        mujoco = adapter._mujoco
        self._mj = mujoco
        model = adapter._sim.model
        self._model = model
        self._scratch = mujoco.MjData(model)
        self._arm_qadr = adapter._qadr[adapter._arm_idx]
        self._arm_dof = adapter._vadr[adapter._arm_idx]
        self._site = adapter._site
        self._lo = adapter._arm_lo + JOINT_MARGIN_RAD
        self._hi = adapter._arm_hi - JOINT_MARGIN_RAD
        self._classify_geoms()

    # -------------------------------------------------------------- geometry classes

    def _classify_geoms(self) -> None:
        mujoco, model = self._mj, self._model
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
        self._is_robot = np.zeros(model.ngeom, dtype=bool)
        self._is_finger = np.zeros(model.ngeom, dtype=bool)
        self._is_cube = np.zeros(model.ngeom, dtype=bool)
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            name = names[body]
            self._is_robot[geom] = name.startswith("ext_")
            self._is_finger[geom] = "gripper_finger" in name
            self._is_cube[geom] = name.startswith("rs_cube")

    def _sync(self) -> None:
        self._scratch.qpos[:] = self._adapter._data().qpos
        self._scratch.qvel[:] = 0.0

    def _blocking_pairs(self, q_arm: np.ndarray) -> frozenset[tuple[int, int]]:
        d, m = self._scratch, self._model
        d.qpos[self._arm_qadr] = q_arm
        self._mj.mj_kinematics(m, d)
        self._mj.mj_collision(m, d)
        pairs: set[tuple[int, int]] = set()
        for i in range(d.ncon):
            contact = d.contact[i]
            if contact.dist >= 0.0:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            ra, rb = self._is_robot[a], self._is_robot[b]
            if not (ra or rb):
                continue
            if (self._is_finger[a] and self._is_cube[b]) or (self._is_finger[b] and self._is_cube[a]):
                continue
            pairs.add((min(a, b), max(a, b)))
        return frozenset(pairs)

    def _free(self, q_arm: np.ndarray, ignore: frozenset[tuple[int, int]]) -> bool:
        return self._blocking_pairs(q_arm) <= ignore

    def _edge_free(self, a: np.ndarray, b: np.ndarray, ignore: frozenset[tuple[int, int]]) -> bool:
        count = max(2, int(np.ceil(np.max(np.abs(b - a)) / EDGE_RESOLUTION_RAD)) + 1)
        return all(self._free(a + (b - a) * t, ignore) for t in np.linspace(0.0, 1.0, count)[1:])

    # -------------------------------------------------------------- inverse kinematics

    def _target_world(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        world_from_base = np.linalg.inv(self._adapter.base_from_world())
        world = world_from_base @ pose.as_matrix()
        # tool rotation -> site rotation
        return world[:3, 3], world[:3, :3] @ TOOL_FROM_SITE.T

    def _ik_once(self, position: np.ndarray, rotation: np.ndarray, q0: np.ndarray) -> tuple[np.ndarray, float, float]:
        d, m, mj = self._scratch, self._model, self._mj
        q = np.clip(q0, self._lo, self._hi).copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        pos_err = rot_err = np.inf
        for _ in range(250):
            d.qpos[self._arm_qadr] = q
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            current_rot = d.site_xmat[self._site].reshape(3, 3)
            e_pos = position - d.site_xpos[self._site]
            e_rot = _rotation_error(rotation, current_rot)
            pos_err, rot_err = float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))
            if pos_err < IK_POSITION_TOL_M and rot_err < IK_ROTATION_TOL_RAD:
                break
            mj.mj_jacSite(m, d, jacp, jacr, self._site)
            jac = np.vstack([jacp[:, self._arm_dof], 0.6 * jacr[:, self._arm_dof]])
            err = np.concatenate([e_pos, 0.6 * e_rot])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 2e-3 * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -0.15, 0.15), self._lo, self._hi)
        return q, pos_err, rot_err

    def _solve_candidates(
        self, pose: Pose, seed: np.ndarray, *, want: int = 6
    ) -> list[tuple[np.ndarray, float, float]]:
        """Converged IK solutions ordered nearest-first to ``seed``."""
        position, rotation = self._target_world(pose)
        out: list[tuple[np.ndarray, float, float]] = []
        starts = [seed] + [self._rng.uniform(self._lo, self._hi) for _ in range(IK_RESTARTS)]
        for start in starts:
            q, pe, re = self._ik_once(position, rotation, start)
            if pe < IK_POSITION_TOL_M and re < IK_ROTATION_TOL_RAD:
                if not any(np.max(np.abs(q - known[0])) < 0.05 for known in out):
                    out.append((q, pe, re))
            if len(out) >= want:
                break
        out.sort(key=lambda item: float(np.linalg.norm(item[0] - seed)))
        return out

    def _validate(self, state: RobotState, arm: str) -> ApiError | None:
        if arm != "primary" or "primary" not in state.joint_positions:
            return ApiError(ErrorCode.NOT_FOUND, f"unknown arm {arm!r}")
        return None

    def solve_ik(
        self, target_pose: Pose, robot_state: RobotState, *, arm: str = "primary"
    ) -> IKResult:
        error = self._validate(robot_state, arm)
        if error is not None:
            return IKResult(ok=False, arm=arm, error=error, embodiment=robot_state.embodiment)
        self._sync()
        seed = np.asarray(robot_state.joint_positions[arm], dtype=np.float64)
        ignore = self._blocking_pairs(seed)
        candidates = self._solve_candidates(target_pose, seed)
        if not candidates:
            return IKResult(
                ok=False,
                arm=arm,
                embodiment=robot_state.embodiment,
                error=ApiError(
                    ErrorCode.IK_UNREACHABLE,
                    "no IK solution within tolerance for the requested pose",
                    details={"provider": PLANNER_NAME},
                ),
            )
        for q, pe, re in candidates:
            if self._free(q, ignore):
                return IKResult(
                    ok=True,
                    joint_positions=q,
                    arm=arm,
                    embodiment=robot_state.embodiment,
                    diagnostics={
                        "provider": PLANNER_NAME,
                        "position_error_m": pe,
                        "orientation_error_rad": re,
                        "collision_aware": True,
                    },
                )
        return IKResult(
            ok=False,
            arm=arm,
            embodiment=robot_state.embodiment,
            error=ApiError(
                ErrorCode.IK_FAILED,
                "every IK solution for the requested pose is in collision",
                details={"provider": PLANNER_NAME},
            ),
        )

    # -------------------------------------------------------------- planning

    def _rrt_connect(
        self, start: np.ndarray, goal: np.ndarray, ignore: frozenset[tuple[int, int]]
    ) -> list[np.ndarray] | None:
        if self._edge_free(start, goal, ignore):
            return [start, goal]
        trees = [[start], [goal]]
        parents: list[list[int]] = [[-1], [-1]]

        def nearest(tree: list[np.ndarray], q: np.ndarray) -> int:
            return int(np.argmin([np.linalg.norm(q - node) for node in tree]))

        def extend(side: int, target: np.ndarray) -> tuple[str, int]:
            tree, par = trees[side], parents[side]
            index = nearest(tree, target)
            delta = target - tree[index]
            dist = float(np.linalg.norm(delta))
            new = target if dist <= RRT_STEP_RAD else tree[index] + delta / dist * RRT_STEP_RAD
            if not self._edge_free(tree[index], new, ignore):
                return "trapped", -1
            tree.append(new)
            par.append(index)
            return ("reached" if dist <= RRT_STEP_RAD else "advanced"), len(tree) - 1

        side = 0
        for _ in range(RRT_ITERATIONS):
            sample = goal if self._rng.random() < 0.05 else self._rng.uniform(self._lo, self._hi)
            status, new_index = extend(side, sample)
            if status != "trapped":
                target = trees[side][new_index]
                other = 1 - side
                while True:
                    status2, other_index = extend(other, target)
                    if status2 != "advanced":
                        break
                if status2 == "reached":
                    a, b = (new_index, other_index) if side == 0 else (other_index, new_index)
                    path_a: list[np.ndarray] = []
                    i = a
                    while i != -1:
                        path_a.append(trees[0][i])
                        i = parents[0][i]
                    path_b: list[np.ndarray] = []
                    i = b
                    while i != -1:
                        path_b.append(trees[1][i])
                        i = parents[1][i]
                    return path_a[::-1] + path_b
            side = 1 - side
        return None

    def _shortcut(self, path: list[np.ndarray], ignore: frozenset[tuple[int, int]]) -> list[np.ndarray]:
        path = list(path)
        for _ in range(60):
            if len(path) <= 2:
                break
            i, j = sorted(self._rng.integers(0, len(path), size=2))
            if j - i < 2:
                continue
            if self._edge_free(path[i], path[j], ignore):
                path = path[: i + 1] + path[j:]
        return path

    def _timed(self, path: list[np.ndarray]) -> np.ndarray:
        waypoints = np.asarray(path)
        seg = np.max(np.abs(np.diff(waypoints, axis=0)), axis=1)
        length = float(np.sum(seg))
        if length < 1e-6:
            return waypoints[:1].repeat(2, axis=0)
        duration = max(MIN_DURATION_S, 1.875 * length / MAX_JOINT_SPEED_RAD_S)
        dt = self._adapter.control_period_s
        count = int(np.ceil(duration / dt)) + 1
        cumulative = np.concatenate([[0.0], np.cumsum(seg)]) / length
        out = np.empty((count, waypoints.shape[1]))
        for k in range(count):
            tau = k / (count - 1)
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5  # minimum-jerk
            index = min(int(np.searchsorted(cumulative, s, side="right")) - 1, len(seg) - 1)
            span = max(cumulative[index + 1] - cumulative[index], 1e-12)
            out[k] = waypoints[index] + (waypoints[index + 1] - waypoints[index]) * (
                (s - cumulative[index]) / span
            )
        out[0], out[-1] = waypoints[0], waypoints[-1]
        return out

    def _trajectory(
        self,
        robot_state: RobotState,
        arm: str,
        waypoints: np.ndarray,
        gripper_position: float | None,
        planner: str,
    ) -> Trajectory:
        gripper = (
            float(robot_state.gripper_positions[arm])
            if gripper_position is None
            else float(gripper_position)
        )
        return Trajectory(
            joint_positions=waypoints,
            dt_s=self._adapter.control_period_s,
            joint_names=robot_state.joint_names[arm],
            planner=planner,
            collision_aware=True,
            expected_start=robot_state,
            arm=arm,
            gripper_positions=np.full(len(waypoints), gripper),
            embodiment=robot_state.embodiment,
        )

    def plan_to_joints(
        self,
        robot_state: RobotState,
        target_joints: np.ndarray,
        *,
        arm: str = "primary",
        gripper_position: float | None = None,
        scene: Any = None,
        context: Any = None,
        time_dilation_factor: float | None = None,
        interpolation_dt_s: float | None = None,
        maximum_trajectory_dt_s: float | None = None,
    ) -> PlanResult:
        del scene, context, interpolation_dt_s, maximum_trajectory_dt_s
        error = self._validate(robot_state, arm)
        if error is not None:
            return PlanResult(ok=False, error=error)
        start = np.asarray(robot_state.joint_positions[arm], dtype=np.float64)
        goal = np.clip(np.asarray(target_joints, dtype=np.float64), self._lo, self._hi)
        self._sync()
        ignore = self._blocking_pairs(start)
        if not self._free(goal, ignore):
            return PlanResult(
                ok=False,
                error=ApiError(
                    ErrorCode.PLANNING_FAILED,
                    "goal configuration is in collision",
                    details={"provider": PLANNER_NAME},
                ),
            )
        path = self._rrt_connect(start, goal, ignore)
        if path is None:
            return PlanResult(
                ok=False,
                error=ApiError(
                    ErrorCode.PLANNING_FAILED,
                    "no collision-free path found",
                    details={"provider": PLANNER_NAME},
                ),
            )
        path = self._shortcut(path, ignore)
        waypoints = self._timed(path)
        if time_dilation_factor is not None and time_dilation_factor < 1.0:
            # A smaller factor means a slower motion: stretch the timeline.
            stretch = 1.0 / float(time_dilation_factor)
            count = int(np.ceil((len(waypoints) - 1) * stretch)) + 1
            grid = np.linspace(0.0, len(waypoints) - 1, count)
            waypoints = np.stack(
                [np.interp(grid, np.arange(len(waypoints)), waypoints[:, j]) for j in range(waypoints.shape[1])],
                axis=1,
            )
        trajectory = self._trajectory(robot_state, arm, waypoints, gripper_position, f"{PLANNER_NAME}_cspace")
        return PlanResult(ok=True, trajectory=trajectory, diagnostics={"provider": PLANNER_NAME})

    def plan_to_pose(
        self,
        robot_state: RobotState,
        target_pose: Pose,
        *,
        arm: str = "primary",
        gripper_position: float | None = None,
        scene: Any = None,
        context: Any = None,
        time_dilation_factor: float | None = None,
        interpolation_dt_s: float | None = None,
        maximum_trajectory_dt_s: float | None = None,
    ) -> PlanResult:
        error = self._validate(robot_state, arm)
        if error is not None:
            return PlanResult(ok=False, error=error)
        self._sync()
        start = np.asarray(robot_state.joint_positions[arm], dtype=np.float64)
        ignore = self._blocking_pairs(start)
        candidates = self._solve_candidates(target_pose, start)
        if not candidates:
            return PlanResult(
                ok=False,
                error=ApiError(
                    ErrorCode.IK_UNREACHABLE,
                    "no IK solution within tolerance for the requested pose",
                    details={"provider": PLANNER_NAME},
                ),
            )
        last: PlanResult | None = None
        for q, _pe, _re in candidates:
            if not self._free(q, ignore):
                continue
            last = self.plan_to_joints(
                robot_state,
                q,
                arm=arm,
                gripper_position=gripper_position,
                time_dilation_factor=time_dilation_factor,
                scene=scene,
                context=context,
                interpolation_dt_s=interpolation_dt_s,
                maximum_trajectory_dt_s=maximum_trajectory_dt_s,
            )
            if last.ok:
                return PlanResult(
                    ok=True,
                    trajectory=Trajectory(
                        joint_positions=last.trajectory.joint_positions,
                        dt_s=last.trajectory.dt_s,
                        joint_names=last.trajectory.joint_names,
                        planner=f"{PLANNER_NAME}_pose",
                        collision_aware=True,
                        expected_start=robot_state,
                        arm=arm,
                        gripper_positions=last.trajectory.gripper_positions,
                        embodiment=robot_state.embodiment,
                    ),
                    diagnostics={"provider": PLANNER_NAME},
                )
        if last is not None:
            return last
        return PlanResult(
            ok=False,
            error=ApiError(
                ErrorCode.PLANNING_FAILED,
                "every IK solution for the requested pose is in collision",
                details={"provider": PLANNER_NAME},
            ),
        )

    def plan_synchronized_motion(
        self, robot_state: RobotState, targets: Mapping[str, Pose], **_: Any
    ) -> SynchronizedPlanResult:
        return SynchronizedPlanResult(
            ok=False,
            error=ApiError(ErrorCode.UNSUPPORTED, "GR-3 exposes a single arm"),
        )


__all__ = ["ARM_JOINT_NAMES", "Gr3MujocoPlanner", "PLANNER_NAME", "matrix_to_pose"]

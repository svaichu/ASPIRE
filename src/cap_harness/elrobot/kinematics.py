"""FK and IK for the ElRobot arm, solved with mink over the scene's own model.

The solver runs on a copy of the scene with the loose objects removed. They carry
hundreds of free degrees of freedom that the arm does not touch, and mink's QP
scales with them. Joint limits are the actuator ranges, since the scene does not
declare joint limits of its own.

Everything in and out is in the robot base frame, quaternions **wxyz** (the
harness convention). Unlike the YAM solver there is no xyzw boundary to convert.

This class carries a mutable ``mink.Configuration`` and is not thread safe; the
provider that wraps it serializes access.
"""

from __future__ import annotations

import mink
import mujoco
import numpy as np
import qpsolvers

from cap_harness.elrobot.config import ARM_JOINTS, BASE_BODY, GRASP_SITE, ElRobotConfig
from cap_harness.elrobot.sim import build_model

_SOLVER = "daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]


class ElRobotKinematics:
    def __init__(
        self,
        config: ElRobotConfig,
        *,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        lm_damping: float = 1.0,
    ) -> None:
        self.config = config
        self._model = build_model(config, gravity_comp=False, objects=False)
        for index, name in enumerate(ARM_JOINTS):
            joint = self._model.joint(name)
            joint.range[:] = (config.joint_limits_lower[index], config.joint_limits_upper[index])
            self._model.jnt_limited[joint.id] = 1
        self._qadr = np.array([self._model.joint(n).qposadr[0] for n in ARM_JOINTS])
        self._base = self._model.body(BASE_BODY).id
        self.configuration = mink.Configuration(self._model)
        self.task = mink.FrameTask(
            frame_name=GRASP_SITE,
            frame_type="site",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=lm_damping,
        )
        self._limits = [mink.ConfigurationLimit(self._model)]
        self._rng = np.random.default_rng(0)

    def model_copy_for_collisions(self) -> mujoco.MjModel:
        """This solver's model (loose items removed) with a small contact margin.

        Margin makes ``mj_collision`` report near-misses, which is what a clearance
        check wants. Returned model is a copy: the solver's own is left untouched.
        """
        model = _copy_model(self._model)
        model.geom_margin[:] = 0.002
        model.geom_gap[:] = 0.0
        return model

    def write_qpos(self, data: mujoco.MjData, joints: np.ndarray) -> None:
        data.qpos[self._qadr] = np.asarray(joints, dtype=np.float64).reshape(7)

    def _base_inverse(self) -> mink.SE3:
        """World-from-base inverse, read from the model's own base body."""
        data = self.configuration.data
        rotation = data.xmat[self._base].reshape(3, 3)
        return mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(rotation), data.xpos[self._base]
        ).inverse()

    def seed(self, joints: np.ndarray) -> None:
        self.configuration.data.qpos[self._qadr] = np.asarray(joints, dtype=np.float64).reshape(7)
        self.configuration.update()

    def forward_kinematics(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grasp-site ``(position, quaternion_wxyz)`` in the base frame."""
        self.seed(joints)
        world = self.configuration.get_transform_frame_to_world(GRASP_SITE, "site")
        local = self._base_inverse() @ world
        return local.translation().copy(), local.rotation().wxyz.copy()

    def pose_error(
        self, joints: np.ndarray, position: np.ndarray, quaternion_wxyz: np.ndarray
    ) -> tuple[float, float]:
        """``(metres, radians)`` between FK at ``joints`` and the target."""
        achieved_position, achieved_quaternion = self.forward_kinematics(joints)
        dot = abs(float(np.dot(achieved_quaternion, _unit(quaternion_wxyz))))
        return (
            float(np.linalg.norm(achieved_position - np.asarray(position, dtype=np.float64))),
            float(2.0 * np.arccos(np.clip(dot, 0.0, 1.0))),
        )

    def _solve_from(
        self,
        seed: np.ndarray,
        target: mink.SE3,
        *,
        dt: float,
        max_iters: int,
        err_threshold: float,
        damping: float,
    ) -> np.ndarray:
        self.seed(seed)
        self.task.set_target(target)
        for _ in range(max_iters):
            velocity = mink.solve_ik(
                self.configuration, [self.task], dt, _SOLVER, damping, limits=self._limits
            )
            self.configuration.integrate_inplace(velocity, dt)
            if np.linalg.norm(self.task.compute_error(self.configuration)) <= err_threshold:
                break
        return self.configuration.q[self._qadr].copy()

    def inverse_kinematics(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        seed: np.ndarray,
        *,
        position_tolerance_m: float = 2e-3,
        orientation_tolerance_rad: float = 2e-2,
        restarts: int = 30,
        dt: float = 0.05,
        max_iters: int = 120,
        err_threshold: float = 1e-4,
        damping: float = 1e-3,
    ) -> tuple[np.ndarray, float, float]:
        """Best joints for a base-frame pose, as ``(joints, metres, radians)``.

        Continues from ``seed`` first, so a motion stays on the branch the arm is
        on. Only if that misses the tolerance are random in-limit restarts tried.
        Among restarts that converge, the one nearest the seed in joint space wins
        (up to ``wanted`` of them are collected), so a solve far from home does not
        flip the elbow for no reason. If nothing converges the best miss is returned:
        the caller checks the residual and decides.
        """
        world_target = self._world_from_base() @ mink.SE3.from_rotation_and_translation(
            mink.SO3(wxyz=_unit(quaternion_wxyz)), np.asarray(position, dtype=np.float64).reshape(3)
        )
        lower, upper = self.config.joint_limits_lower, self.config.joint_limits_upper
        origin = np.asarray(seed, dtype=np.float64).reshape(7)
        converged: list[tuple[np.ndarray, float, float]] = []
        best: tuple[np.ndarray, float, float] | None = None
        wanted = 3
        starts = [origin] + [self._rng.uniform(lower, upper) for _ in range(restarts)]
        for index, start in enumerate(starts):
            joints = self._solve_from(
                start,
                world_target,
                dt=dt,
                max_iters=max_iters,
                err_threshold=err_threshold,
                damping=damping,
            )
            metres, radians = self.pose_error(joints, position, quaternion_wxyz)
            if best is None or (metres + radians) < (best[1] + best[2]):
                best = (joints, metres, radians)
            if metres <= position_tolerance_m and radians <= orientation_tolerance_rad:
                if index == 0:
                    return joints, metres, radians  # the seed's own branch: keep it
                converged.append((joints, metres, radians))
                if len(converged) >= wanted:
                    break
        if converged:
            return min(converged, key=lambda item: float(np.linalg.norm(item[0] - origin)))
        assert best is not None
        return best

    def _world_from_base(self) -> mink.SE3:
        data = self.configuration.data
        return mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(data.xmat[self._base].reshape(3, 3)), data.xpos[self._base]
        )


def _copy_model(model: mujoco.MjModel) -> mujoco.MjModel:
    """A deep copy of a compiled model, via the binary round trip MuJoCo provides."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.mjb")
        mujoco.mj_saveModel(model, path)
        return mujoco.MjModel.from_binary_path(path)


def _unit(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return q / np.linalg.norm(q)


__all__ = ["ElRobotKinematics"]

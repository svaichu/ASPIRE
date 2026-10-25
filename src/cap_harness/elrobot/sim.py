"""MuJoCo plant for the ElRobot pizza station.

The plant is stepped synchronously: a command advances the physics by exactly the
duration it asks for, then returns. That makes every run deterministic and lets a
long trajectory finish faster than real time, at the cost of there being no
background motion between calls -- which is fine here, because nothing else moves
the scene.

Two modelling choices matter to anyone trusting its numbers:

* **Gravity is compensated by the model, not the actuators.** The scene's
  position servos have a low gain (``kp=10``) chosen so the arm could hold itself
  up, and a held pose sagged by up to 0.06 rad -- 3 cm at the gripper, enough to
  fail every Cartesian tolerance. Real ST3215 servos hold their load with a
  stiff internal loop, so the arm bodies get ``gravcomp=1`` instead. Objects keep
  real gravity. This is applied by recompiling through ``MjSpec``; assigning
  ``body_gravcomp`` on a compiled model does nothing, because ``ngravcomp`` is
  fixed at compile time.
* **Joint limits are the actuator ranges.** The scene does not enforce joint
  limits itself; the plant clips every command to them.

Rendering goes through one dedicated thread, because an EGL context is bound to
the thread that created it and the MCP session calls in from several.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import threading
import time
from typing import Any

import mujoco
import numpy as np

from cap_harness.elrobot.config import (
    ARM_JOINTS,
    BASE_BODY,
    CAMERAS,
    GRASP_SITE,
    GRIPPER_JOINT,
    ElRobotConfig,
)
from cap_harness.yam_real.codec import rotation_to_wxyz

#: Depth beyond this is background, reported as 0 (no return) like a real sensor.
MAX_DEPTH_M = 3.0
#: Physics substeps never exceed this, whatever the control period.
SETTLE_POLL_S = 0.02
CONTROL_CONTRACT_VERSION = "elrobot-sim-1"

_OPTICAL_FROM_MUJOCO = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True, slots=True)
class CameraFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    base_from_camera: np.ndarray  # 4x4, optical convention (x right, y down, z forward)
    timestamp_s: float


def build_model(
    config: ElRobotConfig, *, gravity_comp: bool = True, objects: bool = True
) -> mujoco.MjModel:
    """Compile the scene, optionally with compensated arm gravity or without objects."""
    spec = mujoco.MjSpec.from_file(str(config.model_xml))
    if not objects:
        for body in [
            b for b in spec.bodies if any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in b.joints)
        ]:
            spec.detach_body(body)
    if gravity_comp:
        for body in spec.bodies:
            node = body
            while node is not None and node.name != "world":
                if node.name == BASE_BODY:
                    body.gravcomp = 1.0
                    break
                node = node.parent
    return spec.compile()


class SimElRobotStation:
    """The pizza scene plus the ElRobot arm, with the station's read/command surface."""

    def __init__(self, config: ElRobotConfig, *, gravity_comp: bool = True) -> None:
        self.config = config
        self.model = build_model(config, gravity_comp=gravity_comp)
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self._arm_qadr = np.array([self.model.joint(n).qposadr[0] for n in ARM_JOINTS])
        self._arm_vadr = np.array([self.model.joint(n).dofadr[0] for n in ARM_JOINTS])
        self._gripper_qadr = int(self.model.joint(GRIPPER_JOINT).qposadr[0])
        self._arm_act = np.array([self.model.actuator(n).id for n in ARM_JOINTS])
        self._gripper_act = int(self.model.actuator(GRIPPER_JOINT).id)
        self._site = int(self.model.site(GRASP_SITE).id)
        self._base = int(self.model.body(BASE_BODY).id)
        self._renderer: mujoco.Renderer | None = None
        self._render_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="elrobot-render")
        self._closed = False
        self._frames: dict[str, tuple[int, CameraFrame]] = {}
        self.steps = 0
        self.reset()

    # -- state ---------------------------------------------------------------

    def reset(self) -> None:
        """Restore the scene's authored state with the arm at home, then settle."""
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self._arm_qadr] = self.config.home_joints
            self.data.qpos[self._gripper_qadr] = self.config.gripper_to_rad(
                self.config.home_gripper
            )
            self._command(self.config.home_joints, self.config.home_gripper)
            mujoco.mj_forward(self.model, self.data)
            self._step_for(0.5)

    def _command(self, joints: np.ndarray, gripper: float) -> None:
        clipped = np.clip(joints, self.config.joint_limits_lower, self.config.joint_limits_upper)
        self.data.ctrl[self._arm_act] = clipped
        self.data.ctrl[self._gripper_act] = self.config.gripper_to_rad(gripper)

    def _step_for(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)
            self.steps += 1

    def _base_transform(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self._base].copy(), self.data.xmat[self._base].reshape(3, 3).copy()

    def joint_positions(self) -> np.ndarray:
        with self.lock:
            return self.data.qpos[self._arm_qadr].copy()

    def joint_velocities(self) -> np.ndarray:
        with self.lock:
            return self.data.qvel[self._arm_vadr].copy()

    def gripper_position(self) -> float:
        with self.lock:
            return self.config.gripper_from_rad(self.data.qpos[self._gripper_qadr])

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Grasp-site position and wxyz quaternion in the robot base frame."""
        with self.lock:
            origin, rotation = self._base_transform()
            position = rotation.T @ (self.data.site_xpos[self._site] - origin)
            orientation = rotation.T @ self.data.site_xmat[self._site].reshape(3, 3)
            return position, rotation_to_wxyz(orientation)

    def get_observations(self) -> dict[str, np.ndarray]:
        position, quaternion = self.ee_pose()
        return {
            "joint_pos": self.joint_positions(),
            "joint_vel": self.joint_velocities(),
            "gripper_pos": np.array([self.gripper_position()]),
            "ee_pos": position,
            "ee_quat": quaternion,
        }

    # -- actuation -----------------------------------------------------------

    def execute_waypoints(
        self,
        times_s: np.ndarray,
        joints: np.ndarray,
        grippers: np.ndarray,
        *,
        settle_s: float = 0.0,
    ) -> dict[str, Any]:
        """Stream linearly interpolated targets at the control rate, then hold.

        ``times_s`` starts at 0 and is strictly increasing; row ``i`` of ``joints``
        and entry ``i`` of ``grippers`` are the targets at ``times_s[i]``.
        """
        times = np.asarray(times_s, dtype=np.float64).reshape(-1)
        joints = np.asarray(joints, dtype=np.float64).reshape(len(times), -1)
        grippers = np.asarray(grippers, dtype=np.float64).reshape(len(times))
        period = 1.0 / self.config.control_frequency_hz
        commands = 0
        with self.lock:
            clock = 0.0
            while clock < times[-1] - 1e-9:
                clock = min(times[-1], clock + period)
                target = np.array(
                    [np.interp(clock, times, joints[:, axis]) for axis in range(joints.shape[1])]
                )
                self._command(target, float(np.interp(clock, times, grippers)))
                self._step_for(period)
                commands += 1
            self._command(joints[-1], float(grippers[-1]))
            self._step_for(max(settle_s, 0.0))
        return {"success": True, "command_count": commands, "reason": ""}

    def set_gripper(
        self, position: float, *, duration_s: float = 1.0, max_wait_s: float = 4.0
    ) -> dict[str, Any]:
        """Command the jaws and return only once they have stopped moving.

        The gripper is force-limited, so at the cap its speed is set by joint
        damping and a fixed wait would report a half-closed jaw as the result. A
        jaw stalled on an object stops too, which is exactly the reading a grasp
        check wants -- so "stopped" is the terminal condition, not "arrived".
        """
        with self.lock:
            held = self.joint_positions()
            start = self.gripper_position()
            result = self.execute_waypoints(
                [0.0, duration_s], np.vstack([held, held]), [start, float(position)]
            )
            waited = 0.0
            dof = int(self.model.joint(GRIPPER_JOINT).dofadr[0])
            while waited < max_wait_s:
                self._step_for(0.1)
                waited += 0.1
                if abs(self.data.qvel[dof]) < 0.01:
                    break
            result["gripper_wait_s"] = round(waited, 2)
            return result

    def go_home(self, *, duration_s: float = 3.0) -> dict[str, Any]:
        with self.lock:
            start = self.joint_positions()
            grip = self.gripper_position()
            return self.execute_waypoints(
                [0.0, duration_s],
                np.vstack([start, self.config.home_joints]),
                [grip, self.config.home_gripper],
                settle_s=0.5,
            )

    # -- cameras -------------------------------------------------------------

    @property
    def camera_aliases(self) -> tuple[str, ...]:
        return tuple(name for name in CAMERAS if name in self._model_cameras())

    def camera_frame(self, name: str) -> CameraFrame | None:
        """The latest frame, re-rendered only if physics has advanced since.

        Observers poll constantly and a render costs ~150 ms. Between steps the
        scene is bit-identical, so the cached pixels are genuinely current and only
        their timestamp is refreshed -- otherwise an idle scene would read as stale.
        """
        cached = self._frames.get(name)
        if cached is not None and cached[0] == self.steps:
            return replace(cached[1], timestamp_s=time.monotonic())
        frame = self.read_camera(name)
        if frame is not None:
            self._frames[name] = (self.steps, frame)
        return frame

    def _model_cameras(self) -> set[str]:
        return {self.model.camera(i).name for i in range(self.model.ncam)}

    def read_camera(self, name: str | None = None) -> CameraFrame | None:
        if self._closed:
            return None
        return self._render_pool.submit(self._render, name or self.config.camera).result()

    def _render(self, name: str) -> CameraFrame:
        width, height = self.config.camera_resolution
        with self.lock:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height, width)
            mujoco.mj_forward(self.model, self.data)
            camera = self.model.camera(name)
            self._renderer.update_scene(self.data, camera=name)
            self._renderer.disable_depth_rendering()
            rgb = self._renderer.render().copy()
            self._renderer.enable_depth_rendering()
            depth = self._renderer.render().astype(np.float64)
            self._renderer.disable_depth_rendering()
            depth[(depth > MAX_DEPTH_M) | ~np.isfinite(depth)] = 0.0
            focal = (height / 2.0) / np.tan(np.deg2rad(float(camera.fovy[0])) / 2.0)
            intrinsics = np.array(
                [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
            )
            origin, rotation = self._base_transform()
            cam_rotation = self.data.cam_xmat[camera.id].reshape(3, 3)
            transform = np.eye(4)
            transform[:3, :3] = rotation.T @ cam_rotation @ _OPTICAL_FROM_MUJOCO
            transform[:3, 3] = rotation.T @ (self.data.cam_xpos[camera.id] - origin)
        return CameraFrame(rgb, depth, intrinsics, transform, time.monotonic())

    # -- misc ------------------------------------------------------------------

    def control_contract(self) -> dict[str, Any]:
        return {
            "version": CONTROL_CONTRACT_VERSION,
            "plant": "mujoco",
            "control_frequency_hz": self.config.control_frequency_hz,
            "physics_timestep_s": float(self.model.opt.timestep),
            "joint_names": list(ARM_JOINTS),
            "joint_limits_lower": self.config.joint_limits_lower.tolist(),
            "joint_limits_upper": self.config.joint_limits_upper.tolist(),
            "gripper": "0 closed, 1 open (scissor jaws: 52 mm fingertip gap fully open)",
            "gravity_compensated_arm": True,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        def release() -> None:
            renderer, self._renderer = self._renderer, None
            if renderer is not None:
                try:
                    renderer.close()
                except Exception:
                    # EGL teardown raises on some drivers. Drop the handles so the
                    # renderer's own __del__ does not raise the same error again.
                    renderer._gl_context = None
                    renderer._mjr_context = None

        self._render_pool.submit(release).result()
        self._render_pool.shutdown(wait=True)


__all__ = ["CameraFrame", "SimElRobotStation", "build_model"]

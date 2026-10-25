"""An ElRobot plant that lives on the other end of a policy-server connection.

The real arm is wired to a different machine from the one running ASPIRE. That machine
runs a small bridge which, every tick, sends the arm's state and camera frames and asks
``get_action`` what to do (the Isaac-GR00T policy protocol; see ``policy_server``). Here
the "policy" is the ASPIRE agent: its tool calls reach :meth:`RemoteElRobotStation.
execute_waypoints`, which parks a trajectory in a :class:`RemoteLink`; the bridge's next
``get_action`` calls drain it in short chunks, and the call returns once the bridge has
acknowledged the last chunk. Everything above the station (adapter, CapApi, MCP, skills)
is the unchanged sim stack.

Wire units are the real robot's: ``position_norm`` in ``[0, 1]`` per joint, with the
gripper as the eighth entry, converted with :class:`~cap_harness.elrobot.real.RealJointMap`.
Targets are absolute, so a chunk served twice (a lost reply) is harmless.

What is *not* the sim: there is no ground truth and no reset; time is wall-clock; and the
real cameras report no depth, so :class:`RealCameraModel` synthesizes it by intersecting
each pixel's ray with the table plane. That is exact for the table and a few centimetres
short for the tops of tall objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from cap_harness.elrobot.config import ElRobotConfig
from cap_harness.elrobot.real import REAL_DIR, RealJointMap
from cap_harness.elrobot.sim import _OPTICAL_FROM_MUJOCO, CameraFrame, build_model

CONTROL_CONTRACT_VERSION = "elrobot-remote-1"
#: Steps of the control rate sent per ``get_action`` reply (0.5 s at 50 Hz).
CHUNK_STEPS = 25
#: The first waypoint of a trajectory may differ from the arm's reported state by this much.
START_TOLERANCE_RAD = 0.2
#: Wall-clock allowance beyond a trajectory's own duration for the bridge to finish and ack.
ACK_MARGIN_S = 15.0
#: A state older than this is not a reading of the arm any more.
STATE_STALE_S = 3.0
#: The bridge may exceed the commanded per-step motion by this factor before it is a fault.
SPEED_MARGIN = 1.5


class RemoteLinkError(RuntimeError):
    """The bridge is absent, stale, or reported a fault."""


@dataclass(slots=True)
class RemoteState:
    seq: int
    received_s: float
    norm: np.ndarray  # (8,) position_norm, gripper last
    frames: dict[str, np.ndarray]  # alias -> uint8 (H, W, 3)
    depth: dict[str, np.ndarray] = field(default_factory=dict)  # alias -> metres, optional


@dataclass(slots=True)
class _Segment:
    id: int
    targets: np.ndarray  # (N, 8) norm
    dt: float
    next_chunk: int = 0
    done: bool = False
    ok: bool = False
    reason: str = ""

    def chunk_count(self) -> int:
        return -(-len(self.targets) // CHUNK_STEPS)


class RemoteLink:
    """Thread-safe mailbox between the policy endpoint and the station.

    The endpoint thread calls :meth:`update_state` and :meth:`next_command`; the control
    thread calls :meth:`run_segment` and the ``wait_*`` readers. One segment at a time.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._state: RemoteState | None = None
        self._previous: RemoteState | None = None
        self._segment: _Segment | None = None
        self._cancel_pending = False
        self._ids = 0
        self._seq = 0
        self.closed = False

    # -- endpoint side -----------------------------------------------------

    def update_state(
        self,
        norm: np.ndarray,
        frames: dict[str, np.ndarray],
        depth: dict[str, np.ndarray] | None = None,
    ) -> None:
        with self._cond:
            self._seq += 1
            self._previous = self._state
            self._state = RemoteState(
                self._seq, time.monotonic(), np.asarray(norm, np.float64).reshape(8), frames,
                depth or {},
            )
            self._cond.notify_all()

    def handle_ack(self, ack: dict[str, Any] | None, fault: str | None) -> None:
        """Advance or fail the active segment from the bridge's report on its last chunk."""
        with self._cond:
            segment = self._segment
            if segment is None or segment.done:
                return
            if fault:
                self._finish(segment, False, f"bridge fault: {fault}")
            elif ack and int(ack.get("segment", -1)) == segment.id:
                segment.next_chunk = max(segment.next_chunk, int(ack["chunk"]) + 1)
                if segment.next_chunk >= segment.chunk_count():
                    self._finish(segment, True, "")

    def next_command(self) -> tuple[np.ndarray, dict[str, Any]]:
        """``(targets (N, 8), info)`` for the next chunk; N = 0 means hold and poll again."""
        with self._cond:
            if self.closed or self._cancel_pending:
                self._cancel_pending = False
                return np.empty((0, 8)), {"cancel": True, "idle": True, "closed": self.closed}
            segment = self._segment
            if segment is None or segment.done:
                return np.empty((0, 8)), {"idle": True, "poll_s": 0.1}
            start = segment.next_chunk * CHUNK_STEPS
            chunk = segment.targets[start : start + CHUNK_STEPS]
            return chunk, {
                "segment": segment.id,
                "chunk": segment.next_chunk,
                "last": segment.next_chunk == segment.chunk_count() - 1,
                "dt": segment.dt,
                "idle": False,
            }

    # -- station side --------------------------------------------------------

    def run_segment(self, targets: np.ndarray, dt: float, timeout_s: float) -> tuple[bool, str]:
        """Publish ``targets`` and block until the bridge has executed them all."""
        with self._cond:
            if self._segment is not None and not self._segment.done:
                raise RemoteLinkError("a trajectory is already being executed")
            self._ids += 1
            segment = _Segment(self._ids, np.asarray(targets, np.float64).reshape(-1, 8), dt)
            self._segment = segment
            self._cond.notify_all()
            deadline = time.monotonic() + timeout_s
            while not segment.done:
                if self.closed:
                    self._finish(segment, False, "server shut down")
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._cancel_pending = True
                    self._finish(
                        segment, False,
                        f"no acknowledgement from the bridge within {timeout_s:.0f} s "
                        f"(served {segment.next_chunk}/{segment.chunk_count()} chunks)",
                    )
                    break
                self._cond.wait(min(remaining, 0.25))
            return segment.ok, segment.reason

    def _finish(self, segment: _Segment, ok: bool, reason: str) -> None:
        segment.done, segment.ok, segment.reason = True, ok, reason
        self._cond.notify_all()

    def latest(self, *, wait_s: float = 5.0) -> RemoteState:
        """The freshest state, waiting up to ``wait_s`` if there is none yet or it is stale."""
        with self._cond:
            deadline = time.monotonic() + wait_s
            while True:
                state = self._state
                if state is not None and time.monotonic() - state.received_s <= STATE_STALE_S:
                    return state
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    why = "has not connected" if state is None else "state is stale"
                    raise RemoteLinkError(f"the robot bridge {why}; is inference-aspire running?")
                self._cond.wait(remaining)

    def state_after(self, t_s: float, *, wait_s: float = 5.0) -> RemoteState:
        """The first state received at or after monotonic time ``t_s``."""
        with self._cond:
            deadline = time.monotonic() + wait_s
            while self._state is None or self._state.received_s < t_s:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RemoteLinkError("the robot bridge stopped reporting state")
                self._cond.wait(remaining)
            return self._state

    def previous(self) -> RemoteState | None:
        with self._cond:
            return self._previous

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()


class RealCameraModel:
    """Pinhole camera in the robot base frame, with depth from the table plane."""

    def __init__(self, base_from_camera: np.ndarray, vertical_fov_deg: float, table_z: float):
        self.base_from_camera = np.asarray(base_from_camera, np.float64).reshape(4, 4)
        self.vertical_fov_deg = float(vertical_fov_deg)
        self.table_z = float(table_z)
        self._depth_cache: dict[tuple[int, int], np.ndarray] = {}

    @classmethod
    def from_layout(
        cls, config: ElRobotConfig, layout_path: str | Path | None = None
    ) -> RealCameraModel:
        """The camera fitted to the real demonstrations (``pizza-robot/real/layout.json``).

        The fit is already expressed in the robot base frame (the scene's overhead camera is this
        pose pulled back 1.45x along its axis, and ``base_link`` has an identity rotation), so
        it is used as is. Only the table height is read from the scene.
        """
        import mujoco

        layout = json.loads(Path(layout_path or REAL_DIR / "layout.json").read_text())["camera"]
        right, up = np.split(np.asarray(layout["xyaxes_right_up"], np.float64), 2)
        back = np.cross(right, up)  # MuJoCo cameras look along -z
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack([right, up, back]) @ _OPTICAL_FROM_MUJOCO
        transform[:3, 3] = layout["position"]

        model = build_model(config, objects=False)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        top = model.geom("table_top")
        base_z = data.xpos[model.body("base_link").id][2]
        table_z = float(data.geom_xpos[top.id][2] + model.geom_size[top.id][2] - base_z)
        return cls(transform, layout["vertical_fov_deg"], table_z)

    def intrinsics(self, width: int, height: int) -> np.ndarray:
        focal = (height / 2.0) / np.tan(np.deg2rad(self.vertical_fov_deg) / 2.0)
        return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])

    def table_depth(self, width: int, height: int) -> np.ndarray:
        """Distance along the optical axis to the table plane, 0 where the ray misses it."""
        key = (width, height)
        if key not in self._depth_cache:
            k = self.intrinsics(width, height)
            u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
            rays = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1)
            world = rays @ self.base_from_camera[:3, :3].T
            height_above = self.base_from_camera[2, 3] - self.table_z
            with np.errstate(divide="ignore", invalid="ignore"):
                depth = np.where(world[..., 2] < -1e-6, height_above / -world[..., 2], 0.0)
            self._depth_cache[key] = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
        return self._depth_cache[key]


class RemoteElRobotStation:
    """:class:`~cap_harness.elrobot.station.ElRobotStation` backed by a :class:`RemoteLink`."""

    def __init__(
        self,
        config: ElRobotConfig,
        link: RemoteLink,
        cameras: dict[str, RealCameraModel],
        *,
        joint_map: RealJointMap | None = None,
        kinematics: Any = None,
    ) -> None:
        self.config = config
        self.link = link
        self.cameras = cameras
        self.joint_map = joint_map or RealJointMap.from_config(config)
        self._kinematics = kinematics
        self.camera_aliases = tuple(cameras)
        self._period = 1.0 / config.control_frequency_hz

    # -- reads -----------------------------------------------------------------

    def _state(self) -> RemoteState:
        return self.link.latest()

    def joint_positions(self) -> np.ndarray:
        return self.joint_map.to_sim(self._state().norm)[0]

    def gripper_position(self) -> float:
        return float(self.joint_map.to_sim(self._state().norm)[1])

    def joint_velocities(self) -> np.ndarray:
        state, previous = self._state(), self.link.previous()
        if previous is None or state.received_s <= previous.received_s:
            return np.zeros(7)
        delta = self.joint_map.to_sim(state.norm)[0] - self.joint_map.to_sim(previous.norm)[0]
        return delta / (state.received_s - previous.received_s)

    def get_observations(self) -> dict[str, np.ndarray]:
        joints = self.joint_positions()
        if self._kinematics is None:
            position, quaternion = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        else:
            position, quaternion = self._kinematics.forward_kinematics(joints)
        return {
            "joint_pos": joints,
            "joint_vel": self.joint_velocities(),
            "gripper_pos": np.array([self.gripper_position()]),
            "ee_pos": np.asarray(position, np.float64),
            "ee_quat": np.asarray(quaternion, np.float64),
        }

    def camera_frame(self, name: str) -> CameraFrame | None:
        model = self.cameras.get(name)
        state = self._state()
        rgb = state.frames.get(name)
        if model is None or rgb is None:
            return None
        height, width = rgb.shape[:2]
        depth = state.depth.get(name)
        if depth is None or depth.shape != (height, width):
            depth = model.table_depth(width, height)
        return CameraFrame(
            rgb, depth, model.intrinsics(width, height), model.base_from_camera, state.received_s
        )

    # -- actuation ---------------------------------------------------------------

    def execute_waypoints(
        self, times_s: Any, joints: Any, grippers: Any, *, settle_s: float = 0.0
    ) -> dict[str, Any]:
        times = np.asarray(times_s, np.float64).reshape(-1)
        joints = np.asarray(joints, np.float64).reshape(len(times), -1)
        grippers = np.asarray(grippers, np.float64).reshape(len(times))
        joints = np.clip(joints, self.config.joint_limits_lower, self.config.joint_limits_upper)

        current = self.joint_positions()
        start_error = float(np.max(np.abs(joints[0] - current)))
        if start_error > START_TOLERANCE_RAD:
            return {
                "success": False,
                "command_count": 0,
                "reason": (
                    f"trajectory starts {start_error:.3f} rad from the arm's reported pose "
                    f"(limit {START_TOLERANCE_RAD}); re-read the state and plan again"
                ),
            }

        grid = np.arange(0.0, times[-1] + 1e-9, self._period)
        if grid[-1] < times[-1] - 1e-9:
            grid = np.append(grid, times[-1])
        grid = grid[1:] if len(grid) > 1 else grid
        steps = np.column_stack(
            [np.interp(grid, times, joints[:, axis]) for axis in range(joints.shape[1])]
        )
        step_gripper = np.interp(grid, times, grippers)
        targets = self.joint_map.from_sim(steps, step_gripper)

        limit = self.config.joint_speed_limits_rad_s * self._period * SPEED_MARGIN
        jumps = np.abs(np.diff(np.vstack([current, steps]), axis=0))
        if bool(np.any(jumps > limit)):
            return {"success": False, "command_count": 0,
                    "reason": "trajectory exceeds the joint speed limits"}

        ok, reason = self.link.run_segment(targets, self._period, times[-1] + ACK_MARGIN_S)
        if ok:
            time.sleep(max(settle_s, 0.0))
            self.link.state_after(time.monotonic(), wait_s=ACK_MARGIN_S)
        return {"success": ok, "command_count": len(targets), "reason": reason}

    def set_gripper(
        self, position: float, *, duration_s: float = 1.0, max_wait_s: float = 4.0
    ) -> dict[str, Any]:
        """Command the jaws, then wait until the reading stops changing (open, closed, or on an object)."""
        held = self.joint_positions()
        start = self.gripper_position()
        result = self.execute_waypoints(
            [0.0, duration_s], np.vstack([held, held]), [start, float(position)]
        )
        waited, last = 0.0, self.gripper_position()
        while result["success"] and waited < max_wait_s:
            time.sleep(0.2)
            waited += 0.2
            now = self.gripper_position()
            if abs(now - last) < 0.01:
                break
            last = now
        result["gripper_wait_s"] = round(waited, 2)
        return result

    def go_home(self, *, duration_s: float = 3.0) -> dict[str, Any]:
        start, grip = self.joint_positions(), self.gripper_position()
        return self.execute_waypoints(
            [0.0, duration_s],
            np.vstack([start, self.config.home_joints]),
            [grip, self.config.home_gripper],
            settle_s=0.5,
        )

    def reset(self) -> None:
        """A real scene cannot be reset; the arm and items stay as they are."""

    def control_contract(self) -> dict[str, Any]:
        return {
            "version": CONTROL_CONTRACT_VERSION,
            "plant": "remote-real",
            "control_frequency_hz": self.config.control_frequency_hz,
            "joint_names": list(self.config.joint_names),
            "joint_limits_lower": self.config.joint_limits_lower.tolist(),
            "joint_limits_upper": self.config.joint_limits_upper.tolist(),
            "gripper": "0 closed, 1 open; wire unit is position_norm",
            "depth": "synthesized from the table plane unless the bridge sends it",
        }

    def close(self) -> None:
        self.link.close()


__all__ = [
    "CHUNK_STEPS",
    "RealCameraModel",
    "RemoteElRobotStation",
    "RemoteLink",
    "RemoteLinkError",
    "RemoteState",
]

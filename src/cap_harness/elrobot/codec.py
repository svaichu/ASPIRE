"""Convert the ElRobot plant's readings into the shared harness contracts.

The plant already speaks the harness conventions -- quaternions are wxyz, poses
are in ``robot_base`` -- so this layer is mostly assembly. What it does own is the
gripper: the contracts carry ``gripper_position`` as its own normalized field
(0 closed, 1 open), never as a joint 8, and the plant's radians stay below here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    Observation,
    Pose,
    RobotState,
    TaskContext,
)
from cap_harness.elrobot.config import ARM_JOINTS
from cap_harness.yam_real.codec import rotation_to_wxyz

EMBODIMENT = "elrobot"
ARMS = ("arm",)
ARM_DOF = 7
JOINT_NAMES = ARM_JOINTS


def robot_state(env: Any, *, timestamp_s: float | None = None) -> RobotState:
    obs = env.get_observations()
    base_frame = env.config.base_frame
    return RobotState(
        joint_positions={"arm": np.asarray(obs["joint_pos"], dtype=np.float64).reshape(ARM_DOF)},
        joint_velocities={"arm": np.asarray(obs["joint_vel"], dtype=np.float64).reshape(ARM_DOF)},
        end_effector_poses={
            "arm": Pose(
                position=np.asarray(obs["ee_pos"], dtype=np.float64).reshape(3),
                quaternion_wxyz=np.asarray(obs["ee_quat"], dtype=np.float64).reshape(4),
                frame=base_frame,
            )
        },
        gripper_positions={"arm": float(np.asarray(obs["gripper_pos"]).reshape(-1)[0])},
        base_frame=base_frame,
        joint_names={"arm": JOINT_NAMES},
        timestamp_s=timestamp_s,
        embodiment=EMBODIMENT,
    )


def camera_observation(env: Any, name: str) -> CameraObservation | None:
    frame = env.camera_frame(name)
    if frame is None:
        return None
    transform = np.asarray(frame.base_from_camera, dtype=np.float64)
    return CameraObservation(
        rgb=np.asarray(frame.rgb, dtype=np.uint8),
        depth_m=np.asarray(frame.depth_m, dtype=np.float64),
        intrinsics=np.asarray(frame.intrinsics, dtype=np.float64),
        # The camera's own optical frame, so point-cloud conversion applies the
        # extrinsics instead of relabelling camera-frame points as base ones.
        frame=f"{name}_optical",
        camera_pose=Pose(
            position=transform[:3, 3],
            quaternion_wxyz=rotation_to_wxyz(transform[:3, :3]),
            frame=env.config.base_frame,
        ),
        timestamp_s=float(frame.timestamp_s),
    )


def observation(env: Any, *, task_context: TaskContext | None = None) -> Observation:
    cameras = {}
    for name in env.camera_aliases:
        camera = camera_observation(env, name)
        if camera is not None:
            cameras[name] = camera
    if not cameras:
        raise RuntimeError("the station produced no camera frame")
    return Observation(
        cameras=cameras,
        robot_state=robot_state(env),
        task_context=task_context,
    )


def arm_command_targets(action: Any) -> tuple[dict[str, np.ndarray], dict[str, float | None]]:
    joints: dict[str, np.ndarray] = {}
    grippers: dict[str, float | None] = {}
    for side, command in action.arms.items():
        if not isinstance(command, ArmCommand):
            raise ValueError(f"arms[{side!r}] must be an ArmCommand")
        joints[side] = np.asarray(command.target, dtype=np.float64).reshape(-1)[:ARM_DOF]
        grippers[side] = (
            None if command.gripper_position is None else float(command.gripper_position)
        )
    return joints, grippers


__all__ = [
    "ARMS",
    "ARM_DOF",
    "EMBODIMENT",
    "JOINT_NAMES",
    "arm_command_targets",
    "camera_observation",
    "observation",
    "robot_state",
]

"""Scene, joint and camera description for the ElRobot pizza station.

Everything embodiment-specific that a plant, a solver and an adapter must agree
on lives here, so each has exactly one place to read it from. The scene itself is
data under ``pizza-robot/scene/``; this module only locates and interprets it.

Frames: ``robot_base`` is the ``base_link`` body, so +z is up from the table top
and the arm reaches along +y at zero pose. All poses crossing the harness
boundary are expressed in it; MuJoCo's world frame never leaks out.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np

BASE_FRAME = "robot_base"
BASE_BODY = "base_link"
GRASP_SITE = "grasp_site"
ARM_JOINTS = tuple(f"joint{index}" for index in range(1, 8))
GRIPPER_JOINT = "gripper"
#: Camera that carries calibrated RGB-D for localization, then the arm-mounted one.
CAMERAS = ("overhead", "wrist")
DEFAULT_CAMERA = "overhead"

SCENE_ENV = "ASPIRE_ELROBOT_SCENE"
_DEFAULT_SCENE_DIR = Path(__file__).resolve().parents[3] / "pizza-robot" / "scene"


def scene_dir() -> Path:
    override = os.environ.get(SCENE_ENV)
    return Path(override).expanduser() if override else _DEFAULT_SCENE_DIR


@dataclass(frozen=True, slots=True)
class ElRobotConfig:
    """Static description of one ElRobot station."""

    station: str
    model_xml: Path
    home_joints: np.ndarray  # (7,) arm joints, radians
    home_gripper: float  # normalized, 1.0 = open
    joint_limits_lower: np.ndarray
    joint_limits_upper: np.ndarray
    gripper_open_rad: float
    gripper_closed_rad: float
    control_frequency_hz: float = 50.0
    camera_resolution: tuple[int, int] = (640, 480)  # width, height
    base_frame: str = BASE_FRAME
    camera: str = DEFAULT_CAMERA

    @property
    def joint_names(self) -> tuple[str, ...]:
        return ARM_JOINTS

    def gripper_to_rad(self, normalized: float) -> float:
        value = float(np.clip(normalized, 0.0, 1.0))
        return self.gripper_closed_rad + value * (self.gripper_open_rad - self.gripper_closed_rad)

    def gripper_from_rad(self, radians: float) -> float:
        span = self.gripper_open_rad - self.gripper_closed_rad
        return float(np.clip((float(radians) - self.gripper_closed_rad) / span, 0.0, 1.0))


def load_config(station: str = "pizza", *, directory: str | Path | None = None) -> ElRobotConfig:
    """Read ``home.json`` and the actuator ranges of the scene's MJCF."""
    import mujoco

    root = Path(directory).expanduser() if directory else scene_dir()
    xml = root / "elrobot_table.xml"
    home = json.loads((root / "home.json").read_text())
    if tuple(home["joint_names"]) != (*ARM_JOINTS, GRIPPER_JOINT):
        raise ValueError("home.json joint_names must be joint1..joint7 then gripper")
    model = mujoco.MjModel.from_xml_path(str(xml))
    lower, upper = [], []
    for name in ARM_JOINTS:
        low, high = model.actuator(name).ctrlrange
        lower.append(low)
        upper.append(high)
    values = np.asarray(home["home_joints"], dtype=np.float64)
    opened, closed = float(home["gripper_open_rad"]), float(home["gripper_closed_rad"])
    config = ElRobotConfig(
        station=station,
        model_xml=xml,
        home_joints=values[:7].copy(),
        home_gripper=float(np.clip((values[7] - closed) / (opened - closed), 0.0, 1.0)),
        joint_limits_lower=np.asarray(lower, dtype=np.float64),
        joint_limits_upper=np.asarray(upper, dtype=np.float64),
        gripper_open_rad=opened,
        gripper_closed_rad=closed,
    )
    if bool(np.any(config.home_joints < config.joint_limits_lower)) or bool(
        np.any(config.home_joints > config.joint_limits_upper)
    ):
        raise ValueError("home_joints lie outside the actuator ranges")
    return config


__all__ = [
    "ARM_JOINTS",
    "BASE_BODY",
    "BASE_FRAME",
    "CAMERAS",
    "DEFAULT_CAMERA",
    "GRASP_SITE",
    "GRIPPER_JOINT",
    "ElRobotConfig",
    "load_config",
    "scene_dir",
]

"""The real ElRobot's recorded joint space, mapped onto the simulated arm.

The real station records each joint as ``position_norm``: the servo position as a fraction of
the range its calibration sweep found, ``[0, 1]`` from ``range_min`` to ``range_max``. That
calibrated sweep is the arm's full mechanical travel, which is what the URDF joint limits
(and so this sim's actuator ranges) describe, so the map is linear:

    joint angle = lower_limit + norm * (upper_limit - lower_limit)         (joints 1-7)

This was checked rather than assumed (``pizza-robot/real/analyze.py``). The calibrated spans match
the sim's limit spans to 2% on six joints (joint 1 is 7% narrower, the gripper 7% wider). Of the
128 direction choices, 12 keep the fingertip above the table across all 12.8k recorded steps
(touching z = 0 where the operator picks), and the data fixes the direction of joints 2, 3 and
4. Joint 1 mirrors the whole layout; it is fixed by a camera fit, which is worse by 2.7x
(54 px vs 20 px) when joint 1 is flipped. Joints 5, 6 and 7 are left open by position data
(wrist rolls barely move the fingertip), so the all-positive choice is the assumption there.

The gripper is different in convention only: real ``norm`` is high when open (the jaws open at
~0.85 and hold a shred of cheese at ~0.15), the sim's normalized gripper is 1 when open, so the
two agree up to a rescale of the observed open/closed endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

REAL_DIR = Path(__file__).resolve().parents[3] / "pizza-robot" / "real"
#: Observed endpoints of the real gripper's normalized reading, from the demonstrations'
#: 1st and 99th percentiles: fully closed on nothing, fully open.
REAL_GRIPPER_CLOSED = 0.13
REAL_GRIPPER_OPEN = 0.85


@dataclass(frozen=True, slots=True)
class RealEpisode:
    name: str
    task: str
    state: np.ndarray  # (T, 8) position_norm
    action: np.ndarray  # (T, 8) goal_norm
    fps: float


class RealJointMap:
    """Real ``[0, 1]`` readings <-> the sim's joint radians and normalized gripper."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        self.lower = np.asarray(lower, dtype=np.float64).reshape(7)
        self.upper = np.asarray(upper, dtype=np.float64).reshape(7)

    @classmethod
    def from_config(cls, config: object) -> RealJointMap:
        return cls(config.joint_limits_lower, config.joint_limits_upper)

    def to_sim(self, norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(joints_rad, gripper_normalized)`` for one row or a ``(T, 8)`` block."""
        norm = np.asarray(norm, dtype=np.float64)
        joints = self.lower + norm[..., :7] * (self.upper - self.lower)
        gripper = (norm[..., 7] - REAL_GRIPPER_CLOSED) / (REAL_GRIPPER_OPEN - REAL_GRIPPER_CLOSED)
        return joints, np.clip(gripper, 0.0, 1.0)

    def from_sim(self, joints: np.ndarray, gripper: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float64)
        norm = (joints - self.lower) / (self.upper - self.lower)
        grip = REAL_GRIPPER_CLOSED + np.asarray(gripper, dtype=np.float64).reshape(
            norm.shape[:-1]
        ) * (REAL_GRIPPER_OPEN - REAL_GRIPPER_CLOSED)
        return np.concatenate([norm, grip[..., None]], axis=-1)


#: Gripper goals on the real robot: above this the jaws are open, below this they are closed on
#: something or on nothing (the reading is high when open).
GRIPPER_OPEN_GOAL = 0.5
GRIPPER_CLOSED_GOAL = 0.4
#: A close counts as a pick only this near the table, and only this far from where the jaws opened.
PICK_MAX_Z_M = 0.08
PICK_MAX_TRAVEL_M = 0.12


def pick_event(gripper_goal: np.ndarray) -> tuple[int, int] | None:
    """``(open, close)`` frame of the first grasp: the first rise above open, then the first fall."""
    up = np.where(np.asarray(gripper_goal) > GRIPPER_OPEN_GOAL)[0]
    if not len(up):
        return None
    down = np.where(np.asarray(gripper_goal)[up[0] :] < GRIPPER_CLOSED_GOAL)[0]
    return (int(up[0]), int(up[0] + down[0])) if len(down) else None


def is_clean_pick(tip_at_open: np.ndarray, tip_at_close: np.ndarray) -> bool:
    """A grasp at the table close to where the jaws opened (not a release over the plate)."""
    return bool(
        tip_at_close[2] <= PICK_MAX_Z_M
        and np.linalg.norm(np.asarray(tip_at_close) - np.asarray(tip_at_open)) <= PICK_MAX_TRAVEL_M
    )


def load_pick_spots(directory: str | Path | None = None, *, min_picks: int = 5) -> dict[str, tuple[float, float]]:
    """Where operators grasp each topping, xy in the base frame, for toppings with enough picks.

    Written by ``pizza-robot/real/derive_pick_spots.py``. Empty if the file is absent. The
    layout's tray centres were read by eye and miss the grasp by ~5 cm (cheese); these do not.
    """
    path = Path(directory or REAL_DIR) / "pick_spots.json"
    if not path.exists():
        return {}
    spots = json.loads(path.read_text())["spots"]
    return {name: tuple(entry["xy"]) for name, entry in spots.items() if entry["n"] >= min_picks}


def load_real_episodes(directory: str | Path | None = None) -> list[RealEpisode]:
    """The exported demonstrations (``pizza-robot/real/episodes.npz``), one per episode."""
    data = np.load(Path(directory or REAL_DIR) / "episodes.npz")
    episodes = []
    for index, start in enumerate(data["starts"]):
        stop = int(start + data["lengths"][index])
        episodes.append(
            RealEpisode(
                name=str(data["names"][index]),
                task=str(data["tasks"][index]),
                state=data["state"][int(start) : stop],
                action=data["action"][int(start) : stop],
                fps=float(data["fps"][index]),
            )
        )
    return episodes


def load_calibration(directory: str | Path | None = None) -> dict:
    return json.loads((Path(directory or REAL_DIR) / "calibration.json").read_text())


__all__ = [
    "REAL_GRIPPER_CLOSED",
    "REAL_GRIPPER_OPEN",
    "GRIPPER_CLOSED_GOAL",
    "GRIPPER_OPEN_GOAL",
    "PICK_MAX_TRAVEL_M",
    "PICK_MAX_Z_M",
    "RealEpisode",
    "RealJointMap",
    "is_clean_pick",
    "load_calibration",
    "load_pick_spots",
    "load_real_episodes",
    "pick_event",
]

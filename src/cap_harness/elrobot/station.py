"""How an ElRobot station is built, and what the adapter may assume about it.

Sim-only. A real station would need its own transport for the Feetech bus servos
(see ``pizza-robot/PLAN.md``); until then there is deliberately no
``build_real_station`` to be mistaken for one.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from cap_harness.elrobot.config import ElRobotConfig, load_config


@runtime_checkable
class ElRobotStation(Protocol):
    """What the adapter needs from an ElRobot plant."""

    config: ElRobotConfig
    camera_aliases: tuple[str, ...]

    def get_observations(self) -> dict[str, np.ndarray]: ...

    def joint_positions(self) -> np.ndarray: ...

    def execute_waypoints(
        self, times_s: Any, joints: Any, grippers: Any, *, settle_s: float = 0.0
    ) -> dict[str, Any]:
        """The single actuation chokepoint."""

    def set_gripper(self, position: float, **kwargs: Any) -> dict[str, Any]: ...

    def go_home(self, **kwargs: Any) -> dict[str, Any]: ...

    def reset(self) -> None: ...

    def camera_frame(self, name: str) -> Any | None: ...

    def control_contract(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def build_sim_station(
    station: str = "pizza", *, scene_directory: str | None = None
) -> ElRobotStation:
    from cap_harness.elrobot.sim import SimElRobotStation

    return SimElRobotStation(load_config(station, directory=scene_directory))


__all__ = ["ElRobotStation", "build_sim_station"]

"""Clearance-checked approach search for the ElRobot -- what cuRobo would be here.

The motion stack for this station is mink IK plus joint-space interpolation, and
neither knows the bins exist. A hover-then-descend into a bin therefore works at
some positions and drives the wrist base into a wall at others, and the only way
to find out used to be to try it. This module answers *before* moving: for a
fingertip target it searches approach tilts and lean directions for one whose IK
solutions are reachable and whose sampled joint-space path, hover to grasp, keeps
the gripper clear of the static scene.

What it knows: the table, bins, tray and plate -- the scene with every loose item
removed. What it does not know: the loose items themselves (a neighbouring slice
can still be nudged by an open jaw), anything the arm is holding, or dynamics. A
"clear" verdict is necessary, not sufficient; verify a descent from the returned
images as usual.

Pure planning: it builds its own model copy and never touches a running station.

    python -m cap_harness.elrobot.approach X Y Z [--jaw-azimuth D] [--hover Z]
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from cap_harness.elrobot.config import BASE_BODY, ElRobotConfig, load_config
from cap_harness.elrobot.kinematics import ElRobotKinematics
from cap_harness.elrobot.poses import grasp_pose

#: Approaches tried, most conventional first. Tilt is degrees from straight down;
#: lean turns the tilt direction about vertical (0 = away from the base).
DEFAULT_TILTS = (40.0, 30.0, 50.0, 20.0)
DEFAULT_LEANS = (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)
#: A path point counts as blocked only once it overlaps the scene by this much, metres.
#: Fingertips are meant to rest on a bin floor at the grasp, and the pad's lower corner dips a
#: few millimetres below the fingertip at the standard tilt, into a floor that gives
#: (it is a soft contact, and the grips hold with it). A wall or the table in the way shows up
#: as far more than this. Calibrated against pick success: 0.5 mm refused approaches that
#: physically work, and steered cheese toward approaches that grip worse.
PENETRATION_M = 0.0035
PATH_SAMPLES = 8


@dataclass(frozen=True, slots=True)
class Approach:
    tilt_deg: float
    lean_deg: float
    hover_pose: dict
    grasp_pose: dict
    clear: bool
    reachable: bool
    blocked_by: tuple[str, ...] = ()
    joint_travel_rad: float = 0.0

    def as_dict(self) -> dict:
        return {
            "tilt_deg": self.tilt_deg,
            "lean_deg": self.lean_deg,
            "hover_pose": self.hover_pose,
            "grasp_pose": self.grasp_pose,
            "clear": self.clear,
            "reachable": self.reachable,
            "blocked_by": list(self.blocked_by),
            "joint_travel_rad": round(self.joint_travel_rad, 3),
        }


class ApproachPlanner:
    def __init__(self, config: ElRobotConfig | None = None) -> None:
        self.config = config or load_config()
        self._kin = ElRobotKinematics(self.config)
        self._model = self._kin.model_copy_for_collisions()
        self._data = mujoco.MjData(self._model)
        self._arm = _subtree(self._model, self._model.body(BASE_BODY).id)

    def _collisions(self, joints: np.ndarray) -> list[str]:
        self._kin.write_qpos(self._data, joints)
        mujoco.mj_forward(self._model, self._data)
        mujoco.mj_collision(self._model, self._data)
        hits = []
        for index in range(self._data.ncon):
            contact = self._data.contact[index]
            first = self._model.geom_bodyid[contact.geom1]
            second = self._model.geom_bodyid[contact.geom2]
            if contact.dist < -PENETRATION_M and (first in self._arm) != (second in self._arm):
                other = second if first in self._arm else first
                hits.append(self._model.body(other).name or "world")
        return sorted(set(hits))

    def evaluate(
        self,
        position: tuple[float, float, float],
        tilt_deg: float,
        lean_deg: float,
        *,
        hover_z: float,
        jaw_azimuth_deg: float | None,
        start: np.ndarray | None = None,
        jaw_deg: float = 0.0,
    ) -> Approach:
        x, y, z = (float(v) for v in position)
        kw = {} if jaw_azimuth_deg is None else {"jaw_azimuth_deg": jaw_azimuth_deg}
        hover = grasp_pose((x, y, hover_z), tilt_deg, jaw_deg, lean_deg, **kw)
        grasp = grasp_pose((x, y, z), tilt_deg, jaw_deg, lean_deg, **kw)
        origin = np.asarray(self.config.home_joints if start is None else start, dtype=np.float64)
        solved = []
        for pose in (hover, grasp):
            joints, metres, radians = self._kin.inverse_kinematics(
                np.array(pose["position"]),
                np.array(pose["quaternion_wxyz"]),
                solved[-1] if solved else origin,
                restarts=8,
            )
            if metres > 0.005 or radians > 0.05:
                return Approach(tilt_deg, lean_deg, hover, grasp, False, False)
            solved.append(joints)
        # The executor interpolates in joint space, so that is the path to check:
        # start -> hover, then hover -> grasp.
        blocked: set[str] = set()
        travel = 0.0
        for a, b in ((origin, solved[0]), (solved[0], solved[1])):
            travel += float(np.max(np.abs(b - a)))
            for fraction in np.linspace(0.0, 1.0, PATH_SAMPLES):
                blocked.update(self._collisions(a + fraction * (b - a)))
        return Approach(
            tilt_deg, lean_deg, hover, grasp, not blocked, True, tuple(sorted(blocked)), travel
        )

    def plan(
        self,
        position: tuple[float, float, float],
        *,
        hover_z: float = 0.09,
        jaw_azimuth_deg: float | None = None,
        tilts: tuple[float, ...] = DEFAULT_TILTS,
        leans: tuple[float, ...] = DEFAULT_LEANS,
        start: np.ndarray | None = None,
        limit: int = 3,
    ) -> list[Approach]:
        """Up to ``limit`` clear approaches, best first; the best blocked one if none are clear."""
        clear: list[Approach] = []
        best_blocked: Approach | None = None
        for tilt in tilts:
            for lean in leans:
                found = self.evaluate(
                    position,
                    tilt,
                    lean,
                    hover_z=hover_z,
                    jaw_azimuth_deg=jaw_azimuth_deg,
                    start=start,
                )
                if found.clear:
                    clear.append(found)
                    if len(clear) >= limit:
                        return clear
                elif found.reachable and (
                    best_blocked is None or len(found.blocked_by) < len(best_blocked.blocked_by)
                ):
                    best_blocked = found
        return clear or ([best_blocked] if best_blocked else [])


def _subtree(model: mujoco.MjModel, root: int) -> set[int]:
    bodies = {root}
    for index in range(model.nbody):
        parent = model.body_parentid[index]
        if parent in bodies and index != parent:
            bodies.add(index)
    return bodies


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Find a clearance-checked approach to a fingertip target."
    )
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("z", type=float)
    parser.add_argument("--hover", type=float, default=0.09, help="hover height above the table, m")
    parser.add_argument(
        "--jaw-azimuth", type=float, default=None, help="jaw axis in the base frame, deg"
    )
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args(argv)
    found = ApproachPlanner().plan(
        (args.x, args.y, args.z),
        hover_z=args.hover,
        jaw_azimuth_deg=args.jaw_azimuth,
        limit=args.limit,
    )
    print(json.dumps({"approaches": [item.as_dict() for item in found]}, indent=1))
    return 0 if found and found[0].clear else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["Approach", "ApproachPlanner"]

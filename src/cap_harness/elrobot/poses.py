"""Gripper poses for the ElRobot, expressed in the robot base frame.

Pure geometry -- no robot, no scene, no state. Exists because the arm's workspace
is shaped by one wrist: with the jaw axis pinned to the world x axis, a top-down
pose at the salami bin is out of reach, while the same point is reachable once the
approach is yawed to face away from the base and tilted a little off vertical.

Convention (the ``grasp_site`` frame): ``z`` is the approach direction out of the
fingers, ``x`` is the jaw-closing axis, ``y = z cross x``.

* ``tilt_deg`` is the approach's angle from straight down (0 = top-down). The
  approach leans *away* from the base, along the radial direction to the target.
* ``lean_deg`` turns that lean direction about vertical: 0 leans away from the
  base, 180 leans back toward it, +/-90 lean sideways. Use it to keep the wrist
  and jaw bodies on the open side of a bin wall.
* ``jaw_deg`` is the jaw-closing axis, as a horizontal angle about the approach
  from the *tangent* direction: 0 closes the jaws sideways across the radial line
  (the reachable default), 90 closes them along it. Choose it so the closing axis
  runs parallel to a nearby wall -- the open jaws reach 3 cm to each side of the
  fingertip point.
* ``jaw_azimuth_deg``, if given, overrides ``jaw_deg``: the closing axis is the
  horizontal direction at that angle in the base frame (0 = +x, 90 = +y), made
  perpendicular to the approach. Use it to close across an object's short side.
"""

from __future__ import annotations

import numpy as np

from cap_harness.yam_real.codec import rotation_to_wxyz


def grasp_pose(
    position: tuple[float, float, float] | np.ndarray,
    tilt_deg: float = 30.0,
    jaw_deg: float = 0.0,
    lean_deg: float = 0.0,
    jaw_azimuth_deg: float | None = None,
    *,
    frame: str = "robot_base",
) -> dict[str, object]:
    """A ``move_to_pose`` target for the fingertip centre at ``position``."""
    p = np.asarray(position, dtype=np.float64).reshape(3)
    radial = np.array([p[0], p[1], 0.0])
    norm = np.linalg.norm(radial)
    radial = radial / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])
    tilt = np.deg2rad(float(tilt_deg))
    turn = np.deg2rad(float(lean_deg))
    lean = np.array(
        [
            np.cos(turn) * radial[0] - np.sin(turn) * radial[1],
            np.sin(turn) * radial[0] + np.cos(turn) * radial[1],
            0.0,
        ]
    )
    approach = np.array([np.sin(tilt) * lean[0], np.sin(tilt) * lean[1], -np.cos(tilt)])
    tangent = np.cross([0.0, 0.0, 1.0], radial)
    x = tangent - np.dot(tangent, approach) * approach
    x /= np.linalg.norm(x)
    y0 = np.cross(approach, x)
    if jaw_azimuth_deg is not None:
        wish = np.deg2rad(float(jaw_azimuth_deg))
        x = np.array([np.cos(wish), np.sin(wish), 0.0])
        x = x - np.dot(x, approach) * approach
        x /= np.linalg.norm(x)
    else:
        turn_jaw = np.deg2rad(float(jaw_deg))
        x = np.cos(turn_jaw) * x + np.sin(turn_jaw) * y0
    y = np.cross(approach, x)
    quaternion = rotation_to_wxyz(np.column_stack([x, y, approach]))
    return {
        "position": [round(float(v), 4) for v in p],
        "frame": frame,
        "quaternion_wxyz": [round(float(v), 5) for v in quaternion],
    }


__all__ = ["grasp_pose"]


def _main(argv: list[str] | None = None) -> int:
    """``python -m cap_harness.elrobot.poses X Y Z [--tilt D] ...`` prints a pose as JSON."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("z", type=float)
    parser.add_argument("--tilt", type=float, default=30.0, help="degrees from straight down")
    parser.add_argument(
        "--lean", type=float, default=0.0, help="lean direction, 0 = away from base"
    )
    parser.add_argument("--jaw", type=float, default=0.0, help="jaw axis from tangent, degrees")
    parser.add_argument(
        "--jaw-azimuth", type=float, default=None, help="jaw axis in the base frame"
    )
    args = parser.parse_args(argv)
    pose = grasp_pose(
        (args.x, args.y, args.z),
        args.tilt,
        args.jaw,
        args.lean,
        args.jaw_azimuth,
    )
    print(json.dumps(pose))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

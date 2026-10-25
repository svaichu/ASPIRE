"""Next small step when creeping the fingertip toward a localized item.

Pure geometry -- no robot, no scene, no state -- like :mod:`cap_harness.elrobot.poses`. The agent
and its tools run in the cloud while the arm is wired to a machine across a network link, so
every command is one short, bounded move followed by a fresh look. This module decides how short:

* **Two cameras, one estimate.** The overhead camera sees the whole scene but coarsely; the
  wrist camera is sharp only once the fingertip is close. Their estimates are blended by
  distance, and *how far they disagree* is the confidence: agreeing views allow a full step,
  disagreeing views shrink it. Views so far apart that they are probably two different items
  (each camera names its own "center" slice) are not averaged: the wrist is dropped, the overhead
  kept, and the step stays tiny. One camera alone is never fully trusted. The arm holds still
  (``action: "hold"``) only when nothing was localized or confidence is near zero.
* **Step size follows confidence and distance.** Large and quick when far and sure, a few
  millimetres when close or unsure, and never more than a fraction of the remaining distance.
* **Turn gradually.** Pass the fingertip's current orientation and each step also rotates a few
  degrees toward the wanted tilt, so the wrist camera swings toward the table without the one
  large wrist turn that a distant pose needs (and that the planner often cannot reach from Home).
* **Align, then descend.** The step moves sideways at the current height until the fingertip is
  over the item, and only then lowers it to the stop height. It never descends while misaligned
  and never goes below the stop height; the final descent into a bin is the approach planner's.

    python -m cap_harness.elrobot.creep --tip X Y Z --est overhead X Y Z [SCORE] \\
        --est wrist X Y Z [SCORE] [--stop-z 0.07] [--tilt 40]
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from cap_harness.elrobot.poses import grasp_pose

#: Two estimates this close (horizontally, metres) fully agree; this far apart they conflict.
AGREE_M = 0.015
CONFLICT_M = 0.05
#: A single camera, with nothing to cross-check it, is never trusted beyond this.
SINGLE_VIEW_CONFIDENCE = 0.6
#: After dropping a conflicting wrist view, the overhead alone is trusted no more than this.
CONFLICT_CONFIDENCE = 0.4
#: Below this the arm does not move: look again.
HOLD_BELOW = 0.25
#: Fingertip within this of the stop point is "arrived".
DONE_XY_M = 0.008
DONE_Z_M = 0.01
#: Lower the fingertip only once the horizontal error is under this.
ALIGN_XY_M = 0.015
#: Largest step by horizontal distance left, metres.
FAR_STEP_M, MID_STEP_M, NEAR_STEP_M = 0.04, 0.025, 0.012
FAR_M, NEAR_M = 0.10, 0.04
MIN_STEP_M = 0.004
#: A step never covers more than this fraction of what is left, so error in the estimate is
#: corrected by the next look before it can be overshot.
MAX_FRACTION = 0.6
#: Largest rotation per step toward the wanted orientation, degrees (before the confidence scale).
MAX_TURN_DEG = 10.0
#: Orientation within this of the wanted one counts as turned.
DONE_TURN_DEG = 4.0
#: Nominal slow fingertip speed used to size ``max_steps``, m/s.
NOMINAL_SPEED_M_S = 0.05
CONTROL_HZ = 50.0


@dataclass(frozen=True, slots=True)
class Profile:
    """How noisy the estimates are, which sets what counts as agreement and how far to trust one view."""

    agree_m: float = AGREE_M
    conflict_m: float = CONFLICT_M
    single_view: float = SINGLE_VIEW_CONFIDENCE


SIM = Profile()
#: Measured on the real demonstrations (``pizza-robot/scripts/check_creep_real.py``): the overhead and
#: wrist estimates of the same topping differ by a median 27 mm (p90 41 mm) and that gap tracks the
#: error (r = 0.58), so agreement is judged at the cameras' own noise, not the simulator's. One real
#: overhead view alone is 29 mm median error, as good as the pair's worse member.
REAL = Profile(agree_m=0.025, conflict_m=0.07, single_view=0.7)
PROFILES = {"sim": SIM, "real": REAL}


@dataclass(frozen=True, slots=True)
class Estimate:
    """Where one camera puts the item, in the robot base frame."""

    camera: str
    position: tuple[float, float, float]
    score: float = 1.0  # segmentation score in [0, 1]


def fuse(
    estimates: list[Estimate], tip: np.ndarray, profile: Profile = SIM
) -> tuple[np.ndarray, float, list[str]]:
    """``(item position, confidence in [0, 1], notes)`` from one or more camera estimates."""
    if not estimates:
        return np.zeros(3), 0.0, ["no estimate: the item was not localized"]
    tip = np.asarray(tip, dtype=np.float64).reshape(3)
    points = np.array([e.position for e in estimates], dtype=np.float64)
    scores = np.clip([e.score for e in estimates], 0.0, 1.0)
    notes: list[str] = []

    by_camera = {e.camera: i for i, e in enumerate(estimates)}
    if "wrist" in by_camera and "overhead" in by_camera:
        o, w = points[by_camera["overhead"]], points[by_camera["wrist"]]
        gap = float(np.linalg.norm((o - w)[:2]))
        if gap >= profile.conflict_m:
            notes.append(
                f"wrist and overhead are {gap * 1000:.0f} mm apart, likely different items; "
                "following the overhead view only"
            )
            return o, float(min(CONFLICT_CONFIDENCE, scores[by_camera["overhead"]])), notes
        near = float(np.linalg.norm(points.mean(axis=0) - tip))
        wrist_weight = float(np.clip(1.0 - near / 0.20, 0.15, 0.85))
        weights = np.zeros(len(estimates))
        weights[by_camera["wrist"]] = wrist_weight
        weights[by_camera["overhead"]] = 1.0 - wrist_weight
        agreement = float(np.clip((profile.conflict_m - gap) / (profile.conflict_m - profile.agree_m), 0.0, 1.0))
        if agreement < 1.0:
            notes.append(f"cameras disagree by {gap * 1000:.0f} mm")
    else:
        weights = np.ones(len(estimates)) / len(estimates)
        agreement = profile.single_view
        if len(estimates) == 1:
            notes.append(f"only {estimates[0].camera} sees it; confidence capped")
    position = weights @ points
    confidence = agreement * float(np.average(scores, weights=weights))
    return position, float(np.clip(confidence, 0.0, 1.0)), notes


def creep_step(
    tip: np.ndarray,
    estimates: list[Estimate],
    *,
    stop_z: float = 0.07,
    tilt_deg: float = 40.0,
    lean_deg: float = 0.0,
    jaw_deg: float = 0.0,
    jaw_azimuth_deg: float | None = None,
    quaternion_wxyz: tuple[float, float, float, float] | None = None,
    current_quaternion_wxyz: tuple[float, float, float, float] | None = None,
    max_step_m: float = FAR_STEP_M,
    profile: Profile = SIM,
) -> dict[str, object]:
    """One bounded move toward hovering at ``stop_z`` over the item, or why not to move.

    ``tip`` is the fingertip position now (``cap_get_robot_state``); ``estimates`` are fresh
    localizations. Orientation, in order of precedence: ``quaternion_wxyz`` holds that orientation;
    else, given ``current_quaternion_wxyz``, it turns up to ``MAX_TURN_DEG`` per step toward the
    :func:`~cap_harness.elrobot.poses.grasp_pose` for the tilt and lean; else it jumps straight
    to that grasp pose.
    """
    tip = np.asarray(tip, dtype=np.float64).reshape(3)
    item, confidence, notes = fuse(estimates, tip, profile)
    result: dict[str, object] = {"confidence": round(confidence, 3), "notes": notes}
    if not estimates:
        return {**result, "action": "hold", "reason": "localize the item first"}

    stop = np.array([item[0], item[1], stop_z])
    delta = stop - tip
    remaining_xy = float(np.linalg.norm(delta[:2]))
    remaining_z = float(delta[2])
    result.update(
        item_xy=[round(float(item[0]), 4), round(float(item[1]), 4)],
        remaining_xy_m=round(remaining_xy, 4),
        remaining_z_m=round(remaining_z, 4),
    )
    wanted = grasp_pose(stop, tilt_deg, jaw_deg, lean_deg, jaw_azimuth_deg)["quaternion_wxyz"]
    turn_left = (
        _angle_deg(current_quaternion_wxyz, wanted)
        if current_quaternion_wxyz is not None and quaternion_wxyz is None
        else 0.0
    )
    result["remaining_turn_deg"] = round(turn_left, 1)
    if remaining_xy <= DONE_XY_M and abs(remaining_z) <= DONE_Z_M and turn_left <= DONE_TURN_DEG:
        return {**result, "action": "done", "reason": "fingertip is at the stop point"}
    if confidence < HOLD_BELOW:
        return {
            **result,
            "action": "hold",
            "reason": "not confident enough to move; wait for a fresh observation and localize again",
        }

    if tip[2] < stop_z - DONE_Z_M:
        direction, distance, phase = np.array([0.0, 0.0, 1.0]), remaining_z, "rise"
    elif remaining_xy > ALIGN_XY_M:
        direction, distance, phase = np.append(delta[:2] / remaining_xy, 0.0), remaining_xy, "align"
    else:
        direction = delta / max(float(np.linalg.norm(delta)), 1e-9)
        distance, phase = float(np.linalg.norm(delta)), "descend"

    cap = FAR_STEP_M if distance > FAR_M else MID_STEP_M if distance > NEAR_M else NEAR_STEP_M
    scale = 1.0 if confidence >= 0.8 else 0.5 if confidence >= 0.5 else 0.25
    step = min(max_step_m, cap * scale, max(MAX_FRACTION * distance, MIN_STEP_M), distance)
    target = tip + direction * step

    if quaternion_wxyz is not None:
        orientation = tuple(float(v) for v in quaternion_wxyz)
    elif current_quaternion_wxyz is not None:
        orientation = _turn_toward(current_quaternion_wxyz, wanted, MAX_TURN_DEG * scale)
    else:
        orientation = tuple(float(v) for v in grasp_pose(target, tilt_deg, jaw_deg, lean_deg, jaw_azimuth_deg)["quaternion_wxyz"])
    pose = {
        "position": [round(float(v), 4) for v in target],
        "frame": "robot_base",
        "quaternion_wxyz": [round(v, 5) for v in orientation],
    }
    seconds = step / NOMINAL_SPEED_M_S
    return {
        **result,
        "action": "move",
        "phase": phase,
        "step_m": round(float(step), 4),
        "target_pose": pose,
        "turn_deg": round(min(turn_left, MAX_TURN_DEG * scale), 1) if current_quaternion_wxyz is not None and quaternion_wxyz is None else 0.0,
        "max_steps": int(min(200, max(60, math.ceil(seconds * CONTROL_HZ) + 50))),
        "then": "wait for the fresh images this move returns, localize again, and call this again",
    }


def _rotation(q: tuple[float, ...]):
    from scipy.spatial.transform import Rotation

    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w])


def _angle_deg(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float(np.rad2deg((_rotation(a).inv() * _rotation(b)).magnitude()))


def _turn_toward(
    current: tuple[float, ...], wanted: tuple[float, ...], max_deg: float
) -> tuple[float, float, float, float]:
    """``current`` rotated toward ``wanted`` by at most ``max_deg`` degrees (wxyz)."""
    from scipy.spatial.transform import Slerp

    total = _angle_deg(current, wanted)
    if total <= max_deg or total < 1e-6:
        return tuple(float(v) for v in wanted)  # type: ignore[return-value]
    x, y, z, w = Slerp([0.0, 1.0], _rotation(current).__class__.concatenate([_rotation(current), _rotation(wanted)]))(
        [max_deg / total]
    ).as_quat()[0]
    return (float(w), float(x), float(y), float(z))


__all__ = ["PROFILES", "REAL", "SIM", "Estimate", "Profile", "creep_step", "fuse"]


def _main(argv: list[str] | None = None) -> int:
    """``python -m cap_harness.elrobot.creep --tip X Y Z --est CAMERA X Y Z [SCORE] ...``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("--tip", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--est", action="append", nargs="+", default=[], metavar="CAMERA X Y Z [SCORE]",
        help="one localization; repeat for each camera that found the item",
    )
    parser.add_argument("--stop-z", type=float, default=0.07, help="hover height of the fingertip")
    parser.add_argument("--tilt", type=float, default=40.0)
    parser.add_argument("--lean", type=float, default=0.0)
    parser.add_argument("--jaw", type=float, default=0.0)
    parser.add_argument("--jaw-azimuth", type=float, default=None)
    parser.add_argument(
        "--quat", type=float, nargs=4, default=None, metavar=("W", "X", "Y", "Z"),
        help="hold this orientation (the fingertip's current one) instead of tilt/lean",
    )
    parser.add_argument(
        "--cur-quat", type=float, nargs=4, default=None, metavar=("W", "X", "Y", "Z"),
        help="the fingertip's current orientation: turn gradually toward tilt/lean from here",
    )
    parser.add_argument(
        "--profile", choices=sorted(PROFILES), default="sim",
        help="'real' judges agreement at the real cameras' measured noise (use on the real robot)",
    )
    parser.add_argument("--max-step", type=float, default=FAR_STEP_M, help="metres")
    args = parser.parse_args(argv)

    estimates = []
    for raw in args.est:
        if len(raw) not in (4, 5):
            parser.error("--est takes CAMERA X Y Z [SCORE]")
        estimates.append(
            Estimate(raw[0], tuple(float(v) for v in raw[1:4]), float(raw[4]) if len(raw) == 5 else 1.0)
        )
    print(
        json.dumps(
            creep_step(
                np.array(args.tip), estimates, stop_z=args.stop_z, tilt_deg=args.tilt,
                lean_deg=args.lean, jaw_deg=args.jaw, jaw_azimuth_deg=args.jaw_azimuth,
                quaternion_wxyz=tuple(args.quat) if args.quat else None,
                current_quaternion_wxyz=tuple(args.cur_quat) if args.cur_quat else None,
                max_step_m=args.max_step, profile=PROFILES[args.profile],
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

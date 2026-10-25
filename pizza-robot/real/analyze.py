"""What the real demonstrations say about the real station, in the sim's terms.

Reads ``episodes.npz`` / ``calibration.json`` / ``frames/`` (see ``export_real.py``) and, using
only the sim's own kinematics, answers four questions:

1. Does recorded ``position_norm`` map linearly onto the sim's joint limits, and in which
   direction? (calibrated spans vs limit spans; every direction choice scored against "the
   fingertip never goes below the table")
2. How does the real arm move? (rest pose, joint speeds, gripper behaviour)
3. Where is the camera? A pinhole camera is fitted to fingertip positions (forward kinematics)
   against where the black gripper appears in ``cam0``.
4. Where are the pizza and trays? Landmarks read off the median ``cam0`` image are unprojected
   through that camera onto the table.

Writes ``layout.json``, which the scene edit (docs/pizza-robot.md) is derived from.

    .venv-system2/bin/python pizza-robot/real/analyze.py
"""

import itertools
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from cap_harness import elrobot  # noqa: F401  (headless GL before mujoco loads)
from cap_harness.elrobot.config import load_config
from cap_harness.elrobot.kinematics import ElRobotKinematics
from cap_harness.elrobot.real import RealJointMap, load_calibration, load_real_episodes

HERE = Path(__file__).resolve().parent
IMG_W, IMG_H, SRC_W, SRC_H = 224, 224, 640, 480  # cam0 frames are 640x480 squashed to 224x224

# Landmarks in the median cam0 image (224x224 px), read off by eye from median_cam0.png
# (the median of the first and last frames of every episode, which removes the arm and hands).
# They are inputs, not measurements: pizza ellipse centre and its horizontal/vertical extent,
# and the three tray centres.
PIZZA_PX = {
    "centre": (112, 130),
    "left": (68, 130),
    "right": (155, 130),
    "top": (112, 73),
    "bottom": (112, 188),
}
TRAYS_PX = {"salami": (170, 100), "olives": (198, 98), "cheese": (185, 152)}
PIZZA_TABLE_Z = 0.01  # top of the pizza base, metres above the table


class Fingertip:
    """Forward kinematics of the fingertip site in the base frame, many rows at a time."""

    def __init__(self, config):
        self.kin = ElRobotKinematics(config)
        self.model, self.data = self.kin.model, self.kin.configuration.data
        self.site = self.model.site("grasp_site").id
        self.base = self.model.body("base_link").id

    def __call__(self, joints):
        out = np.empty((len(joints), 3))
        for i, q in enumerate(joints):
            self.data.qpos[self.kin._qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            out[i] = self.data.site_xpos[self.site] - self.data.xpos[self.base]
        return out


def section(title):
    print(f"\n== {title}")


def check_mapping(config, tip, state):
    section("1. Joint mapping")
    calibration = load_calibration()
    span = (np.array(calibration["range_max"]) - np.array(calibration["range_min"])) * 360 / 4096
    sim = np.rad2deg(
        np.append(
            config.joint_limits_upper - config.joint_limits_lower,
            config.gripper_closed_rad - config.gripper_open_rad,
        )
    )
    for name, real, ours in zip(calibration["joint_order"], span, sim, strict=True):
        print(
            f"  {name:8s} calibrated span {real:6.1f} deg   sim limit span {ours:6.1f} deg   ratio {real / ours:.3f}"
        )
    sample = np.arange(0, len(state), 4)
    lo, hi = config.joint_limits_lower, config.joint_limits_upper
    scored = []
    for signs in itertools.product((1, -1), repeat=7):
        norm = np.where(np.array(signs) > 0, state[sample, :7], 1 - state[sample, :7])
        z = tip(lo + norm * (hi - lo))[:, 2]
        scored.append(((z < -0.01).mean(), signs, float(z.min())))
    scored.sort(key=lambda row: row[0])
    consistent = [row for row in scored if row[0] == 0.0]
    print(
        f"  {len(consistent)} of 128 direction choices keep the fingertip above the table; all-positive is one of them: "
        f"min z {scored[[r[1] for r in scored].index((1,) * 7)][2]:+.4f} m"
    )
    flip = lambda row: "".join("+" if s > 0 else "-" for s in row[1])
    print("  consistent choices (joint1..7):", ", ".join(flip(r) for r in consistent))
    print(f"  worst choice puts the fingertip below the table {scored[-1][0]:.0%} of the time")
    determined = [j for j in range(7) if len({r[1][j] for r in consistent}) == 1]
    print(
        f"  direction fixed by the data for joints {[j + 1 for j in determined]}; "
        f"left open: {[j + 1 for j in range(7) if j not in determined]}"
    )


def motion_statistics(config, episodes):
    section("2. How the real arm moves")
    mapper = RealJointMap.from_config(config)
    firsts = np.array([e.state[0] for e in episodes])
    rest_joints, _ = mapper.to_sim(firsts.mean(axis=0))
    print(
        "  rest pose (mean first frame), sim radians:",
        np.round(rest_joints, 3),
        "std norm",
        np.round(firsts[:, :7].std(axis=0), 3),
    )
    speeds = np.concatenate(
        [np.abs(np.diff(mapper.to_sim(e.state)[0], axis=0)) * e.fps for e in episodes]
    )
    print(
        "  joint speed rad/s  p50:",
        np.round(np.percentile(speeds, 50, axis=0), 3),
        "\n                     p99:",
        np.round(np.percentile(speeds, 99, axis=0), 2),
        "\n                     max:",
        np.round(speeds.max(axis=0), 2),
    )
    steps = np.concatenate([e.action - e.state for e in episodes])
    print(
        "  |goal - position| (norm) median",
        np.round(np.median(np.abs(steps), axis=0), 3),
        " p99",
        np.round(np.percentile(np.abs(steps), 99, axis=0), 3),
    )
    lengths = np.array([len(e.state) / e.fps for e in episodes])
    print(
        f"  episode length {lengths.min():.0f}-{lengths.max():.0f} s, median {np.median(lengths):.0f} s, {np.mean([e.fps for e in episodes]):.1f} fps"
    )
    return {
        "rest_joints_rad": rest_joints.tolist(),
        "joint_speed_p99_rad_s": np.percentile(speeds, 99, axis=0).tolist(),
        "joint_speed_max_rad_s": speeds.max(axis=0).tolist(),
        "episode_seconds_median": float(np.median(lengths)),
    }


def fingertip_pixels(config, tip, episodes, flip_joint1=False):
    """(x, y, z, u, v): fingertip position by FK against the lowest dark-gripper pixel in cam0."""
    mapper = RealJointMap.from_config(config)
    points = []
    for episode in episodes:
        for k in range(1, 7):
            index = int(round(np.linspace(0, 1, 8)[k] * (len(episode.state) - 1)))
            row = episode.state[index].copy()
            if flip_joint1:
                row[0] = 1 - row[0]
            position = tip(mapper.to_sim(row)[0][None])[0]
            if position[2] > 0.14:
                continue
            image = np.asarray(
                Image.open(HERE / "frames" / f"{episode.name}_f{k}_cam0.png").convert("RGB")
            ).astype(float)
            dark = image.max(axis=2) < 60
            roi = np.zeros_like(dark)
            roi[60:205, 38:205] = True  # the white board: excludes the dark room around it
            labels, count = ndimage.label(ndimage.binary_opening(dark & roi))
            if count == 0:
                continue
            sizes = ndimage.sum(dark & roi, labels, range(1, count + 1))
            if sizes.max() < 40:
                continue
            ys, xs = np.nonzero(labels == int(np.argmax(sizes)) + 1)
            lowest = int(
                np.argmax(ys)
            )  # the arm enters from the top, so its tip is the lowest dark pixel
            points.append((*position, xs[lowest], ys[lowest]))
    return np.array(points)


_TOP_DOWN = np.array(
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1.0]]
).T  # columns: image-right, image-down, forward


def project(par, points):
    centre, rotation = par[:3], Rot.from_rotvec(par[3:6]).as_matrix() @ _TOP_DOWN
    camera = (points - centre) @ rotation
    return np.c_[
        SRC_W / 2 + par[6] * camera[:, 0] / camera[:, 2],
        SRC_H / 2 + par[6] * camera[:, 1] / camera[:, 2],
    ]


def unproject(par, pixel, z):
    rotation = Rot.from_rotvec(par[3:6]).as_matrix() @ _TOP_DOWN
    u, v = pixel[0] * SRC_W / IMG_W, pixel[1] * SRC_H / IMG_H
    ray = rotation @ np.array([(u - SRC_W / 2) / par[6], (v - SRC_H / 2) / par[6], 1.0])
    return (par[:3] + (z - par[2]) / ray[2] * ray)[:2]


def fit_camera(points, quiet=False):
    if not quiet:
        section("3. Camera")
    design = np.c_[np.ones(len(points)), points[:, :3]]
    keep = np.ones(len(points), bool)
    for _ in range(4):  # drop detections that are not the gripper: iteratively trimmed affine fit
        cu = np.linalg.lstsq(design[keep], points[keep, 3], rcond=None)[0]
        cv = np.linalg.lstsq(design[keep], points[keep, 4], rcond=None)[0]
        error = np.hypot(design @ cu - points[:, 3], design @ cv - points[:, 4])
        keep = error < max(6, np.percentile(error[keep], 80))
    inliers = points[keep]
    uv = np.c_[inliers[:, 3] * SRC_W / IMG_W, inliers[:, 4] * SRC_H / IMG_H]
    residual = lambda par: (project(par, inliers[:, :3]) - uv).ravel()
    lower = [-0.3, -0.2, 0.25, -0.9, -0.9, -0.9, 200]
    upper = [0.8, 0.9, 1.6, 0.9, 0.9, 0.9, 1500]
    best = None
    for height, focal, tilt in itertools.product((0.35, 0.6, 0.9), (300, 600, 900), (0.0, -0.5)):
        fit = least_squares(
            residual,
            [0.2, 0.3, height, tilt, 0, 0, focal],
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=8,
        )
        best = fit if best is None or fit.cost < best.cost else best
    par = best.x
    error = np.hypot(*(project(par, inliers[:, :3]) - uv).T)
    vfov = np.rad2deg(2 * np.arctan(SRC_H / 2 / par[6]))
    if quiet:
        return float(np.median(error))
    print(
        f"  {keep.sum()} of {len(points)} detections used; camera at {np.round(par[:3], 3)} m, "
        f"tilt {np.round(np.rad2deg(par[3:6]), 1)} deg from straight down, vertical fov {vfov:.0f} deg"
    )
    print(
        f"  reprojection error (640x480 px): median {np.median(error):.0f}, p90 {np.percentile(error, 90):.0f}"
    )
    axes = Rot.from_rotvec(par[3:6]).as_matrix() @ _TOP_DOWN
    return par, {
        "position": par[:3].tolist(),
        "xyaxes_right_up": [*axes[:, 0], *(-axes[:, 1])],
        "forward": axes[:, 2].tolist(),
        "vertical_fov_deg": float(vfov),
        "reprojection_median_px_640": float(np.median(error)),
        "detections_used": int(keep.sum()),
    }


def layout(par):
    section("4. Pizza and trays")
    centre = unproject(par, PIZZA_PX["centre"], PIZZA_TABLE_Z)
    half_width = (
        np.linalg.norm(
            unproject(par, PIZZA_PX["left"], PIZZA_TABLE_Z)
            - unproject(par, PIZZA_PX["right"], PIZZA_TABLE_Z)
        )
        / 2
    )
    half_height = (
        np.linalg.norm(
            unproject(par, PIZZA_PX["top"], PIZZA_TABLE_Z)
            - unproject(par, PIZZA_PX["bottom"], PIZZA_TABLE_Z)
        )
        / 2
    )
    trays = {name: unproject(par, px, 0.02) for name, px in TRAYS_PX.items()}
    print(
        f"  pizza centre {np.round(centre, 3)} m, radius {half_width:.3f} (width) / {half_height:.3f} (height, perspective-stretched)"
    )
    for name, xy in trays.items():
        print(f"  {name:7s} tray centre {np.round(xy, 3)}")
    return {
        "pizza": {"centre_xy": centre.tolist(), "radius_m": float(half_width)},
        "trays_xy": {k: v.tolist() for k, v in trays.items()},
    }


def main():
    config = load_config()
    episodes = load_real_episodes()
    state = np.concatenate([e.state for e in episodes])
    print(f"{len(episodes)} real episodes, {len(state)} steps")
    tip = Fingertip(config)
    check_mapping(config, tip, state)
    motion = motion_statistics(config, episodes)
    par, camera = fit_camera(fingertip_pixels(config, tip, episodes))
    mirrored = fit_camera(fingertip_pixels(config, tip, episodes, flip_joint1=True), quiet=True)
    print(
        f"  joint 1 direction: mirrored hypothesis fits with median error {mirrored:.0f} px vs {camera['reprojection_median_px_640']:.0f} px "
        "(a real camera cannot be a mirror, so the better fit is the physical one)"
    )
    camera["joint1_mirrored_median_px"] = mirrored
    result = {
        "camera": camera,
        **layout(par),
        "motion": motion,
        "provenance": "derived by pizza-robot/real/analyze.py from ApoorvaBajaj/pizza-topping-camera-cleaned; base frame, metres",
    }
    (HERE / "layout.json").write_text(json.dumps(result, indent=2) + "\n")
    print("\nwrote", HERE / "layout.json")
    sys.stdout.flush()
    os._exit(0)  # skip EGL teardown noise


if __name__ == "__main__":
    main()

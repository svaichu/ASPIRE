r"""Fit the real wrist camera's mount, field of view and aspect from the demonstrations.

The scene model's ``wrist`` camera is the design placement; nothing says the real one is mounted
the same way, and a wrong mount puts every wrist-view estimate in the wrong place. What the real
data does give is a known landmark seen from many arm poses: the pizza, a disc of known centre
and radius (``layout.json``) whose red sauce is easy to mask. With the arm's pose at each frame
from forward kinematics, the right camera is the one that projects that disc onto the sauce.

Parameters (relative to the design mount, in the camera's own frame): a rotation vector (3),
a translation (3), a vertical field of view and a horizontal/vertical focal ratio (the frames
are squashed from 640x480 to 224x224, so the horizontal scale may differ). Fitted with
Nelder-Mead on 1 - mean IoU over the training episodes and scored on the rest.

Outcome on the 22 salami/cheese episodes: the fit does not beat the design mount (held-out IoU
0.383 -> 0.385), and ``--sensitivity`` shows why that is good news rather than a failed fit: the
score falls on both sides of the design mount for every parameter (rotations of 0.1-0.2 rad,
shifts of 3-6 cm, field of view +-10-20 deg), so the data supports the assumed mount to about
that precision. Nothing loads a fitted file, so ``--out`` is only written when asked for.

    .venv-system2/bin/python pizza-robot/real/fit_wrist.py DENSE_DIR [--sensitivity] [--out FILE]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from cap_harness import elrobot  # noqa: F401
from cap_harness.elrobot.config import load_config
from cap_harness.elrobot.real import REAL_DIR, RealJointMap
from cap_harness.elrobot.remote import RealCameraModel, WristCameraModel
from cap_harness.elrobot.segmentation import _hsv

SIZE = 112  # fit at half resolution
#: The sauce is the inner part of the pizza; the crust ring is pale and not masked.
SAUCE_RADIUS_FRACTION = 0.8


def sauce_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = _hsv(rgb)
    hue = (hsv[..., 0] + 180.0) % 360.0 - 180.0
    return (np.abs(hue) < 22.0) & (hsv[..., 1] > 0.33) & (hsv[..., 2] > 0.18)


def predicted_disc(pose: np.ndarray, fov_deg: float, aspect: float, centre, radius) -> np.ndarray:
    """Pixels whose ray meets the table inside the pizza's sauce disc."""
    f = (SIZE / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    u, v = np.meshgrid(np.arange(SIZE) + 0.5, np.arange(SIZE) + 0.5)
    rays = np.stack([(u - SIZE / 2) / (f * aspect), (v - SIZE / 2) / f, np.ones_like(u)], -1) @ pose[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (0.01 - pose[2, 3]) / rays[..., 2]
    hit = (rays[..., 2] < -1e-6) & (t > 0)
    point = pose[:3, 3] + rays * np.where(hit, t, 0.0)[..., None]
    return hit & (np.hypot(point[..., 0] - centre[0], point[..., 1] - centre[1]) < radius * SAUCE_RADIUS_FRACTION)


def perturb(pose: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``pose`` composed with a mount correction expressed in the camera's own frame."""
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec(params[:3]).as_matrix()
    delta[:3, 3] = params[3:6]
    return pose @ delta


def load_frames(dense: Path, stride: int, rig) -> dict[str, list]:
    frames: dict[str, list] = {}
    for directory in sorted(glob.glob(str(dense / "*"))):
        name = os.path.basename(directory)
        series = np.load(directory + "/series.npz")
        for t in series["frame_index"][::stride]:
            joints, _ = rig["map"].to_sim(series["state"][int(t)])
            image = Image.open(f"{directory}/f{int(t):04d}_cam1.jpg").convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            mask = sauce_mask(np.asarray(image))
            frames.setdefault(name, []).append((rig["wrist"].pose(joints), mask))
    return frames


def mean_iou(params: np.ndarray, frames: list, centre, radius) -> float:
    ious = []
    for pose, observed in frames:
        predicted = predicted_disc(perturb(pose, params), params[6], params[7], centre, radius)
        union = int((predicted | observed).sum())
        if union > 0.02 * SIZE * SIZE:  # skip frames that see neither
            ious.append(float((predicted & observed).sum()) / union)
    return float(np.mean(ious)) if ious else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dense_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="write the fitted mount here (nothing loads it)")
    parser.add_argument("--sensitivity", action="store_true", help="print IoU for deliberately wrong mounts and stop")
    parser.add_argument("--stride", type=int, default=6)
    args = parser.parse_args()

    config = load_config("pizza")
    overhead = RealCameraModel.from_layout(config)
    rig = {"map": RealJointMap.from_config(config), "wrist": WristCameraModel(config, overhead.table_z)}
    layout = json.loads((REAL_DIR / "layout.json").read_text())["pizza"]
    centre, radius = layout["centre_xy"], layout["radius_m"]
    frames = load_frames(args.dense_dir, args.stride, rig)
    names = sorted(frames)
    train = [f for i, n in enumerate(names) if i % 2 == 0 for f in frames[n]]
    test = [f for i, n in enumerate(names) if i % 2 == 1 for f in frames[n]]
    print(f"{len(names)} episodes: {len(train)} training frames, {len(test)} held-out frames")

    start = np.array([0, 0, 0, 0, 0, 0, rig["wrist"].vertical_fov_deg, 1.0])
    if args.sensitivity:
        everything = train + test
        print(f"design mount IoU {mean_iou(start, everything, centre, radius):.3f} over {len(everything)} frames")
        for label, index, deltas in (
            ("rotate about camera x (rad)", 0, (-0.2, -0.1, 0.1, 0.2)), ("rotate about camera y (rad)", 1, (-0.2, -0.1, 0.1, 0.2)),
            ("rotate about camera z (rad)", 2, (-0.2, -0.1, 0.1, 0.2)), ("translate camera x (m)", 3, (-0.06, -0.03, 0.03, 0.06)),
            ("translate camera y (m)", 4, (-0.06, -0.03, 0.03, 0.06)), ("translate camera z (m)", 5, (-0.06, -0.03, 0.03, 0.06)),
            ("vertical fov (deg)", 6, (-20, -10, 10, 20)), ("aspect fx/fy", 7, (-0.3, -0.15, 0.15, 0.3)),
        ):
            scores = []
            for delta in deltas:
                shifted = start.copy()
                shifted[index] += delta
                scores.append(f"{delta:+g}: {mean_iou(shifted, everything, centre, radius):.3f}")
            print(f"  {label:30s}" + "   ".join(scores))
        return 0
    print(f"design mount     train IoU {mean_iou(start, train, centre, radius):.3f}   held-out IoU {mean_iou(start, test, centre, radius):.3f}")

    scale = np.array([0.15, 0.15, 0.15, 0.02, 0.02, 0.02, 8.0, 0.1])
    best, best_loss = start, 1.0 - mean_iou(start, train, centre, radius)
    rng = np.random.default_rng(0)
    for restart in range(6):
        x0 = start + (rng.normal(size=8) * scale * (0.0 if restart == 0 else 0.7))
        result = minimize(
            lambda z: 1.0 - mean_iou(start + z * scale, train, centre, radius) + 0.002 * float(np.sum(z**2)),
            (x0 - start) / scale, method="Nelder-Mead", options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-4},
        )
        params = start + result.x * scale
        loss = 1.0 - mean_iou(params, train, centre, radius)
        print(f"  restart {restart}: train IoU {1 - loss:.3f}")
        if loss < best_loss:
            best, best_loss = params, loss
    print(f"fitted mount     train IoU {1 - best_loss:.3f}   held-out IoU {mean_iou(best, test, centre, radius):.3f}")
    print("rotvec", np.round(best[:3], 3), "translation m", np.round(best[3:6], 3), "fov", round(best[6], 1), "aspect", round(best[7], 3))
    if args.out is None:
        return 0
    args.out.write_text(json.dumps({
        "rotvec_rad": best[:3].tolist(), "translation_m": best[3:6].tolist(),
        "vertical_fov_deg": float(best[6]), "aspect_fx_over_fy": float(best[7]),
        "train_iou": 1 - best_loss, "heldout_iou": mean_iou(best, test, centre, radius),
        "design_heldout_iou": mean_iou(start, test, centre, radius),
        "note": "fitted by pizza-robot/real/fit_wrist.py from the pizza disc seen in the wrist frames",
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

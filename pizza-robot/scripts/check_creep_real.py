r"""Replay the real demonstrations through the creep pipeline and compare with what the operator did.

No robot and no simulator: the real joint series give the fingertip (forward kinematics), the real
frames give what the cameras saw, and the operator's own next moves are the reference. For each
clean pick (gripper opens, then closes near the table) and each sampled frame of its approach:

* **perception** -- where the segmenter puts the topping (overhead and wrist), against where the
  operator then closed the jaws, in millimetres;
* **step** -- the move ``creep_step`` would command from the real fingertip, against the
  fingertip's actual displacement over the next ``--lookahead`` frames: direction agreement
  (cosine), and whether it points to where the operator ended up;
* **joint-space prediction** -- that step solved with IK from the real joints, against the
  operator's recorded joint goal ``--lookahead`` frames later, versus "stay where you are".

Frames come from ``pizza-robot/real/export_dense.py`` (run it first):

    .venv-system2/bin/python pizza-robot/scripts/check_creep_real.py DENSE_DIR [--lookahead 5]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from cap_harness import elrobot  # noqa: F401  (selects headless GL before mujoco loads)
from cap_harness.contracts import CameraObservation, Pose
from cap_harness.elrobot.config import load_config
from cap_harness.elrobot.creep import PROFILES, Estimate, creep_step
from cap_harness.elrobot.kinematics import ElRobotKinematics
from cap_harness.elrobot.real import REAL_DIR, RealJointMap, is_clean_pick, pick_event
from cap_harness.elrobot.real_segmentation import RealSceneSegmentationProvider, plane_points
from cap_harness.elrobot.remote import RealCameraModel, WristCameraModel
from cap_harness.elrobot.segmentation import ColorSegmentationProvider
from cap_harness.yam_real.codec import rotation_to_wxyz

STOP_ABOVE_PICK_M = 0.04


class Rig:
    def __init__(self) -> None:
        self.config = load_config("pizza")
        self.joint_map = RealJointMap.from_config(self.config)
        self.kin = ElRobotKinematics(self.config)
        self.overhead = RealCameraModel.from_layout(self.config)
        self.wrist = WristCameraModel(self.config, self.overhead.table_z)
        self.sim_seg = ColorSegmentationProvider()
        self.real_seg = RealSceneSegmentationProvider()
        self._landmarks = {n: tuple(xy) for n, xy in json.loads((REAL_DIR / "layout.json").read_text())["trays_xy"].items()}

    def provider_without(self, episode: str, kind: str) -> RealSceneSegmentationProvider:
        """The real segmenter with ``kind``'s tray centred on the *other* episodes' grasps.

        ``pick_spots.json`` is derived from every demonstration, so scoring a frame against its own
        episode's grasp with it would be leaky; leaving that episode out keeps the check honest.
        """
        spots = json.loads((REAL_DIR / "pick_spots.json").read_text())["spots"].get(kind)
        others = [np.array(p[:2]) for n, p in (spots or {}).get("episodes", {}).items() if n != episode]
        if len(others) < 5:
            return RealSceneSegmentationProvider({**self.real_seg.trays, **{k: v for k, v in self._landmarks.items() if k == kind}})
        return RealSceneSegmentationProvider({**self._landmarks, kind: tuple(np.median(others, axis=0))})

    def observation(self, rgb: np.ndarray, camera: str, joints: np.ndarray) -> CameraObservation:
        model = self.overhead if camera == "overhead" else self.wrist
        pose = model.pose(joints)
        return CameraObservation(
            rgb=rgb,
            depth_m=np.zeros(rgb.shape[:2]),  # unused by the segmenters
            intrinsics=model.intrinsics(rgb.shape[1], rgb.shape[0]),
            frame=f"{camera}_optical",
            camera_pose=Pose(position=pose[:3, 3], quaternion_wxyz=rotation_to_wxyz(pose[:3, :3]), frame="robot_base"),
        )

    def locate(self, provider, observation, query: str) -> tuple[np.ndarray, float, float] | None:
        """``(xy on the table, score, fraction of the mask that is inside the image's tray area)``."""
        result = provider.segment_text(observation, query)
        if not result.ok:
            return None
        segmentation = result.segmentations[0]
        points = plane_points(observation)[segmentation.mask]
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return None
        return points.mean(axis=0)[:2], float(segmentation.score), float(segmentation.mask.sum())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dense_dir", type=Path)
    parser.add_argument("--lookahead", type=int, default=5, help="frames (10 fps) of operator motion to compare with")
    parser.add_argument("--stride", type=int, default=4, help="frames between samples")
    parser.add_argument("--window", type=int, default=60, help="frames before the jaws open to start from")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="real")
    parser.add_argument("--out", type=Path, default=None, help="write the raw per-frame rows as JSON")
    args = parser.parse_args()

    rig = Rig()
    rows, picks = [], []
    for directory in sorted(glob.glob(str(args.dense_dir / "*"))):
        name = os.path.basename(directory)
        series = np.load(directory + "/series.npz")
        state, action = series["state"], series["action"]
        event = pick_event(action[:, 7])
        if event is None:
            continue
        joints, _ = rig.joint_map.to_sim(state)
        tips = np.array([rig.kin.forward_kinematics(q)[0] for q in joints])
        opened, closed = event
        pick = tips[closed]
        if not is_clean_pick(tips[opened], pick):
            continue
        kind = "salami" if "salami" in name else "cheese"
        picks.append((name, kind, pick))
        provider = rig.provider_without(name, kind)
        stop = np.array([pick[0], pick[1], pick[2] + STOP_ABOVE_PICK_M])
        a_joints, _ = rig.joint_map.to_sim(action)
        frame_index = set(series["frame_index"].tolist())

        for t in range(max(0, opened - args.window), closed, args.stride):
            if t not in frame_index or t + args.lookahead >= len(state):
                continue
            tip = tips[t]
            quat = tuple(float(v) for v in rig.kin.forward_kinematics(joints[t])[1])
            toward = pick - tip
            operator = tips[t + args.lookahead] - tip
            row = {"episode": name, "kind": kind, "t": t, "to_pick_cm": float(np.linalg.norm(toward) * 100),
                   "operator_cm": float(np.linalg.norm(operator) * 100)}
            later = rig.kin.forward_kinematics(joints[t + args.lookahead])[1]
            row["operator_turn_deg"] = float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(quat, later)))))))

            images = {c: np.asarray(Image.open(f"{directory}/f{t:04d}_cam{i}.jpg").convert("RGB")) for i, c in enumerate(("overhead", "wrist"))}
            estimates = {}
            for camera in ("overhead", "wrist"):
                obs = rig.observation(images[camera], camera, joints[t])
                for label, seg in (("sim", rig.sim_seg), ("real", provider)):
                    found = rig.locate(seg, obs, kind)
                    row[f"{label}_{camera}_found"] = found is not None
                    if found is not None:
                        xy, score, area = found
                        row[f"{label}_{camera}_err_mm"] = float(np.linalg.norm(xy - pick[:2]) * 1000)
                        row[f"{label}_{camera}_area_px"] = area
                        if label == "real":
                            estimates[camera] = Estimate(camera, (xy[0], xy[1], 0.02), score)
                            row[f"real_{camera}_xy"] = [float(xy[0]), float(xy[1])]

            variants = {
                "oracle": [Estimate("overhead", (pick[0], pick[1], 0.02), 1.0), Estimate("wrist", (pick[0], pick[1], 0.02), 1.0)],
                "overhead": [estimates["overhead"]] if "overhead" in estimates else [],
                "both": list(estimates.values()),
            }
            for label, ests in variants.items():
                out = creep_step(tip, ests, stop_z=float(stop[2]), quaternion_wxyz=quat, profile=PROFILES[args.profile])
                if out.get("item_xy"):
                    row[f"{label}_item_err_mm"] = float(np.linalg.norm(np.array(out["item_xy"]) - pick[:2]) * 1000)
                row[f"{label}_action"] = out["action"]
                if out["action"] != "move":
                    continue
                step = np.array(out["target_pose"]["position"]) - tip
                row[f"{label}_step_cm"] = float(np.linalg.norm(step) * 100)
                row[f"{label}_cos_operator"] = cosine(step, operator)
                row[f"{label}_cos_pick"] = cosine(step, toward)
                row[f"{label}_cosxy_pick"] = cosine(step[:2], toward[:2])
                row[f"{label}_cosxy_operator"] = cosine(step[:2], operator[:2])
                # How much closer to the pick (cm, horizontally) each would leave the fingertip.
                row[f"{label}_gain_cm"] = float((np.linalg.norm(toward[:2]) - np.linalg.norm(toward[:2] - step[:2])) * 100)
                row["operator_gain_cm"] = float((np.linalg.norm(toward[:2]) - np.linalg.norm(toward[:2] - operator[:2])) * 100)
                solved, metres, _ = rig.kin.inverse_kinematics(
                    np.array(out["target_pose"]["position"]), np.array(out["target_pose"]["quaternion_wxyz"]), joints[t]
                )
                if metres < 0.01:
                    goal = a_joints[t + args.lookahead]
                    row[f"{label}_joint_err"] = float(np.mean(np.abs(solved - goal)))
                    row[f"{label}_joint_cos"] = cosine(solved - joints[t], goal - joints[t])
                    row[f"{label}_joint_move"] = float(np.linalg.norm(solved - joints[t]))
                    row["operator_joint_move"] = float(np.linalg.norm(goal - joints[t]))
                    row["stay_joint_err"] = float(np.mean(np.abs(joints[t] - goal)))
            rows.append(row)

    if args.out:
        args.out.write_text(json.dumps(rows, default=float))
    report(rows, picks, args.lookahead)
    return 0


def _vals(rows, key):
    return np.array([r[key] for r in rows if key in r and r[key] == r[key]], dtype=np.float64)


def report(rows, picks, lookahead) -> None:
    print(f"\n{len(picks)} clean picks: " + ", ".join(f"{n}" for n, _, _ in picks))
    print(f"{len(rows)} sampled frames (approach window, lookahead {lookahead} frames = {lookahead / 10:.1f} s)\n")

    print("PERCEPTION  (xy error of the topping estimate vs where the operator closed the jaws)")
    print(f"  {'provider/camera':22s} {'kind':7s} {'found':>6s} {'median mm':>10s} {'p90 mm':>8s}")
    for label in ("sim", "real"):
        for camera in ("overhead", "wrist"):
            for kind in ("salami", "cheese"):
                sub = [r for r in rows if r["kind"] == kind]
                if not sub:
                    continue
                found = np.mean([r[f"{label}_{camera}_found"] for r in sub])
                err = _vals(sub, f"{label}_{camera}_err_mm")
                med = f"{np.median(err):10.0f}" if len(err) else f"{'-':>10s}"
                p90 = f"{np.percentile(err, 90):8.0f}" if len(err) else f"{'-':>8s}"
                print(f"  {label + '/' + camera:22s} {kind:7s} {found:6.0%} {med} {p90}")

    print("\nSTEP vs OPERATOR  (does our next move point where the operator went?)")
    print(f"  {'estimates':10s} {'move':>6s} {'hold':>6s} {'done':>6s} {'step cm':>8s} {'cosxy(pick)':>12s} {'cosxy(op)':>10s} {'gain cm':>8s} {'item err mm':>12s}")
    op = _vals(rows, "operator_cm")
    for label in ("oracle", "overhead", "both"):
        acts = [r.get(f"{label}_action") for r in rows]
        n = max(len(acts), 1)
        step = _vals(rows, f"{label}_step_cm")
        nan = float("nan")
        print(
            f"  {label:10s} {acts.count('move') / n:6.0%} {acts.count('hold') / n:6.0%} {acts.count('done') / n:6.0%} "
            f"{np.median(step) if len(step) else nan:8.1f} "
            f"{np.median(_vals(rows, f'{label}_cosxy_pick')) if len(step) else nan:12.2f} "
            f"{np.median(_vals(rows, f'{label}_cosxy_operator')) if len(step) else nan:10.2f} "
            f"{np.median(_vals(rows, f'{label}_gain_cm')) if len(step) else nan:8.2f} "
            f"{np.median(_vals(rows, f'{label}_item_err_mm')) if len(_vals(rows, f'{label}_item_err_mm')) else nan:12.0f}"
        )
    print(f"  operator over the same {lookahead / 10:.1f} s: moves median {np.median(op):.1f} cm (p90 {np.percentile(op, 90):.1f}), "
          f"horizontal gain toward the pick median {np.median(_vals(rows, 'operator_gain_cm')):.2f} cm; "
          f"distance left: median {np.median(_vals(rows, 'to_pick_cm')):.1f} cm")

    turn = _vals(rows, "operator_turn_deg")
    print(f"  operator also turns the wrist a median {np.median(turn):.1f} deg (p90 {np.percentile(turn, 90):.1f}) in that time; creep holds the orientation.")
    print("\nJOINT-SPACE PREDICTION  (mean |joint error| in rad against the operator's goal 0.5 s later)")
    stay = _vals(rows, "stay_joint_err")
    print(f"  stay still      median {np.median(stay):.3f}   (operator's own joint move: median {np.median(_vals(rows, 'operator_joint_move')):.3f} rad-norm)")
    for label in ("oracle", "overhead", "both"):
        err = _vals(rows, f"{label}_joint_err")
        if len(err):
            skill = 1.0 - np.median(err) / np.median(stay)
            print(
                f"  {label:10s}      median {np.median(err):.3f}   skill vs staying still {skill:+.0%}   "
                f"joint-delta cos {np.median(_vals(rows, f'{label}_joint_cos')):+.2f}   our move {np.median(_vals(rows, f'{label}_joint_move')):.3f} rad-norm   (n={len(err)})"
            )


if __name__ == "__main__":
    raise SystemExit(main())

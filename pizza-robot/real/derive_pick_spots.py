r"""Where operators actually grasp each topping, from the real demonstrations.

The layout's tray positions were read by eye off the overhead image, and the cheese landmark sits
~47 mm from where every operator grabs it. This finds each demonstration's grasp (the first close
of the jaws after they open, near the table), takes the fingertip there by forward kinematics, and
writes the per-topping median to ``pick_spots.json``. ``RealSceneSegmentationProvider`` centres
its tray disc on a spot when it rests on at least 5 picks (cheese: 8 picks, ±16 mm; salami: 4 picks
spread over ±47 mm, too few and too varied to use, so it keeps the layout landmark).

    .venv-system2/bin/python pizza-robot/real/derive_pick_spots.py DENSE_DIR
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from cap_harness import elrobot  # noqa: F401
from cap_harness.elrobot.config import load_config
from cap_harness.elrobot.kinematics import ElRobotKinematics
from cap_harness.elrobot.real import REAL_DIR, RealJointMap, is_clean_pick, pick_event


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dense_dir", type=Path)
    parser.add_argument("--out", type=Path, default=REAL_DIR / "pick_spots.json")
    args = parser.parse_args()

    config = load_config("pizza")
    joint_map, kin = RealJointMap.from_config(config), ElRobotKinematics(config)
    picks: dict[str, dict[str, list[float]]] = {}
    for directory in sorted(glob.glob(str(args.dense_dir / "*"))):
        name = os.path.basename(directory)
        series = np.load(directory + "/series.npz")
        event = pick_event(series["action"][:, 7])
        if event is None:
            continue
        joints, _ = joint_map.to_sim(series["state"])
        opened, closed = event
        tip_open, tip_close = (kin.forward_kinematics(joints[i])[0] for i in event)
        if is_clean_pick(tip_open, tip_close):
            kind = "salami" if "salami" in name else "cheese" if "cheese" in name else "olives"
            picks.setdefault(kind, {})[name] = tip_close.tolist()

    spots = {}
    for kind, episodes in picks.items():
        points = np.array(list(episodes.values()))
        spots[kind] = {
            "xy": np.median(points[:, :2], axis=0).round(4).tolist(),
            "z": round(float(np.median(points[:, 2])), 4),
            "n": len(points),
            "std_xy_mm": (points[:, :2].std(axis=0) * 1000).round(0).tolist(),
            "episodes": {name: np.round(p, 4).tolist() for name, p in episodes.items()},
        }
        print(f"{kind}: n={len(points)} xy {spots[kind]['xy']} std {spots[kind]['std_xy_mm']} mm")
    args.out.write_text(json.dumps({
        "spots": spots,
        "provenance": "fingertip (forward kinematics) at the first jaw close after opening, near the table; "
                      "derived by pizza-robot/real/derive_pick_spots.py from ApoorvaBajaj/pizza-topping-camera-cleaned",
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

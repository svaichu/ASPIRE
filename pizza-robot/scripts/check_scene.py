"""Milestone-1 check: offscreen cameras render at 1280x960 and the home pose settles.

    MUJOCO_GL=egl .venv-system2/bin/python pizza-robot/scripts/check_scene.py [out_dir]
"""
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

SCENE = Path(__file__).resolve().parents[1] / "scene"
WIDTH, HEIGHT = 1280, 960


def main(out_dir: Path | None) -> int:
    home = json.loads((SCENE / "home.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(SCENE / "elrobot_table.xml"))
    data = mujoco.MjData(model)
    names, target = home["joint_names"], np.array(home["home_joints"])
    mujoco.mj_forward(model, data)
    data.ctrl[: len(target)] = target  # actuators are declared in joint order
    for _ in range(4000):
        mujoco.mj_step(model, data)
    qpos = np.array([data.qpos[model.joint(n).qposadr].item() for n in names])  # by name, never index
    err = np.abs(qpos - target).max()
    ok = err <= home["settle_tolerance_rad"]
    print(f"home settle max error {err:.3f} rad ({'ok' if ok else 'FAIL'})")

    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    for cam in ("overhead", "wrist"):
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        ok &= frame.shape == (HEIGHT, WIDTH, 3) and frame.std() > 1.0
        print(f"{cam}: {frame.shape}")
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(out_dir / f"{cam}.png")
    sys.stdout.flush()
    os._exit(0 if ok else 1)  # skip Renderer teardown, which raises EGLError on this driver


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)

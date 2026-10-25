"""Station sanity check: home pose, both cameras at their configured size, and the layout.

    .venv-system2/bin/python pizza-robot/scripts/check_scene.py [out_dir]

Builds the same plant the MCP server uses (gravity-compensated arm, stiff contacts)
rather than loading the XML raw, so what it reports is what an agent will get.
"""

import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from cap_harness import elrobot  # noqa: F401  (headless GL before mujoco loads)
from cap_harness.elrobot.station import build_sim_station


def main(out_dir: Path | None) -> int:
    env = build_sim_station()
    ok = True
    error = float(np.max(np.abs(env.joint_positions() - env.config.home_joints)))
    ok &= error < 0.01
    print(f"home error {error:.4f} rad, gripper {env.gripper_position():.2f} (1 = open)")
    model, data = env.model, env.data
    base = data.xpos[model.body("base_link").id]
    for prefix, expected in (("bin_salami_disc", 9), ("bin_cheese_shred", 9)):
        names = [
            model.body(i).name for i in range(model.nbody) if model.body(i).name.startswith(prefix)
        ]
        heights = np.array([data.xpos[model.body(n).id][2] - base[2] for n in names])
        settled = len(names) == expected and bool(np.all(heights > 0.005))
        ok &= settled
        print(
            f"{prefix}: {len(names)} items, z {heights.min():.3f}..{heights.max():.3f}, {'ok' if settled else 'FAIL'}"
        )
    for camera in env.camera_aliases:
        frame = env.camera_frame(camera)
        ok &= frame is not None and frame.rgb.std() > 1.0
        print(f"{camera}: rgb {frame.rgb.shape}, depth>0 {(frame.depth_m > 0).mean():.0%}")
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame.rgb).save(out_dir / f"{camera}.png")
    sys.stdout.flush()
    os._exit(0 if ok else 1)  # skip EGL teardown, which raises on some drivers


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)

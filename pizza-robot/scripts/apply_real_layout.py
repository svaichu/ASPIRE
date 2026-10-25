"""Move the pizza, the two topping containers, the camera stand and the home pose to match the real station.

Reads ``pizza-robot/real/layout.json`` (derived from the real demonstrations by
``pizza-robot/real/analyze.py``) and rewrites the matching parts of
``pizza-robot/scene/elrobot_table.xml`` and ``home.json``. Idempotent. It adds nothing: the
toppings and the containers' sizes are untouched; only positions, the pizza's size and the
camera's pose change.

    .venv-system2/bin/python pizza-robot/scripts/apply_real_layout.py

Frames: the scene is authored in MuJoCo world coordinates; the robot base sits at
``BASE_WORLD``, so base-frame coordinates (what the analysis and the agent use) are world minus that.
"""

import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scene" / "elrobot_table.xml"
HOME = ROOT / "scene" / "home.json"
LAYOUT = json.loads((ROOT / "real" / "layout.json").read_text())

BASE_WORLD = np.array([0.0, -0.36, 0.75])
# Container centres, base frame. The real trays are small (about 5 x 9 cm) and sit in a column next
# to the pizza on the side away from it; these containers must be 18-20 cm wide so the open jaws
# clear the walls, so they cannot sit where the real trays do. They keep the real arrangement --
# pizza on one side, toppings on the other, nothing in front of the robot -- and are placed to
# stay inside the arm's reach and clear of the pizza.
SALAMI_BIN_XY = (-0.04, 0.18)
CHEESE_TRAY_XY = (-0.255, 0.14)
# Chosen by search (see docs/pizza-robot.md): of 56 x 110 candidate centre pairs, 22 keep every one of
# the 18 items reachable at the standard 40 degree approach, leave >= 8 mm between each container and
# the pizza and each other, and stay clear of the robot base; this is the one nearest the real trays.
# It assumes the pizza centre below, 1.2 cm +x of the fitted 0.248 m (inside the fit's error).
PIZZA_CENTRE_NUDGE_X = 0.012
# Offsets of the items inside their containers (unchanged from the validated layout).
SALAMI_OFFSETS = [(dx, dy) for dx in (-0.06, 0.0, 0.06) for dy in (-0.06, 0.0, 0.06)]
CHEESE_OFFSETS = [(dx, dy) for dx in (-0.045, 0.0, 0.045) for dy in (-0.04, 0.0, 0.04)]
# The fitted camera frames the pizza dead centre with the real trays at its right edge. These
# containers sit further out, so the camera is pulled back along its own optical axis (same
# direction, same point looked at) until the whole layout is in view.
CAMERA_ZOOM_OUT = 1.45
CAMERA_BOOM_M = 0.12  # the stand's pole stands this far behind the camera, away from the robot


def world(xy):
    return np.asarray(xy) + BASE_WORLD[:2]


def set_body_pos(xml, name, pos):
    pattern = re.compile(rf'(<body name="{re.escape(name)}" pos=")[^"]*(")')
    assert pattern.search(xml), name
    return pattern.sub(
        lambda m: m.group(1) + " ".join(f"{v:.4f}" for v in pos) + m.group(2), xml, count=1
    )


def main():
    xml = SCENE.read_text()

    pizza = LAYOUT["pizza"]
    centre = world(pizza["centre_xy"]) + (PIZZA_CENTRE_NUDGE_X, 0.0)
    xml = set_body_pos(xml, "pizza_plate", (centre[0], centre[1], BASE_WORLD[2]))
    xml = re.sub(
        r'(<body name="pizza_plate"[^>]*>\s*<geom type="cylinder" size=")[^"]*(")',
        lambda m: f"{m.group(1)}{pizza['radius_m']:.3f} 0.005{m.group(2)}",
        xml,
        count=1,
    )

    for body, xy in (("bin_salami", SALAMI_BIN_XY), ("bin_cheese", CHEESE_TRAY_XY)):
        c = world(xy)
        xml = set_body_pos(xml, body, (c[0], c[1], BASE_WORLD[2]))
    for index, (dx, dy) in enumerate(SALAMI_OFFSETS):
        c = world(SALAMI_BIN_XY) + (dx, dy)
        xml = set_body_pos(xml, f"bin_salami_disc{index}", (c[0], c[1], 0.7750))
    for index, (dx, dy) in enumerate(CHEESE_OFFSETS):
        c = world(CHEESE_TRAY_XY) + (dx, dy)
        xml = set_body_pos(xml, f"bin_cheese_shred{index}", (c[0], c[1], 0.7800))

    camera = LAYOUT["camera"]
    fitted = np.asarray(camera["position"])
    forward = np.asarray(camera["forward"])
    looked_at = fitted + forward * (
        -fitted[2] / forward[2]
    )  # where the optical axis meets the table
    cam_world = looked_at + CAMERA_ZOOM_OUT * (fitted - looked_at) + BASE_WORLD
    pole = np.array([cam_world[0], cam_world[1] + CAMERA_BOOM_M])
    height = cam_world[2] - BASE_WORLD[2]  # camera height above the table
    stand = f"""<body name="camera_stand" pos="{pole[0]:.4f} {pole[1]:.4f} {BASE_WORLD[2]:.4f}">
      <geom type="cylinder" size="0.015 {(height + 0.03) / 2:.4f}" pos="0 0 {(height + 0.03) / 2:.4f}" rgba="0.2 0.2 0.2 1" />
      <geom type="box" pos="0 {-CAMERA_BOOM_M / 2:.4f} {height + 0.015:.4f}" size="0.012 {CAMERA_BOOM_M / 2:.4f} 0.012" rgba="0.2 0.2 0.2 1" />
      <geom type="box" pos="0 {-CAMERA_BOOM_M:.4f} {height:.4f}" size="0.025 0.02 0.015" rgba="0.1 0.1 0.1 1" />
      <camera name="overhead" pos="0 {-CAMERA_BOOM_M:.4f} {height - 0.02:.4f}" xyaxes="{" ".join(f"{v:.5f}" for v in camera["xyaxes_right_up"])}" fovy="{camera["vertical_fov_deg"]:.1f}" />
    </body>"""
    xml, count = re.subn(
        r'<body name="camera_stand".*?</body>', stand, xml, count=1, flags=re.DOTALL
    )
    assert count == 1
    SCENE.write_text(xml)

    home = json.loads(HOME.read_text())
    rest = LAYOUT["motion"]["rest_joints_rad"]
    home["home_joints"] = [round(v, 4) for v in rest] + [home["gripper_open_rad"]]
    # Per-joint speed limits: the real demonstrations' 99th-percentile joint speeds, floored at 0.8
    # rad/s because the wrist rolls barely move in the demos and would otherwise crawl.
    p99 = np.asarray(LAYOUT["motion"]["joint_speed_p99_rad_s"])
    home["joint_speed_limits_rad_s"] = [round(float(v), 2) for v in np.maximum(p99, 0.8)]
    HOME.write_text(json.dumps(home, indent=2) + "\n")
    print(
        f"pizza centre {np.round(np.array(pizza['centre_xy']) + (PIZZA_CENTRE_NUDGE_X, 0.0), 3)} r={pizza['radius_m']:.3f}; salami bin {SALAMI_BIN_XY}; cheese tray {CHEESE_TRAY_XY}"
    )
    print(
        f"camera at {np.round(cam_world - BASE_WORLD, 3)} (base frame), looking at {np.round(looked_at, 3)}, vfov {camera['vertical_fov_deg']:.0f} deg; home {home['home_joints']}"
    )


if __name__ == "__main__":
    main()

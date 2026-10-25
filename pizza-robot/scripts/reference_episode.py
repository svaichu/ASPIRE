"""A scripted reference for the pizza task, written against the agent's own tool surface.

It is the recipe in ``.agents/skills/manipulation-tasks/pizza-assembly.md`` as running code:
localize with ``localize_object``, plan a clearance-checked approach with
``cap_harness.elrobot.approach``, move with ``move_to_pose``, grip with ``set_gripper``, and
judge every step from the gripper reading and the overhead camera -- never from simulator
state. Scene truth appears only in the final score, which an agent never gets.

    .venv-system2/bin/python pizza-robot/scripts/reference_episode.py [--salami N] [--cheese N]
"""

import argparse
import os
import sys

import numpy as np

from cap_harness import elrobot  # noqa: F401  (headless GL before mujoco loads)
from cap_harness.agent.elrobot_server import build_elrobot_runtime
from cap_harness.contracts import MotionStrategy
from cap_harness.elrobot.approach import ApproachPlanner
from cap_harness.elrobot.ik import ElRobotIKProvider
from cap_harness.elrobot.kinematics import ElRobotKinematics
from cap_harness.elrobot.segmentation import ColorSegmentationProvider
from cap_harness.elrobot.station import build_sim_station

STRATEGY = MotionStrategy(ik_solver="mink")
PIZZA = np.array([0.26, 0.235])
PIZZA_RADIUS = 0.178
# The arm cannot reach the whole pizza (it is 36 cm across and the far side is past the arm's
# reach), so toppings go where it can: the near-left part, which is also where the real operator
# placed them. Slots are spread out so a release never lands on, or drags, something already placed.
SLOT_CENTRE = np.array([0.22, 0.22])
SALAMI_SLOTS = [
    SLOT_CENTRE + 0.055 * np.array([np.cos(a), np.sin(a)]) for a in np.deg2rad([90, 210, 330])
]
CHEESE_SLOTS = [
    SLOT_CENTRE + np.array(o) for o in ([0.0, 0.0], [0.0, 0.025], [0.0, -0.025], [0.025, 0.0])
]
# Item recipes. grasp_z is the item's centre height; the gripper reading after closing says
# whether something is between the fingers: ~0.00 empty, ~0.1 a cheese stick, ~0.55 a slice.
RECIPES = {
    "salami": dict(grasp_z=0.016, hover=0.07, jaw_azimuth=None, held=(0.35, 0.75)),
    "cheese": dict(grasp_z=0.025, hover=0.07, jaw_azimuth=90.0, held=(0.08, 0.35)),
}
QUERIES = {
    "salami": ["salami center", "salami left", "salami right", "salami top", "salami bottom"],
    "cheese": ["cheese center", "cheese left", "cheese right", "cheese top", "cheese bottom"],
}
DROP_HOVER, DROP_Z = 0.11, 0.035


#: Moves run at the real arm's speeds, so a long one takes many 20 ms ticks; the API's default
#: ``max_steps=120`` (2.4 s) refuses them. 800 ticks = 16 s.
MAX_STEPS = 800


def go(api, pose):
    return api.move_to_pose(pose, strategy=STRATEGY, max_steps=MAX_STEPS)


def area_on_plate(api, kind):
    """Pixels of ``kind`` the overhead camera sees on the pizza. Touching items merge into one blob,
    so area, not instance count, is what stays comparable before and after a placement.
    """
    found = api.segment_text("overhead", f"{kind} on the plate")
    return sum(int(item.mask.sum()) for item in found.segmentations) if found.ok else 0


def pick(api, planner, kind, query, log):
    """Pick the item ``query`` names. Returns its image area (truthy) with the item held, else 0."""
    recipe = RECIPES[kind]
    located = api.localize_object(query, camera_name="overhead")
    if not located.ok:
        log(f"  {query}: not found ({located.error.message[:60]})")
        return 0
    xy = located.geometry.pose.position[:2]
    item_area = int(located.segmentation.mask.sum())
    options = planner.plan(
        (xy[0], xy[1], recipe["grasp_z"]),
        hover_z=recipe["hover"],
        jaw_azimuth_deg=recipe["jaw_azimuth"],
        limit=3,
    )
    if not options or not options[0].clear:
        log(f"  {query}: no clear approach to {np.round(xy, 3)}")
        return 0
    for option in options:
        api.open_gripper()
        if not go(api, option.hover_pose).ok or not go(api, option.grasp_pose).ok:
            log(f"  {query}: tilt {option.tilt_deg:.0f} could not reach the item")
            go(api, option.hover_pose)
            continue
        api.set_gripper(0.0)
        reading = api.get_robot_state().gripper_positions["arm"]
        low, high = recipe["held"]
        if not low <= reading <= high:
            log(
                f"  {query}: tilt {option.tilt_deg:.0f} grip reads {reading:.2f}, expected {low}-{high}"
            )
            api.open_gripper()
            go(api, option.hover_pose)
            continue
        if not go(api, option.hover_pose).ok:
            api.open_gripper()
            continue
        # Lifting is the real test. A slice or stick that slips out lets the closed-commanded
        # jaws keep closing, so the reading falls; one still held reads as it did.
        after = api.get_robot_state().gripper_positions["arm"]
        if abs(after - reading) > 0.08 or not low <= after <= high:
            log(f"  {query}: lost it on the lift (reading {reading:.2f} -> {after:.2f})")
            api.open_gripper()
            continue
        return item_area
    return 0


def release_on(api, planner, kind, slot, log):
    recipe = RECIPES[kind]
    options = planner.plan(
        (slot[0], slot[1], DROP_Z),
        hover_z=DROP_HOVER,
        jaw_azimuth_deg=recipe["jaw_azimuth"],
        limit=2,
    )
    for option in options:
        if go(api, option.hover_pose).ok and go(api, option.grasp_pose).ok:
            api.open_gripper()
            go(api, option.hover_pose)
            return True
    log(f"  no clear way onto the plate at {np.round(slot, 3)}")
    return False


def run(api, planner, salami, cheese, log=print):
    api.go_home()
    placed = {"salami": 0, "cheese": 0}
    for kind, wanted, slots in (("salami", salami, SALAMI_SLOTS), ("cheese", cheese, CHEESE_SLOTS)):
        for slot in slots[:wanted]:
            for query in QUERIES[kind]:
                log(f"{kind} -> {np.round(slot, 3)} via '{query}'")
                before = area_on_plate(api, kind)
                item_area = pick(api, planner, kind, query, log)
                if not item_area:
                    continue
                if release_on(api, planner, kind, slot, log):
                    api.go_home()  # clear the camera's view of the pizza before looking
                    gained = area_on_plate(api, kind) - before
                    log(
                        f"  placed; overhead sees +{gained} px of {kind} on the pizza (item is ~{item_area} px)"
                    )
                    if gained >= 0.5 * item_area:
                        placed[kind] += 1
                        break
                api.open_gripper()
    api.go_home()
    return placed


def score(env):
    """Ground truth, for the harness's eyes only: what is actually on the plate?"""
    model, data = env.model, env.data
    base = data.xpos[model.body("base_link").id]
    counts = {"salami": 0, "cheese": 0}
    for index in range(model.nbody):
        name = model.body(index).name
        for kind, prefix in (("salami", "bin_salami_disc"), ("cheese", "bin_cheese_shred")):
            if name.startswith(prefix):
                position = data.xpos[index] - base
                if (
                    np.linalg.norm(position[:2] - PIZZA) < PIZZA_RADIUS - 0.01
                    and position[2] > 0.008
                ):
                    counts[kind] += 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salami", type=int, default=3)
    parser.add_argument("--cheese", type=int, default=4)
    args = parser.parse_args(argv)
    env = build_sim_station()
    _, runtime = build_elrobot_runtime(
        env,
        allow_motion=True,
        ik_providers={"mink": ElRobotIKProvider(ElRobotKinematics(env.config))},
        segmentation_provider=ColorSegmentationProvider(),
    )
    placed = run(runtime.session.api, ApproachPlanner(env.config), args.salami, args.cheese)
    truth = score(env)
    print(f"agent-visible: {placed}   on plate (ground truth): {truth}")
    ok = truth == {"salami": args.salami, "cheese": args.cheese}
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()

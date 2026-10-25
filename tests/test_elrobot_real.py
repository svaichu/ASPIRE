"""The real demonstrations (ApoorvaBajaj/pizza-topping-camera-cleaned) as seen by the sim.

Uses the exported arrays under ``pizza-robot/real/`` (made with ``robotdataset`` by
``export_real.py``), so the suite needs neither the dataset nor torch.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from cap_harness import elrobot  # noqa: F401  (headless GL before mujoco loads)
from cap_harness.elrobot.config import load_config
from cap_harness.elrobot.real import (
    REAL_DIR,
    REAL_GRIPPER_CLOSED,
    REAL_GRIPPER_OPEN,
    RealJointMap,
    load_calibration,
    load_real_episodes,
)

pytestmark = pytest.mark.skipif(
    not (REAL_DIR / "episodes.npz").exists(), reason="run pizza-robot/real/export_real.py first"
)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mapper():
    return RealJointMap.from_config(load_config())


def test_export_matches_the_dataset_card():
    episodes = load_real_episodes()
    assert len(episodes) == 30 and sum(len(e.state) for e in episodes) == 12816
    assert {e.task.split()[2] for e in episodes} == {"cheese", "olives", "salami"}
    assert all(e.state.shape[1] == 8 and 9.5 < e.fps < 10.5 for e in episodes)


def test_calibrated_spans_match_the_sim_limit_spans():
    """The premise of the linear map: the calibration sweep IS the URDF travel."""
    config = load_config()
    calibration = load_calibration()
    span = (
        (np.array(calibration["range_max"]) - np.array(calibration["range_min"]))[:7]
        * 2
        * np.pi
        / 4096
    )
    sim = config.joint_limits_upper - config.joint_limits_lower
    assert np.all(np.abs(span / sim - 1) < 0.08)


def test_map_round_trips_and_stays_inside_the_limits(mapper):
    state = np.concatenate([e.state for e in load_real_episodes()])
    joints, gripper = mapper.to_sim(state)
    config = load_config()
    assert np.all(joints >= config.joint_limits_lower - 1e-9)
    assert np.all(joints <= config.joint_limits_upper + 1e-9)
    assert gripper.min() >= 0.0 and gripper.max() <= 1.0
    back = mapper.from_sim(joints, gripper)
    np.testing.assert_allclose(back[:, :7], state[:, :7], atol=1e-6)
    inside = (state[:, 7] >= REAL_GRIPPER_CLOSED) & (state[:, 7] <= REAL_GRIPPER_OPEN)
    np.testing.assert_allclose(back[inside, 7], state[inside, 7], atol=1e-6)


def test_the_real_gripper_is_open_when_high():
    """High norm = open on the real station, as in the sim's normalized gripper."""
    mapper = RealJointMap.from_config(load_config())
    _, open_ = mapper.to_sim(np.array([0.5] * 7 + [REAL_GRIPPER_OPEN]))
    _, closed = mapper.to_sim(np.array([0.5] * 7 + [REAL_GRIPPER_CLOSED]))
    assert open_ == pytest.approx(1.0) and closed == pytest.approx(0.0)


def test_the_real_fingertip_never_goes_below_the_table(mapper):
    from cap_harness.elrobot.kinematics import ElRobotKinematics

    kin = ElRobotKinematics(load_config())
    state = np.concatenate([e.state for e in load_real_episodes()])[::15]
    heights = [kin.forward_kinematics(q)[0][2] for q in mapper.to_sim(state)[0]]
    assert min(heights) > -0.01


def test_scene_matches_the_recovered_layout():
    layout = json.loads((REAL_DIR / "layout.json").read_text())
    config = load_config()
    xml = (ROOT / "pizza-robot/scene/elrobot_table.xml").read_text()
    import re

    pizza = re.search(
        r'<body name="pizza_plate" pos="([^"]+)">\s*<geom type="cylinder" size="([^ ]+)', xml
    )
    x, y, _ = (float(v) for v in pizza.group(1).split())
    assert (
        np.hypot(x - layout["pizza"]["centre_xy"][0], (y + 0.36) - layout["pizza"]["centre_xy"][1])
        < 0.02
    )
    assert float(pizza.group(2)) == pytest.approx(layout["pizza"]["radius_m"], abs=0.002)
    np.testing.assert_allclose(config.home_joints, layout["motion"]["rest_joints_rad"], atol=1e-3)


def test_replaying_a_real_demonstration_is_tracked_closely():
    spec = importlib.util.spec_from_file_location(
        "replay_real", ROOT / "pizza-robot/scripts/replay_real.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from cap_harness.elrobot.station import build_sim_station

    env = build_sim_station()
    try:
        episode = next(e for e in load_real_episodes() if e.name == "symmetric-salami3")
        stats = module.replay(env, episode, RealJointMap.from_config(env.config))
    finally:
        env.close()
    assert stats["rms_rad"] < 0.02 and stats["p99_rad"] < 0.08
    assert stats["lowest_fingertip_z_m"] > -0.01

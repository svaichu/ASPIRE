"""Creeping toward an item: the step rules, the wrist camera model, and a closed loop in the sim."""

from __future__ import annotations

import numpy as np
import pytest

from cap_harness import elrobot  # noqa: F401  (selects headless GL before mujoco loads)
from cap_harness.elrobot import creep
from cap_harness.elrobot.creep import Estimate, creep_step, fuse
from cap_harness.elrobot.poses import grasp_pose

pytest.importorskip("mujoco")

TIP = np.array([0.15, 0.10, 0.20])
ITEM = (0.25, 0.25, 0.016)


def _est(camera, dx=0.0, dy=0.0, score=1.0):
    return Estimate(camera, (ITEM[0] + dx, ITEM[1] + dy, ITEM[2]), score)


class TestStepRules:
    def test_nothing_localized_holds(self):
        out = creep_step(TIP, [])
        assert out["action"] == "hold" and "target_pose" not in out

    def test_agreeing_cameras_allow_a_full_step_far_away(self):
        out = creep_step(TIP, [_est("overhead"), _est("wrist", 0.003)])
        assert out["action"] == "move" and out["phase"] == "align"
        assert out["step_m"] == pytest.approx(0.04)
        assert out["confidence"] > 0.9

    def test_one_camera_alone_takes_a_smaller_step(self):
        both = creep_step(TIP, [_est("overhead"), _est("wrist")])["step_m"]
        alone = creep_step(TIP, [_est("overhead")])
        assert alone["step_m"] < both and alone["confidence"] <= creep.SINGLE_VIEW_CONFIDENCE

    def test_disagreement_shrinks_the_step(self):
        sure = creep_step(TIP, [_est("overhead"), _est("wrist", 0.004)])
        unsure = creep_step(TIP, [_est("overhead"), _est("wrist", 0.03)])
        assert unsure["step_m"] < sure["step_m"]
        assert any("disagree" in note for note in unsure["notes"])

    def test_a_wrist_view_of_a_different_item_is_dropped_not_averaged(self):
        wrong = _est("wrist", 0.08)  # 8 cm off: the wrist camera is looking at another slice
        out = creep_step(TIP, [_est("overhead"), wrong])
        assert out["action"] == "move" and out["confidence"] <= creep.CONFLICT_CONFIDENCE
        assert out["step_m"] <= 0.25 * creep.FAR_STEP_M + 1e-9  # tiny
        assert out["item_xy"] == pytest.approx([ITEM[0], ITEM[1]])  # overhead only
        assert any("different items" in note for note in out["notes"])

    def test_holds_when_confidence_is_near_zero(self):
        out = creep_step(TIP, [_est("overhead", score=0.1)])
        assert out["action"] == "hold" and "fresh observation" in out["reason"]

    def test_low_segmentation_score_counts_against_confidence(self):
        confident = creep_step(TIP, [_est("overhead"), _est("wrist")])
        weak = creep_step(TIP, [_est("overhead", score=0.4), _est("wrist", score=0.4)])
        assert weak["confidence"] < confident["confidence"] and weak["step_m"] < confident["step_m"]

    def test_step_shrinks_with_distance_and_is_never_most_of_what_is_left(self):
        near = np.array([0.238, 0.25, 0.07])  # 12 mm from the stop point sideways
        out = creep_step(near, [_est("overhead"), _est("wrist")], stop_z=0.07)
        assert out["step_m"] == pytest.approx(0.6 * 0.012)  # a fraction of what is left

    def test_aligns_sideways_before_descending_and_never_dives_while_misaligned(self):
        high = creep_step(np.array([0.15, 0.10, 0.20]), [_est("overhead"), _est("wrist")])
        assert high["target_pose"]["position"][2] == pytest.approx(0.20)  # same height
        over = creep_step(np.array([0.251, 0.249, 0.20]), [_est("overhead"), _est("wrist")])
        assert over["phase"] == "descend" and over["target_pose"]["position"][2] < 0.20

    def test_rises_first_when_below_the_stop_height(self):
        low = creep_step(np.array([0.15, 0.10, 0.03]), [_est("overhead"), _est("wrist")])
        assert low["phase"] == "rise"
        assert low["target_pose"]["position"][:2] == pytest.approx([0.15, 0.10])

    def test_never_goes_below_the_stop_height(self):
        out = creep_step(np.array([0.25, 0.25, 0.09]), [_est("overhead"), _est("wrist")], stop_z=0.07)
        assert out["target_pose"]["position"][2] >= 0.07 - 1e-9

    def test_done_at_the_stop_point(self):
        out = creep_step(np.array([0.2505, 0.2495, 0.07]), [_est("overhead"), _est("wrist")])
        assert out["action"] == "done"

    def test_held_orientation_and_max_steps_bounds(self):
        out = creep_step(TIP, [_est("overhead"), _est("wrist")], quaternion_wxyz=(0, 1, 0, 0))
        assert out["target_pose"]["quaternion_wxyz"] == [0, 1, 0, 0]
        assert 60 <= out["max_steps"] <= 200

    def test_wrist_dominates_up_close_and_overhead_far_away(self):
        wrist_off = [_est("overhead"), _est("wrist", 0.01)]
        far, _, _ = fuse(wrist_off, np.array([0.0, -0.2, 0.3]))
        close, _, _ = fuse(wrist_off, np.array([0.25, 0.24, 0.08]))
        assert abs(far[0] - ITEM[0]) < abs(close[0] - ITEM[0])


class TestWristCamera:
    def test_pose_matches_the_simulators_own_wrist_camera(self):
        from cap_harness.elrobot.config import load_config
        from cap_harness.elrobot.remote import RealCameraModel, WristCameraModel
        from cap_harness.elrobot.station import build_sim_station

        env = build_sim_station()
        try:
            try:
                env.camera_frame("wrist")
            except Exception as exc:  # no EGL on this machine
                pytest.skip(f"offscreen rendering unavailable: {exc}")
            model = WristCameraModel(env.config, RealCameraModel.from_layout(env.config).table_z)
            for joints in (env.config.home_joints, env.config.home_joints + 0.2):
                env.execute_waypoints(
                    [0.0, 1.0], np.vstack([env.joint_positions(), joints]), [1.0, 1.0], settle_s=0.3
                )
                truth = env.read_camera("wrist")
                np.testing.assert_allclose(model.pose(env.joint_positions()), truth.base_from_camera, atol=1e-6)
            assert model.vertical_fov_deg == pytest.approx(90.0)
        finally:
            env.close()

    def test_table_depth_matches_the_simulators_depth_image_over_the_table(self):
        from cap_harness.elrobot.kinematics import ElRobotKinematics
        from cap_harness.elrobot.poses import grasp_pose
        from cap_harness.elrobot.remote import RealCameraModel, WristCameraModel
        from cap_harness.elrobot.station import build_sim_station

        env = build_sim_station()
        try:
            # At Home the camera looks at the gripper's own body; hover over the table instead.
            hover = grasp_pose((0.22, 0.22, 0.15), 40.0)
            joints, _, _ = ElRobotKinematics(env.config).inverse_kinematics(
                np.array(hover["position"]), np.array(hover["quaternion_wxyz"]), env.config.home_joints
            )
            env.execute_waypoints(
                [0.0, 4.0], np.vstack([env.joint_positions(), joints]), [1.0, 1.0], settle_s=0.5
            )
            try:
                frame = env.read_camera("wrist")
            except Exception as exc:
                pytest.skip(f"offscreen rendering unavailable: {exc}")
            model = WristCameraModel(env.config, RealCameraModel.from_layout(env.config).table_z)
            depth = model.table_depth(
                frame.rgb.shape[1], frame.rgb.shape[0], model.pose(env.joint_positions())
            )
            both = (depth > 0) & (frame.depth_m > 0)
            assert both.mean() > 0.3
            # Bins, plate and items stand above the table, so not every pixel agrees.
            assert (np.abs(depth[both] - frame.depth_m[both]) < 0.02).mean() > 0.6
        finally:
            env.close()


@pytest.fixture
def rig():
    from cap_harness.agent.elrobot_server import build_elrobot_runtime
    from cap_harness.elrobot.ik import ElRobotIKProvider
    from cap_harness.elrobot.kinematics import ElRobotKinematics
    from cap_harness.elrobot.segmentation import ColorSegmentationProvider
    from cap_harness.elrobot.station import build_sim_station

    env = build_sim_station()
    try:
        env.camera_frame("overhead")
    except Exception as exc:
        env.close()
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    _, runtime = build_elrobot_runtime(
        env,
        allow_motion=True,
        ik_providers={"mink": ElRobotIKProvider(ElRobotKinematics(env.config))},
        segmentation_provider=ColorSegmentationProvider(),
    )
    yield env, runtime.session.api
    runtime.close()


def _localize(api, query, camera):
    result = api.localize_object(query, camera_name=camera)
    if not result.ok:
        return None
    return Estimate(camera, tuple(result.geometry.pose.position), float(result.segmentation.score))


def _look(api):
    """Overhead names the item; the wrist is asked for that very item ("salami at X,Y")."""
    overhead = _localize(api, "salami center", "overhead")
    if overhead is None:
        return []
    x, y = overhead.position[:2]
    wrist = _localize(api, f"salami at {x:.3f},{y:.3f}", "wrist")
    return [overhead] + ([wrist] if wrist else [])


def test_closed_loop_creep_reaches_hover_over_a_salami_slice(rig):
    """Scout over the region, then look / take one bounded step / look again until over the slice."""
    from cap_harness.contracts import MotionStrategy

    env, api = rig
    strategy = MotionStrategy(ik_solver="mink")
    from cap_harness.elrobot.approach import ApproachPlanner

    overhead = _localize(api, "salami center", "overhead")
    assert overhead is not None
    x, y = overhead.position[:2]
    # The planner picks a tilt/lean that is reachable and clear; creep holds that orientation.
    approach = ApproachPlanner(env.config).plan((x, y, 0.02), hover_z=0.09)[0]
    assert approach.clear
    quaternion = tuple(approach.hover_pose["quaternion_wxyz"])
    hover = np.array(approach.hover_pose["position"])
    # One coarse move to a viewing pose beside and above it; creeping covers the last few centimetres.
    for offset in ([0.0, 0.04, 0.05], [-0.03, 0.03, 0.05], [0.03, 0.03, 0.05], [0.0, 0.0, 0.05]):
        scout = {**approach.hover_pose, "position": [float(v) for v in hover + offset]}
        if api.move_to_pose(scout, max_steps=800, strategy=strategy).ok:  # not every side is reachable
            break
    else:
        pytest.fail("no reachable scout pose beside the hover pose")

    sizes, held, looks = [], 0, 0
    for looks in range(1, 61):
        pose = api.get_robot_state().end_effector_poses["arm"]
        # Translate only, holding the approach's orientation, so a step never needs a wrist turn.
        out = creep_step(np.asarray(pose.position), _look(api), stop_z=0.09, quaternion_wxyz=quaternion)
        if out["action"] == "done":
            break
        if out["action"] == "hold":
            held += 1
            continue
        sizes.append(out["step_m"])
        result = api.move_to_pose(out["target_pose"], max_steps=out["max_steps"], strategy=strategy)
        assert result.ok, (out, result.error)
    else:
        pytest.fail(f"did not arrive in 60 looks; last: {out}")

    assert max(sizes) <= creep.FAR_STEP_M + 1e-9  # every command stayed small
    assert len(sizes) >= 3  # it really crept, not one jump
    final = np.asarray(api.get_robot_state().end_effector_poses["arm"].position)
    truth = _localize(api, "salami center", "overhead")
    assert truth is not None
    assert np.linalg.norm(final[:2] - np.array(truth.position[:2])) < 0.02
    assert final[2] == pytest.approx(0.09, abs=0.015)


def test_at_xy_asks_the_wrist_camera_for_the_item_the_overhead_camera_found(rig):
    """Each camera's own 'center' can be a different slice; 'at X,Y' pins the same one."""
    from cap_harness.contracts import MotionStrategy

    env, api = rig
    overhead = _localize(api, "salami left", "overhead")  # an edge slice, not the middle one
    assert overhead is not None
    hover = grasp_pose((overhead.position[0], overhead.position[1], 0.14), 40.0)
    result = api.move_to_pose(hover, max_steps=800, strategy=MotionStrategy(ik_solver="mink"))
    assert result.ok, result.error
    x, y = overhead.position[:2]
    pinned = _localize(api, f"salami at {x:.3f},{y:.3f}", "wrist")
    assert pinned is not None
    assert np.linalg.norm(np.array(pinned.position[:2]) - np.array([x, y])) < 0.02

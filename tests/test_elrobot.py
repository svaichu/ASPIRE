"""ElRobot pizza station: plant, kinematics, adapter interlock, perception, and grasping.

The last two classes drive real physics (a salami disc and a cheese stick, bin to
plate) because the hard-won facts about this gripper -- scissor jaws, tapered
blades, soft contacts -- only show up in contact, and a mock would agree with
whatever it was told.
"""

from __future__ import annotations

import numpy as np
import pytest

from cap_harness import elrobot  # noqa: F401  (selects headless GL before mujoco loads)
from cap_harness.contracts import MotionStrategy, Pose
from cap_harness.elrobot.adapter import ElRobotAdapter
from cap_harness.elrobot.ik import ElRobotIKProvider
from cap_harness.elrobot.kinematics import ElRobotKinematics
from cap_harness.elrobot.poses import grasp_pose
from cap_harness.elrobot.registry import ElRobotRegistryError, ElRobotTaskRegistry
from cap_harness.elrobot.segmentation import ColorSegmentationProvider
from cap_harness.elrobot.station import build_sim_station
from cap_harness.errors import ErrorCode

mujoco = pytest.importorskip("mujoco")
STRATEGY = MotionStrategy(ik_solver="mink")
#: The real pizza (fitted from the demonstrations) and the slot the pick-and-place tests release at.
PIZZA = np.array([0.26, 0.235])
PIZZA_RADIUS = 0.178
SLOT = (0.22, 0.22)
#: Real-speed moves are long; the API default of 120 ticks (2.4 s) refuses them.
MAX_STEPS = 800


def _build_api(allow_motion: bool = True):
    from cap_harness.agent.elrobot_server import build_elrobot_runtime

    env = build_sim_station()
    try:
        env.camera_frame("overhead")
    except Exception as exc:  # no EGL on this machine
        env.close()
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    _, runtime = build_elrobot_runtime(
        env,
        allow_motion=allow_motion,
        ik_providers={"mink": ElRobotIKProvider(ElRobotKinematics(env.config))},
        segmentation_provider=ColorSegmentationProvider(),
    )
    return env, runtime


@pytest.fixture
def rig():
    env, runtime = _build_api()
    yield env, runtime.session.api
    runtime.close()


def _base(env):
    return env.data.xpos[env.model.body("base_link").id].copy()


def _body(env, name):
    return env.data.xpos[env.model.body(name).id] - _base(env)


class TestPlant:
    def test_starts_at_home_with_open_gripper(self, rig):
        env, _ = rig
        np.testing.assert_allclose(env.joint_positions(), env.config.home_joints, atol=2e-3)
        assert env.gripper_position() == pytest.approx(1.0, abs=1e-2)

    def test_tracks_a_joint_motion_tightly(self, rig):
        env, _ = rig
        target = np.array([0.4, 0.3, -1.0, 0.6, 0.5, 0.8, 0.4])
        env.execute_waypoints(
            [0.0, 2.0], np.vstack([env.joint_positions(), target]), [1.0, 1.0], settle_s=0.5
        )
        assert np.max(np.abs(env.joint_positions() - target)) < 0.01

    def test_commands_are_clipped_to_joint_limits(self, rig):
        env, _ = rig
        env.execute_waypoints([0.0, 1.0], np.vstack([env.joint_positions()] * 2) + 9.0, [1.0, 1.0])
        assert np.all(env.joint_positions() <= env.config.joint_limits_upper + 0.01)

    def test_gripper_zero_is_closed_and_one_is_open(self, rig):
        env, _ = rig
        env.set_gripper(0.0)
        assert env.gripper_position() < 0.05
        env.set_gripper(1.0)
        assert env.gripper_position() > 0.95

    def test_camera_reports_calibrated_rgbd_in_the_base_frame(self, rig):
        env, _ = rig
        frame = env.camera_frame("overhead")
        assert frame.rgb.shape == (480, 640, 3) and frame.depth_m.shape == (480, 640)
        assert (frame.depth_m > 0).mean() > 0.5
        rotation = frame.base_from_camera[:3, :3]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
        assert frame.base_from_camera[2, 3] > 0.3  # overhead: above the table

    def test_scene_reset_restores_the_authored_layout(self, rig):
        env, _ = rig
        model = env.model
        body = model.body("bin_salami_disc4")
        before = _body(env, "bin_salami_disc4").copy()
        address = model.jnt_qposadr[body.jntadr[0]]
        env.data.qpos[address : address + 3] += [0.2, 0.2, 0.1]  # knock the slice off
        env.reset()
        np.testing.assert_allclose(_body(env, "bin_salami_disc4"), before, atol=2e-3)


class TestKinematics:
    def test_fk_matches_the_plant(self, rig):
        env, _ = rig
        kin = ElRobotKinematics(env.config)
        position, quaternion = kin.forward_kinematics(env.joint_positions())
        expected_position, expected_quaternion = env.ee_pose()
        np.testing.assert_allclose(position, expected_position, atol=1e-6)
        assert abs(float(quaternion @ expected_quaternion)) == pytest.approx(1.0, abs=1e-6)

    def test_ik_round_trips_a_reachable_pose(self, rig):
        env, _ = rig
        kin = ElRobotKinematics(env.config)
        goal = np.array([0.3, -0.5, 0.4, 0.8, 0.2, 0.9, -0.3])
        position, quaternion = kin.forward_kinematics(goal)
        solution, metres, radians = kin.inverse_kinematics(
            position, quaternion, env.config.home_joints
        )
        assert metres < 2e-3 and radians < 2e-2
        assert np.all(solution >= env.config.joint_limits_lower - 1e-6)
        assert np.all(solution <= env.config.joint_limits_upper + 1e-6)

    def test_provider_refuses_an_unreachable_pose_before_any_motion(self, rig):
        env, api = rig
        far = Pose([0.0, 1.5, 0.3], [0, 1, 0, 0], "robot_base")
        result = api.solve_ik(far, backend="mink")
        assert not result.ok and result.error.code is ErrorCode.IK_UNREACHABLE

    def test_provider_rejects_a_foreign_frame(self, rig):
        _, api = rig
        result = api.solve_ik(Pose([0.0, 0.2, 0.1], [0, 1, 0, 0], "world"), backend="mink")
        assert not result.ok and result.error.code is ErrorCode.INVALID_REQUEST


class TestPoses:
    @pytest.mark.parametrize("tilt", [0.0, 20.0, 45.0])
    @pytest.mark.parametrize("position", [(0.2, 0.15, 0.02), (-0.2, 0.15, 0.02), (0.0, 0.25, 0.05)])
    def test_approach_leans_away_from_the_base_by_the_requested_tilt(self, position, tilt):
        pose = Pose(**grasp_pose(position, tilt))
        approach = pose.as_matrix()[:3, 2]
        assert approach[2] == pytest.approx(-np.cos(np.deg2rad(tilt)), abs=1e-4)
        radial = np.array([position[0], position[1]]) / np.hypot(position[0], position[1])
        assert float(approach[:2] @ radial) == pytest.approx(np.sin(np.deg2rad(tilt)), abs=1e-4)

    def test_jaw_axis_is_horizontal_tangent_by_default_and_follows_azimuth(self):
        matrix = Pose(**grasp_pose((0.2, 0.15, 0.02), 30.0)).as_matrix()
        assert abs(matrix[2, 0]) < 0.6  # closing axis stays nearly horizontal
        steered = Pose(**grasp_pose((0.2, 0.15, 0.02), 30.0, jaw_azimuth_deg=90.0)).as_matrix()
        assert abs(steered[1, 0]) > 0.9  # closes along +/- y

    def test_quaternion_is_unit(self):
        assert np.linalg.norm(grasp_pose((0.1, 0.2, 0.03), 30)["quaternion_wxyz"]) == pytest.approx(
            1.0, abs=1e-4
        )


class TestAdapterInterlock:
    def test_unauthorized_motion_is_refused_before_the_plant_is_touched(self):
        env = build_sim_station()
        try:
            adapter = ElRobotAdapter(env, allow_physical_motion=False)
            before = env.joint_positions().copy()
            for result in (adapter.set_gripper(0.0), adapter.go_home()):
                assert not result.ok and result.error.code is ErrorCode.SAFETY_INTERLOCK
            assert adapter.command_rpc_count == 0
            np.testing.assert_array_equal(env.joint_positions(), before)
        finally:
            env.close()

    def test_one_arm_aliases(self):
        env = build_sim_station()
        try:
            adapter = ElRobotAdapter(env)
            assert adapter.resolve_arm("primary") == "arm" and adapter.resolve_arm("arm") == "arm"
            assert adapter.embodiment == "elrobot" and adapter.dof == 7
        finally:
            env.close()

    def test_state_uses_the_shared_contract(self, rig):
        env, api = rig
        state = api.get_robot_state()
        assert state.embodiment == "elrobot" and state.base_frame == "robot_base"
        assert state.joint_positions["arm"].shape == (7,)
        assert state.end_effector_poses["arm"].frame == "robot_base"
        assert 0.95 <= state.gripper_positions["arm"] <= 1.0

    def test_home_reports_verified_diagnostics(self, rig):
        _, api = rig
        api.move_to_joints(np.array([0.3, 0.2, -0.5, 0.3, 0.1, 0.4, 0.2]))
        result = api.go_home()
        assert result.ok
        assert result.diagnostics["home_verified"] is True
        assert result.diagnostics["grippers_open"] == {"arm": True}

    def test_blocked_motion_is_reported_not_swallowed(self, rig):
        env, api = rig
        # Drive the fingertips into the table: the arm cannot track and must say so.
        api.go_home()
        result = api.move_to_pose(
            grasp_pose((0.0, 0.22, -0.03), 20.0), strategy=STRATEGY, max_steps=MAX_STEPS
        )
        assert not result.ok


class TestRegistry:
    def test_resolves_known_tasks_and_rejects_others(self):
        registry = ElRobotTaskRegistry()
        assert registry.resolve("pizza_assembly:0").task_name == "pizza_assembly"
        with pytest.raises(ElRobotRegistryError):
            registry.resolve("unknown")
        with pytest.raises(ElRobotRegistryError):
            registry.resolve("pizza_assembly:3")


class TestPerception:
    def test_finds_every_salami_slice_and_every_cheese_stick(self, rig):
        _, api = rig
        assert len(api.segment_text("overhead", "salami").segmentations) == 9
        assert len(api.segment_text("overhead", "cheese").segmentations) == 9

    def test_localizes_the_plate_within_a_centimetre(self, rig):
        _, api = rig
        located = api.localize_object("plate", camera_name="overhead")
        assert located.ok
        assert np.linalg.norm(located.geometry.pose.position[:2] - PIZZA) < 0.01
        assert located.geometry.extents[0] == pytest.approx(2 * PIZZA_RADIUS, abs=0.02)

    def test_a_qualifier_selects_the_instance(self, rig):
        env, api = rig
        centre = api.localize_object("salami center", camera_name="overhead").geometry.pose.position
        assert np.linalg.norm(centre[:2] - _body(env, "bin_salami_disc4")[:2]) < 0.01
        left = api.localize_object("salami left", camera_name="overhead").geometry.pose.position
        right = api.localize_object("salami right", camera_name="overhead").geometry.pose.position
        assert left[0] > centre[0] > right[0]  # the overhead view is mirrored: image-left is +x

    def test_unknown_query_is_refused_with_the_vocabulary(self, rig):
        _, api = rig
        result = api.localize_object("unicorn", camera_name="overhead")
        assert not result.ok and "vocabulary" in str(result.error.details)


def _pick_and_place(env, api, name, query, *, tilt, jaw_azimuth=None, grasp_z):
    located = api.localize_object(query, camera_name="overhead").geometry.pose.position
    assert np.linalg.norm(located[:2] - _body(env, name)[:2]) < 0.01, "localized the wrong item"
    xy = located[:2]
    jaw = {} if jaw_azimuth is None else {"jaw_azimuth_deg": jaw_azimuth}

    def go(z, at=xy):
        result = api.move_to_pose(
            grasp_pose((at[0], at[1], z), tilt, **jaw), strategy=STRATEGY, max_steps=MAX_STEPS
        )
        assert result.ok, result.error
        return result

    api.open_gripper()
    go(0.07)
    go(grasp_z)
    api.set_gripper(0.0)
    go(0.10)
    assert _body(env, name)[2] > 0.06, "the item did not follow the lift"
    go(0.12, SLOT)
    go(0.05, SLOT)
    api.open_gripper()
    go(0.12, SLOT)
    return _body(env, name)


class TestGrasping:
    def test_salami_slice_goes_from_bin_to_plate(self, rig):
        env, api = rig
        final = _pick_and_place(
            env, api, "bin_salami_disc4", "salami center", tilt=40.0, grasp_z=0.016
        )
        assert np.linalg.norm(final[:2] - PIZZA) < PIZZA_RADIUS and final[2] > 0.005

    def test_cheese_stick_follows_the_lift(self, rig):
        """A stick is gripped and lifted. (Carrying it across the table is not reliable
        yet -- a pinched cylinder can twist out of the pads on a long swing -- so the end-to-end
        placement is covered by salami, and cheese by the reference episode's ``>= 1``.).
        """
        env, api = rig
        located = api.localize_object(
            "cheese center", camera_name="overhead"
        ).geometry.pose.position
        for z in (0.07, 0.025):
            result = api.move_to_pose(
                grasp_pose((located[0], located[1], z), 40.0, jaw_azimuth_deg=90.0),
                strategy=STRATEGY,
                max_steps=MAX_STEPS,
            )
            assert result.ok, result.error
        api.set_gripper(0.0)
        result = api.move_to_pose(
            grasp_pose((located[0], located[1], 0.07), 40.0, jaw_azimuth_deg=90.0),
            strategy=STRATEGY,
            max_steps=MAX_STEPS,
        )
        assert result.ok and _body(env, "bin_cheese_shred4")[2] > 0.045

    def test_an_empty_close_is_distinguishable_from_a_grasp(self, rig):
        env, api = rig
        api.set_gripper(0.0)
        empty = api.get_robot_state().gripper_positions["arm"]
        api.open_gripper()
        api.move_to_pose(
            grasp_pose((-0.04, 0.18, 0.07), 40.0), strategy=STRATEGY, max_steps=MAX_STEPS
        )
        api.move_to_pose(
            grasp_pose((-0.04, 0.18, 0.016), 40.0), strategy=STRATEGY, max_steps=MAX_STEPS
        )
        api.set_gripper(0.0)
        held = api.get_robot_state().gripper_positions["arm"]
        assert empty < 0.02 and held > 0.2  # fingers stall on the slice


class TestPlateAwarePerception:
    def test_items_on_the_plate_are_not_bin_items(self, rig):
        env, api = rig
        model, data = env.model, env.data
        base = _base(env)
        for index, (x, y) in enumerate([(0.22, 0.275), (0.17, 0.19)]):
            body = model.body(f"bin_salami_disc{index}")
            address = model.jnt_qposadr[body.jntadr[0]]
            data.qpos[address : address + 3] = [x + base[0], y + base[1], base[2] + 0.03]
        env.execute_waypoints([0, 1], np.vstack([env.joint_positions()] * 2), [1.0, 1.0])
        assert len(api.segment_text("overhead", "salami").segmentations) == 7
        assert len(api.segment_text("overhead", "salami on the plate").segmentations) == 2


class TestApproachPlanner:
    @pytest.fixture(scope="class")
    @classmethod
    def planner(cls):
        from cap_harness.elrobot.approach import ApproachPlanner
        from cap_harness.elrobot.config import load_config

        return ApproachPlanner(load_config())

    def test_finds_a_clear_approach_to_a_slice_in_the_bin(self, planner):
        found = planner.plan((-0.04, 0.18, 0.016), hover_z=0.07, limit=1)
        assert found and found[0].clear and found[0].reachable

    def test_reports_a_pose_below_the_table_as_blocked(self, planner):
        found = planner.plan((-0.04, 0.18, -0.01), hover_z=0.07, limit=1)
        assert not found or not found[0].clear

    def test_corner_slice_is_clear_at_the_standard_tilt(self, planner):
        found = planner.plan(
            (0.02, 0.24, 0.016), hover_z=0.07, tilts=(40.0,), leans=(0.0,), limit=1
        )
        assert found and found[0].clear and found[0].tilt_deg == 40.0


def test_reference_episode_assembles_the_pizza():
    """The scripted recipe, end to end: it is the proof the skills describe something achievable."""
    import importlib.util
    from pathlib import Path

    from cap_harness.elrobot.approach import ApproachPlanner

    path = Path(__file__).resolve().parents[1] / "pizza-robot/scripts/reference_episode.py"
    spec = importlib.util.spec_from_file_location("reference_episode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env, runtime = _build_api()
    try:
        placed = module.run(
            runtime.session.api, ApproachPlanner(env.config), 3, 2, log=lambda _m: None
        )
        truth = module.score(env)
        assert truth["salami"] == 3 and truth["cheese"] >= 1, truth
        assert placed["salami"] == 3  # the overhead camera saw what really happened
    finally:
        runtime.close()

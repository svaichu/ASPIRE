"""What the real demonstrations taught: camera geometry, real-scene segmentation, pick spots.

Image tests use the frames ``pizza-robot/real/export_real.py`` leaves in ``pizza-robot/real/frames``
(frame 0 of an episode: the arm is out of view) and skip when they are absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from cap_harness import elrobot  # noqa: F401  (selects headless GL before mujoco loads)
from cap_harness.contracts import CameraObservation, Pose
from cap_harness.elrobot import real
from cap_harness.elrobot.config import load_config
from cap_harness.elrobot.creep import PROFILES, REAL, SIM, Estimate, creep_step, fuse
from cap_harness.elrobot.real_segmentation import (
    OVERHEAD_SCORE,
    RealSceneSegmentationProvider,
    plane_points,
)
from cap_harness.elrobot.remote import RealCameraModel
from cap_harness.elrobot.segmentation import ColorSegmentationProvider
from cap_harness.yam_real.codec import rotation_to_wxyz

pytest.importorskip("mujoco")
PIL = pytest.importorskip("PIL.Image")
FRAMES = real.REAL_DIR / "frames"


@pytest.fixture(scope="module")
def overhead():
    return RealCameraModel.from_layout(load_config("pizza"))


def _observation(model, rgb, frame="overhead_optical"):
    pose = model.pose(np.zeros(7))
    return CameraObservation(
        rgb=rgb,
        depth_m=np.zeros(rgb.shape[:2]),
        intrinsics=model.intrinsics(rgb.shape[1], rgb.shape[0]),
        frame=frame,
        camera_pose=Pose(position=pose[:3, 3], quaternion_wxyz=rotation_to_wxyz(pose[:3, :3]), frame="robot_base"),
    )


def _frame(name):
    path = FRAMES / name
    if not path.exists():
        pytest.skip(f"{path} is not exported")
    return np.asarray(PIL.open(path).convert("RGB"))


class TestCameraGeometry:
    def test_real_frames_are_squashed_so_the_focal_lengths_differ(self, overhead):
        """cam0 frames are 640x480 shrunk to 224x224: fx = f*224/640, fy = f*224/480."""
        k = overhead.intrinsics(224, 224)
        assert k[0, 0] / k[1, 1] == pytest.approx(480 / 640)
        assert overhead.source_size == (640, 480)
        assert k[0, 2] == 112 and k[1, 2] == 112

    def test_a_frame_of_the_native_size_is_unchanged(self, overhead):
        k = overhead.intrinsics(640, 480)
        assert k[0, 0] == pytest.approx(k[1, 1])

    def test_the_pizza_centre_lands_on_the_sauce(self, overhead):
        """The fitted pizza centre projects onto the real sauce blob: measured offset (1.4, 5.2) px
        over 21 episodes (std 2.7, 1.6), i.e. about 1.6 cm vertically."""
        import json

        from scipy import ndimage

        from cap_harness.elrobot.segmentation import _hsv

        pizza = json.loads((real.REAL_DIR / "layout.json").read_text())["pizza"]
        t, k = overhead.base_from_camera, overhead.intrinsics(224, 224)
        cam = t[:3, :3].T @ (np.array([*pizza["centre_xy"], 0.0]) - t[:3, 3])
        centre = np.array([k[0, 0] * cam[0] / cam[2] + k[0, 2], k[1, 1] * cam[1] / cam[2] + k[1, 2]])

        hsv = _hsv(_frame("salami1_f0_cam0.png"))
        hue = (hsv[..., 0] + 180.0) % 360.0 - 180.0
        red = (np.abs(hue) < 20) & (hsv[..., 1] > 0.4) & (hsv[..., 2] > 0.2)
        labels, _ = ndimage.label(ndimage.binary_opening(red, iterations=2))
        sauce = labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)  # the largest red blob
        ys, xs = np.nonzero(sauce)
        assert abs(centre[0] - xs.mean()) < 5 and abs(centre[1] - ys.mean()) < 9


class TestRealSceneSegmentation:
    def test_hue_segmenter_confuses_salami_with_sauce_and_misses_cheese(self, overhead):
        """The failure that motivated the real-scene provider, kept as a regression of the premise."""
        rgb = _frame("salami1_f0_cam0.png")
        sim = ColorSegmentationProvider().segment_text(_observation(overhead, rgb), "salami")
        assert sim.ok and sum(int(s.mask.sum()) for s in sim.segmentations) > 3000  # the pizza, in pieces
        assert not ColorSegmentationProvider().segment_text(_observation(overhead, _frame("cheese1_f0_cam0.png")), "cheese").ok

    def test_salami_is_found_in_its_tray_not_on_the_pizza(self, overhead):
        rgb = _frame("salami1_f0_cam0.png")
        observation = _observation(overhead, rgb)
        result = RealSceneSegmentationProvider().segment_text(observation, "salami")
        assert result.ok
        mask = result.segmentations[0].mask
        assert mask.sum() < 600  # a pile, not the 5000+ px pizza
        points = plane_points(observation)[mask]
        tray = np.array(RealSceneSegmentationProvider().trays["salami"])
        assert np.all(np.hypot(*(points[:, :2] - tray).T) < 0.05)  # inside the tray disc

    def test_cheese_is_found_in_its_bowl(self, overhead):
        result = RealSceneSegmentationProvider().segment_text(
            _observation(overhead, _frame("cheese1_f0_cam0.png")), "cheese"
        )
        assert result.ok and result.segmentations[0].mask.sum() > 30

    def test_overhead_score_is_fixed_and_wrist_score_follows_coverage(self, overhead):
        rgb = _frame("cheese1_f0_cam0.png")
        over = RealSceneSegmentationProvider().segment_text(_observation(overhead, rgb), "cheese")
        wrist = RealSceneSegmentationProvider().segment_text(
            _observation(overhead, rgb, frame="wrist_optical"), "cheese"
        )
        assert over.segmentations[0].score == OVERHEAD_SCORE
        assert wrist.segmentations[0].score != OVERHEAD_SCORE

    def test_unknown_topping_is_refused_with_the_vocabulary(self, overhead):
        result = RealSceneSegmentationProvider().segment_text(_observation(overhead, _frame("cheese1_f0_cam0.png")), "mushroom")
        assert not result.ok and "salami" in result.error.details["vocabulary"]

    def test_a_tray_outside_the_view_is_reported_as_such(self, overhead):
        provider = RealSceneSegmentationProvider({"salami": (5.0, 5.0), "cheese": (0, 0), "olives": (0, 0)})
        result = provider.segment_text(_observation(overhead, _frame("cheese1_f0_cam0.png")), "salami")
        assert not result.ok and "not in this camera's view" in result.error.message


class TestPickEvents:
    def test_first_close_after_opening(self):
        goal = np.array([0.14] * 5 + [0.8] * 5 + [0.14] * 5 + [0.8] * 3)
        assert real.pick_event(goal) == (5, 10)

    def test_never_opened_or_never_closed_is_not_a_pick(self):
        assert real.pick_event(np.full(10, 0.14)) is None
        assert real.pick_event(np.array([0.14, 0.8, 0.8, 0.8])) is None

    def test_a_pick_is_near_the_table_and_near_where_the_jaws_opened(self):
        assert real.is_clean_pick(np.array([0, 0.3, 0.1]), np.array([0, 0.32, 0.04]))
        assert not real.is_clean_pick(np.array([0, 0.3, 0.1]), np.array([0, 0.32, 0.2]))  # not at the table
        assert not real.is_clean_pick(np.array([0, 0.1, 0.2]), np.array([0.2, 0.3, 0.04]))  # opened far away

    def test_pick_spots_need_enough_demonstrations(self, tmp_path):
        import json

        (tmp_path / "pick_spots.json").write_text(json.dumps({"spots": {
            "cheese": {"xy": [-0.03, 0.34], "n": 8}, "salami": {"xy": [0.01, 0.16], "n": 4}}}))
        assert real.load_pick_spots(tmp_path) == {"cheese": (-0.03, 0.34)}
        assert real.load_pick_spots(tmp_path / "missing") == {}


class TestCalibratedProfile:
    def test_real_noise_is_not_penalised_like_simulator_noise(self):
        """27 mm between the cameras is the median on real frames: nearly full agreement there."""
        tip = np.array([0.0, 0.2, 0.2])
        estimates = [Estimate("overhead", (0.0, 0.30, 0.02), 0.7), Estimate("wrist", (0.0, 0.327, 0.02), 0.7)]
        _, sim_conf, _ = fuse(estimates, tip, SIM)
        _, real_conf, _ = fuse(estimates, tip, REAL)
        assert real_conf > 1.4 * sim_conf
        assert real_conf == pytest.approx(0.7, abs=0.05)

    def test_a_real_overhead_view_alone_still_moves_but_cautiously(self):
        out = creep_step(np.array([0.0, 0.2, 0.2]), [Estimate("overhead", (0.0, 0.34, 0.02), OVERHEAD_SCORE)],
                         stop_z=0.07, profile=REAL)
        assert out["action"] == "move" and out["confidence"] == pytest.approx(0.49)

    def test_profiles_are_selectable(self):
        assert set(PROFILES) == {"sim", "real"} and REAL.agree_m > SIM.agree_m

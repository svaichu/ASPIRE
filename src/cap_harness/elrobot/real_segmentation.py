"""Segmentation for the *real* pizza station, where colour alone cannot tell the items apart.

The sim segmenter (:mod:`cap_harness.elrobot.segmentation`) keys on hue, and on the real frames
that fails twice over, measured on ``ApoorvaBajaj/pizza-topping-camera-cleaned``: salami and
pizza sauce have the same colour (hue 0-12 deg, saturation 0.3-0.5, value 0.3-0.4), so "salami"
lights up the whole pizza, and the real cheese is pale cream, not the sim's yellow, so "cheese"
finds nothing. The frames are also dark and noisy.

What the real scene does have is a fixed layout: the toppings sit in known trays. This provider
therefore localizes by *where* first -- a disc on the table around the tray's known position,
projected into whichever camera is asked -- and uses colour only to say what part of that disc
is the topping rather than tray or table. The estimate is the topping inside its tray, not a
single slice, and its score reflects how much of the disc looks like the topping (an arm over
the tray lowers it). ``at X,Y`` and the spatial qualifiers of the sim segmenter work the same.

Pixels and the camera pose only: no scene state, so what it can find is what a camera sees.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from cap_harness.contracts import CameraObservation, Segmentation, SegmentationSet
from cap_harness.elrobot.real import REAL_DIR, load_pick_spots
from cap_harness.elrobot.segmentation import _hsv, _order
from cap_harness.errors import ApiError, ErrorCode

MIN_AREA_PX = 12
#: Height above the table of the plane the tray disc is intersected with, metres.
PLANE_Z_M = 0.02


@dataclass(frozen=True, slots=True)
class Topping:
    name: str
    aliases: tuple[str, ...]
    tray: str  # key in layout.json trays_xy
    radius_m: float
    hue_deg: tuple[float, float]
    saturation: tuple[float, float]
    value: tuple[float, float]


#: Measured in the arm-free median overhead frame: salami hue ~0-12 sat 0.3-0.4 val 0.3-0.4;
#: cheese hue 40-80 sat 0.05-0.15 val 0.4-0.6; olives hue 30-45 sat 0.14-0.2 val 0.35-0.4.
TOPPINGS: tuple[Topping, ...] = (
    Topping("salami", ("salami", "pepperoni", "meat"), "salami", 0.045, (-25.0, 25.0), (0.22, 1.0), (0.12, 0.85)),
    Topping("cheese", ("cheese", "shred", "shreds"), "cheese", 0.05, (30.0, 95.0), (0.0, 0.3), (0.3, 1.0)),
    Topping("olives", ("olives", "olive"), "olives", 0.035, (25.0, 60.0), (0.08, 0.35), (0.25, 0.9)),
)


def plane_points(observation: CameraObservation, z: float = PLANE_Z_M) -> np.ndarray:
    """``(H, W, 3)`` base-frame points where each pixel's ray meets the plane ``z`` (NaN if it misses)."""
    height, width = np.asarray(observation.rgb).shape[:2]
    k = np.asarray(observation.intrinsics, dtype=np.float64)
    pose = observation.camera_pose
    w, x, y, zq = pose.quaternion_wxyz
    rotation = Rotation.from_quat([x, y, zq, w]).as_matrix()
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    rays = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1) @ rotation.T
    origin = np.asarray(pose.position, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z - origin[2]) / rays[..., 2]
    t = np.where((rays[..., 2] < -1e-6) & (t > 0), t, np.nan)
    return origin + rays * t[..., None]


class RealSceneSegmentationProvider:
    """``SegmentationProvider`` for the real station: tray position first, colour second."""

    def __init__(self, trays_xy: dict[str, tuple[float, float]] | None = None) -> None:
        """``trays_xy`` overrides where each tray disc is centred.

        By default a topping with enough demonstrated grasps is centred on where operators grasp
        it (``pick_spots.json``); the rest on the layout's by-eye tray position. Measured leave-
        one-out on the cheese picks that moves overhead error 44 -> 29 mm and wrist 31 -> 21 mm.
        """
        if trays_xy is None:
            layout = json.loads((REAL_DIR / "layout.json").read_text())["trays_xy"]
            trays_xy = {**{name: tuple(xy) for name, xy in layout.items()}, **load_pick_spots()}
        self.trays = trays_xy

    def _fail(self, message: str, **details: object) -> SegmentationSet:
        return SegmentationSet(
            ok=False, error=ApiError(code=ErrorCode.NO_SEGMENTATION, message=message, details=details)
        )

    def segment_text(self, observation: CameraObservation, query: str) -> SegmentationSet:
        text = str(query).lower()
        topping = next((t for t in TOPPINGS if any(a in text for a in t.aliases)), None)
        if topping is None:
            return self._fail(
                f"no real-scene topping matches {query!r}",
                vocabulary={t.name: list(t.aliases) for t in TOPPINGS},
            )
        cx, cy = self.trays[topping.tray]
        points = plane_points(observation)
        disc = np.hypot(points[..., 0] - cx, points[..., 1] - cy) < topping.radius_m  # NaN -> False
        if int(disc.sum()) < MIN_AREA_PX:
            return self._fail(
                f"the {topping.name} tray is not in this camera's view",
                hint="move the fingertip over the tray, or ask the overhead camera",
            )
        hsv = _hsv(np.asarray(observation.rgb))
        hue = (hsv[..., 0] + 180.0) % 360.0 - 180.0 if topping.hue_deg[0] < 0 else hsv[..., 0]
        looks_like = (
            (hue >= topping.hue_deg[0]) & (hue <= topping.hue_deg[1])
            & (hsv[..., 1] >= topping.saturation[0]) & (hsv[..., 1] <= topping.saturation[1])
            & (hsv[..., 2] >= topping.value[0]) & (hsv[..., 2] <= topping.value[1])
        )
        mask = ndimage.binary_opening(looks_like & disc, iterations=1)
        labels, count = ndimage.label(mask)
        areas = ndimage.sum(mask, labels, index=np.arange(1, count + 1)) if count else []
        coverage = float(mask.sum()) / float(disc.sum())
        found = [
            _segmentation(labels == (index + 1), topping.name, coverage, observation, wrist=_is_wrist(observation))
            for index in np.argsort(-np.asarray(areas, dtype=np.float64))
            if areas[index] >= MIN_AREA_PX
        ]
        if not found:
            return self._fail(
                f"no {topping.name}-looking pixels inside its tray from this camera",
                hint="the arm may be over the tray, or the tray may be out of view",
            )
        found = _order(found, text, observation)
        return SegmentationSet(
            ok=True,
            segmentations=found,
            diagnostics={"class": topping.name, "instances": len(found), "method": "tray-roi", "coverage": round(coverage, 3)},
        )

    def segment_points(self, observation, points_px, *, point_labels=None) -> SegmentationSet:
        return self._fail("point prompts are not supported by the real-scene segmenter")


def _is_wrist(observation: CameraObservation) -> bool:
    return str(observation.frame).lower().startswith("wrist")


#: Overhead reliability, fixed. On the real cheese picks the overhead coverage score did not
#: predict error at all (r = +0.24; mid-score frames were the most accurate), so scoring by it only
#: made the creep freeze on frames as good as the ones it acted on. The wrist score does track
#: error (r = -0.37: 26 mm below 0.4, 8 mm above 0.7), so there it is kept.
OVERHEAD_SCORE = 0.7


def _segmentation(
    component: np.ndarray, label: str, coverage: float, observation: CameraObservation, *, wrist: bool
) -> Segmentation:
    ys, xs = np.nonzero(component)
    return Segmentation(
        mask=component,
        label=label,
        score=float(np.clip(0.25 + 0.6 * coverage, 0.0, 0.85)) if wrist else OVERHEAD_SCORE,
        camera_name="camera",
        frame=observation.frame,
        box_xyxy=np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64),
    )


__all__ = ["RealSceneSegmentationProvider", "TOPPINGS", "plane_points"]

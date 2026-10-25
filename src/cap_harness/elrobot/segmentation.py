"""Colour-based segmentation for the pizza scene -- a stand-in for SAM3 in simulation.

SAM3 needs a GPU service, gated weights and a separate environment. The pizza
scene is flat-shaded geometry in a handful of distinct colours, so a classical
hue/saturation segmenter localizes its ingredients well enough to run the whole
agent loop with nothing else installed. It uses **pixels only** -- no scene
state, no object poses -- so what an agent can localize is what a camera could
see, and swapping SAM3 in later changes quality, not the interface.

It is NOT general: the vocabulary below is tuned to ``elrobot_table.xml`` and an
unknown query is refused with the vocabulary rather than guessed at. Instances
are 4-connected components of the colour class, largest first, so the first
segmentation of "salami" is one disc (or a few touching ones), not the whole bin.
Touching discs are split at their necks (see ``_split_touching``).

``cap_localize_object`` returns only the first instance, so the query itself picks
which one. A qualifier word reorders the instances: ``center``/``middle`` (closest to
the centroid of all of them), ``left``/``right``/``top``/``bottom`` (extreme in the
image, as the camera sees it -- the overhead view is mirrored, image-left is +x in
the base frame), ``smallest``. With no qualifier the largest, least-occluded one
comes first.

``at X,Y`` (base-frame metres, e.g. ``salami at 0.22,0.25``) names the instance by where it is
instead of how it looks: the one whose projection lands nearest that point in this camera. Use it
to ask the wrist camera for the very item the overhead camera found, since each camera's own
"center" or "left" can be a different slice.

Items lying on the plate are a different population from items waiting in a bin, so a
query for an ingredient skips whatever is on the plate -- "salami center" stays
"the middle of the bin" as the plate fills up. Ask for "salami on the plate" to get
the placed ones instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
from scipy import ndimage

from cap_harness.contracts import CameraObservation, Segmentation, SegmentationSet
from cap_harness.errors import ApiError, ErrorCode

MIN_AREA_PX = 25


@dataclass(frozen=True, slots=True)
class ColorClass:
    name: str
    aliases: tuple[str, ...]
    hue_deg: tuple[float, float]
    saturation: tuple[float, float]
    value: tuple[float, float]
    split: bool = True  # separate touching instances; False keeps a region whole


#: Ranges are measured from rendered frames of the shipped scene (see
#: tests/test_elrobot_segmentation.py), not read from the XML: lighting and
#: shading shift the authored colours.
CLASSES: tuple[ColorClass, ...] = (
    ColorClass(
        "salami",
        ("salami", "pepperoni", "red disc", "meat"),
        (-12.0, 12.0),
        (0.45, 1.0),
        (0.2, 1.0),
    ),
    ColorClass("cheese", ("cheese", "shred", "yellow"), (38.0, 62.0), (0.55, 1.0), (0.45, 1.0)),
    ColorClass(
        "plate", ("plate", "pizza", "dough", "base"), (38.0, 70.0), (0.1, 0.45), (0.85, 1.0)
    ),
)


def _hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB (uint8) to (hue degrees, saturation, value)."""
    x = rgb.astype(np.float64) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    top, bottom = x.max(axis=-1), x.min(axis=-1)
    delta = top - bottom
    safe = np.where(delta == 0, 1.0, delta)
    hue = np.where(
        top == r,
        ((g - b) / safe) % 6.0,
        np.where(top == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
    )
    hue = np.where(delta == 0, 0.0, hue) * 60.0
    saturation = np.where(top == 0, 0.0, delta / np.where(top == 0, 1.0, top))
    return np.stack([hue, saturation, top], axis=-1)


def _class_mask(hsv: np.ndarray, cls: ColorClass) -> np.ndarray:
    hue = (hsv[..., 0] + 180.0) % 360.0 - 180.0  # so red wraps to a single signed range
    return (
        (hue >= cls.hue_deg[0])
        & (hue <= cls.hue_deg[1])
        & (hsv[..., 1] >= cls.saturation[0])
        & (hsv[..., 1] <= cls.saturation[1])
        & (hsv[..., 2] >= cls.value[0])
        & (hsv[..., 2] <= cls.value[1])
    )


def _split_touching(mask: np.ndarray, core_fraction: float = 0.6) -> np.ndarray:
    """Label instances, separating blobs that touch at a neck (adjacent discs).

    Each connected component is cut at its distance-transform necks: pixels deeper
    than ``core_fraction`` of the component's own maximum depth seed instances, and
    every other pixel joins its nearest seed. A single convex region has one seed
    and comes back whole; a clump of equal discs comes back as one label per disc.
    """
    components, count = ndimage.label(mask)
    labels = np.zeros(mask.shape, dtype=np.int32)
    depth = ndimage.distance_transform_edt(mask)
    next_label = 0
    for index in range(1, count + 1):
        region = components == index
        cores, core_count = ndimage.label(region & (depth > core_fraction * depth[region].max()))
        if core_count <= 1:
            next_label += 1
            labels[region] = next_label
            continue
        _, nearest = ndimage.distance_transform_edt(cores == 0, return_indices=True)
        assigned = cores[nearest[0], nearest[1]]
        labels[region] = assigned[region] + next_label
        next_label += core_count
    return labels


def _plate_region(hsv: np.ndarray) -> np.ndarray:
    """The plate and anything sitting on it: the plate colour with its holes filled."""
    plate = next(c for c in CLASSES if c.name == "plate")
    mask = ndimage.binary_closing(_class_mask(hsv, plate), iterations=3)
    # Dilated so that rim shading, which blends the pizza into the table, still counts as pizza.
    return ndimage.binary_dilation(ndimage.binary_fill_holes(mask), iterations=5)


def _on_plate(segmentation: Segmentation, region: np.ndarray) -> bool:
    rows, cols = np.nonzero(segmentation.mask)
    return bool(region[int(rows.mean()), int(cols.mean())])


_AT_POINT = re.compile(r"\bat\s*\(?\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)?")
#: Height above the table of the point an ``at X,Y`` query is projected from, metres.
AT_POINT_Z_M = 0.02


def _project_to_pixels(observation: CameraObservation, point_base: np.ndarray) -> np.ndarray | None:
    """``(row, col)`` of a base-frame point in this camera, or None if it is behind it."""
    from scipy.spatial.transform import Rotation

    pose = observation.camera_pose
    w, x, y, z = pose.quaternion_wxyz
    rotation = Rotation.from_quat([x, y, z, w]).as_matrix()
    in_camera = rotation.T @ (point_base - np.asarray(pose.position, dtype=np.float64))
    if in_camera[2] <= 1e-6:
        return None
    k = np.asarray(observation.intrinsics, dtype=np.float64)
    return np.array(
        [k[1, 1] * in_camera[1] / in_camera[2] + k[1, 2], k[0, 0] * in_camera[0] / in_camera[2] + k[0, 2]]
    )


def _order(
    found: list[Segmentation], text: str, observation: CameraObservation | None = None
) -> list[Segmentation]:
    """Reorder instances by a spatial qualifier in the query; stable when there is none."""
    if len(found) < 2:
        return found
    centres = np.array([np.argwhere(s.mask).mean(axis=0) for s in found])  # (row, col)
    words = set(text.lower().replace(",", " ").split())
    at = _AT_POINT.search(text.lower())
    if at and observation is not None and observation.camera_pose is not None:
        target = _project_to_pixels(
            observation, np.array([float(at.group(1)), float(at.group(2)), AT_POINT_Z_M])
        )
        if target is not None:
            return [found[i] for i in np.argsort(np.linalg.norm(centres - target, axis=1), kind="stable")]
    if words & {"center", "centre", "middle", "central"}:
        key = np.linalg.norm(centres - centres.mean(axis=0), axis=1)
    elif words & {"left", "leftmost"}:
        key = centres[:, 1]
    elif words & {"right", "rightmost"}:
        key = -centres[:, 1]
    elif words & {"top", "upper", "topmost", "far"}:
        key = centres[:, 0]
    elif words & {"bottom", "lower", "bottommost", "near"}:
        key = -centres[:, 0]
    elif words & {"smallest"}:
        key = np.array([s.mask.sum() for s in found], dtype=np.float64)
    else:
        return found
    return [found[i] for i in np.argsort(key, kind="stable")]


def _resolve(query: str) -> ColorClass | None:
    text = query.lower()
    for cls in CLASSES:
        if cls.name in text or any(alias in text for alias in cls.aliases):
            return cls
    return None


class ColorSegmentationProvider:
    """``SegmentationProvider`` over the scene's colour classes."""

    def _fail(self, message: str, **details: object) -> SegmentationSet:
        return SegmentationSet(
            ok=False,
            error=ApiError(code=ErrorCode.NO_SEGMENTATION, message=message, details=details),
        )

    def _instances(
        self, observation: CameraObservation, cls: ColorClass, label: str
    ) -> list[Segmentation]:
        mask = _class_mask(_hsv(np.asarray(observation.rgb)), cls)
        mask = ndimage.binary_opening(mask, iterations=1)
        labels = _split_touching(mask) if cls.split else ndimage.label(mask)[0]
        count = int(labels.max())
        if count == 0:
            return []
        areas = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
        return [
            _segmentation(labels == (index + 1), label, float(areas[index]), observation)
            for index in np.argsort(-areas)
            if areas[index] >= MIN_AREA_PX
        ]

    def segment_text(self, observation: CameraObservation, query: str) -> SegmentationSet:
        cls = _resolve(str(query))
        if cls is None:
            return self._fail(
                f"no colour class matches {query!r}",
                vocabulary={c.name: list(c.aliases) for c in CLASSES},
            )
        instances = self._instances(observation, cls, str(query))
        if cls.name != "plate" and instances:
            region = _plate_region(_hsv(np.asarray(observation.rgb)))
            wanted_on_plate = "plate" in str(query).lower()
            instances = [s for s in instances if _on_plate(s, region) == wanted_on_plate]
        instances = _order(instances, str(query), observation)
        if not instances:
            return self._fail(
                f"no {cls.name}-coloured region is visible from this camera",
                hint="it may be occluded by the arm or outside the view; try another camera",
            )
        return SegmentationSet(
            ok=True,
            segmentations=instances,
            diagnostics={"class": cls.name, "instances": len(instances), "method": "hsv"},
        )

    def segment_points(
        self,
        observation: CameraObservation,
        points_px: np.ndarray,
        *,
        point_labels: np.ndarray | None = None,
    ) -> SegmentationSet:
        points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        if point_labels is not None:
            positives = points[np.asarray(point_labels).reshape(-1) > 0]
            points = positives if len(positives) else points
        hsv = _hsv(np.asarray(observation.rgb))
        height, width = hsv.shape[:2]
        u, v = int(round(points[0, 0])), int(round(points[0, 1]))
        if not (0 <= u < width and 0 <= v < height):
            return self._fail("prompt point lies outside the image", point=[u, v])
        for cls in CLASSES:
            if _class_mask(hsv, cls)[v, u]:
                instances = self._instances(observation, cls, cls.name)
                for segmentation in instances:
                    if segmentation.mask[v, u]:
                        return SegmentationSet(ok=True, segmentations=[segmentation])
        return self._fail("the prompt point is not on a known ingredient colour", point=[u, v])


def _segmentation(
    component: np.ndarray, label: str, area: float, observation: CameraObservation
) -> Segmentation:
    ys, xs = np.nonzero(component)
    return Segmentation(
        mask=component,
        label=label,
        score=float(min(0.95, 0.5 + area / 4000.0)),
        camera_name="camera",
        frame=observation.frame,
        box_xyxy=np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64),
    )


__all__ = ["CLASSES", "ColorClass", "ColorSegmentationProvider"]

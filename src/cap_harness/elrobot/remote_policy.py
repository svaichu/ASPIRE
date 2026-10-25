"""The GR00T-protocol face of a :class:`~cap_harness.elrobot.remote.RemoteLink`.

Observation (bridge -> server), the Isaac-GR00T layout::

    state:    {"state": float32 (1, 1, 8)}        # position_norm, gripper last
    video:    {alias: uint8 (1, 1, H, W, 3)}      # alias in {"overhead"}, see RemoteCameraModel
    depth:    {alias: float32 (1, 1, H, W)}       # optional, metres
    language: {"task": [[str]]}                   # optional, informational

``options`` (bridge -> server): ``{"ack": {"segment", "chunk"}, "fault": str}`` reports the
chunk just executed, or why it was abandoned.

Action (server -> bridge): ``{"action": float32 (1, N, 8)}`` absolute ``position_norm`` targets
at ``info["dt"]`` spacing. ``N = 0`` with ``info["idle"]`` means hold and poll again;
``info["cancel"]`` means drop whatever is queued and stop moving.

Requires the ``policy_server`` package (``/workspace/policy-server``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from policy_server import BasePolicy

from cap_harness.elrobot.remote import RemoteLink


def _last(array: Any) -> np.ndarray:
    """Drop the batch and time axes: the newest entry of a (B, T, ...) block."""
    array = np.asarray(array)
    return array[0, -1] if array.ndim >= 3 else array


class AspirePolicy(BasePolicy):
    def __init__(self, link: RemoteLink, *, aliases: tuple[str, ...], **kw: Any) -> None:
        super().__init__(**kw)
        self.link = link
        self.aliases = aliases

    def check_observation(self, observation: dict[str, Any]) -> None:
        state = np.asarray(observation.get("state", {}).get("state", ()))
        if state.size < 8 or not np.isfinite(state).all():
            raise ValueError("observation['state']['state'] must hold 8 finite position_norm values")
        for alias, frames in observation.get("video", {}).items():
            frames = np.asarray(frames)
            if alias not in self.aliases:
                raise ValueError(f"unknown camera {alias!r}; this station has {list(self.aliases)}")
            if frames.dtype != np.uint8 or frames.shape[-1] != 3:
                raise ValueError(f"video[{alias!r}] must be uint8 (..., H, W, 3)")

    def _get_action(self, observation, options=None):
        options = options or {}
        # Ack first: it concerns the chunk executed *before* this observation was taken.
        self.link.handle_ack(options.get("ack"), options.get("fault"))
        self.link.update_state(
            _last(observation["state"]["state"]).reshape(-1)[:8],
            {a: _last(v).astype(np.uint8, copy=False) for a, v in observation.get("video", {}).items()},
            {a: _last(v).astype(np.float64) for a, v in observation.get("depth", {}).items()},
        )
        targets, info = self.link.next_command()
        return {"action": targets.astype(np.float32)[None]}, info

    def reset(self, options=None):
        return {}

    def get_modality_config(self):
        return {
            "state": {"keys": ["state"], "dim": 8, "unit": "position_norm, gripper last"},
            "video": {"keys": list(self.aliases)},
            "action": {"keys": ["action"], "dim": 8, "unit": "position_norm, absolute"},
            "protocol": "aspire-remote-1",
        }

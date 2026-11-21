"""Live CAP MCP server for the GR-3 humanoid simulator.

Serves the same MCP surface as the YAM System 2 server (``cap_*`` primitives, ``observe``,
``get_job``, reviewed ``cap_program_*`` programs) over one persistent GR-3 pick scene::

    MUJOCO_GL=egl /root/gr3-humanoid/.venv/bin/python -m cap_harness.gr3.live --port 8222

MuJoCo rendering contexts are bound to the thread that created them, while the agent session
runs each job on a fresh thread. Every simulator access therefore goes through one dedicated
thread (:class:`SimThread`); jobs, MCP reads and the camera monitor are serialized on it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import functools
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

import numpy as np

from cap_harness.gr3.adapter import Gr3Adapter

logger = logging.getLogger("cap_harness.gr3.live")

GR3_MCP_DIRECT_TOOLS = frozenset(
    {
        "get_robot_state",
        "get_task_context",
        "localize_object",
        "move_to_joints",
        "move_to_pose",
        "set_gripper",
        "open_gripper",
        "close_gripper",
        "gr3.get_controller_metadata",
        "gr3.get_task_metadata",
    }
)
GR3_MCP_READ_ONLY_TOOLS = frozenset(
    {
        "get_robot_state",
        "get_task_context",
        "localize_object",
        "gr3.get_controller_metadata",
        "gr3.get_task_metadata",
    }
)
# The in-process planner on the simulator's own model, as in ``cap-harness run --benchmark gr3``.
GR3_MOTION_STRATEGY = {
    "ik_solver": "curobo",
    "trajectory_planner": "curobo",
    "pose_planner": "curobo-integrated",
}
LIVE_HORIZON = 10**9
"""A live session has no episode horizon; ``reset_scene`` starts a new scene instead."""


class SimThread:
    """Run every simulator call on one thread; calls from that thread run inline."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gr3-sim", initializer=self._mark
        )

    def _mark(self) -> None:
        self._local.owner = True

    def call(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if getattr(self._local, "owner", False):
            return function(*args, **kwargs)
        return self._pool.submit(function, *args, **kwargs).result()

    def bind(self, function: Callable[..., Any]) -> Callable[..., Any]:
        """Keep the signature and annotations the registry and MCP schemas read."""

        @functools.wraps(function)
        def bound(*args: Any, **kwargs: Any) -> Any:
            return self.call(function, *args, **kwargs)

        return bound

    def close(self) -> None:
        self._pool.shutdown(wait=True)


class _SimBoundApi:
    """Present the CAP API to the session with every method marshalled to the sim thread."""

    def __init__(self, api: Any, sim: SimThread) -> None:
        self._api = api
        self._sim = sim

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._api, name)
        return self._sim.bind(value) if callable(value) else value


class Gr3LiveAdapter(Gr3Adapter):
    """GR-3 adapter with the station-monitor hook the live session polls."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._scene_generation = 0
        self._monitor_key: tuple[int, int] | None = None
        self._monitor_frames: dict[str, np.ndarray] = {}

    def reset(self, task_ref: Any, seed: int) -> Any:
        observation = super().reset(task_ref, seed)
        self._scene_generation += 1
        return observation

    def monitor_snapshot(self) -> dict[str, Any]:
        # The scene only changes when the simulator steps: re-render on a new step or scene.
        key = (self._scene_generation, self._elapsed_steps)
        if key != self._monitor_key:
            self._monitor_frames = {
                name: self.render_rgb(name) for name in self.metadata.camera_names
            }
            self._monitor_key = key
        state = self.get_robot_state()
        return {
            "cameras": self._monitor_frames,
            "camera_metadata": {
                name: {"sequence": self._elapsed_steps, "age_ms": 0.0, "stale": False, "missing": False}
                for name in self._monitor_frames
            },
            "state": {
                **{f"{arm}_joint_pos": np.asarray(q, dtype=float) for arm, q in state.joint_positions.items()},
                **{
                    f"{arm}_gripper_pos": np.asarray([g], dtype=float)
                    for arm, g in state.gripper_positions.items()
                },
            },
            "timestamp_s": time.time(),
            "observation_seq": self._elapsed_steps,
        }


class JobVideoRecorder:
    """Write one MP4 per camera for every motion job the live session runs.

    The session brackets each job with ``begin_operation``/``end_operation``; the adapter calls
    ``after_step`` with every simulator step's observation, whose cameras are already rendered.
    Jobs that never step the simulator (reads, resets) leave no files.
    """

    def __init__(self, root: Path, *, fps: Callable[[], float]) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._lock = threading.Lock()
        self._sequence = 0
        self._segment: Path | None = None
        self._label = ""
        self._writers: dict[str, Any] = {}
        self._frames = 0

    # -------------------------------------------------- session operation controller

    def begin_operation(self, *, kind: str, label: str) -> None:
        with self._lock:
            self._close_segment()
            self._sequence += 1
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")[:60] or kind
            stamp = datetime.now(timezone.utc).strftime("%H%M%S")
            self._segment = self.root / f"{self._sequence:03d}_{stamp}_{slug}"
            self._label = label

    def end_operation(self, *, kind: str, label: str) -> None:
        with self._lock:
            self._close_segment()

    def close(self) -> None:
        with self._lock:
            self._close_segment()

    # -------------------------------------------------- adapter run observer

    def on_reset(self, observation: Any, metadata: Any) -> None:
        pass

    def before_step(self, action: Any) -> None:
        pass

    def after_step(self, action: Any, result: Any) -> None:
        cameras = getattr(getattr(result, "observation", None), "cameras", None) or {}
        with self._lock:
            if self._segment is None or not cameras:
                return
            import imageio.v2 as imageio

            for name, camera in cameras.items():
                writer = self._writers.get(name)
                if writer is None:
                    self._segment.mkdir(parents=True, exist_ok=True)
                    writer = imageio.get_writer(
                        self._segment / f"{name}.mp4",
                        fps=float(self._fps()),
                        codec="libx264",
                        macro_block_size=1,
                    )
                    self._writers[name] = writer
                writer.append_data(np.asarray(camera.rgb, dtype=np.uint8))
            self._frames += 1

    def _close_segment(self) -> None:
        if self._writers:
            for writer in self._writers.values():
                try:
                    writer.close()
                except Exception as exc:
                    logger.warning("video finalize failed: %s", exc)
            entry = {
                "job": self._label,
                "dir": str(self._segment),
                "videos": sorted(f"{name}.mp4" for name in self._writers),
                "frames": self._frames,
                "fps": float(self._fps()),
            }
            with (self.root / "videos.jsonl").open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
            logger.info("saved %d-frame video for %s: %s", self._frames, self._label, self._segment)
        self._writers = {}
        self._frames = 0
        self._segment = None
        self._label = ""


class _ScenePlanner:
    """The GR-3 planner caches the scene's MuJoCo model; rebuild it after every scene reset."""

    def __init__(self, adapter: Gr3LiveAdapter) -> None:
        self._adapter = adapter
        self._generation: int | None = None
        self._planner: Any = None

    def _current(self) -> Any:
        if self._generation != self._adapter._scene_generation:
            self._planner = self._adapter.planner()
            self._generation = self._adapter._scene_generation
        return self._planner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._current(), name)


class Gr3LiveRuntime:
    """One persistent GR-3 scene, its CAP API, and the agent session over it."""

    def __init__(
        self,
        *,
        task: str = "pick_red_cube",
        seed: int = 1,
        camera_width: int = 800,
        camera_height: int = 512,
        sam3_url: str | None = None,
        record_dir: Path | None = None,
    ) -> None:
        from cap_harness.agent.program_catalog import CapProgramCatalog, CapProgramDefinition
        from cap_harness.agent.session import LiveAgentSession
        from cap_harness.api import CapApi
        from cap_harness.contracts import MotionStrategy
        from cap_harness.gr3.registry import Gr3TaskRegistry
        from cap_harness.providers.graspnet.client import ContactGraspNetProvider
        from cap_harness.providers.sam3.client import DEFAULT_SAM3_URL, Sam3Provider
        from cap_harness.registry import GR3_PUBLIC_TOOL_NAMES, ToolRegistry

        self.sim = SimThread()
        self.registry = Gr3TaskRegistry()
        self.videos: JobVideoRecorder | None = None
        if record_dir is not None:
            self.videos = JobVideoRecorder(record_dir, fps=lambda: self.adapter.control_frequency)
        self.adapter = Gr3LiveAdapter(
            registry=self.registry,
            camera_width=camera_width,
            camera_height=camera_height,
            horizon=LIVE_HORIZON,
            run_observer=self.videos,
        )
        self.task = task
        self.seed = seed
        self.sim.call(self.adapter.reset, self.registry.resolve(task), seed)

        def build_api() -> Any:
            planner = _ScenePlanner(self.adapter)
            return CapApi(
                self.adapter,
                segmentation_provider=Sam3Provider(
                    base_url=sam3_url or os.environ.get("CAP_HARNESS_SAM3_URL", DEFAULT_SAM3_URL)
                ),
                grasp_providers={"contact-graspnet": ContactGraspNetProvider()},
                ik_providers={"curobo": planner},
                trajectory_planning_providers={"curobo": planner},
                integrated_pose_planning_providers={"curobo-integrated": planner},
                default_camera="rs_agentview",
                default_motion_strategy=MotionStrategy(**GR3_MOTION_STRATEGY),
            )

        api = self.sim.call(build_api)
        native = api.register_tools(ToolRegistry(public_extension_allowlist=GR3_PUBLIC_TOOL_NAMES))
        registry = ToolRegistry(public_extension_allowlist=GR3_PUBLIC_TOOL_NAMES)
        for spec in native.all_specs():
            registry.register(
                self.sim.bind(spec.function),
                name=spec.name,
                layer=spec.layer,
                capability=spec.capability,
                public=spec.public,
            )
        catalog = CapProgramCatalog(
            (
                CapProgramDefinition("observe", "Read the current robot state.", False),
                CapProgramDefinition(
                    "go_home", "Open the right hand and return the right arm to its reset pose."
                ),
                CapProgramDefinition(
                    "pick_red_cube",
                    "Localize the red cube, grasp it with the right hand (tool tilted 20 deg), "
                    "and lift it 0.22 m above its grasp height. Verify the hold from the images.",
                ),
                CapProgramDefinition(
                    "pick_green_cube",
                    "Localize the green cube, grasp it with the right hand (tool tilted 20 deg), "
                    "and lift it 0.22 m above its grasp height. Verify the hold from the images.",
                ),
            ),
            package="cap_harness.gr3.programs",
        )
        self.session = LiveAgentSession(
            _SimBoundApi(api, self.sim),
            registry,
            program_catalog=catalog,
            event_sink=self._log_event,
            operation_controller=self.videos,
        )

    # ------------------------------------------------------------------ operator-side evidence

    def _log_event(self, event: Any) -> None:
        """Log native task success to the server console only; it is never sent to the agent."""
        logger.info("[%s] %s", event.role, event.text)
        if event.role == "tool" and event.text.startswith("finished "):
            try:
                success = self.sim.call(self.adapter.check_success)
                logger.info(
                    "operator evidence: scene task %s (checks only its own cube) native success=%s",
                    self.task,
                    success,
                )
            except Exception as exc:  # evidence logging must never fail a job
                logger.warning("native success check failed: %s", exc)

    # ------------------------------------------------------------------ extra MCP tools

    def _reset_scene(self, args: dict[str, Any]) -> dict[str, Any]:
        task = str(args.get("task", self.task))
        seed = int(args.get("seed", self.seed))
        metadata = self.registry.resolve(task)

        def reset() -> dict[str, Any]:
            self.adapter.reset(metadata, seed)
            self.task, self.seed = task, seed
            return {"ok": True, "task": task, "seed": seed, "language": metadata.language}

        job_id = self.session._start_job(
            kind="tool",
            label=f"reset_scene {task} seed={seed}",
            phase="perceive",
            operation=lambda: self.sim.call(reset),
        )
        job = self.session.wait(job_id, timeout_s=None)
        return {"ok": bool(job and not job["error"]), "job": job}

    def extra_tools(self) -> dict[str, Any]:
        return {
            "cap_sim_reset_scene": (
                "Rebuild the GR-3 simulator scene (both cubes return to their seeded start "
                "and the robot to its reset pose). Simulation only.",
                {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "enum": list(self.registry.available_tasks)},
                        "seed": {"type": "integer", "minimum": 0},
                    },
                    "additionalProperties": False,
                },
                self._reset_scene,
                True,
            )
        }

    def bridge_options(self) -> dict[str, Any]:
        return {
            "direct_tools": GR3_MCP_DIRECT_TOOLS,
            "read_only_tools": GR3_MCP_READ_ONLY_TOOLS,
            "motion_strategy": GR3_MOTION_STRATEGY,
            "extra_tools": self.extra_tools(),
        }

    def close(self) -> None:
        self.session.close()
        try:
            self.sim.call(self.adapter.close)
        finally:
            self.sim.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="pick_red_cube", help="initial GR-3 task scene")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8222)
    parser.add_argument("--camera-width", type=int, default=800)
    parser.add_argument("--camera-height", type=int, default=512)
    parser.add_argument("--sam3-url", help="default: $CAP_HARNESS_SAM3_URL or the shared service")
    parser.add_argument(
        "--record-dir",
        type=Path,
        help="where per-job camera videos are saved (default: outputs/gr3-live/<UTC time>)",
    )
    parser.add_argument("--no-video", action="store_true", help="do not record videos")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    import uvicorn

    from cap_harness.agent.server import create_app

    runtime = Gr3LiveRuntime(
        task=args.task,
        seed=args.seed,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        sam3_url=args.sam3_url,
        record_dir=None
        if args.no_video
        else args.record_dir
        or Path("outputs/gr3-live") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    try:
        app = create_app(runtime.session, **runtime.bridge_options())
        logger.info("GR-3 CAP MCP at http://%s:%d/mcp/", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

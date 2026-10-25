"""System 2 MCP server for the ElRobot pizza-table simulation.

Same session, MCP bridge and operator steering as the YAM server; only the
embodiment underneath differs. Serves MCP at ``http://127.0.0.1:8222/mcp/``.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from cap_harness import elrobot as _elrobot  # noqa: F401  (sets MUJOCO_GL before mujoco loads)

from .embedded import EmbeddedAgentRuntime
from .server import create_app


def build_elrobot_runtime(
    env: Any,
    *,
    allow_motion: bool = False,
    event_sink: Any = None,
    tool_result_sink: Any = None,
    **api_options: Any,
) -> tuple[Any, EmbeddedAgentRuntime]:
    """Attach to a caller-owned station; construction performs no motion."""
    from cap_harness.api import CapApi
    from cap_harness.contracts import TaskContext
    from cap_harness.elrobot.adapter import ElRobotAdapter
    from cap_harness.registry import ELROBOT_PUBLIC_TOOL_NAMES, ToolRegistry
    from cap_harness.yam_real.control.runtime import RuntimeControl
    from cap_harness.yam_real.live_adapter import LiveYamAdapter

    raw = ElRobotAdapter(
        env,
        allow_physical_motion=allow_motion,
        task_context=TaskContext(
            suite="system2",
            task_id=0,
            task_name="system2",
            language="Follow the operator goal.",
            family="elrobot",
        ),
    )
    # The live wrapper and control worker are embodiment-neutral: one serialized
    # actuation thread, and a monitor snapshot built from `camera_aliases`.
    control = RuntimeControl(raw)
    adapter = LiveYamAdapter(raw, control)
    try:
        api = CapApi(adapter, default_camera=env.config.camera, **api_options)
        registry = api.register_tools(
            ToolRegistry(public_extension_allowlist=ELROBOT_PUBLIC_TOOL_NAMES)
        )
        runtime = EmbeddedAgentRuntime(
            api,
            registry,
            operation_controller=control,
            event_sink=event_sink,
            tool_result_sink=tool_result_sink,
        )
    except BaseException:
        control.close()
        raise
    return adapter, runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir", type=Path, help="Directory with elrobot_table.xml + home.json"
    )
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8222)
    parser.add_argument(
        "--sam3-url",
        help="Use a SAM3 service instead of the built-in colour segmenter",
    )
    args = parser.parse_args(argv)

    import uvicorn

    from cap_harness.elrobot.ik import ElRobotIKProvider
    from cap_harness.elrobot.kinematics import ElRobotKinematics
    from cap_harness.elrobot.station import build_sim_station

    with ExitStack() as stack:
        env = build_sim_station(scene_directory=str(args.scene_dir) if args.scene_dir else None)
        stack.callback(env.close)
        options: dict[str, Any] = {
            "ik_providers": {"mink": ElRobotIKProvider(ElRobotKinematics(env.config))}
        }
        if args.sam3_url:
            from cap_harness.providers.sam3 import Sam3Provider

            options["segmentation_provider"] = Sam3Provider(base_url=args.sam3_url)
        else:
            from cap_harness.elrobot.segmentation import ColorSegmentationProvider

            options["segmentation_provider"] = ColorSegmentationProvider()
        _, runtime = build_elrobot_runtime(env, allow_motion=args.allow_motion, **options)
        stack.callback(runtime.close)
        uvicorn.run(create_app(runtime.session), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

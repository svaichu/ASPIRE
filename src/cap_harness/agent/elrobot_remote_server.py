"""System 2 for a *real* ElRobot, served over a GR00T-style policy connection.

Runs the MCP session of :mod:`.elrobot_server` with a remote plant. The machine wired to
the arm runs ``inference-aspire`` (in ``inference-client``), which connects to ``--policy-port``
and executes whatever the agent's tools command. Nothing is actuated until a bridge is
connected, and not at all without ``--allow-motion``.

    python -m cap_harness.agent.elrobot_remote_server --allow-motion
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import threading
from typing import Any

from .elrobot_server import build_elrobot_runtime
from .server import create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--host", default="127.0.0.1", help="MCP host")
    parser.add_argument("--port", type=int, default=8222, help="MCP port")
    parser.add_argument("--policy-host", default="0.0.0.0", help="bridge-facing ZMQ bind address")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--api-token", help="require this token from the bridge")
    parser.add_argument("--layout", help="layout.json holding the real camera fit")
    parser.add_argument("--sam3-url")
    args = parser.parse_args(argv)

    import uvicorn
    from policy_server import PolicyServer

    from cap_harness.elrobot.config import load_config
    from cap_harness.elrobot.ik import ElRobotIKProvider
    from cap_harness.elrobot.kinematics import ElRobotKinematics
    from cap_harness.elrobot.remote import RealCameraModel, RemoteElRobotStation, RemoteLink
    from cap_harness.elrobot.remote_policy import AspirePolicy

    config = load_config("pizza")
    link = RemoteLink()
    cameras = {"overhead": RealCameraModel.from_layout(config, args.layout)}
    env = RemoteElRobotStation(
        config, link, cameras,
        kinematics=ElRobotKinematics(config),  # its own instance: FK must not reseed the IK solver
    )

    with ExitStack() as stack:
        stack.callback(env.close)
        options: dict[str, Any] = {
            "ik_providers": {"mink": ElRobotIKProvider(ElRobotKinematics(config))}
        }
        if args.sam3_url:
            from cap_harness.providers.sam3 import Sam3Provider

            options["segmentation_provider"] = Sam3Provider(base_url=args.sam3_url)
        else:
            from cap_harness.elrobot.segmentation import ColorSegmentationProvider

            options["segmentation_provider"] = ColorSegmentationProvider()
        _, runtime = build_elrobot_runtime(env, allow_motion=args.allow_motion, **options)
        stack.callback(runtime.close)

        server = PolicyServer(
            AspirePolicy(link, aliases=env.camera_aliases),
            host=args.policy_host, port=args.policy_port, api_token=args.api_token,
        )
        thread = threading.Thread(target=server.run, name="policy-server", daemon=True)
        thread.start()
        print(f"policy endpoint on {args.policy_host}:{args.policy_port}; waiting for the bridge")
        try:
            uvicorn.run(create_app(runtime.session), host=args.host, port=args.port)
        finally:
            link.close()
            server.running = False

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

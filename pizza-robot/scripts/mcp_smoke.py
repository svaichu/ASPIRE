"""Smoke test over the real MCP interface: one slice, bin to plate, exactly as an agent would.

Needs ``scripts/start_pizza_system2.sh --allow-motion`` running. It uses only the MCP tools
and the two pure-geometry CLI helpers the skills name, so a pass shows the whole surface --
server, tool schemas, perception, planner, motion, gripper -- works end to end.

    .venv-system2/bin/python pizza-robot/scripts/mcp_smoke.py [--url http://127.0.0.1:8222/mcp/]
"""

import argparse
import asyncio
import json
import subprocess
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

SLOT = (0.22, 0.275)  # a reachable spot on the pizza (see the task context)
PIZZA, PIZZA_RADIUS = (0.26, 0.235), 0.178
MAX_STEPS = 800  # moves run at real-arm speeds; the default 120 ticks is too short


def planner(x, y, z, hover):
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "cap_harness.elrobot.approach",
            str(x),
            str(y),
            str(z),
            "--hover",
            str(hover),
            "--limit",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    approaches = json.loads(out.stdout)["approaches"]
    if not approaches or not approaches[0]["clear"]:
        raise SystemExit(f"no clear approach to ({x:.3f}, {y:.3f}, {z})")
    return approaches[0]


async def main(url):
    async with (
        streamablehttp_client(url) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        async def call(name, arguments=None):
            result = await session.call_tool(name, arguments or {})
            data = json.loads(next(c.text for c in result.content if c.type == "text"))
            job = data.get("job", data)
            outcome = job.get("result", data.get("result", data)) if isinstance(job, dict) else data
            ok = bool(data.get("ok", True)) and not result.isError
            return ok, data, outcome

        ok, data, _ = await call("cap_program_go_home")
        assert ok, f"home failed: {data}"
        ok, data, _ = await call("cap_localize_object", {"query": "salami center"})
        assert ok, f"localize failed: {data}"
        pose = data["result"]["geometry"]["pose"]["position"]
        x, y = pose[0], pose[1]
        print(f"localized a slice at ({x:.3f}, {y:.3f})")

        approach = planner(x, y, 0.016, 0.07)
        print(f"approach: tilt {approach['tilt_deg']:.0f} lean {approach['lean_deg']:.0f}")
        assert (await call("cap_set_gripper", {"position": 1.0}))[0]
        for pose_key in ("hover_pose", "grasp_pose"):
            ok, data, _ = await call(
                "cap_move_to_pose", {"target_pose": approach[pose_key], "max_steps": MAX_STEPS}
            )
            assert ok, f"{pose_key} failed: {json.dumps(data)[:300]}"
        assert (await call("cap_set_gripper", {"position": 0.0}))[0]
        _, state, _ = await call("cap_get_robot_state")
        reading = state["result"]["gripper_positions"]["arm"]
        ok, data, _ = await call(
            "cap_move_to_pose", {"target_pose": approach["hover_pose"], "max_steps": MAX_STEPS}
        )
        assert ok
        _, state, _ = await call("cap_get_robot_state")
        after = state["result"]["gripper_positions"]["arm"]
        print(f"gripper reading {reading:.2f} -> {after:.2f} after the lift")
        assert 0.35 <= reading <= 0.75 and abs(after - reading) < 0.08, "grasp not confirmed"

        drop = planner(SLOT[0], SLOT[1], 0.035, 0.11)
        for pose_key in ("hover_pose", "grasp_pose"):
            ok, data, _ = await call(
                "cap_move_to_pose", {"target_pose": drop[pose_key], "max_steps": MAX_STEPS}
            )
            assert ok, f"{pose_key} to the plate failed: {json.dumps(data)[:300]}"
        assert (await call("cap_set_gripper", {"position": 1.0}))[0]
        assert (
            await call(
                "cap_move_to_pose", {"target_pose": drop["hover_pose"], "max_steps": MAX_STEPS}
            )
        )[0]
        assert (await call("cap_program_go_home"))[0]

        ok, data, _ = await call("cap_localize_object", {"query": "salami on the plate"})
        assert ok, "the slice is not visible on the plate"
        seen = data["result"]["geometry"]["pose"]["position"]
        error = ((seen[0] - SLOT[0]) ** 2 + (seen[1] - SLOT[1]) ** 2) ** 0.5
        print(f"slice now at ({seen[0]:.3f}, {seen[1]:.3f}), {error * 1000:.0f} mm from the slot")
        # Release and retreat nudge the slice, so the check is 'on the pizza', not 'on the aim point'.
        on_pizza = (
            (seen[0] - PIZZA[0]) ** 2 + (seen[1] - PIZZA[1]) ** 2
        ) ** 0.5 < PIZZA_RADIUS - 0.02
        assert on_pizza and error < 0.08


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8222/mcp/")
    asyncio.run(main(parser.parse_args().url))
    print("MCP smoke test passed")

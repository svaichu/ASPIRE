"""Single-arm ElRobot embodiment on the MuJoCo pizza-table scene.

Same layering as :mod:`cap_harness.yam_real`, sized to one arm::

    CapApi                 embodiment-agnostic
      -> ElRobotAdapter    adapter.py -- interlock, normalize, delegate
        -> station         sim.py     -- MuJoCo plant, cameras, actuation

Sim-only: there is no real-hardware station (ElRobot's Feetech bus servos need
their own transport, see ``pizza-robot/PLAN.md``).
"""

import os

# Headless offscreen rendering. MuJoCo reads this when it is first imported, so it
# has to be set here, before any submodule pulls mujoco in. A user-set value wins.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

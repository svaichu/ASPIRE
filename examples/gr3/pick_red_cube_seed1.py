# GR-3: pick up the red cube. The hand cannot point straight down at the cube from this stance,
# so approach with the tool axis tilted 20 degrees off vertical (reachable region of the arm).
report = {}
GRASP_QUAT = (0.0664522807, 0.3768696111, 0.9098437265, 0.1604299972)  # tool +z tilted 20 deg from straight down
CUBE_HALF = 0.025

opened = open_gripper()
cube = localize_object("red cube", camera_name="rs_agentview")
report["localized"] = cube.ok

if cube.ok:
    points = cube.point_cloud.points
    xs = sorted([float(p[0]) for p in points])
    ys = sorted([float(p[1]) for p in points])
    zs = sorted([float(p[2]) for p in points])
    n = len(points)
    cx, cy = xs[n // 2], ys[n // 2]
    top = zs[int(0.95 * (n - 1))]
    cz = top - CUBE_HALF
    frame = get_robot_state().base_frame
    hover = Pose(position=(cx, cy, cz + 0.12), quaternion_wxyz=GRASP_QUAT, frame=frame)
    approached = move_to_pose(hover, tolerance=0.02, max_steps=400)
    report["approached"] = approached.ok
else:
    approached = None

if approached is not None and approached.ok:
    grasp = Pose(position=(cx, cy, cz + 0.03), quaternion_wxyz=GRASP_QUAT, frame=frame)
    descended = move_to_pose(grasp, tolerance=0.015, max_steps=300)
    report["descended"] = descended.ok
else:
    descended = None

if descended is not None and descended.ok:
    closed = close_gripper()
    report["closed"] = closed.ok
    lift = Pose(position=(cx, cy, cz + 0.25), quaternion_wxyz=GRASP_QUAT, frame=frame)
    lifted = move_to_pose(lift, tolerance=0.02, max_steps=400)
    report["lifted"] = lifted.ok

result = report

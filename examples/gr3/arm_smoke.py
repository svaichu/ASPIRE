# Harness smoke test for the GR-3 embodiment: no perception, arm and gripper only.
report = {}
opened = open_gripper()
report["opened"] = opened.ok
state = get_robot_state()
pose = state.end_effector_poses["primary"]
report["frame"] = state.base_frame
target = Pose(
    position=pose.position + (0.12, 0.0, 0.18),
    quaternion_wxyz=pose.quaternion_wxyz,
    frame=pose.frame,
)
moved = move_to_pose(target, tolerance=0.02, max_steps=400)
report["moved"] = moved.ok
closed = close_gripper()
report["closed"] = closed.ok
result = report

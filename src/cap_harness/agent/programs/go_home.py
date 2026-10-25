# Home releases every gripper before the single whole-robot home motion.
for arm in get_robot_state().joint_positions:
    result = set_gripper(1.0, arm=arm)
    if not result.ok:
        break
else:
    result = go_home(arm="both")

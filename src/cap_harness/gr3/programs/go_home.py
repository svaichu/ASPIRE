# Open the hand, then return the right arm to the pose it had when the scene was reset.
opened = open_gripper()
homed = go_home()
result = {"ok": bool(opened.ok and homed.ok), "opened": opened.ok, "homed": homed.ok}

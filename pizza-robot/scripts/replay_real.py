"""Replay a real demonstration in the simulator and measure how closely the sim follows it.

The real episode's recorded ``goal_norm`` targets are mapped onto the sim's joints
(``cap_harness.elrobot.real``) and streamed at the recording rate. Reported per episode: how far
the simulated joints lag the commanded ones, and the lowest the fingertip gets (the real operator
touches the table, so it should reach z ~ 0 and never go meaningfully below). Optionally saves
real-camera / sim-camera pairs.

    .venv-system2/bin/python pizza-robot/scripts/replay_real.py [--episode NAME ...] [--frames DIR]
"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from cap_harness import elrobot  # noqa: F401  (headless GL before mujoco loads)
from cap_harness.elrobot.real import RealJointMap, load_real_episodes
from cap_harness.elrobot.station import build_sim_station

REAL_FRAMES = Path(__file__).resolve().parents[1] / "real" / "frames"


def replay(env, episode, mapper, frames_dir=None):
    joints, gripper = mapper.to_sim(episode.action)  # what the operator commanded
    env.reset()
    env.execute_waypoints(
        [0.0, 0.5], np.vstack([env.joint_positions(), joints[0]]), [1.0, gripper[0]]
    )
    dt = 1.0 / episode.fps
    errors, lowest = [], np.inf
    saved = set(np.round(np.linspace(0, len(joints) - 1, 8)).astype(int)) if frames_dir else set()
    for index in range(1, len(joints)):
        env.execute_waypoints(
            [0.0, dt],
            np.vstack([joints[index - 1], joints[index]]),
            [gripper[index - 1], gripper[index]],
        )
        errors.append(np.abs(env.joint_positions() - joints[index]))
        lowest = min(lowest, float(env.ee_pose()[0][2]))
        if index in saved:
            k = sorted(saved).index(index)
            pair = [env.camera_frame("overhead").rgb]
            real = REAL_FRAMES / f"{episode.name}_f{k}_cam0.png"
            if real.exists():
                pair.insert(0, np.asarray(Image.open(real).convert("RGB").resize((640, 480))))
            Image.fromarray(np.hstack(pair)).save(Path(frames_dir) / f"{episode.name}_{k}.png")
    errors = np.array(errors)
    return {
        "rms_rad": float(np.sqrt((errors**2).mean())),
        "p99_rad": float(np.percentile(errors, 99)),
        "per_joint_rms": np.sqrt((errors**2).mean(axis=0)).round(3).tolist(),
        "lowest_fingertip_z_m": lowest,
        "steps": len(joints),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", nargs="*", help="episode names (default: one per task)")
    parser.add_argument("--frames", type=Path, help="save real|sim image pairs here")
    args = parser.parse_args(argv)
    episodes = load_real_episodes()
    if args.episode:
        chosen = [e for e in episodes if e.name in args.episode]
    else:  # the first episode of each task
        chosen = list({e.task: e for e in reversed(episodes)}.values())
    if args.frames:
        args.frames.mkdir(parents=True, exist_ok=True)
    env = build_sim_station()
    mapper = RealJointMap.from_config(env.config)
    for episode in chosen:
        stats = replay(env, episode, mapper, args.frames)
        print(
            f"{episode.name:24s} {stats['steps']:4d} steps  tracking rms {stats['rms_rad']:.3f} rad "
            f"(p99 {stats['p99_rad']:.3f})  lowest fingertip z {stats['lowest_fingertip_z_m']:+.3f} m"
        )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

r"""Export the real ElRobot pizza demonstrations into small arrays the ASPIRE sim tooling can read.

Loads ``ApoorvaBajaj/pizza-topping-camera-cleaned`` (30 teleoperated episodes of Station parquet,
~10 fps, 2 cameras, 8 joints) through ``robotdataset.StationParquetDataset`` and writes:

  real/episodes.npz      state / action in [0, 1] (joint ``position_norm`` / ``goal_norm``),
                         concatenated, with per-episode offsets, names and task strings
  real/calibration.json  per-joint servo tick ranges (range_min/range_max), the thing that turns
                         [0, 1] back into a joint angle
  real/frames/<ep>_f<k>_cam{0,1}.png   8 frames per episode, evenly spread (not committed; ~25 MB)

``robotdataset`` and torch live in the finetune-groot environment, not .venv-system2, so run:

    /workspace/finetune-groot/.venv/bin/python pizza-robot/real/export_real.py \\
        --data-dir /workspace/finetune-groot/outputs/pizza_hf/cleaned

The loader hands back the normalised state/action and the images; the tick ranges are constant
per-joint metadata it does not surface, so they are read straight from one parquet file.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from robotdataset.station_dataset import StationParquetDataset

HERE = Path(__file__).resolve().parent
FRAMES_PER_EPISODE = 8


def main(data_dir: Path) -> None:
    files = sorted(data_dir.glob("*.parquet"))
    dataset = StationParquetDataset(data_dir=str(data_dir), batch_size=1)
    steps = dataset._storage._storage
    state = steps["observation", "state"].numpy().astype(np.float32)
    action = steps["action"].numpy().astype(np.float32)
    starts = np.array([dataset._episode_starts[i] for i in range(len(files))])
    lengths = np.array([dataset._episode_lengths[i] for i in range(len(files))])
    assert int(lengths.sum()) == len(state), "episode index does not cover the storage"

    tasks, ranges, fps = [], [], []
    for path in files:
        table = pq.read_table(path, columns=["task", "joints", "timestamp_ns_since_episode_start"])
        tasks.append(table.column("task")[0].as_py())
        joints = table.column("joints")[0].as_py()
        ranges.append([(j["range_min"], j["range_max"]) for j in joints])
        ts = np.sort(np.asarray(table.column(2).to_pylist(), dtype=np.int64))
        fps.append(1e9 / float(np.median(np.diff(ts))))
    ranges = np.array(ranges)
    assert (ranges == ranges[0]).all(), "calibration differs between episodes"

    np.savez_compressed(
        HERE / "episodes.npz",
        state=state,
        action=action,
        starts=starts,
        lengths=lengths,
        names=np.array([p.stem.removeprefix("dataset-") for p in files]),
        tasks=np.array(tasks),
        fps=np.array(fps, dtype=np.float32),
    )
    (HERE / "calibration.json").write_text(
        json.dumps(
            {
                "joint_order": [
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                    "joint7",
                    "gripper",
                ],
                "range_min": ranges[0, :, 0].tolist(),
                "range_max": ranges[0, :, 1].tolist(),
                "ticks_per_rev": 4096,
                "note": "position_norm = (ticks - range_min) / (range_max - range_min)",
            },
            indent=2,
        )
        + "\n"
    )

    frames = HERE / "frames"
    frames.mkdir(exist_ok=True)
    images = {k: steps["observation", "images", k] for k in ("cam0", "cam1")}
    for index, path in enumerate(files):
        name = path.stem.removeprefix("dataset-")
        for k, fraction in enumerate(np.linspace(0.0, 1.0, FRAMES_PER_EPISODE)):
            row = int(starts[index] + round(fraction * (lengths[index] - 1)))
            for camera, tensor in images.items():
                Image.fromarray(tensor[row].numpy()).save(frames / f"{name}_f{k}_{camera}.png")
    print(
        f"{len(files)} episodes, {len(state)} steps, fps {np.mean(fps):.1f}, tasks {sorted(set(tasks))}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    main(parser.parse_args().data_dir)

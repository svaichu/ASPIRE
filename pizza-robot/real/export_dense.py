r"""Dump dense frames and full-rate joint series of the real pizza demonstrations for offline checks.

``export_real.py`` keeps 8 frames per episode, enough for the camera fit but not for judging a
motion step by step. This writes, per selected episode, every ``--every``-th frame as the
dataset's own JPEGs (no re-encoding) plus the full-rate joint series:

  <out>/<episode>/f<frame>_cam{0,1}.jpg      cam0 = overhead, cam1 = wrist
  <out>/<episode>/series.npz                 state / action ([0, 1], one row per frame),
                                             frame_index of the saved JPEGs, fps

Pure pyarrow, so it runs in the finetune-groot environment (``.venv-system2`` has no pyarrow):

    /workspace/finetune-groot/.venv/bin/python pizza-robot/real/export_dense.py OUT_DIR \
        --data-dir /workspace/finetune-groot/outputs/pizza_hf/cleaned --names salami cheese
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--names", nargs="+", default=["salami", "cheese"],
                        help="episode name prefixes, e.g. salami matches salami1 and symmetric-salami1")
    parser.add_argument("--every", type=int, default=2)
    args = parser.parse_args()

    for path in sorted(args.data_dir.glob("dataset-*.parquet")):
        name = path.stem.removeprefix("dataset-")
        if not any(key in name for key in args.names):
            continue
        rows = pq.read_table(path, columns=["joints", "images", "task", "timestamp_ns_since_episode_start"]).to_pylist()
        rows.sort(key=lambda row: row["timestamp_ns_since_episode_start"])
        state = np.array([[j["position_norm"] for j in row["joints"]] for row in rows], np.float32)
        action = np.array([[j["goal_norm"] for j in row["joints"]] for row in rows], np.float32)
        stamps = np.array([row["timestamp_ns_since_episode_start"] for row in rows], np.int64)
        directory = args.out / name
        directory.mkdir(parents=True, exist_ok=True)
        kept = list(range(0, len(rows), args.every))
        for index in kept:
            for cam, image in enumerate(rows[index]["images"]):
                (directory / f"f{index:04d}_cam{cam}.jpg").write_bytes(image["jpeg"])
        np.savez(
            directory / "series.npz",
            state=state,
            action=action,
            frame_index=np.array(kept),
            fps=1e9 / float(np.median(np.diff(stamps))),
            task=rows[0].get("task", ""),
        )
        print(f"{name}: {len(rows)} frames, {len(kept)} exported")


if __name__ == "__main__":
    main()

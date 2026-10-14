"""Measure where the 5-shot demos grasp the bowl -> data/demo_grasp.json (per-task median grip-site offset from
the bowl centre at the close frame, world frame, cm). Only needed to re-derive that file.

  python -m phone2libero.demo_grasp --hdf5-dir <LIBERO libero_spatial HDF5 demos>

The 5-shot dataset has no object poses, so each demo is matched to its source demo in LIBERO's original HDF5
files: the source whose start state, loaded with the standard reset, renders the closest first image. The demo's
recorded actions are replayed up to the close frame (first gripper close command), which puts the bowl where
it is at the grasp; the recorded grip site minus the bowl's mesh centre is the offset. The HDF5 files exist for
the table-level tasks only, so those are the tasks measured.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

TASKS = [0, 1, 2, 6, 8]


def measure_task(task: int, ds, hdf5_dir: Path) -> list[np.ndarray]:
    import h5py

    from . import libero as L

    env = L.make_env(task)
    lang = env.task_description
    path = next(p for p in glob.glob(f"{hdf5_dir}/*.hdf5") if Path(p).name.startswith(lang.replace(" ", "_")))
    with h5py.File(path, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        env._init_states = np.stack([f["data"][k]["states"][0] for k in keys])  # the source demos' start states

    def start(j: int) -> None:
        env.init_state_id = j
        env.reset(seed=0)

    renders = []
    for j in range(len(keys)):
        start(j)
        renders.append(env.render().astype(float))

    hf = ds.hf_dataset
    ep, ti = np.array(hf["episode_index"]), np.array(hf["task_index"])
    task_index = int(ds.meta.tasks.loc[lang].task_index)
    offsets = []
    for e in np.unique(ep[ti == task_index]):
        rows = np.where(ep == e)[0]
        image0 = ds[int(rows[0])]["observation.images.image"].permute(1, 2, 0).numpy() * 255
        state = np.stack([np.asarray(hf[int(i)]["observation.state"]) for i in rows])
        action = np.stack([np.asarray(hf[int(i)]["action"]) for i in rows])
        start(int(np.argmin([np.abs(r - image0).mean() for r in renders])))
        close = next(k for k in range(1, len(action)) if action[k, 6] > 0 >= action[k - 1, 6])
        for a in action[:close]:
            env.step(a.astype(np.float32))
        offsets.append(100 * (state[close, :3] - L.mesh_centre(env, L.target_object(env))))
    env.close()
    return offsets


def main(hdf5_dir: Path, out: Path = Path("data/demo_grasp.json")) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .dataset import FIVE_SHOT

    ds = LeRobotDataset("local/libero_spatial_5shot", root=FIVE_SHOT)
    res = {}
    for t in TASKS:
        o = np.array(measure_task(t, ds, hdf5_dir))
        med = np.median(o, axis=0)
        res[str(t)] = {
            "world_rel_x_cm": float(med[0]),
            "world_rel_y_cm": float(med[1]),
            "height_vs_bowl_centre_cm": float(med[2]),
            "n_demos": len(o),
        }
        print(f"task {t}: grip site - bowl centre = {np.round(med, 1)} cm (median of {len(o)} demos)")
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    import tyro

    tyro.cli(main)

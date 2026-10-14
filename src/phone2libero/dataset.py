"""LeRobot datasets in the 5-shot set's exact format: export passing replays as episodes, and merge datasets.

  python -m phone2libero.dataset build --results outputs/grid/table/replays.jsonl --root data/phone_table
  python -m phone2libero.dataset merge --out data/train data/phone_table data/phone_raised ...

Selection, per task: up to per_task replays that passed every check, each init state used at most once, takes
spread evenly (the take used least so far goes first). Each one is re-run from a fresh reset with cameras on
and must pass again (closed-loop, so it does not depend on the grid run's exact state).

Frame convention of the 5-shot set: the episode ends 3 frames after the first success; reward -1 before
success, +1 from it; mc_returns = discounted return, gamma 0.98. Its metadata says 10 fps although the demos
run at LIBERO's 20 Hz control rate; we copy the metadata as is, so the sets merge and train identically.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro

FIVE_SHOT = Path("data/5shot/libero_spatial")  # just download-5shot (SemyonXu616/HF_LIBERO_5_SHOT)
TAIL, GAMMA = 3, 0.98
_DEFAULT_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def features() -> dict:
    info = json.loads((FIVE_SHOT / "meta" / "info.json").read_text())
    return {
        k: {kk: vv for kk, vv in v.items() if kk != "fps"}
        for k, v in info["features"].items()
        if k not in _DEFAULT_KEYS
    }


def create(root: Path, overwrite: bool):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        if not overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite")
        shutil.rmtree(root)
    return LeRobotDataset.create(
        f"local/{root.name}", fps=10, features=features(), root=root, robot_type="panda", use_videos=False
    )


def add_episode(ds, frames: list[dict], actions: list[np.ndarray], first_success: int, task: str) -> int:
    """Append frames[: first_success + TAIL] with the 5-shot reward and return columns. Returns its length."""
    n = first_success + TAIL
    reward = np.where(np.arange(n) >= first_success, 1.0, -1.0)
    ret, acc = np.zeros(n), 0.0
    for k in range(n - 1, -1, -1):
        acc = reward[k] + GAMMA * acc
        ret[k] = acc
    for k in range(n):
        ds.add_frame(
            {
                **frames[k],
                "action": np.asarray(actions[k], dtype=np.float32),
                "reward": np.array([reward[k]], dtype=np.float32),
                "mc_returns": np.array([ret[k]], dtype=np.float32),
                "task": task,
            }
        )
    ds.save_episode()
    return n


def finish(ds, root: Path) -> None:
    """Finalize, then copy the 5-shot set's metadata conventions so lerobot's aggregate_datasets accepts the merge:
    fps 10.0 inside every feature, the same per-episode stats columns, and plain floats for scalar stats."""
    import pandas as pd

    ds.finalize()
    ref = json.loads((FIVE_SHOT / "meta" / "info.json").read_text())
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text())
    assert set(info["features"]) == set(ref["features"]), "feature names differ from the 5-shot set"
    info["features"], info["fps"] = ref["features"], ref["fps"]
    path.write_text(json.dumps(info, indent=4))
    ref_ep = pd.read_parquet(sorted((FIVE_SHOT / "meta" / "episodes").glob("*/*.parquet"))[0])
    for f in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        ep = pd.read_parquet(f)
        ep = ep[[c for c in ep.columns if not c.startswith("stats/") or c in ref_ep.columns]]
        for c in ep.columns:
            if c.startswith("stats/") and not isinstance(ref_ep[c].iloc[0], np.ndarray):
                ep[c] = ep[c].map(lambda v: np.asarray(v).reshape(-1)[0]).astype(ref_ep[c].dtype)
        ep.to_parquet(f)


def select(rows: list[dict], per_task: int) -> list[dict]:
    passed = {(r["task"], r["init"], r["take"]): r for r in rows if r["ok"]}
    uses: Counter = Counter()
    chosen = []
    for task in sorted({k[0] for k in passed}):
        cands = [r for k, r in passed.items() if k[0] == task]
        used_inits: set[int] = set()
        while sum(c["task"] == task for c in chosen) < per_task:
            free = [r for r in cands if r["init"] not in used_inits]
            if not free:
                break
            r = min(free, key=lambda r: (uses[r["take"]], r["take"], r["init"]))  # deterministic tie-break
            chosen.append(r)
            used_inits.add(r["init"])
            uses[r["take"]] += 1
    return chosen


@dataclass
class Build:
    """Export passing replays from a grid run as a LeRobot dataset."""

    results: Path  # <grid out>/replays.jsonl
    root: Path
    per_task: int = 20
    spare: int = 0  # extra candidates per task, used in selection order to replace replays that fail on re-run
    tasks: list[int] | None = None
    overwrite: bool = False


@dataclass
class Merge:
    """The 5-shot set (first, so its episodes keep indices 0-49) plus extra datasets, via lerobot's
    aggregate_datasets (stats recomputed for the merged set)."""

    extra: tyro.conf.Positional[list[Path]]
    out: Path


def build(cfg: Build) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from . import libero as L
    from .plan import make_plan
    from .replay import run, scene_of
    from .takes import load_takes

    rows = [json.loads(x) for x in cfg.results.read_text().splitlines()]
    rows = [r for r in rows if cfg.tasks is None or r["task"] in cfg.tasks]
    chosen = select(rows, cfg.per_task + cfg.spare)  # within a task, a larger cap only adds candidates at the end
    ds = create(cfg.root, cfg.overwrite)
    takes = {t.name: t for t in load_takes()}
    manifest = []
    for task in sorted({c["task"] for c in chosen}):
        env = L.make_env(task)
        kept = 0
        for c in [c for c in chosen if c["task"] == task]:
            if kept == cfg.per_task:
                break
            L.reset(env, c["init"], cameras=True)
            rep = run(env, make_plan(takes[c["take"]], scene_of(env, task)), record=True)
            if not rep.ok or rep.first_success is None:
                print(f"task {task} init {c['init']} {c['take']}: failed on re-run ({rep.reason}), skipped", flush=True)
                continue
            n = add_episode(ds, rep.frames, rep.actions, rep.first_success, env.task_description)
            kept += 1
            manifest.append({k: c[k] for k in ("task", "init", "take")} | {"n_frames": n})
            print(f"task {task} init {c['init']} {c['take']}: {n} frames", flush=True)
        env.close()
    finish(ds, cfg.root)
    (cfg.root / "meta" / "phone2libero_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} episodes -> {cfg.root}")


def merge(cfg: Merge) -> None:
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if cfg.out.exists():
        shutil.rmtree(cfg.out)
    roots = [FIVE_SHOT, *cfg.extra]
    aggregate_datasets(
        repo_ids=[f"local/{r.name}" for r in roots],
        aggr_repo_id=f"local/{cfg.out.name}",
        roots=roots,
        aggr_root=cfg.out,
    )
    ds = LeRobotDataset(f"local/{cfg.out.name}", root=cfg.out)
    print(f"{cfg.out}: {ds.num_episodes} episodes, {ds.num_frames} frames, {len(ds.meta.tasks)} tasks")


def main() -> None:
    cmd = tyro.cli(Annotated[Build, tyro.conf.subcommand("build")] | Annotated[Merge, tyro.conf.subcommand("merge")])
    {Build: build, Merge: merge}[type(cmd)](cmd)


if __name__ == "__main__":
    main()

"""Evaluate a policy on libero_spatial over several seeds, and compare policies with paired statistics.

  python -m phone2libero.evaluate run --policy <pretrained_model or HF repo> --name mymodel
  python -m phone2libero.evaluate compare baseline mymodel [--out table.md]

`run` calls lerobot-eval once per seed (256 px, batch 10). By default it runs init states 0-9 of every task, so
every model faces the same 1000 (seed, task, init state) episodes; `--episodes 50` runs init states 0-49, of
which 10-49 are held out from the test episodes and score choices made before a test eval
(`compare --inits 10 50`). Results and videos: outputs/evals/<name>/seed_<s>/.

`compare`: success per task and overall, and against the reference model the difference, paired by scene (task
and init state). The 95% intervals come from a bootstrap that resamples init states within each task and keeps
all seeds of a scene together: outcomes of one scene are correlated across seeds, so resampling single episodes
gives intervals that are too narrow. They cover evaluation noise only; each model is one training run.
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro

from .libero import OBS_SIZE, RAISED_TASKS, SUITE, TABLE_TASKS

EVALS = Path("outputs/evals")
SEEDS = list(range(0, 10000, 1000))


@dataclass
class Run:
    """lerobot-eval over several seeds; seeds already evaluated are skipped."""

    policy: str
    name: str
    seeds: list[int] = field(default_factory=lambda: SEEDS)
    task_ids: list[int] | None = None  # default: all 10 tasks
    episodes: int = 10  # per task: init states 0 .. episodes - 1
    n_action_steps: int | None = None  # actions executed per predicted chunk; default: the model's own


@dataclass
class Compare:
    """Tables of success rates and paired differences against a reference model."""

    names: tyro.conf.Positional[list[str]]
    ref: str = "baseline"
    inits: tuple[int, int] = (0, 10)  # init states scored, [start, end); (10, 50) for held-out runs
    out: Path | None = None


def run(cfg: Run) -> None:
    for s in cfg.seeds:
        out = EVALS / cfg.name / f"seed_{s}"
        if (out / "eval_info.json").exists():
            print(f"skip {cfg.name} seed {s}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "lerobot-eval",
            f"--policy.path={cfg.policy}",
            "--env.type=libero",
            f"--env.task={SUITE}",
            "--env.init_states=true",
            f"--env.observation_height={OBS_SIZE}",
            f"--env.observation_width={OBS_SIZE}",
            "--eval.batch_size=10",
            f"--eval.n_episodes={cfg.episodes}",
            f"--seed={s}",
            f"--output_dir={out}",
        ]
        if cfg.n_action_steps is not None:
            cmd.append(f"--policy.n_action_steps={cfg.n_action_steps}")
        if cfg.task_ids is not None:
            cmd.append(f"--env.task_ids=[{','.join(map(str, cfg.task_ids))}]")
        print(f"eval {cfg.name} seed {s}", flush=True)
        with open(f"{out}.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def load(name: str) -> dict[tuple[int, int, int], bool]:
    """(seed, task, episode) -> success."""
    out = {}
    for f in sorted((EVALS / name).glob("seed_*/eval_info.json")):
        seed = int(f.parent.name.split("_")[1])
        for t in json.loads(f.read_text())["per_task"]:
            for e, s in enumerate(t["metrics"]["successes"]):
                out[(seed, t["task_id"], e)] = bool(s)
    if not out:
        raise SystemExit(f"no finished evals in {EVALS / name}")
    return out


def _by_task(values: dict[tuple[int, int, int], float], tasks: list[int]) -> list[np.ndarray]:
    """Per-scene means over seeds, one array per task (a scene is one task and init state)."""
    scenes = defaultdict(list)
    for (_, t, e), v in values.items():
        if t in tasks:
            scenes[(t, e)].append(v)
    rows = defaultdict(list)
    for (t, _), v in sorted(scenes.items()):
        rows[t].append(np.mean(v))
    return [np.array(r) for r in rows.values()]


def _boot(rows: list[np.ndarray]) -> tuple[float, float, float]:
    """Mean over scenes in %, with a 95% interval from resampling scenes within each task."""
    rng = np.random.default_rng(0)
    total = sum(r[rng.integers(0, len(r), (10000, len(r)))].sum(1) for r in rows)
    boots = total / sum(len(r) for r in rows)
    return 100 * np.concatenate(rows).mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5)


def rate(res: dict, tasks: list[int]) -> tuple[float, float, float]:
    """Success in %, with a 95% interval."""
    return _boot(_by_task({k: float(v) for k, v in res.items()}, tasks))


def paired_diff(a: dict, b: dict, tasks: list[int]) -> tuple[float, float, float, int]:
    """Mean of a - b over the episodes both have, in points, with a 95% interval; and the number of episodes."""
    d = {k: float(a[k]) - float(b[k]) for k in a if k in b and k[1] in tasks}
    return *_boot(_by_task(d, tasks)), len(d)


def compare(cfg: Compare) -> None:
    first, end = cfg.inits
    res = {n: {k: v for k, v in load(n).items() if first <= k[2] < end} for n in cfg.names}
    groups = [("all 10", list(range(10))), ("table-level", TABLE_TASKS), ("raised", RAISED_TASKS)]
    L = [
        "| model | seeds | " + " | ".join(g for g, _ in groups) + " | " + " | ".join(f"t{t}" for t in range(10)) + " |",
        "|---|---|" + "---|" * (len(groups) + 10),
    ]
    for n, r in res.items():
        cells = []
        for _, ts in groups:
            p, lo, hi = rate(r, ts)
            cells.append(f"{p:.1f} [{lo:.0f}-{hi:.0f}]")
        cells += [f"{rate(r, [t])[0]:.0f}" for t in range(10)]
        L.append(f"| {n} | {len({k[0] for k in r})} | " + " | ".join(cells) + " |")
    if cfg.ref in res:
        L += [
            "",
            f"Difference vs {cfg.ref} (points, paired by scene, 95% interval):",
            "",
            "| model | " + " | ".join(g for g, _ in groups) + " | episodes paired |",
            "|---|" + "---|" * (len(groups) + 1),
        ]
        for n, r in res.items():
            if n == cfg.ref:
                continue
            diffs = [paired_diff(r, res[cfg.ref], ts) for _, ts in groups]
            L.append(
                f"| {n} | "
                + " | ".join(f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]" for d, lo, hi, _ in diffs)
                + f" | {diffs[0][3]} |"
            )
    text = "\n".join(L)
    print(text)
    if cfg.out:
        cfg.out.write_text(text + "\n")


def main() -> None:
    cmd = tyro.cli(Annotated[Run, tyro.conf.subcommand("run")] | Annotated[Compare, tyro.conf.subcommand("compare")])
    {Run: run, Compare: compare}[type(cmd)](cmd)


if __name__ == "__main__":
    main()

"""Evaluate a policy on libero_spatial over several seeds, and compare policies with paired statistics.

  python -m phone2libero.evaluate run --policy <pretrained_model or HF repo> --name mymodel
  python -m phone2libero.evaluate compare baseline mymodel [--out table.md]

`run` calls lerobot-eval once per seed (256 px, init states 0-9, 10 episodes per task, batch 10), so every
model faces the same 1000 (seed, task, init state) episodes. Results and videos: outputs/evals/<name>/seed_<s>/.

`compare`: success per task and overall with 95% Wilson intervals; against the reference model, the difference
with a 95% paired bootstrap interval. Episodes are paired by (seed, task, init state): both models start from
the same scene with the same env seed, which removes scene difficulty from the comparison.
"""

from __future__ import annotations

import json
import subprocess
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


@dataclass
class Compare:
    """Tables of success rates and paired differences against a reference model."""

    names: tyro.conf.Positional[list[str]]
    ref: str = "baseline"
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
            "--eval.n_episodes=10",
            f"--seed={s}",
            f"--output_dir={out}",
        ]
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
    return out


def wilson(k: int, n: int) -> tuple[float, float]:
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def rate(res: dict, tasks: list[int]) -> tuple[float, int, int]:
    v = [s for (_, t, _), s in res.items() if t in tasks]
    return 100 * float(np.mean(v)), int(sum(v)), len(v)


def paired_diff(a: dict, b: dict, tasks: list[int], n_boot: int = 10000) -> tuple[float, float, float, int]:
    """Mean of a - b over the episodes both have, in points, with a 95% bootstrap interval."""
    keys = sorted(k for k in a if k in b and k[1] in tasks)
    d = np.array([a[k] - b[k] for k in keys], dtype=float)
    boots = d[np.random.default_rng(0).integers(0, len(d), (n_boot, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5), len(keys)


def compare(cfg: Compare) -> None:
    res = {n: load(n) for n in cfg.names}
    groups = [("all 10", list(range(10))), ("table-level", TABLE_TASKS), ("raised", RAISED_TASKS)]
    L = [
        "| model | seeds | " + " | ".join(g for g, _ in groups) + " | " + " | ".join(f"t{t}" for t in range(10)) + " |",
        "|---|---|" + "---|" * (len(groups) + 10),
    ]
    for n, r in res.items():
        cells = []
        for _, ts in groups:
            p, k, m = rate(r, ts)
            lo, hi = wilson(k, m)
            cells.append(f"{p:.1f} [{lo:.0f}-{hi:.0f}]")
        cells += [f"{rate(r, [t])[0]:.0f}" for t in range(10)]
        L.append(f"| {n} | {len({k[0] for k in r})} | " + " | ".join(cells) + " |")
    if cfg.ref in res:
        L += [
            "",
            f"Difference vs {cfg.ref} (points, paired by seed/task/init state, 95% bootstrap):",
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

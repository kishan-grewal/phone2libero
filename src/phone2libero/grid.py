"""Replay every take on every (task, init state), without rendering; one JSON line per replay.

  python -m phone2libero.grid run --tasks 0 1 2 6 8 --out outputs/grid/table
  python -m phone2libero.grid report --out outputs/grid/table

Each worker owns one (task, init state): one reset, a checkpoint, then every take from that same state
(restore is bit-exact). Re-running skips the (task, init) jobs already in <out>/replays.jsonl.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Annotated

import tyro

INITS = list(range(10, 50))  # 0-9 are the eval episodes


def _job(task: int, init: int) -> list[dict]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from . import libero as L
    from .plan import make_plan
    from .replay import run, scene_of
    from .takes import load_takes

    takes = load_takes()
    env = L.make_env(task)
    L.reset(env, init, cameras=False)
    scene = scene_of(env, task)
    snap = L.checkpoint(env)
    rows = []
    for take in takes:
        L.restore(env, snap)
        rep = run(env, make_plan(take, scene))
        rows.append(
            {
                "task": task,
                "init": init,
                "take": take.name,
                "video": take.video,
                "flagged": bool(take.flags),
                "ok": rep.ok,
                "reason": rep.reason,
                "success_final": rep.success_final,
                "first_success": rep.first_success,
                "lift_cm": 100 * rep.lift_m,
                "slip_cm": 100 * rep.slip_m,
                "other_shift_cm": 100 * rep.other_shift_m,
                "other_worst": rep.other_worst,
                "track_err_cm": 100 * rep.track_err_m,
            }
        )
    env.close()
    return rows


def _pool(workers: int, jobs: list[tuple], results: Path) -> None:
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex, results.open("a") as f:
        futs = {ex.submit(_job, *j): j for j in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            rows = fut.result()
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            task, init = futs[fut]
            print(
                f"[{n}/{len(jobs)}] task {task} init {init}: {sum(r['ok'] for r in rows)}/{len(rows)} pass "
                f"({time.time() - t0:.0f} s)",
                flush=True,
            )


@dataclass
class Run:
    """Replay every take on the given tasks and init states."""

    out: Path
    tasks: list[int]
    inits: list[int] = field(default_factory=lambda: INITS)
    workers: int = 12


@dataclass
class Report:
    """Pass rates by task, recording and take, plus the failure reasons: <out>/report.md."""

    out: Path


def run_grid(cfg: Run) -> None:
    cfg.out.mkdir(parents=True, exist_ok=True)
    results = cfg.out / "replays.jsonl"
    done = set()
    if results.exists():
        done = {(r["task"], r["init"]) for r in map(json.loads, results.read_text().splitlines())}
    jobs = [(t, i) for t in cfg.tasks for i in cfg.inits if (t, i) not in done]
    print(f"{len(jobs)} (task, init) jobs, {len(done)} already done", flush=True)
    _pool(cfg.workers, jobs, results)


def report(cfg: Report) -> None:
    import pandas as pd

    df = pd.DataFrame([json.loads(x) for x in (cfg.out / "replays.jsonl").read_text().splitlines()])
    pct = lambda s: f"{100 * s.mean():.0f}% ({s.sum()}/{len(s)})"  # noqa: E731
    lines = [
        f"# Retargeted replays: {len(df)}",
        "",
        f"Pass rate: {pct(df.ok)}; LIBERO success alone: {pct(df.success_final)}",
        "",
    ]
    for by, title in (("task", "task"), ("video", "recording")):
        lines += [f"## By {title}", "", f"| {title} | pass | LIBERO success | median lift cm |", "|---|---|---|---|"]
        lines += [
            f"| {k} | {pct(g.ok)} | {pct(g.success_final)} | {g.lift_cm.median():.1f} |" for k, g in df.groupby(by)
        ]
        lines.append("")
    reasons = df[~df.ok].reason.str.split("; ").explode().str.replace(r"[\d.]+ cm", "", regex=True).str.strip()
    lines += ["## Failure reasons (a replay can have several)", "", "| reason | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in reasons.value_counts().items()]
    per_ti = df.groupby(["task", "init"]).ok.mean()
    lines += ["", f"(task, init) pairs where no take passes: {int((per_ti == 0).sum())} of {len(per_ti)}"]
    (cfg.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    cmd = tyro.cli(Annotated[Run, tyro.conf.subcommand("run")] | Annotated[Report, tyro.conf.subcommand("report")])
    {Run: run_grid, Report: report}[type(cmd)](cmd)


if __name__ == "__main__":
    main()

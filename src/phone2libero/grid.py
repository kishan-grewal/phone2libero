"""Replay every take on every (task, init state), without rendering; one JSON line per replay.

  python -m phone2libero.grid run --tasks 0 1 2 6 8 --out outputs/grid/table
  python -m phone2libero.grid pick-side --tasks 3 5 7
  python -m phone2libero.grid run --tasks 9 --sides 1 -1 --out outputs/grid/raised
  python -m phone2libero.grid report --out outputs/grid/table

Each worker owns one (task, init state): one reset, a checkpoint, then every take from that same state
(restore is bit-exact). Re-running skips the (task, init, side) jobs already in <out>/replays.jsonl.
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

import numpy as np
import tyro

INITS = list(range(10, 50))  # 0-9 are the eval episodes


def _job(task: int, init: int, side: int | None = None, take_names: list[str] | None = None) -> list[dict]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from . import libero as L
    from .plan import make_plan
    from .replay import ENCLOSED, run, scene_of
    from .takes import load_takes

    takes = [t for t in load_takes() if take_names is None or t.name in take_names]
    env = L.make_env(task)
    L.reset(env, init, cameras=False)
    scene = scene_of(env, task, side)
    snap = L.checkpoint(env)
    rows = []
    for take in takes:
        L.restore(env, snap)
        rep = run(env, make_plan(take, scene), track_bowl=task in ENCLOSED)
        rows.append(
            {
                "task": task,
                "init": init,
                "take": take.name,
                "video": take.video,
                "side": side,
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
            task, init, side = futs[fut][:3]
            print(
                f"[{n}/{len(jobs)}] task {task} init {init}"
                + (f" side {side:+d}" if side else "")
                + f": {sum(r['ok'] for r in rows)}/{len(rows)} pass ({time.time() - t0:.0f} s)",
                flush=True,
            )


@dataclass
class Run:
    """Replay every take on the given tasks and init states."""

    out: Path
    tasks: list[int]
    inits: list[int] = field(default_factory=lambda: INITS)
    sides: list[int] | None = None  # raised tasks: run each (task, init) once per grasp side (+1 left, -1 right)
    workers: int = 12


@dataclass
class PickSide:
    """Raised tasks: try both grasp sides on 10 init states x every 5th take; store the better side per task."""

    tasks: list[int]
    out: Path = Path("outputs/grid/pick_side")
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
        done = {(r["task"], r["init"], r.get("side")) for r in map(json.loads, results.read_text().splitlines())}
    jobs = [(t, i, s) for t in cfg.tasks for i in cfg.inits for s in (cfg.sides or [None]) if (t, i, s) not in done]
    print(f"{len(jobs)} (task, init, side) jobs, {len(done)} already done", flush=True)
    _pool(cfg.workers, jobs, results)


def pick_side(cfg: PickSide) -> None:
    from .replay import RAISED_SIDES
    from .takes import load_takes

    cfg.out.mkdir(parents=True, exist_ok=True)
    results = cfg.out / "replays.jsonl"
    results.unlink(missing_ok=True)
    names = [t.name for t in load_takes()][::5]
    _pool(cfg.workers, [(t, i, s, names) for t in cfg.tasks for i in range(10, 20) for s in (1, -1)], results)
    rows = [json.loads(x) for x in results.read_text().splitlines()]
    sides = json.loads(RAISED_SIDES.read_text()) if RAISED_SIDES.exists() else {}
    for t in cfg.tasks:
        rate = {s: np.mean([r["ok"] for r in rows if r["task"] == t and r["side"] == s]) for s in (1, -1)}
        sides[str(t)] = max(rate, key=rate.get)
        print(f"task {t}: left (+1) {100 * rate[1]:.0f}%, right (-1) {100 * rate[-1]:.0f}% -> {sides[str(t)]:+d}")
    RAISED_SIDES.write_text(json.dumps(sides, indent=1))


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
    for by, title in (("task", "task"), ("video", "recording"), ("side", "grasp side")):
        if df[by].isna().all():
            continue
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
    cmd = tyro.cli(
        Annotated[Run, tyro.conf.subcommand("run")]
        | Annotated[PickSide, tyro.conf.subcommand("pick-side")]
        | Annotated[Report, tyro.conf.subcommand("report")]
    )
    {Run: run_grid, PickSide: pick_side, Report: report}[type(cmd)](cmd)


if __name__ == "__main__":
    main()

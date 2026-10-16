"""Plan phone takes without reading simulator state: a small CNN estimates what replay.scene_of reads from the sim
(target bowl centre, plate centre, bowl and plate bottom heights) from the agentview image and the task id.

  python -m phone2libero.scene sensitivity                  # replay pass rate vs noise on the planned scene
  python -m phone2libero.scene generate --split val         # labelled frames (and --split train)
  python -m phone2libero.scene train --name a128            # estimator
  python -m phone2libero.scene accuracy a128                # errors in cm vs the per-task mean, overlays
  python -m phone2libero.scene check-reset --name a128      # Check A: takes planned at reset
  python -m phone2libero.scene check-takeover --name a128   # Check B: takes planned at takeover states (--look)
  python -m phone2libero.scene demos --name a128            # phone demos planned from the estimate

Init states 10-39 make training data, 40-49 validate; 0-9 (the eval episodes) are never used. Frame rendering runs
on any EGL device; the results were rendered with Mesa (__EGL_VENDOR_LIBRARY_FILENAMES=.../50_mesa.json).
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro

from .plan import SIM_PLACE_DZ, SimScene
from .replay import ENCLOSED, grasp_offset

OUT = Path("outputs/scene")
TRAIN_INITS, VAL_INITS = range(10, 40), range(40, 50)
# What scene_of reads from the sim, as one vector (metres, world frame)
LABELS = ("bowl_x", "bowl_y", "bowl_z", "plate_x", "plate_y", "plate_z", "bowl_lo", "plate_lo")


def labels(env) -> np.ndarray:
    from . import libero as L

    target = L.target_object(env)
    return np.concatenate(
        [
            L.mesh_centre(env, target),
            L.mesh_centre(env, "plate_1"),
            [L.mesh_box(env, f"{target}_g0")[0][2], L.mesh_box(env, "plate_1_g0")[0][2]],
        ]
    )


def scene_from(y: np.ndarray, eef0: np.ndarray, task: int, side: int | None = None) -> SimScene:
    """replay.scene_of with its sim reads replaced by y (LABELS order); eef0 is the grip site from proprio."""
    c0, bowl_lo, plate_lo = y[:3], y[6], y[7]
    place_z = None
    if bowl_lo > plate_lo + 0.01:  # raised start: place relative to the plate, not the start height
        place_z = float(plate_lo + (c0[2] - bowl_lo) + SIM_PLACE_DZ)
    return SimScene(
        bowl_c0=np.array(c0, dtype=float),
        plate_c=np.array(y[3:6], dtype=float),
        eef0=np.array(eef0, dtype=float),
        grasp_offset=grasp_offset(task, side),
        place_z=place_z,
        clear_lift=ENCLOSED[task]["clear_lift"] if task in ENCLOSED else 0.0,
    )


def sides_of(task: int) -> tuple[int | None, ...]:
    return (1, -1) if task == 9 else (None,)  # task 9: the passing rim side depends on the scene (grid.py)


def _pool(fn, jobs: list[tuple], workers: int, out: Path | None = None) -> list:
    """Run fn(*job) in spawned workers; rows (lists of dicts) are appended to out as JSON lines when given."""
    t0, rows = time.time(), []
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        futs = [ex.submit(fn, *j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            rows += res
            if out is not None:
                with out.open("a") as f:
                    f.writelines(json.dumps(r) + "\n" for r in res)
            if n % max(1, len(jobs) // 20) == 0 or n == len(jobs):
                print(f"[{n}/{len(jobs)}] {time.time() - t0:.0f} s", flush=True)
    return rows


def _reset_labels(task: int, inits: list[int]) -> list[dict]:
    from . import libero as L

    env = L.make_env(task)
    rows = []
    for init in inits:
        L.reset(env, init, cameras=False)
        rows.append({"task": task, "init": init, "y": labels(env).tolist()})
    env.close()
    return rows


def task_means(workers: int = 4) -> dict[int, np.ndarray]:
    """Per-task mean of the labels at reset over the training init states: the no-camera guess."""
    rows = _pool(_reset_labels, [(t, list(TRAIN_INITS)) for t in range(10)], workers)
    return {t: np.mean([r["y"] for r in rows if r["task"] == t], axis=0) for t in range(10)}


# ---------------------------------------------------------------------------------------------- sensitivity

# (part, sigma cm): Gaussian noise per axis on the planned bowl centre (x, y, z) and plate centre (x, y)
NOISE = (
    [("all", s) for s in (0.5, 1.0, 1.5, 2.0, 3.0)]
    + [("bowl_xy", s) for s in (1.0, 2.0, 3.0)]
    + [("bowl_z", s) for s in (1.0, 2.0)]
    + [("plate_xy", s) for s in (1.0, 2.0, 3.0)]
)
_PARTS = {
    "all": [1, 1, 1, 1, 1],
    "bowl_xy": [1, 1, 0, 0, 0],
    "bowl_z": [0, 0, 1, 0, 0],
    "plate_xy": [0, 0, 0, 1, 1],
}
TAKES_PER_INIT = 6


def _noisy(scene: SimScene, z: np.ndarray, part: str, sigma_cm: float) -> SimScene:
    d = sigma_cm / 100 * z * np.array(_PARTS[part])
    return replace(scene, bowl_c0=scene.bowl_c0 + d[:3], plate_c=scene.plate_c + np.r_[d[3:], 0.0])


def _sensitivity_job(task: int, init: int, mean_y: list[float]) -> list[dict]:
    from . import libero as L
    from .plan import make_plan
    from .replay import run, scene_of
    from .takes import load_takes

    takes = load_takes()
    k = init - VAL_INITS[0]
    picked = [takes[(TAKES_PER_INIT * k + j) % len(takes)] for j in range(TAKES_PER_INIT)]  # every take over 40-49
    env = L.make_env(task)
    L.reset(env, init, cameras=False)
    eef0 = L.gripper_pos(env)
    snap = L.checkpoint(env)
    rows = []
    for side in sides_of(task):
        L.restore(env, snap)  # the second side must start from reset too, not from the last replay's end
        scene = scene_of(env, task, side)
        for take in picked:
            # the same direction at every sigma, so the curve is paired across noise levels
            z = np.random.default_rng([task, init, takes.index(take)]).standard_normal(5)
            conds = [("sim", 0.0, scene), ("task_mean", 0.0, scene_from(np.array(mean_y), eef0, task, side))]
            conds += [(p, s, _noisy(scene, z, p, s)) for p, s in NOISE]
            for part, sigma, sc in conds:
                L.restore(env, snap)
                rep = run(env, make_plan(take, sc), track_bowl=task in ENCLOSED)
                rows.append(
                    {
                        "task": task,
                        "init": init,
                        "take": take.name,
                        "side": side,
                        "part": part,
                        "sigma_cm": sigma,
                        "bowl_err_cm": float(100 * np.linalg.norm(sc.bowl_c0 - scene.bowl_c0)),
                        "plate_err_cm": float(100 * np.linalg.norm(sc.plate_c[:2] - scene.plate_c[:2])),
                        "ok": rep.ok,
                        "reason": rep.reason,
                    }
                )
    env.close()
    return rows


@dataclass
class Sensitivity:
    """Replay takes on init states 40-49 with Gaussian noise added to the planned bowl and plate centres, and with
    the per-task mean scene (no camera); pass rate vs noise -> <out>/sensitivity.md."""

    out: Path = OUT / "sensitivity"
    tasks: tuple[int, ...] = tuple(range(10))
    workers: int = 4


def _done(results: Path) -> set[tuple[int, int]]:
    """(task, init) jobs already in a results file (a job's rows are written together when it ends)."""
    if not results.exists():
        return set()
    return {(r["task"], r["init"]) for r in map(json.loads, results.read_text().splitlines())}


def sensitivity(cfg: Sensitivity) -> None:
    cfg.out.mkdir(parents=True, exist_ok=True)
    results = cfg.out / "replays.jsonl"
    done = _done(results)
    todo = [(t, i) for t in cfg.tasks for i in VAL_INITS if (t, i) not in done]
    if todo:
        means = task_means(cfg.workers)
        _pool(_sensitivity_job, [(t, i, means[t].tolist()) for t, i in todo], cfg.workers, results)
    sensitivity_report(cfg.out)


def sensitivity_report(out: Path) -> None:
    import pandas as pd

    df = pd.DataFrame([json.loads(x) for x in (out / "replays.jsonl").read_text().splitlines()])
    df["side"] = df.side.astype(float).fillna(0)  # None (one grasp side) breaks the join's keys
    key = ["task", "init", "take", "side"]
    base = df[df.part == "sim"].set_index(key).ok.rename("base")
    df = df.join(base, on=key)
    lines = [
        "# Replay pass rate vs noise on the planned scene",
        "",
        f"Init states 40-49, {TAKES_PER_INIT} takes per init state (every take used), task 9 both rim sides:"
        f" {len(base)} replays per row. Noise is Gaussian per axis on the bowl centre (x, y, z) and/or the plate"
        " centre (x, y). 'kept' = passing replays (at sigma 0) that still pass.",
        "",
        "| scene | sigma cm | pass | kept | table tasks | raised tasks |",
        "|---|---|---|---|---|---|",
    ]
    table = df.task.isin([0, 1, 2, 6, 8])
    for (part, sigma), g in df.groupby(["part", "sigma_cm"], sort=False):
        kept = g[g.base].ok.mean()
        lines.append(
            f"| {part} | {sigma:g} | {100 * g.ok.mean():.1f} | {100 * kept:.1f} | {100 * g[table].ok.mean():.1f}"
            f" | {100 * g[~table].ok.mean():.1f} |"
        )
    lines += ["", "Per task, pass %:", "", "| scene | sigma | " + " | ".join(f"t{t}" for t in range(10)) + " |"]
    lines.append("|---|---|" + "---|" * 10)
    for (part, sigma), g in df.groupby(["part", "sigma_cm"], sort=False):
        per = g.groupby("task").ok.mean()
        lines.append(
            f"| {part} | {sigma:g} | " + " | ".join(f"{100 * per.get(t, np.nan):.0f}" for t in range(10)) + " |"
        )
    (out / "sensitivity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=150)
    sim = df[df.part == "sim"].ok.mean()
    for part, colour in (("all", "#2a78d6"), ("bowl_xy", "#eb6834"), ("bowl_z", "#1baf7a"), ("plate_xy", "#eda100")):
        g = df[df.part == part].groupby("sigma_cm").ok.mean()
        ax.plot([0, *g.index], [100 * sim, *(100 * g.values)], marker="o", color=colour, label=part)
    ax.axhline(100 * df[df.part == "task_mean"].ok.mean(), color="#9a9893", ls="--", label="per-task mean scene")
    ax.set_xlabel("noise on the planned scene, sigma per axis (cm)")
    ax.set_ylabel("replays passing (%)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "sensitivity.png")


# ---------------------------------------------------------------------------------------------- labelled frames

# The policy's own successful episodes; each records its init state, and its actions replay exactly from reset.
OWN = [
    Path("data/selfplay_baseline_49"),
    Path("data/selfplay_step2sp49"),
    *sorted(Path("outputs/round2").glob("own_*")),
    *sorted(Path("outputs/round3").glob("own_*")),
]
SOURCES = ("reset", "own", "own_shift", "own_cross", "take", "take_miss", "moved")
HOLD = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)  # OSC delta 0: the arm holds still, gripper open
SETTLE_STEPS = 15
EMPTY_DZ, CLOSED_STEPS = 0.06, 10  # recover.py's takeover trigger: closed this long, this far above the bowl


def own_episodes() -> dict[tuple[int, int], list[tuple[str, int]]]:
    """(task, init) -> the own-success episodes recorded on it, as (dataset root, episode index)."""
    out: dict[tuple[int, int], list[tuple[str, int]]] = {}
    for root in OWN:
        m = json.loads(next((root / "meta").glob("*_manifest.json")).read_text())
        eps = [(int(t), ep["init"]) for t, log in m["tasks"].items() for ep in log.get("own", log.get("used", []))]
        for e, key in enumerate(eps):  # episodes are stored in manifest order (checked against their lengths)
            out.setdefault(key, []).append((str(root), e))
    return out


_ACTIONS: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}


def own_actions(root: str, episode: int) -> tuple[np.ndarray, np.ndarray]:
    """(actions, observation.state) of one recorded episode."""
    import pyarrow.parquet as pq

    if root not in _ACTIONS:
        cols = ["episode_index", "action", "observation.state"]
        t = [pq.read_table(f, columns=cols).to_pandas() for f in sorted(Path(root).glob("data/*/*.parquet"))]
        df = __import__("pandas").concat(t)
        _ACTIONS[root] = {
            int(e): (np.stack(g["action"]).astype(np.float32), np.stack(g["observation.state"]))
            for e, g in df.groupby("episode_index")
        }
    return _ACTIONS[root][episode]


def proprio(obs: dict) -> np.ndarray:
    """observation.state of the 5-shot format: eef position, axis-angle, finger qpos."""
    from . import libero as L

    return np.concatenate([obs["robot0_eef_pos"], L.axis_angle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])


def snapshot_frame(env) -> dict:
    """Both cameras rendered now, in the dataset's image convention (agentview 256, wrist downsampled to 128).

    A forced observation update also advances every observable's sampling timer, which shifts when the proprio
    observations sample on later steps; the observables and their cache are put back afterwards."""
    from . import libero as L
    from .replay import frame_of

    r = L.raw(env)
    saved = [(o, dict(vars(o))) for o in r._observables.values()]
    cache = dict(r._obs_cache)
    L.set_cameras_enabled(env, True)
    obs = r._get_observations(force_update=True)
    for o, attrs in saved:
        o.__dict__.update(attrs)
    r._obs_cache.clear()
    r._obs_cache.update(cache)
    f = frame_of(obs)
    w = f["observation.images.image2"].reshape(128, 2, 128, 2, 3).mean((1, 3))
    return {"agent": f["observation.images.image"], "wrist": (w + 0.5).astype(np.uint8), "state": proprio(obs)}


def _free_joint(env, obj: str) -> tuple[int, int]:
    import mujoco

    from . import libero as L

    r = L.raw(env)
    m = r.sim.model._model
    j = m.body_jntadr[r.obj_body_id[obj]]
    assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE, obj
    return m.jnt_qposadr[j], m.jnt_dofadr[j]


def move_objects(env, shifts: dict[str, np.ndarray], max_other_m: float = 0.01) -> bool:
    """Move objects by planar offsets, then let them settle with the arm still. False (state left as is) if an
    object that was not moved shifts by more than max_other_m: the move made objects overlap."""
    import mujoco

    from . import libero as L

    r = L.raw(env)
    m, d = r.sim.model._model, r.sim.data._data
    before = {n: r.sim.data.body_xpos[i][:2].copy() for n, i in r.obj_body_id.items()}
    for obj, dxy in shifts.items():
        q, v = _free_joint(env, obj)
        d.qpos[q : q + 2] += dxy
        d.qvel[v : v + 6] = 0
    mujoco.mj_forward(m, d)
    for _ in range(SETTLE_STEPS):
        env._env.step(HOLD)
    after = {n: r.sim.data.body_xpos[i][:2] for n, i in r.obj_body_id.items()}
    return all(np.linalg.norm(after[n] - before[n]) <= max_other_m for n in before if n not in shifts)


def _ring(rng, lo_cm: float, hi_cm: float) -> np.ndarray:
    a, r = rng.uniform(0, 2 * np.pi), rng.uniform(lo_cm, hi_cm) / 100
    return r * np.array([np.cos(a), np.sin(a)])


class Recorder:
    """Collects labelled frames of one (task, init) shard."""

    def __init__(self, env, task: int, init: int):
        self.env, self.task, self.init = env, task, init
        self.rows: list[dict] = []
        self.traj = 0

    def frame(self, source: str, step: int, closed: int, bowl0: np.ndarray) -> None:
        y = labels(self.env)
        f = snapshot_frame(self.env)
        eef = f["state"][:3]
        f |= {
            "y": y,
            "source": SOURCES.index(source),
            "traj": self.traj,
            "step": step,
            # recover.py would take over here (gripper closed on nothing above the bowl)
            "trigger": closed >= CLOSED_STEPS and eef[2] > y[2] + EMPTY_DZ,
            "held": closed >= CLOSED_STEPS and np.linalg.norm(eef - y[:3]) < 0.08 and y[2] > bowl0[2] + 0.02,
            "moved_cm": 100 * np.linalg.norm(y[:3] - bowl0),
        }
        self.rows.append(f)

    def play(self, source: str, actions: np.ndarray, every: int, bowl0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Step through recorded actions from the current state, a frame every `every` steps. Returns, after each
        action, the observation.state and the grip site read from the sim (determinism checks)."""
        from . import libero as L

        states, grips, closed = [], [], 0
        for k, a in enumerate(actions):
            if k % every == 0:
                self.frame(source, k, closed, bowl0)
            obs, _, _, _ = self.env._env.step(np.asarray(a, dtype=np.float32))
            states.append(proprio(obs))
            grips.append(L.gripper_pos(self.env))
            closed = closed + 1 if a[6] > 0 else 0
        self.traj += 1
        return np.array(states), np.array(grips)

    def save(self, path: Path) -> None:
        arrays = {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        arrays |= {"task": np.full(len(self.rows), self.task), "init": np.full(len(self.rows), self.init)}
        np.savez(path, **arrays)


def _generate_job(task: int, init: int, split: str, out: str, every: int) -> list[dict]:
    """All labelled frames of one (task, init). Seeded by (task, init), so a rerun (e.g. with another renderer)
    reproduces the same states."""
    from . import libero as L
    from .plan import make_plan
    from .replay import run, scene_of
    from .takes import load_takes

    rng = np.random.default_rng([task, init, 7])
    takes = load_takes()
    env = L.make_env(task)
    rec = Recorder(env, task, init)
    log = []

    def fresh() -> tuple[np.ndarray, SimScene]:
        L.reset(env, init, cameras=False)
        return labels(env)[:3], scene_of(env, task, sides_of(task)[rec.traj % len(sides_of(task))])

    bowl0, _ = fresh()
    rec.frame("reset", 0, 0, bowl0)

    # the policy's own successes: exactly (train), with the bowl moved at reset (train), from another init (val)
    own = own_episodes()
    if split == "train":
        for root, e in own.get((task, init), []):
            actions, stored = own_actions(root, e)
            fresh()
            L.raw(env)._get_observations(force_update=True)  # as the rollout did after reset (sampling phase)
            states, _ = rec.play("own", actions, every, bowl0)
            log.append({"root": root, "episode": e, "state_max_diff": float(np.abs(states[:-1] - stored[1:]).max())})
            fresh()
            if move_objects(env, {L.target_object(env): _ring(rng, 1.5, 5.0)}):
                rec.play("own_shift", actions, every, labels(env)[:3])
    else:
        donors = [k for k in own if k[0] == task and k[1] in TRAIN_INITS]
        for j in rng.choice(len(donors), size=2, replace=False):
            root, e = own[donors[j]][0]
            fresh()
            rec.play("own_cross", own_actions(root, e)[0], every, bowl0)

    # phone takes planned from the true scene, and from a bowl 2.5-5 cm off (a missed grasp, as in a takeover)
    n_takes = 1 if split == "train" else 2
    for j in range(2 * n_takes):
        take = takes[int(rng.integers(len(takes)))]
        _, scene = fresh()
        source = "take" if j < n_takes else "take_miss"
        if source == "take_miss":
            scene = replace(scene, bowl_c0=scene.bowl_c0 + np.r_[_ring(rng, 2.5, 5.0), 0.0])
        snap = L.checkpoint(env)
        rep = run(env, make_plan(take, scene), track_bowl=task in ENCLOSED)
        L.restore(env, snap)
        _, grips = rec.play(source, np.array(rep.actions), every, bowl0)  # open loop: the same trajectory again
        diff = float(np.abs(grips - np.array(rep.grip_trace)).max())
        log.append({"take": take.name, "source": source, "ok": rep.ok, "state_max_diff": diff})

    # bowl, plate and the other bowl moved at reset, then the arm swept to random points above the table
    for _ in range(5 if split == "train" else 3):
        fresh()
        shifts = {L.target_object(env): _ring(rng, 0.0, 6.0), "plate_1": _ring(rng, 0.0, 4.0)}
        if task in (0, 1, 2, 6, 8):
            shifts["akita_black_bowl_2"] = _ring(rng, 0.0, 4.0)
        if not move_objects(env, shifts):
            continue
        b0, closed = labels(env)[:3], 0
        rec.frame("moved", 0, 0, b0)
        for k in range(1, 61):
            if k % 20 == 1:
                goal = np.array([rng.uniform(-0.3, 0.2), rng.uniform(-0.35, 0.35), rng.uniform(1.05, 1.3)])
                grip = rng.choice([-1.0, 1.0])
            a = HOLD.copy()
            a[:3] = np.clip((goal - L.gripper_pos(env)) / 0.05, -0.4, 0.4)  # <= ~0.4 m/s, about the policy's speed
            a[6] = grip
            env._env.step(a)
            closed = closed + 1 if grip > 0 else 0
            if k % 20 == 0:
                rec.frame("moved", k, closed, b0)
        rec.traj += 1
    env.close()
    rec.save(Path(out) / f"t{task}_i{init}.npz")
    summary = {"task": task, "init": init, "frames": len(rec.rows)}
    return [summary | x for x in log] or [summary]


@dataclass
class Generate:
    """Labelled frames (images, labels, proprio) for one split -> <out>/<name>/t<task>_i<init>.npz."""

    split: str = "train"  # train: init states 10-39; val: 40-49
    name: str | None = None  # output folder; default: the split (e.g. val_nvidia for a rerun with another renderer)
    out: Path = OUT / "data"
    inits: tuple[int, ...] = ()  # default: the split's init states
    every: int = 5  # steps between recorded frames of a trajectory
    workers: int = 4


def generate(cfg: Generate) -> None:
    d = cfg.out / (cfg.name or cfg.split)
    d.mkdir(parents=True, exist_ok=True)
    inits = cfg.inits or (TRAIN_INITS if cfg.split == "train" else VAL_INITS)
    jobs = [
        (t, i, cfg.split, str(d), cfg.every) for t in range(10) for i in inits if not (d / f"t{t}_i{i}.npz").exists()
    ]
    rows = _pool(_generate_job, jobs, cfg.workers, d / "log.jsonl")
    print(f"{len(jobs)} shards, {sum(r['frames'] for r in {(r['task'], r['init']): r for r in rows}.values())} frames")


# ---------------------------------------------------------------------------------------------- camera

# The agentview camera, identical in every libero_spatial task (MuJoCo cam_xpos, cam_xmat with the camera axes as
# columns, fovy 45 deg): a calibrated camera, the way the phone rig is calibrated, not a read of the scene.
CAM_POS = np.array([0.6586131746834771, 0.0, 1.6103500240372424])
CAM_MAT = np.array(
    [
        [-7.268184498698815e-08, -6.282664504107425e-01, 7.779982437565529e-01],
        [9.999999999999858e-01, 7.268184498698815e-08, 1.521152659944569e-07],
        [-1.521152659944569e-07, 7.779982437565529e-01, 6.282664504107283e-01],
    ]
)
IMG = 256  # dataset image size; pixel coordinates below are in this image
FOCAL = IMG / 2 / np.tan(np.radians(45 / 2))


def project(p: np.ndarray) -> np.ndarray:
    """World points (..., 3) -> (col, row) in the dataset image, which is the raw render rotated 180 deg."""
    c = (p - CAM_POS) @ CAM_MAT  # camera frame: x right, y up, looking along -z
    return np.stack([IMG / 2 - FOCAL * c[..., 0] / -c[..., 2], IMG / 2 - FOCAL * c[..., 1] / -c[..., 2]], -1)


def backproject(col, row, z):
    """Pixel (col, row) of the dataset image and a height z -> the world point on that pixel's ray (torch)."""
    import torch

    cam = torch.stack([(IMG / 2 - col) / FOCAL, (IMG / 2 - row) / FOCAL, -torch.ones_like(col)], -1)
    ray = cam @ torch.as_tensor(CAM_MAT.T, dtype=cam.dtype, device=cam.device)
    t = (z - CAM_POS[2]) / ray[..., 2]
    return torch.as_tensor(CAM_POS, dtype=cam.dtype, device=cam.device) + t[..., None] * ray


# ---------------------------------------------------------------------------------------------- model


def _net():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def conv(cin, cout, k=3, s=1, d=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, k, s, padding=d * (k // 2), dilation=d, bias=False),
            nn.GroupNorm(8, cout),
            nn.ReLU(inplace=True),
        )

    class Estimator(nn.Module):
        """Agentview image + task id (+ optionally the wrist image and proprio) -> LABELS.

        Two task-conditioned heatmaps over the agentview image locate the target bowl's and the plate's centres
        (soft-argmax, sub-pixel); a head predicts their heights; the camera model puts each centre on its pixel's
        ray at that height. Inputs: uint8 images (B, H, W, 3) in the dataset convention, task (B,), state (B, 8).
        """

        def __init__(self, res: int = 128, wrist: bool = False):
            super().__init__()
            self.res, self.use_wrist = res, wrist
            self.down = nn.Sequential(conv(5, 32, 5, 2), conv(32, 64, 3, 2), conv(64, 64))  # 1/4
            self.deep = nn.Sequential(conv(64, 128, 3, 2), conv(128, 128, d=2), conv(128, 128, d=4), conv(128, 128))
            self.task = nn.Embedding(10, 32)
            self.film = nn.Linear(32, 2 * 192)
            self.fuse = nn.Sequential(conv(192, 96), conv(96, 96))
            self.heat = nn.Conv2d(96, 2, 1)
            self.off = nn.Conv2d(96, 4, 1)  # per cell, each point's sub-cell offset (col, row), in cells
            nn.init.zeros_(self.off.weight)
            nn.init.zeros_(self.off.bias)
            n = 2 * 96 + 32
            if wrist:
                self.wrist = nn.Sequential(
                    conv(3, 32, 5, 2), conv(32, 64, 3, 2), conv(64, 64, 3, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten()
                )
                n += 64 + 8
            self.head = nn.Sequential(nn.Linear(n, 256), nn.ReLU(inplace=True), nn.Linear(256, 6))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

        @staticmethod
        def _img(x, res):
            x = x.permute(0, 3, 1, 2).float() / 255
            if x.shape[-1] != res:
                x = F.interpolate(x, size=(res, res), mode="area")
            return (x - 0.5) / 0.25

        def forward(self, agent, task, wrist=None, state=None):
            x = self._img(agent, self.res)
            b, _, h, w = x.shape
            ys, xs = torch.meshgrid(
                torch.linspace(-1, 1, h, device=x.device), torch.linspace(-1, 1, w, device=x.device), indexing="ij"
            )
            f4 = self.down(torch.cat([x, xs.expand(b, 1, h, w), ys.expand(b, 1, h, w)], 1))
            f8 = self.deep(f4)
            f = torch.cat([f4, F.interpolate(f8, size=f4.shape[-2:], mode="bilinear", align_corners=False)], 1)
            emb = self.task(task)
            g, beta = self.film(emb).chunk(2, -1)
            f = self.fuse(f * (1 + g[..., None, None]) + beta[..., None, None])
            hh, ww = f.shape[-2:]
            z = [emb]
            if self.use_wrist:
                s = (state - state.new_tensor(STATE_MEAN)) / state.new_tensor(STATE_STD)
                z += [self.wrist(self._img(wrist, wrist.shape[1])), s]
            # Geometry in float32: under bf16 autocast, a 1 m coordinate is only resolved to ~0.4 cm
            with torch.autocast(device_type=x.device.type, enabled=False):
                f = f.float()
                p = self.heat(f).flatten(2).softmax(-1)  # (B, 2, cells)
                off = torch.tanh(self.off(f)).view(b, 2, 2, hh * ww)  # (B, point, col/row, cells)
                cy, cx = torch.meshgrid(
                    (torch.arange(hh, device=x.device) + 0.5) * IMG / hh,
                    (torch.arange(ww, device=x.device) + 0.5) * IMG / ww,
                    indexing="ij",
                )
                col = (p * (cx.flatten() + IMG / ww * off[:, :, 0])).sum(-1)
                row = (p * (cy.flatten() + IMG / hh * off[:, :, 1])).sum(-1)
                z = torch.cat([torch.einsum("bkn,bcn->bkc", p, f.flatten(2)).flatten(1)] + [v.float() for v in z], 1)
                o = self.head(z)
                zb, zp = 0.95 + 0.1 * o[:, 0], 0.911 + 0.1 * o[:, 2]  # m; plate centre 91.1 cm at every reset
                bowl = backproject(col[:, 0], row[:, 0], zb)
                if self.use_wrist:
                    bowl = bowl + torch.cat([0.02 * o[:, 4:6], torch.zeros_like(o[:, :1])], 1)
                plate = backproject(col[:, 1], row[:, 1], zp)
                lows = torch.stack([zb - 0.0284 - 0.01 * o[:, 1], zp - 0.0085 - 0.01 * o[:, 3]], 1)
            return torch.cat([bowl, plate, lows], 1), torch.stack([col, row], -1)

    return Estimator


STATE_MEAN = np.array([-0.1, 0.0, 1.05, 0.0, 0.0, 0.0, 0.02, -0.02])
STATE_STD = np.array([0.2, 0.2, 0.15, 3.0, 3.0, 3.0, 0.02, 0.02])


def load_model(path: Path, device: str = "cpu"):
    import torch

    ck = torch.load(path, map_location=device, weights_only=False)
    model = _net()(**ck["config"]).to(device)
    model.load_state_dict(ck["weights"])
    return model.eval()


def estimate(model, frame: dict, task: int) -> np.ndarray:
    """One frame (snapshot_frame) -> LABELS estimate."""
    import torch

    dev = next(model.parameters()).device
    with torch.no_grad():
        y, _ = model(
            torch.as_tensor(frame["agent"][None], device=dev),
            torch.tensor([task], device=dev),
            torch.as_tensor(frame["wrist"][None], device=dev),
            torch.as_tensor(frame["state"][None], dtype=torch.float32, device=dev),
        )
    return y[0].double().cpu().numpy()


# ---------------------------------------------------------------------------------------------- training


def load_split(d: Path, res: int, wrist: bool = True) -> dict[str, np.ndarray]:
    """All shards of a split in preallocated arrays (one copy in RAM); agentview downsampled to res here, once."""
    files = sorted(d.glob("t*_i*.npz"))
    sizes = [len(np.load(f)["task"]) for f in files]  # reads one small member per shard
    out: dict[str, np.ndarray] = {}
    i = 0
    for f, n in zip(files, sizes, strict=True):
        z = np.load(f)
        for k in z.files:
            if k == "wrist" and not wrist:
                continue
            v = z[k]
            if k == "agent" and res != v.shape[1]:
                m = v.shape[1] // res
                v = (v.reshape(len(v), res, m, res, m, 3).mean((2, 4)) + 0.5).astype(np.uint8)
            if k not in out:
                out[k] = np.empty((sum(sizes), *v.shape[1:]), v.dtype)
            out[k][i : i + n] = v
        i += n
    return out


def _batch(data: dict, idx: np.ndarray, device: str) -> tuple:
    import torch

    t = lambda k, dt=None: torch.as_tensor(data[k][idx], device=device, dtype=dt) if k in data else None  # noqa: E731
    return t("agent"), t("task", torch.long), t("wrist"), t("state", torch.float32), t("y", torch.float32)


def _augment(img):
    """Lighting and sensor noise only: the camera is fixed, so no crops or shifts."""
    import torch

    x = img.float()
    b = x.shape[0]
    gain = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
    shift = torch.empty(b, 1, 1, 3, device=x.device).uniform_(-8, 8)
    x = x * gain + shift + 3 * torch.randn_like(x)
    return x.clamp(0, 255).to(torch.uint8)


def predict(model, data: dict, device: str, batch: int = 256) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for i in range(0, len(data["y"]), batch):
            a, t, w, s, _ = _batch(data, np.arange(i, min(i + batch, len(data["y"]))), device)
            out.append(model(a, t, w, s)[0].float().cpu().numpy())
    return np.concatenate(out)


def errors(pred: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame errors in cm."""
    d = 100 * (pred - y)
    raised = lambda v: v[:, 6] > v[:, 7] + 0.01  # noqa: E731  (scene_of's raised-start test)
    return {
        "bowl": np.linalg.norm(d[:, :3], axis=1),
        "bowl_xy": np.linalg.norm(d[:, :2], axis=1),
        "bowl_z": np.abs(d[:, 2]),
        "plate_xy": np.linalg.norm(d[:, 3:5], axis=1),
        "bowl_lo": np.abs(d[:, 6]),
        "raised_wrong": raised(pred) != raised(y),
    }


@dataclass
class Train:
    """Train an estimator on <data>/train, select on <data>/val -> <out>/<name>.pt (+ val predictions)."""

    name: str
    res: int = 128
    wrist: bool = False
    steps: int = 20000
    batch: int = 64
    lr: float = 1e-3
    data: Path = OUT / "data"
    out: Path = OUT / "models"


def train(cfg: Train) -> None:
    import torch
    import torch.nn.functional as F

    cfg.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    data = load_split(cfg.data / "train", cfg.res, cfg.wrist)
    val = load_split(cfg.data / "val", cfg.res, wrist=cfg.wrist)
    print(f"train {len(data['y'])} frames, val {len(val['y'])}", flush=True)
    model = _net()(res=cfg.res, wrist=cfg.wrist).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    best, log, t0 = np.inf, [], time.time()
    for step in range(1, cfg.steps + 1):
        a, t, w, s, y = _batch(data, rng.integers(len(data["y"]), size=cfg.batch), dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred, _ = model(_augment(a), t, None if w is None else _augment(w), s)
        loss = F.smooth_l1_loss(100 * pred.float(), 100 * y, beta=0.5)  # cm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == cfg.steps:
            model.eval()
            e = errors(predict(model, val, dev), val["y"])
            model.train()
            score = float(np.mean(e["bowl"]) + 0.5 * np.mean(e["plate_xy"]))
            log.append({"step": step, "loss": float(loss.detach()), **{k: float(np.mean(v)) for k, v in e.items()}})
            print(json.dumps(log[-1]) + f" ({time.time() - t0:.0f} s)", flush=True)
            if score < best:
                best = score
                ck = {"config": {"res": cfg.res, "wrist": cfg.wrist}, "weights": model.state_dict(), "step": step}
                torch.save(ck, cfg.out / f"{cfg.name}.pt")
    (cfg.out / f"{cfg.name}.json").write_text(
        json.dumps({"train": vars(cfg) | {"data": str(cfg.data), "out": str(cfg.out)}, "log": log}, indent=1)
    )


# ---------------------------------------------------------------------------------------------- accuracy


@dataclass
class Accuracy:
    """Errors of trained estimators on the validation frames (init states 40-49), by source, task and renderer,
    against the per-task mean guess -> <out>/accuracy.md, and overlay figures."""

    names: tyro.conf.Positional[list[str]]
    data: Path = OUT / "data"
    models: Path = OUT / "models"
    out: Path = OUT


def accuracy(cfg: Accuracy) -> None:
    mean_y = _train_means(cfg.data)
    lines = ["# Estimator accuracy on held-out init states 40-49", ""]
    for split in ("val", "val_nvidia"):
        if not (cfg.data / split).exists():
            continue
        frames = {}
        for name in cfg.names:
            model = load_model(cfg.models / f"{name}.pt", "cuda")
            val = load_split(cfg.data / split, model.res, wrist=model.use_wrist)
            frames[name] = errors(predict(model, val, "cuda"), val["y"])
            del model
        frames["task mean"] = errors(np.stack([mean_y[t] for t in val["task"]]), val["y"])
        src = np.array(SOURCES)[val["source"]]
        groups = {"all": np.ones(len(src), bool), "reset": src == "reset"}
        groups |= {s: src == s for s in ("take", "take_miss", "own_cross", "moved")}
        groups |= {"takeover-like (trigger)": val["trigger"], "bowl held": val["held"]}
        lines += [
            f"## {split}: {len(src)} frames",
            "",
            "Mean error, cm (bowl 3-D / bowl xy / bowl z / plate xy), and frames whose raised-start test flips",
            "",
        ]
        lines += ["| frames | n | " + " | ".join(frames) + " |", "|---|---|" + "---|" * len(frames)]
        for g, m in groups.items():
            cells = [
                f"{e['bowl'][m].mean():.2f} / {e['bowl_xy'][m].mean():.2f} / {e['bowl_z'][m].mean():.2f} /"
                f" {e['plate_xy'][m].mean():.2f}, {100 * e['raised_wrong'][m].mean():.1f}%"
                for e in frames.values()
            ]
            lines.append(f"| {g} | {m.sum()} | " + " | ".join(cells) + " |")
        best, base = frames[cfg.names[0]], frames["task mean"]
        task = val["task"]
        mid = (src != "reset") & ~val["held"]
        rows = {"reset": src == "reset", "takeover-like": val["trigger"], "mid-episode": mid}
        lines += [
            "",
            f"{cfg.names[0]} vs the per-task mean, per task: mean error, cm, bowl 3-D / plate xy ('mid-episode': every"
            " frame but reset and bowl-held frames)",
            "",
            "| frames | guess | " + " | ".join(f"t{t}" for t in range(10)) + " |",
            "|---|---|" + "---|" * 10,
        ]
        for g, m in rows.items():
            for label, e in ((cfg.names[0], best), ("task mean", base)):
                cells = [
                    f"{e['bowl'][m & (task == t)].mean():.2f} / {e['plate_xy'][m & (task == t)].mean():.2f}"
                    for t in range(10)
                ]
                lines.append(f"| {g} | {label} | " + " | ".join(cells) + " |")
        lines.append("")
        if split == "val":
            error_figure(best["bowl"][mid], base["bowl"][mid], cfg.names[0], cfg.out / "figures")
    (cfg.out / "accuracy.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    overlay(cfg.names[0], cfg.data / "val", cfg.models, cfg.out / "figures")


def error_figure(est: np.ndarray, base: np.ndarray, name: str, out: Path) -> None:
    """Estimator error against the per-task mean's error, i.e. how far the bowl is from its usual place."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    edges = np.array([0, 1, 2, 3, 5, 8, 12, 20])
    k = np.digitize(base, edges) - 1
    keep = [b for b in range(len(edges) - 1) if (k == b).sum() >= 20]
    x = [base[k == b].mean() for b in keep]
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=150)
    ax.plot([0, edges[-1]], [0, edges[-1]], color="#9a9893", ls="--", lw=1.5, label="per-task mean (error = distance)")
    ax.plot(x, [est[k == b].mean() for b in keep], color="#2a78d6", marker="o", lw=2, label=f"{name}, mean")
    ax.plot(x, [np.quantile(est[k == b], 0.9) for b in keep], color="#2a78d6", ls=":", lw=1.5, label=f"{name}, p90")
    ax.set_xlim(0, max(x) + 1)
    ax.set_ylim(0, max(x) + 1)
    ax.set_xlabel("bowl's distance from its per-task mean position (cm)")
    ax.set_ylabel("bowl centre error (cm)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "error_vs_distance.png")


def _train_means(data: Path) -> dict[int, np.ndarray]:
    """Per-task mean labels of the training reset frames (init states 10-39): the no-camera guess."""
    out: dict[int, list] = {}
    for f in sorted((data / "train").glob("t*_i*.npz")):
        z = np.load(f)
        out.setdefault(int(z["task"][0]), []).append(z["y"][z["source"] == SOURCES.index("reset")][0])
    return {t: np.mean(v, axis=0) for t, v in out.items()}


def overlay(name: str, split: Path, models: Path, out: Path, per_source: int = 6) -> None:
    """Predicted (orange) and true (green) bowl and plate centres drawn on validation frames."""
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    model = load_model(models / f"{name}.pt", "cuda")
    val = load_split(split, IMG, wrist=model.use_wrist)
    pred = predict(model, val, "cuda")
    rng = np.random.default_rng(0)
    for src in ("reset", "take", "take_miss", "own_cross", "moved"):
        idx = np.flatnonzero(np.array(SOURCES)[val["source"]] == src)
        if not len(idx):
            continue
        if src != "reset":  # mid-episode: prefer frames recover.py would take over at
            trig = idx[val["trigger"][idx]]
            idx = trig if len(trig) >= per_source else idx
        tiles = []
        for i in rng.choice(idx, size=min(per_source, len(idx)), replace=False):
            img = np.ascontiguousarray(val["agent"][i])
            for y, colour in ((val["y"][i], (40, 200, 40)), (pred[i], (255, 140, 0))):
                for p in (y[:3], y[3:6]):
                    c, r = project(p)
                    cv2.circle(img, (int(round(c)), int(round(r))), 4, colour, 1, cv2.LINE_AA)
            err = 100 * np.linalg.norm(pred[i, :3] - val["y"][i, :3])
            cv2.putText(
                img, f"t{val['task'][i]} {err:.1f}cm", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
            tiles.append(img)
        cv2.imwrite(str(out / f"overlay_{src}.png"), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------------------------- end-to-end checks


CHECK_TAKES = 10  # takes per init state at reset: init states 40-44 use every take once, 45-49 again


def _check_reset_job(task: int, init: int, y_est: list[float], eef0: list[float], mean_y: list[float]) -> list[dict]:
    from . import libero as L
    from .plan import make_plan
    from .replay import run, scene_of
    from .takes import load_takes

    takes = load_takes()
    k = init - VAL_INITS[0]
    picked = [takes[(CHECK_TAKES * k + j) % len(takes)] for j in range(CHECK_TAKES)]
    env = L.make_env(task)
    L.reset(env, init, cameras=False)
    snap = L.checkpoint(env)
    rows = []
    for side in sides_of(task):
        L.restore(env, snap)  # the second side must start from reset too, not from the last replay's end
        scenes = {
            "sim": scene_of(env, task, side),
            "estimate": scene_from(np.array(y_est), np.array(eef0), task, side),
            "task_mean": scene_from(np.array(mean_y), np.array(eef0), task, side),
        }
        for take in picked:
            for cond, sc in scenes.items():
                L.restore(env, snap)
                rep = run(env, make_plan(take, sc), track_bowl=task in ENCLOSED)
                rows.append(
                    {
                        "task": task,
                        "init": init,
                        "side": side,
                        "take": take.name,
                        "cond": cond,
                        "ok": rep.ok,
                        "reason": rep.reason,
                    }
                )
    env.close()
    return rows


@dataclass
class CheckReset:
    """Check A: CHECK_TAKES takes per init state 40-49 of every task, planned at reset from sim state, from the
    estimate (on the reset frames in <data>/val) and from the per-task mean -> <out>/check_reset.jsonl."""

    name: str
    tasks: tuple[int, ...] = tuple(range(10))  # tasks to (re)run; other tasks' rows in the results are kept
    data: Path = OUT / "data"
    models: Path = OUT / "models"
    out: Path = OUT
    workers: int = 4


def check_reset(cfg: CheckReset) -> None:
    model = load_model(cfg.models / f"{cfg.name}.pt", "cuda")
    val = load_split(cfg.data / "val", model.res, wrist=model.use_wrist)
    at = np.flatnonzero(val["source"] == SOURCES.index("reset"))
    pred = predict(model, {k: v[at] for k, v in val.items()}, "cuda")
    mean_y = _train_means(cfg.data)
    jobs = [
        (t, int(val["init"][i]), pred[j].tolist(), val["state"][i, :3].tolist(), mean_y[t].tolist())
        for j, i in enumerate(at)
        for t in [int(val["task"][i])]
        if t in cfg.tasks
    ]
    res = cfg.out / "check_reset.jsonl"
    kept = [x for x in res.read_text().splitlines() if json.loads(x)["task"] not in cfg.tasks] if res.exists() else []
    res.write_text("".join(x + "\n" for x in kept))
    _pool(_check_reset_job, jobs, cfg.workers, res)
    check_report(cfg.out)


TAKEOVER_EVERY = 25  # chunk boundaries at n_action_steps 25, as in the round-2 and round-3 collections
MAX_SNAPS, SNAP_SPACING, STALL_STEP = 2, 30, 150  # recover.py
TAKES_PER_STATE = 3  # recover.py tries up to 4; 3 keeps the run under an hour on 4 workers


LOOK_MOVE, LOOK_STEPS = np.array([-0.10, 0.0, 0.10]), 20  # m, control steps


def look_pose(env):
    """Open the gripper and move the hand 10 cm up and 10 cm back toward the robot: a closed gripper over the bowl
    hides it from the agentview camera, which looks down at ~45 deg from the far side. Returns the new state."""
    from . import libero as L

    goal = L.gripper_pos(env) + LOOK_MOVE
    for _ in range(LOOK_STEPS):
        a = HOLD.copy()
        a[:3] = np.clip((goal - L.gripper_pos(env)) / 0.05, -0.5, 0.5)
        env._env.step(a)
    return L.checkpoint(env)


def _check_takeover_job(task: int, init: int, weights: str, mean_y: list[float], look: bool) -> list[dict]:
    """Synthetic failures on one held-out scene: the policy's own recorded actions from another init state, and
    phone takes planned 2.5-5 cm off, played open loop; recover.py's trigger picks the takeover states. From each,
    up to TAKES_PER_STATE takes are planned from sim state, from the estimate and from the per-task mean."""
    import torch

    from . import libero as L
    from .plan import make_plan
    from .replay import run, scene_of
    from .takes import load_takes

    torch.set_num_threads(1)  # the workers share the CPU
    rng = np.random.default_rng([task, init, 11])
    model = load_model(Path(weights), "cpu")
    takes = load_takes()
    env = L.make_env(task)
    own = own_episodes()
    donors = [k for k in own if k[0] == task and k[1] in TRAIN_INITS]
    actors = [("own_cross", own_actions(*own[donors[j]][0])[0]) for j in rng.choice(len(donors), 2, replace=False)]
    for _ in range(4):  # mis-planned takes fail more often than replayed own episodes
        L.reset(env, init, cameras=False)
        sc = scene_of(env, task, sides_of(task)[0])
        sc = replace(sc, bowl_c0=sc.bowl_c0 + np.r_[_ring(rng, 2.5, 5.0), 0.0])
        actors.append(("take_miss", np.array(run(env, make_plan(takes[int(rng.integers(50))], sc)).actions)))
    L.reset(env, init, cameras=False)
    bowl_reset, rows = labels(env)[:3], []
    for a_i, (kind, actions) in enumerate(actors):
        L.reset(env, init, cameras=False)
        snaps, closed = [], 0
        for k, a in enumerate(actions):
            env._env.step(np.asarray(a, dtype=np.float32))
            closed = closed + 1 if a[6] > 0 else 0
            if env._env.check_success():
                break
            n = k + 1
            if n % TAKEOVER_EVERY:
                continue
            bowl_z = L.mesh_centre(env, L.target_object(env))[2]
            empty = closed >= CLOSED_STEPS and L.gripper_pos(env)[2] > bowl_z + EMPTY_DZ
            if (
                (empty or n >= STALL_STEP)
                and len(snaps) < MAX_SNAPS
                and (not snaps or n - snaps[-1][0] >= SNAP_SPACING)
            ):
                snaps.append((n, L.checkpoint(env)))
        for s_i, (n, snap) in enumerate(snaps):
            L.restore(env, snap)
            if look:
                snap = look_pose(env)  # every plan then starts from the new state
            frame = snapshot_frame(env)
            y_true, y_est = labels(env), estimate(model, frame, task)
            eef0 = frame["state"][:3]
            e = errors(y_est[None], y_true[None])
            start = (TAKES_PER_STATE * (10 * a_i + s_i + 7 * init)) % len(takes)
            for side in sides_of(task):
                L.restore(env, snap)
                scenes = {
                    "sim": scene_of(env, task, side),
                    "estimate": scene_from(y_est, eef0, task, side),
                    "task_mean": scene_from(np.array(mean_y), eef0, task, side),
                }
                for j in range(TAKES_PER_STATE):
                    take = takes[(start + j) % len(takes)]
                    for cond, sc in scenes.items():
                        L.restore(env, snap)
                        rep = run(env, make_plan(take, sc), track_bowl=task in ENCLOSED)
                        rows.append(
                            {
                                "task": task,
                                "init": init,
                                "actor": kind,
                                "actor_i": a_i,
                                "step": n,
                                "state": s_i,
                                "side": side,
                                "take": take.name,
                                "cond": cond,
                                "ok": rep.ok,
                                "reason": rep.reason,
                                "bowl_err_cm": float(e["bowl"][0]),
                                "plate_err_cm": float(e["plate_xy"][0]),
                                "bowl_moved_cm": float(100 * np.linalg.norm(y_true[:3] - bowl_reset)),
                            }
                        )
    env.close()
    return rows


@dataclass
class CheckTakeover:
    """Check B: recovery takeovers on init states 40-49 (task 4 has none, as in recover.py), planned from sim state,
    from the estimate and from the per-task mean -> <out>/check_takeover[_look].jsonl."""

    name: str
    look: bool = False  # first open the gripper and lift the hand out of the camera's view (look_pose)
    data: Path = OUT / "data"
    models: Path = OUT / "models"
    out: Path = OUT
    workers: int = 4


def check_takeover(cfg: CheckTakeover) -> None:
    mean_y = _train_means(cfg.data)
    w = str(cfg.models / f"{cfg.name}.pt")
    jobs = [(t, i, w, mean_y[t].tolist(), cfg.look) for t in range(10) if t not in ENCLOSED for i in VAL_INITS]
    res = cfg.out / f"check_takeover{'_look' if cfg.look else ''}.jsonl"
    res.unlink(missing_ok=True)
    _pool(_check_takeover_job, jobs, cfg.workers, res)
    check_report(cfg.out)


def check_report(out: Path) -> None:
    """check_<kind>.md: pass rates by planning source, paired."""
    import pandas as pd

    for kind in ("reset", "takeover", "takeover_look"):
        f = out / f"check_{kind}.jsonl"
        if not f.exists():
            continue
        df = pd.DataFrame([json.loads(x) for x in f.read_text().splitlines()])
        df["side"] = df.side.astype(float).fillna(0)  # None (one grasp side) would drop out of the pivot
        key = ["task", "init", "side", "take"] + (["actor_i", "state"] if kind != "reset" else [])
        wide = df.pivot_table(index=key, columns="cond", values="ok", aggfunc="first").astype(float)
        conds = [c for c in ("sim", "estimate", "task_mean") if c in wide]
        title = {"reset": "A: takes planned at reset", "takeover": "B: takes planned at takeover states"}
        title["takeover_look"] = "B, look first: takeover states, hand lifted out of view before estimating"
        lines = [f"# Check {title[kind]}, init states 40-49", ""]
        if kind != "reset":
            st = df.groupby(["task", "init", "actor_i", "state"]).first()
            lines += [
                f"{len(st)} takeover states ({(st.actor == 'own_cross').sum()} from the policy's recorded actions,"
                f" {(st.actor == 'take_miss').sum()} from mis-planned takes); estimate error at those states:"
                f" bowl {st.bowl_err_cm.mean():.2f} cm mean ({st.bowl_err_cm.quantile(0.9):.2f} p90), plate"
                f" {st.plate_err_cm.mean():.2f}; bowl moved from its reset position {st.bowl_moved_cm.median():.1f} cm"
                f" median ({(st.bowl_moved_cm > 2).mean() * 100:.0f}% moved > 2 cm).",
                "",
            ]
            rec = df.groupby(["task", "init", "actor_i", "state", "cond"]).ok.any().unstack()
            lines += ["Recovered (any of the state's takes x sides passes; recover.py's rule):", ""]
            lines += [
                "| plan from | recovered | per task " + " ".join(f"t{t}" for t in sorted(df.task.unique())) + " |",
                "|---|---|---|",
            ]
            for c in conds:
                per = rec[c].groupby(level="task").mean()
                lines.append(
                    f"| {c} | {100 * rec[c].mean():.1f}% ({int(rec[c].sum())}/{len(rec)}) | "
                    + " ".join(f"{100 * v:.0f}" for v in per)
                    + " |"
                )
            lines.append("")
        lines += ["Per replay (same takes, same states):", ""]
        lines += [
            "| plan from | pass | per task " + " ".join(f"t{t}" for t in sorted(df.task.unique())) + " |",
            "|---|---|---|",
        ]
        for c in conds:
            per = wide[c].groupby(level="task").mean()
            lines.append(
                f"| {c} | {100 * wide[c].mean():.1f}% ({int(wide[c].sum())}/{len(wide)}) | "
                + " ".join(f"{100 * v:.0f}" for v in per)
                + " |"
            )
        if "estimate" in wide:
            d = wide["estimate"] - wide["sim"]
            groups = d.groupby(level=["task", "init"]).mean()
            boot = [groups.sample(len(groups), replace=True, random_state=i).mean() for i in range(2000)]
            lines += [
                "",
                f"estimate - sim: {100 * d.mean():+.1f} points [{100 * np.quantile(boot, 0.025):+.1f},"
                f" {100 * np.quantile(boot, 0.975):+.1f}] (95% bootstrap over scenes); replays passing with sim and"
                f" failing with the estimate: {int(((wide['sim'] == 1) & (wide['estimate'] == 0)).sum())}, the"
                f" reverse: {int(((wide['sim'] == 0) & (wide['estimate'] == 1)).sum())}",
            ]
        (out / f"check_{kind}.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


# ---------------------------------------------------------------------------------------------- demos


def _demos_job(task: int, weights: str, root: str, per_task: int, tries: int) -> list[dict]:
    """Phone demos for one task planned from the estimate at reset: init states 10, 11, ... in order, each used at
    most once, takes spread evenly (least used first), the first passing replay per init state kept."""
    from collections import Counter

    import torch

    from . import libero as L
    from .dataset import add_episode, create, finish
    from .plan import make_plan
    from .replay import run
    from .takes import load_takes

    torch.set_num_threads(2)
    model = load_model(Path(weights), "cpu")
    takes = load_takes()
    env = L.make_env(task)
    ds = create(Path(root), overwrite=True)
    uses, kept = Counter(), []
    for init in range(10, 50):
        if len(kept) == per_task:
            break
        L.reset(env, init, cameras=True)
        frame = snapshot_frame(env)
        scene = scene_from(estimate(model, frame, task), frame["state"][:3], task)
        for take in sorted(takes, key=lambda t: (uses[t.name], t.name))[:tries]:
            L.reset(env, init, cameras=True)
            rep = run(env, make_plan(take, scene), record=True, track_bowl=task in ENCLOSED)
            if rep.ok and rep.first_success is not None:
                n = add_episode(ds, rep.frames, rep.actions, rep.first_success, env.task_description)
                uses[take.name] += 1
                kept.append({"task": task, "init": init, "take": take.name, "n_frames": n})
                print(f"task {task} init {init} {take.name}: {n} frames", flush=True)
                break
    env.close()
    finish(ds, Path(root))
    (Path(root) / "meta" / "phone2libero_manifest.json").write_text(json.dumps(kept, indent=1))
    return kept


@dataclass
class Demos:
    """Phone demos (as dataset.py builds them from the grid) with every take planned from the estimate."""

    name: str = "a128"
    out: Path = Path("outputs/scene/phone")
    tasks: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 6, 7, 8])  # the released model's phone tasks
    per_task: int = 20
    tries: int = 6  # takes per init state before moving on
    workers: int = 4


def demos(cfg: Demos) -> None:
    jobs = [
        (t, str(OUT / "models" / f"{cfg.name}.pt"), str(cfg.out / f"t{t}"), cfg.per_task, cfg.tries) for t in cfg.tasks
    ]
    with ProcessPoolExecutor(cfg.workers, mp_context=get_context("spawn")) as ex:
        for fut in as_completed([ex.submit(_demos_job, *j) for j in jobs]):
            kept = fut.result()
            print(f"task {kept[0]['task'] if kept else '?'}: {len(kept)} demos", flush=True)


def main() -> None:
    cmd = tyro.cli(
        Annotated[Sensitivity, tyro.conf.subcommand("sensitivity")]
        | Annotated[Generate, tyro.conf.subcommand("generate")]
        | Annotated[Train, tyro.conf.subcommand("train")]
        | Annotated[Accuracy, tyro.conf.subcommand("accuracy")]
        | Annotated[CheckReset, tyro.conf.subcommand("check-reset")]
        | Annotated[CheckTakeover, tyro.conf.subcommand("check-takeover")]
        | Annotated[Demos, tyro.conf.subcommand("demos")]
    )
    fn = {Sensitivity: sensitivity, Generate: generate, Train: train, Accuracy: accuracy}
    fn |= {CheckReset: check_reset, CheckTakeover: check_takeover, Demos: demos}
    fn[type(cmd)](cmd)


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

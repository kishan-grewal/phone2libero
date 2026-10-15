"""README media: the hero GIF (phone take | the same take retargeted onto the robot | the trained policy), a grid of
one take retargeted onto all ten tasks, a retry by the trained policy, and the results chart. Needs data/raw
(just download-videos) and evals.

  python tools/media.py hero --policy-eval <evals name>
  python tools/media.py scenes
  python tools/media.py diagram
  python tools/media.py retry --policy-eval <evals name>
  python tools/media.py results baseline <model> ...
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path("assets")
EVALS = Path("outputs/evals")
H = 240  # panel height, px
FONT = ImageFont.load_default(size=17)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    w = d.textlength(text, font=FONT)
    d.rectangle([0, 0, w + 16, 28], fill=(0, 0, 0))
    d.text((8, 5), text, fill=(255, 255, 255), font=FONT)
    return np.asarray(im)


DONE = "success: bowl on plate"  # LIBERO ends the episode here, so the robot never lets go


def _banner(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    w = d.textlength(text, font=FONT)
    h = im.height
    d.rectangle([0, h - 28, w + 16, h], fill=(20, 110, 60))
    d.text((8, h - 23), text, fill=(255, 255, 255), font=FONT)
    return np.asarray(im)


def _fit(img: np.ndarray, h: int = H) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((round(img.shape[1] * h / img.shape[0]), h), Image.LANCZOS))


def _save_gif(frames: list[np.ndarray], path: Path, fps: float) -> None:
    ims = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=96) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=round(1000 / fps), loop=0, optimize=True)
    print(f"-> {path} ({path.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


def _replay(take_name: str, task: int, init: int, side: int | None = None) -> tuple[list[np.ndarray], object, object]:
    from phone2libero import libero as L
    from phone2libero.plan import make_plan
    from phone2libero.replay import ENCLOSED, run, scene_of
    from phone2libero.takes import load_takes

    take = next(t for t in load_takes() if t.name == take_name)
    env = L.make_env(task)
    L.reset(env, init, cameras=True)
    plan = make_plan(take, scene_of(env, task, side))
    rep = run(env, plan, record=True, track_bowl=task in ENCLOSED)
    env.close()
    return [f["observation.images.image"] for f in rep.frames], rep, take


def _phone_frames(video: str, t0: float, t1: float) -> tuple[np.ndarray, list[np.ndarray]]:
    import av

    times, frames = [], []
    with av.open(f"data/raw/{video}.mp4") as c:
        s = c.streams.video[0]
        c.seek(int(max(t0 - 1, 0) / s.time_base), stream=s)
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base)
            if t < t0:
                continue
            if t > t1:
                break
            times.append(t)
            frames.append(fr.to_ndarray(format="rgb24")[:, 160:1760])  # trim the empty card edges
    return np.array(times), frames


def _policy_frames(name: str, task: int) -> list[np.ndarray]:
    """A successful episode of `task` from a model's seed-0 eval videos."""
    import imageio.v3 as iio

    info = json.loads((EVALS / name / "seed_0" / "eval_info.json").read_text())
    succ = next(t for t in info["per_task"] if t["task_id"] == task)["metrics"]["successes"]
    ep = succ.index(True)
    return list(iio.imread(EVALS / name / "seed_0" / "videos" / f"libero_spatial_{task}" / f"eval_episode_{ep}.mp4"))


@dataclass
class Hero:
    """Phone take | the same take retargeted onto the robot | the trained policy on an eval episode."""

    policy_eval: str  # evals name of the headline model (outputs/evals/<name>)
    take: str = "G2_gap12/01"
    task: int = 0
    init: int = 10
    fps: float = 10.0


@dataclass
class Scenes:
    """One take retargeted onto all ten tasks (an init state where it passes, from the grid results)."""

    take: str = "G1_gap16/03"
    fps: float = 10.0


@dataclass
class Diagram:
    """The pipeline: capture -> retargeting -> phone demos -> train, and the self-improvement loop."""


@dataclass
class Retry:
    """A late success from a model's eval videos (it missed, then tried again), shown at 2x speed."""

    policy_eval: str
    task: int = 2
    seed: int = 0
    episode: int = 2
    fps: float = 10.0


@dataclass
class Results:
    """Bar chart of success per task for the given eval names (first = reference)."""

    names: tyro.conf.Positional[list[str]]
    labels: list[str] | None = None


def hero(cfg: Hero) -> None:
    from phone2libero.plan import APPROACH_WINDOW, HZ, T_BLEND, TIME_SCALE

    sim, rep, take = _replay(cfg.take, cfg.task, cfg.init)
    assert rep.ok, rep.reason
    takes = json.loads(Path(f"data/takes/{take.video}/results.json").read_text())["takes"]
    t_start = takes[int(cfg.take[-2:]) - 1]["take"]["t_start"]  # video time when the hand enters
    t_a = take.events["close"] - APPROACH_WINDOW
    tau = np.arange(len(sim)) / HZ
    phone_t = t_start + t_a + (tau - T_BLEND) / TIME_SCALE  # the plan's time map, so both show the same moment
    times, phone = _phone_frames(take.video, phone_t.min(), phone_t.max())
    policy = _policy_frames(cfg.policy_eval, cfg.task)
    n = max(len(sim), len(policy))
    step = round(HZ / cfg.fps)
    frames = []
    for k in range(0, n, step):
        i = min(k, len(sim) - 1)
        p = phone[int(np.argmin(np.abs(times - phone_t[i])))]
        j = min(k, len(policy) - 1)
        sim_f = _label(_fit(sim[i]), "my take, on the robot")
        pol_f = _label(_fit(policy[j]), "trained policy, new scene")
        row = [
            _label(_fit(p), f"my hand, phone ({1 / TIME_SCALE:.1f}x speed)"),
            _banner(sim_f, DONE) if i >= rep.first_success else sim_f,
            _banner(pol_f, DONE) if j == len(policy) - 1 else pol_f,
        ]
        frames.append(np.concatenate(row, 1))
    ASSETS.mkdir(exist_ok=True)
    _save_gif(frames, ASSETS / "hero.gif", cfg.fps)


def scenes(cfg: Scenes) -> None:
    from phone2libero.plan import HZ

    rows = [json.loads(x) for g in Path("outputs/grid").glob("*/replays.jsonl") for x in g.read_text().splitlines()]
    tiles = []
    for task in range(10):
        r = min((r for r in rows if r["task"] == task and r["take"] == cfg.take and r["ok"]), key=lambda r: r["init"])
        sim, rep, _ = _replay(cfg.take, task, r["init"], r.get("side"))
        tiles.append([_label(_fit(f, 160), f"task {task}") for f in sim])
    n = max(len(t) for t in tiles)
    step = round(HZ / cfg.fps)
    frames = []
    for k in range(0, n, step):
        t = [tile[min(k, len(tile) - 1)] for tile in tiles]
        frames.append(np.concatenate([np.concatenate(t[:5], 1), np.concatenate(t[5:], 1)], 0))
    ASSETS.mkdir(exist_ok=True)
    _save_gif(frames, ASSETS / "one_take_ten_scenes.gif", cfg.fps)


def diagram(cfg: Diagram) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    boxes = {
        "capture": (0.0, 1.0, "1. Capture", "phone video\n→ 50 takes of my hand"),
        "retarget": (1.0, 1.0, "2. Retargeting", "each take → closed-loop\nrobot replay, checked"),
        "demos": (2.0, 1.0, "3. Phone demos", "20 passing replays\nper task"),
        "train": (3.0, 1.0, "4. Train SmolVLA", "5-shot demos\n+ everything so far"),
        "run": (3.0, 0.0, "5. Run the policy", "keep its successes;\na phone take finishes\neach failure"),
    }
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150)
    w, h = 0.82, 0.62
    for x, y, title, body in boxes.values():
        loop = y == 0.0
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor="#e8f1fb" if loop else "#f3f2ef",
                edgecolor="#2a78d6" if loop else "#9a9893",
                lw=1.2,
            )
        )
        ax.text(x, y + 0.17, title, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x, y - 0.07, body, ha="center", va="center", fontsize=8.5, linespacing=1.3)

    def arrow(a, b, rad=0.0, color="#555555"):
        ax.add_patch(
            FancyArrowPatch(
                a, b, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.2, connectionstyle=f"arc3,rad={rad}"
            )
        )

    for x in (0.0, 1.0, 2.0):
        arrow((x + w / 2, 1.0), (x + 1 - w / 2, 1.0))
    arrow((3.0 + 0.12, 1.0 - h / 2), (3.0 + 0.12, h / 2), rad=-0.0, color="#2a78d6")
    arrow((3.0 - 0.12, h / 2), (3.0 - 0.12, 1.0 - h / 2), color="#2a78d6")
    ax.text(3.0 + 0.25, 0.5, "each round:\nnew data,\nretrain", ha="left", va="center", fontsize=8.5, color="#2a78d6")
    ax.set_xlim(-0.5, 3.95)
    ax.set_ylim(-0.4, 1.4)
    ax.axis("off")
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "pipeline.png")
    print(f"-> {ASSETS / 'pipeline.png'}")


def retry(cfg: Retry) -> None:
    import imageio.v3 as iio

    video = EVALS / cfg.policy_eval / f"seed_{cfg.seed}" / "videos" / f"libero_spatial_{cfg.task}"
    frames = list(iio.imread(video / f"eval_episode_{cfg.episode}.mp4"))
    step = 2 * round(20 / cfg.fps)  # LIBERO runs at 20 Hz
    out = [_label(_fit(f), f"trained policy, step {k}") for k, f in enumerate(frames) if k % step == 0]
    out += [_banner(out[-1], DONE)] * round(1.5 * cfg.fps)
    ASSETS.mkdir(exist_ok=True)
    _save_gif(out, ASSETS / "retry.gif", cfg.fps)


def results(cfg: Results) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from phone2libero.evaluate import load, rate

    labels = cfg.labels or cfg.names
    colors = ["#9a9893", "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"]
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    w = 0.8 / len(cfg.names)
    for j, (n, lab) in enumerate(zip(cfg.names, labels)):
        r = load(n)
        vals = [rate(r, [t])[0] for t in range(10)] + [rate(r, list(range(10)))[0]]
        ax.bar(
            np.arange(11) + (j - (len(cfg.names) - 1) / 2) * w,
            vals,
            w,
            label=f"{lab} ({vals[-1]:.1f}%)",
            color=colors[j % len(colors)],
        )
    ax.set_xticks(range(11), [f"task {t}" for t in range(10)] + ["all"])
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "results.png")
    print(f"-> {ASSETS / 'results.png'}")


def main() -> None:
    cmd = tyro.cli(
        Annotated[Hero, tyro.conf.subcommand("hero")]
        | Annotated[Scenes, tyro.conf.subcommand("scenes")]
        | Annotated[Diagram, tyro.conf.subcommand("diagram")]
        | Annotated[Retry, tyro.conf.subcommand("retry")]
        | Annotated[Results, tyro.conf.subcommand("results")]
    )
    os.environ.setdefault("MUJOCO_GL", "egl")
    {Hero: hero, Scenes: scenes, Diagram: diagram, Retry: retry, Results: results}[type(cmd)](cmd)


if __name__ == "__main__":
    main()

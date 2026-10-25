"""README media: the hero GIF and MP4 (phone take | the same take retargeted onto the robot | the trained policy), a grid of
one take retargeted onto all ten tasks, the retargeting figure, a retry by the trained policy, and the results
chart. Needs data/raw (just download-videos) and evals.

  python tools/media.py hero --policy-eval <evals name>
  python tools/media.py scenes
  python tools/media.py retarget
  python tools/media.py diagram
  python tools/media.py retry --policy-eval <evals name>
  python tools/media.py montage-rollouts   # the policy's episodes for the montage (GPU, ~10 min)
  python tools/media.py montage
  python tools/media.py results baseline <model> ...
"""

from __future__ import annotations

import json
import os
import subprocess
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


def _ffmpeg(frames: list[np.ndarray], fps: float, args: list[str], path: Path) -> None:
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps)]
    subprocess.run([*cmd, "-i", "-", *args, str(path)], input=b"".join(f.tobytes() for f in frames), check=True)
    print(f"-> {path} ({path.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


def _save_gif(frames: list[np.ndarray], path: Path, fps: float) -> None:
    """One 256-colour palette for the whole clip (per-frame palettes shimmer), dithered to hide banding."""
    vf = "split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
    _ffmpeg(frames, fps, ["-vf", vf, "-loop", "0"], path)


def _save_mp4(frames: list[np.ndarray], path: Path, fps: float, scale: int = 2) -> None:
    vf = f"scale=iw*{scale}:ih*{scale}:flags=lanczos"
    _ffmpeg(frames, fps, ["-vf", vf, "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart"], path)


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
class Retarget:
    """One take on two scenes, side by side with the phone: the bowl and grip paths along the bowl -> plate line."""

    take: str = "G2_gap12/01"
    init: int = 10


MONTAGE = Path("outputs/montage")
MONTAGE_TAKES = ("G1_gap16/03", "G2_gap12/01", "G3_gap2/05")  # one per start distance: 16, 12, 2 cm


@dataclass
class MontageRollouts:
    """The released policy's clean episodes (a single grasp, then success), several per task, so the montage can
    pick ones whose timing already matches."""

    policy: str = "kishan-grewal/phone2libero-smolvla"
    per_task: int = 10


@dataclass
class Montage:
    """A video for sharing: my take in the middle, the policy on all ten tasks around it (0-3 above, 4 left, 5 right,
    6-9 below), one cycle per take. My take is sped up to the robot's pace; each tile starts, grasps and lands with
    my hand, mostly at 1x or faster. The first cycle opens on the middle of my video, playing until the hand is above the
    bowl, then pauses; the camera walks back to the full layout over zoom_s seconds and then everything plays. No labels."""

    tile: int = 512  # px; the sim's 256 x 256 frames upscaled 2x
    gap: int = 16
    min_speed: float = 0.9  # slowest a tile may play; slowed sim footage looks wrong
    zoom_s: float = 2.0  # seconds of frozen pull-back at the start of the first cycle; 0 = none
    zoom_shift: float = 60  # world px the opening view is left of my video's middle, to keep the bowl in it
    output: str = "montage_2.mp4"


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
    _save_mp4(frames, Path("outputs/hero.mp4"), cfg.fps)


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


# task, title, the geom the bowl starts on (None: the table)
RETARGET_SCENES = ((0, "task 0: bowl on the table", None), (3, "task 3: bowl on the cookie box", "cookies_1_g0"))


def _sim_side(take, task: int, init: int, support: str | None) -> dict:
    """Replay `take` on one scene; paths in cm along the bowl -> plate line (plate centre 0) and above the table."""
    from phone2libero import libero as L
    from phone2libero.plan import make_plan
    from phone2libero.replay import run, scene_of

    env = L.make_env(task)
    L.reset(env, init, cameras=True)
    sc = scene_of(env, task)
    plate_lo, plate_hi = L.mesh_box(env, "plate_1_g0")
    table = plate_lo[2]  # the plate lies on the table
    u = (sc.plate_c[:2] - sc.bowl_c0[:2]) / np.linalg.norm(sc.plate_c[:2] - sc.bowl_c0[:2])

    def side(p: np.ndarray) -> np.ndarray:
        return 100 * np.stack([(p[:, :2] - sc.plate_c[:2]) @ u, p[:, 2] - table], 1)

    box = None
    if support:
        lo, hi = L.mesh_box(env, support)
        corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
        s = 100 * (corners - sc.plate_c[:2]) @ u
        box = (s.min(), s.max(), 100 * (hi[2] - table))
    plan = make_plan(take, sc)
    rep = run(env, plan, record=True)
    env.close()
    assert rep.ok, rep.reason
    n = len(rep.grip_trace)
    return {
        "frames": [f["observation.images.image"] for f in rep.frames],
        "bowl": side(np.array(rep.bowl_trace)),
        "grip": side(np.array(rep.grip_trace)),
        "closed": np.r_[plan.grip > 0, np.zeros(n - len(plan.grip), bool)],
        "plate_r": 50 * (plate_hi[0] - plate_lo[0]),
        "box": box,
    }


def retarget(cfg: Retarget) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from phone2libero.plan import APPROACH_WINDOW, HZ, T_BLEND, TIME_SCALE
    from phone2libero.takes import load_takes

    take = next(t for t in load_takes() if t.name == cfg.take)
    ev = take.events
    res = json.loads(Path(f"data/takes/{take.video}/results.json").read_text())
    t_start = res["takes"][int(cfg.take[-2:]) - 1]["take"]["t_start"]
    with open(f"data/takes/{cfg.take.replace('/', '/take_')}.csv") as f:
        bowl_z = np.genfromtxt(f, delimiter=",", names=True)["bowl_z_cm"]  # bowl centre; Take keeps only its base
    t_a = ev["close"] - APPROACH_WINDOW
    t_show = ev["lift"] + 0.45 * (ev["placed"] - ev["lift"])  # mid-carry, take clock
    k_show = round((T_BLEND + (t_show - t_a) * TIME_SCALE) * HZ)  # the same moment in the replay
    sims = [_sim_side(take, task, cfg.init, support) for task, _, support in RETARGET_SCENES]

    bowl_c, hand_c, ink, muted, fill = "#e0662f", "#2a78d6", "#222222", "#8a8780", "#d9d6cf"
    plt.rcParams.update(
        {
            "font.family": ["Lato", "DejaVu Sans"],
            "font.size": 9,
            "axes.edgecolor": muted,
            "axes.labelcolor": "#444444",
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )
    fig = plt.figure(figsize=(8.6, 7.4), dpi=200)
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1, 1.55], hspace=0.22, wspace=0.02, left=0.01, right=0.985, top=0.965, bottom=0.075
    )

    def side(ax, bowl, hand, closed, plate_r, show, last):
        ax.axhline(0, color=muted, lw=0.8, zorder=0)
        ax.add_patch(plt.Rectangle((-plate_r, 0), 2 * plate_r, 0.9, color=fill, lw=0))
        ax.text(0, -0.6, "plate", ha="center", va="top", fontsize=7.5, color=muted)
        ax.plot(*np.where(closed[:, None], hand, np.nan).T, color=hand_c, lw=2.4, solid_capstyle="round")
        ax.plot(*bowl.T, color=bowl_c, lw=2.0, solid_capstyle="round")
        ax.plot(*bowl[show], "o", ms=6, mfc="white", mec=ink, mew=1.2, zorder=6)
        x0, x1 = bowl[0, 0], bowl[-1, 0]
        ax.annotate(
            "", (x1, 24), (x0, 24), arrowprops=dict(arrowstyle="<->", color=muted, lw=0.7, shrinkA=0, shrinkB=0)
        )
        ax.text(
            (x0 + x1) / 2, 24.6, f"bowl moves {x1 - x0:.0f} cm", ha="center", va="bottom", fontsize=7.5, color=muted
        )
        ax.set_aspect("equal")
        ax.set_xlim(-36, 24)
        ax.set_ylim(-3, 28)
        ax.set_yticks([0, 10, 20])
        ax.set_xticks([-30, -20, -10, 0, 10, 20])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("height (cm)")
        if last:
            ax.set_xlabel("along the bowl → plate line (cm), plate centre at 0")
        else:
            ax.tick_params(labelbottom=False)

    times, phone = _phone_frames(take.video, t_start + t_show - 0.05, t_start + t_show + 0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(phone[0][120:930, 130:1350])
    ax.axis("off")
    ax.set_title("my hand (phone, side-on)", loc="left", fontsize=10, color=ink, pad=4)
    m = (take.t >= t_a) & (take.t <= ev["release"] + 1.0)
    held = (take.t >= ev["close"]) & (take.t < ev["release"])
    ax = fig.add_subplot(gs[0, 1])
    plate_r = res["scene"]["checks"]["plate_outline"]["diameter_cm"] / 2
    show = int(np.argmin(np.abs(take.t[m] - t_show)))
    side(ax, np.stack([take.bowl_x, bowl_z], 1)[m], take.hand[m], held[m], plate_r, show, False)
    ax.text(-12.5, 4.5, "bowl centre", color=bowl_c, fontsize=8.5)
    ax.annotate(
        "pinch point",
        take.hand[m][held[m].nonzero()[0][-1]],
        xytext=(10, -2),
        textcoords="offset points",
        color=hand_c,
        fontsize=8.5,
        va="top",
    )
    ax.text(23.5, 28, "○ = the frame on the left", fontsize=7.5, color=muted, va="top", ha="right")

    for r, ((_, title, _), sim) in enumerate(zip(RETARGET_SCENES, sims), 1):
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(sim["frames"][min(k_show, len(sim["frames"]) - 1)][8:248])
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=10, color=ink, pad=4)
        ax = fig.add_subplot(gs[r, 1])
        side(ax, sim["bowl"], sim["grip"], sim["closed"], sim["plate_r"], k_show, r == len(sims))
        if sim["box"]:
            a, b, h = sim["box"]
            ax.add_patch(plt.Rectangle((a, 0), b - a, h, color=fill, lw=0))
            ax.text((a + b) / 2, -0.6, "cookie box", ha="center", va="top", fontsize=7.5, color=muted)
        if r == 1:
            ax.annotate(
                "grip site",
                sim["grip"][sim["closed"].nonzero()[0][-1]],
                xytext=(10, -2),
                textcoords="offset points",
                color=hand_c,
                fontsize=8.5,
                va="top",
            )
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "retarget.png")
    print(f"-> {ASSETS / 'retarget.png'}")


def montage_rollouts(cfg: MontageRollouts) -> None:
    from phone2libero.libero import make_env
    from phone2libero.policy import Policy
    from phone2libero.selfplay import episode

    pol = Policy(cfg.policy)
    MONTAGE.mkdir(parents=True, exist_ok=True)
    for task in range(10):
        env = make_env(task)
        got = 0
        for init in range(10, 50):  # held-out init states, one episode each so the scenes differ
            first, frames, actions = episode(env, pol, init, seed=init)
            closed = np.array([a[6] > 0 for a in actions[: (first or 0) + 1]])
            onsets = np.flatnonzero(closed[1:] & ~closed[:-1]) + 1
            if first is None or len(onsets) != 1:
                continue
            np.savez_compressed(
                MONTAGE / f"task{task}_init{init}.npz",
                frames=np.stack([f["observation.images.image"] for f in frames[: first + 1]]),
                grasp=onsets[0],
            )
            got += 1
            if got == cfg.per_task:
                break
        print(f"task {task}: {got} clean episodes, last init {init}", flush=True)
        env.close()


def _phone_stream(video: str, times: np.ndarray):
    """The frame of data/raw/<video>.mp4 shown at each of `times` (s, increasing): the latest one not after it."""
    import av

    with av.open(f"data/raw/{video}.mp4") as c:
        s = c.streams.video[0]
        c.seek(int(max(times[0] - 1, 0) / s.time_base), stream=s)
        frames = c.decode(s)
        prev = nxt = next(frames)
        for t in times:
            while float(nxt.pts * s.time_base) <= t:
                prev, nxt = nxt, next(frames)
            yield prev.to_ndarray(format="rgb24")


def _blit(canvas: np.ndarray, src: np.ndarray, rect: tuple, zoom: float, cx: float) -> None:
    """Draw `src` at world rect (x, y, w, h) as seen by a camera centred at x = cx (y at the canvas's middle),
    showing zoom x the canvas's width and height."""
    H, W = canvas.shape[:2]
    x, y = (rect[0] - (cx - W * zoom / 2)) / zoom, (rect[1] - H / 2 * (1 - zoom)) / zoom
    w, h = rect[2] / zoom, rect[3] / zoom
    x0, y0, x1, y1 = max(round(x), 0), max(round(y), 0), min(round(x + w), W), min(round(y + h), H)
    if x1 <= x0 or y1 <= y0:
        return
    sh, sw = src.shape[:2]
    box = ((x0 - x) / w * sw, (y0 - y) / h * sh, (x1 - x) / w * sw, (y1 - y) / h * sh)  # the visible part of src
    canvas[y0:y1, x0:x1] = np.asarray(
        Image.fromarray(src).resize((x1 - x0, y1 - y0), Image.LANCZOS, box=tuple(np.clip(box, 0, (sw, sh, sw, sh))))
    )


def _hover_time(take_name: str) -> float:
    """When the hand is above the bowl and about to go down: the last time before it touches the rim (`close`)
    that its height is within 1 cm of the highest it was in the second before."""
    from phone2libero.takes import load_takes

    tk = next(t for t in load_takes() if t.name == take_name)
    close = tk.events["close"]
    window = (tk.t > close - 1) & (tk.t <= close)
    return float(tk.t[window & (tk.hand[:, 1] > tk.hand[window, 1].max() - 1)][-1])


def montage(cfg: Montage) -> None:
    import av

    hz, fps = 20, 30  # LIBERO's control rate; the output's frame rate
    pool = {}
    for f in sorted(MONTAGE.glob("task*_init*.npz")):
        task = int(f.stem.split("_")[0][4:])
        d = np.load(f)
        pool.setdefault(task, []).append((f, int(d["grasp"]), len(d["frames"]) - 1))
    grasps = [g for eps in pool.values() for _, g, _ in eps]
    carries = [e - g for eps in pool.values() for _, g, e in eps]
    # on-screen times of the grasp and of the carry, a little under the typical robot's so most tiles play fast
    t_grasp, t_carry = 0.92 * np.median(grasps) / hz, 0.92 * np.median(carries) / hz

    t, g = cfg.tile, cfg.gap
    cells = [(c * (t + g), 0) for c in range(4)]  # tasks 0-3
    cells += [(0, t + g), (3 * (t + g), t + g)]  # tasks 4, 5
    cells += [(c * (t + g), 2 * (t + g)) for c in range(4)]  # tasks 6-9
    cw, ch = 2 * t + g, t  # the phone video in the middle
    W, H = 4 * t + 3 * g, 3 * t + 2 * g
    out_path = MONTAGE / cfg.output
    out = av.open(str(out_path), "w")
    stream = out.add_stream("libx264", rate=fps, options={"crf": "18"})
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"

    used: set[Path] = set()
    for take_name in MONTAGE_TAKES:
        video, k = take_name.split("/")
        rec = json.loads(Path(f"data/takes/{video}/results.json").read_text())["takes"][int(k) - 1]
        t_start, ev = rec["take"]["t_start"], rec["events_s"]
        speed = (ev["placed"] - ev["close"]) / t_carry  # my take's speed-up
        first = take_name == MONTAGE_TAKES[0] and cfg.zoom_s > 0
        if first:  # play from the hand entering to the hand above the bowl, hold there while the camera walks back
            hover = _hover_time(take_name)
            lead = (ev["close"] - hover) / speed  # on-screen s from the pause to my hand's grasp
            n_pre, n_hold = round(hover / speed * fps), round(cfg.zoom_s * fps)
            tt0 = t_grasp - lead  # the tiles' clock at the pause
        else:
            hover, n_pre, n_hold, tt0 = ev["close"] - speed * t_grasp, 0, 0, 0.0
        n_out = n_pre + n_hold + int((ev["release"] + 0.5 - hover) / speed * fps)
        tiles = []
        for task in range(10):
            # the unused episode whose approach and carry need the least retiming, slowing down counted 3x
            def cost(e):
                v = np.log([e[1] / hz / t_grasp, (e[2] - e[1]) / hz / t_carry])
                return np.where(v < 0, -3 * v, v).sum()

            f, gr, end = min((e for e in pool[task] if e[0] not in used), key=cost)
            used.add(f)
            va, vc = gr / hz / t_grasp, (end - gr) / hz / t_carry
            if min(va, vc) < cfg.min_speed:
                raise RuntimeError(f"{f.name}: would play at {min(va, vc):.2f}x; roll out more episodes")
            frames = [np.asarray(Image.fromarray(x).resize((t, t), Image.LANCZOS)) for x in np.load(f)["frames"]]
            tiles.append((frames, gr, end))
            print(f"{take_name} task {task}: {f.stem}, approach {va:.2f}x, carry {vc:.2f}x")
        print(f"{take_name}: my take at {speed:.2f}x" + (f", paused at {hover:.2f} s" if first else f", starting {hover:+.2f} s from the hand entering"))
        idx = np.arange(n_out)
        play = np.maximum(idx - n_pre - n_hold, 0) / fps  # on-screen s since the pause
        times = t_start + np.where(idx < n_pre, speed * idx / fps, hover + speed * play)
        for i, img in enumerate(_phone_stream(video, times)):
            canvas = np.zeros((H, W, 3), np.uint8)
            u = np.clip((i - n_pre) / n_hold, 0, 1) if n_hold else 1.0
            u = u * u * (3 - 2 * u)
            zoom = (ch / H) ** (1 - u)  # from my video's full slot height to the whole layout
            cx = W / 2 - cfg.zoom_shift * (1 - u)
            tt = tt0 * i / n_pre if i < n_pre else tt0 + play[i]
            for (x, y), (frames, gr, end) in zip(cells, tiles):
                # start together, grasp with my hand, land as I place, then hold while I let go
                step = np.interp(tt, [0, t_grasp, t_grasp + t_carry], [0, gr, end])
                _blit(canvas, frames[round(step)], (x, y, t, t), zoom, cx)
            rows = round(img.shape[1] * ch / cw)
            _blit(canvas, img[20 : 20 + rows], (t + g, t + g, cw, ch), zoom, cx)
            for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
                out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()
    print(f"-> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


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
        | Annotated[Retarget, tyro.conf.subcommand("retarget")]
        | Annotated[MontageRollouts, tyro.conf.subcommand("montage-rollouts")]
        | Annotated[Montage, tyro.conf.subcommand("montage")]
        | Annotated[Diagram, tyro.conf.subcommand("diagram")]
        | Annotated[Retry, tyro.conf.subcommand("retry")]
        | Annotated[Results, tyro.conf.subcommand("results")]
    )
    os.environ.setdefault("MUJOCO_GL", "egl")
    {
        Hero: hero,
        Scenes: scenes,
        Retarget: retarget,
        MontageRollouts: montage_rollouts,
        Montage: montage,
        Diagram: diagram,
        Retry: retry,
        Results: results,
    }[type(cmd)](cmd)


if __name__ == "__main__":
    main()

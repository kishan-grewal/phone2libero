"""Replay a retargeted plan in LIBERO with a closed-loop end-effector tracker, and check that it really worked.

Actions are exactly what the 5-shot data uses: LIBERO's OSC_POSE controller in delta mode, 7-D in [-1, 1]:
xyz delta (x 0.05 m), axis-angle delta (x 0.5 rad), gripper (-1 open, +1 close). Each control step the action
is recomputed from the current grip site toward the plan's target LEAD steps ahead, so the recorded actions are
the ones actually sent. Orientation is held at the reset pose (the 5-shot demos barely rotate the gripper).

  python -m phone2libero.replay --take G2_gap12/01 --task 0 --init 10   # video + plot in outputs/replay/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import libero as L
from .plan import SIM_PLACE_DZ, Plan, SimScene

POS_SCALE, ROT_SCALE = 0.05, 0.5  # OSC output limits at action = +-1
GAIN, LEAD = 1.0, 2  # action = (target[k + LEAD] - grip site) / POS_SCALE * GAIN
SETTLE_STEPS = 10  # extra steps holding the last target, so a bowl that slides off afterwards still fails
MAX_OTHER_SHIFT_M = 0.02  # every other object (plate included) moves less than this
MAX_SLIP_M = 0.03  # while carried, the bowl-to-gripper distance grows by less than this
MIN_LIFT_M = 0.05

# Grasp offsets (grip site minus bowl centre while held, world frame).
# Tasks 0, 1, 2, 6, 8: the per-task median of the 5-shot demos at their close frame (demo_grasp.py), all on the
# bowl's left or right rim with the fingers closing along world y.
DEMO_GRASP = Path("data/demo_grasp.json")
# Raised tasks (3, 5, 7, 9) have no source demos to measure, so they use the same pattern: the left (+1) or right
# (-1) rim at the measured tasks' typical radius and height. The side per task comes from a small sim trial
# (grid.py pick-side); for task 9 the better side depends on the init state, so its grid tries both.
RAISED_SIDES = Path("data/raised_sides.json")
RULE_RADIUS_M, RULE_HEIGHT_M = 0.045, 0.008
# Task 4 (bowl in the open top drawer): the drawer walls block both side rims, so the grip site goes on the
# near-left rim, close to where the 5-shot demos grasp it; the bowl must rise clear_lift before it moves
# (plan.py), and the carry is steered by the bowl's own position (run(track_bowl=True)).
ENCLOSED = {4: {"offset": np.array([-0.035, 0.035, 0.0]), "clear_lift": 0.14}}


def grasp_offset(task: int, side: int | None = None) -> np.ndarray:
    demo = json.loads(DEMO_GRASP.read_text())
    if str(task) in demo:
        g = demo[str(task)]
        return np.array([g["world_rel_x_cm"], g["world_rel_y_cm"], g["height_vs_bowl_centre_cm"]]) / 100
    if task in ENCLOSED:
        return ENCLOSED[task]["offset"]
    if side is None:
        side = json.loads(RAISED_SIDES.read_text())[str(task)]
    return np.array([0.0, side * RULE_RADIUS_M, RULE_HEIGHT_M])


def scene_of(env, task: int, side: int | None = None) -> SimScene:
    """The scene a plan needs, read from the env just after reset."""
    target = L.target_object(env)
    bowl_lo = L.mesh_box(env, f"{target}_g0")[0][2]
    plate_lo = L.mesh_box(env, "plate_1_g0")[0][2]
    c0 = L.mesh_centre(env, target)
    place_z = None
    if bowl_lo > plate_lo + 0.01:  # raised start: place relative to the plate, not the start height
        place_z = float(plate_lo + (c0[2] - bowl_lo) + SIM_PLACE_DZ)
    return SimScene(
        bowl_c0=c0,
        plate_c=L.mesh_centre(env, "plate_1"),
        eef0=L.gripper_pos(env),
        grasp_offset=grasp_offset(task, side),
        place_z=place_z,
        clear_lift=ENCLOSED[task]["clear_lift"] if task in ENCLOSED else 0.0,
    )


def frame_of(obs: dict) -> dict:
    return L.frame(
        obs["agentview_image"],
        obs["robot0_eye_in_hand_image"],
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        obs["robot0_gripper_qpos"],
    )


@dataclass
class Replay:
    success_final: bool = False
    first_success: int | None = None
    lift_m: float = 0.0
    slip_m: float = 0.0
    other_shift_m: float = 0.0
    other_worst: str = ""
    track_err_m: float = 0.0  # median grip-site distance to the plan's current target
    ok: bool = False
    reason: str = ""
    actions: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    grip_trace: list = field(default_factory=list)
    bowl_trace: list = field(default_factory=list)


def run(env, plan: Plan, record: bool = False, track_bowl: bool = False) -> Replay:
    """Track the plan from the current (just reset) state; record=True keeps frames (cameras must be on).

    track_bowl (task 4): from lift to release, steer so the bowl centre follows the planned bowl path (target =
    planned centre + the current grip-to-bowl offset). A rim grasp in the drawer leaves the bowl hanging a few
    cm off the planned offset, which otherwise lands it on the plate's rim.
    """
    lib, r = env._env, L.raw(env)  # LIBERO env (step, success test) and the robosuite env inside it
    target = L.target_object(env)
    ctrl = r.robots[0].controller
    r0 = np.array(ctrl.ee_ori_mat)
    others = {n: r.sim.data.body_xpos[i][:2].copy() for n, i in r.obj_body_id.items() if n != target}
    bowl_z0 = L.mesh_centre(env, target)[2]
    obs = r._get_observations(force_update=True) if record else None

    n = len(plan.pos)
    rep = Replay()
    errs = []
    for k in range(n + SETTLE_STEPS):
        eef = L.gripper_pos(env)
        tgt = plan.pos[min(k + LEAD, n - 1)]
        if track_bowl and plan.steps["lift"] <= k < plan.steps["release"]:
            tgt = tgt - plan.grasp_offset + (eef - L.mesh_centre(env, target))
        errs.append(np.linalg.norm(plan.pos[min(k, n - 1)] - eef))
        a = np.zeros(7, dtype=np.float32)
        a[:3] = np.clip((tgt - eef) / POS_SCALE * GAIN, -1, 1)
        a[3:6] = np.clip(Rot.from_matrix(r0 @ np.array(ctrl.ee_ori_mat).T).as_rotvec() / ROT_SCALE, -1, 1)
        a[6] = plan.grip[min(k, n - 1)]
        if record:
            rep.frames.append(frame_of(obs))
        rep.actions.append(a)
        obs, _, _, _ = lib.step(a)
        rep.grip_trace.append(L.gripper_pos(env))
        rep.bowl_trace.append(L.mesh_centre(env, target))
        if rep.first_success is None and lib.check_success():
            rep.first_success = k

    grip, bowl = np.array(rep.grip_trace), np.array(rep.bowl_trace)
    rep.success_final = bool(lib.check_success())
    rep.lift_m = float(bowl[:, 2].max() - bowl_z0)
    a, b = plan.steps["lift"] + 10, plan.steps["release"]
    if b > a:
        d = np.linalg.norm(grip[a:b] - bowl[a:b], axis=1)
        rep.slip_m = float(d.max() - d[0])
    shift = {nm: float(np.linalg.norm(r.sim.data.body_xpos[r.obj_body_id[nm]][:2] - p)) for nm, p in others.items()}
    rep.other_worst = max(shift, key=shift.get)
    rep.other_shift_m = shift[rep.other_worst]
    rep.track_err_m = float(np.median(errs))
    reasons = []
    if not rep.success_final:
        reasons.append("not on plate at end")
    if rep.lift_m < MIN_LIFT_M:
        reasons.append(f"lift {100 * rep.lift_m:.1f} cm")
    if rep.slip_m > MAX_SLIP_M:
        reasons.append(f"slip {100 * rep.slip_m:.1f} cm")
    if rep.other_shift_m > MAX_OTHER_SHIFT_M:
        reasons.append(f"{rep.other_worst} moved {100 * rep.other_shift_m:.1f} cm")
    rep.ok = not reasons
    rep.reason = "; ".join(reasons)
    return rep


def main(
    take: str = "G2_gap12/01",
    task: int = 0,
    init: int = 10,
    side: int | None = None,
    out: Path = Path("outputs/replay"),
) -> None:
    """Retarget one take onto one task and init state; write a video, a plan-vs-actual plot and the checks.

    Args:
        take: video/take number, e.g. G1_gap16/05 (see data/takes).
        task: libero_spatial task id, 0-9.
        init: init state; 10-49 are the ones used for data (0-9 are the eval episodes).
        side: raised tasks only, grasp the bowl's left (+1) or right (-1) rim; default: data/raised_sides.json.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lerobot.utils.io_utils import write_video

    from .plan import HZ, make_plan
    from .takes import load_takes

    out.mkdir(parents=True, exist_ok=True)
    tk = next(t for t in load_takes() if t.name == take)
    env = L.make_env(task)
    L.reset(env, init, cameras=True)
    plan = make_plan(tk, scene_of(env, task, side))
    rep = run(env, plan, record=True, track_bowl=task in ENCLOSED)
    env.close()

    tag = f"{take.replace('/', '_')}_task{task}_init{init}"
    frames = [np.concatenate([f["observation.images.image"], f["observation.images.image2"]], 1) for f in rep.frames]
    write_video(str(out / f"{tag}.mp4"), np.stack(frames), fps=HZ)

    t, tp = np.arange(len(rep.grip_trace)) / HZ, np.arange(len(plan.pos)) / HZ
    grip, bowl = np.array(rep.grip_trace), np.array(rep.bowl_trace)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, dpi=120)
    for i, (ax, lab) in enumerate(zip(axes, ("x (m)", "y (m)", "z (m)"))):
        ax.plot(tp, plan.pos[:, i], color="#9a9893", lw=1.5, ls="--", label="plan (grip site)")
        ax.plot(t, grip[:, i], color="#2a78d6", lw=1.5, label="gripper")
        ax.plot(t, bowl[:, i], color="#eb6834", lw=1.5, label="bowl centre")
        for s in plan.steps.values():
            ax.axvline(s / HZ, color="#d6d5d0", lw=0.8)
        ax.set_ylabel(lab)
    for k, s in plan.steps.items():
        axes[0].text(s / HZ, axes[0].get_ylim()[1], k, fontsize=7, ha="center", va="bottom")
    axes[0].legend(frameon=False, fontsize=7, ncol=3, loc="lower right")
    axes[-1].set_xlabel("time (s), 20 Hz control")
    fig.tight_layout()
    fig.savefig(out / f"{tag}.png")

    res = {
        "take": take,
        "task": task,
        "init": init,
        "ok": rep.ok,
        "reason": rep.reason,
        "first_success_step": rep.first_success,
        "lift_cm": 100 * rep.lift_m,
        "slip_cm": 100 * rep.slip_m,
        "other_shift_cm": 100 * rep.other_shift_m,
        "track_err_cm": 100 * rep.track_err_m,
    }
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"-> {out / tag}.mp4")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)

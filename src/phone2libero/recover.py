"""Recovery data: run a trained policy on held-out init states; when it misses the bowl or stalls, finish the
episode with a phone take planned from wherever the arm is, and keep the policy's own successes as well.

  python -m phone2libero.recover --policy <pretrained_model> --own data/own --recovered data/recovered

Why: the policy almost never retries after a missed grasp (step3_40k: 8 of 781 successes come after step 150 of
280), and filtered behaviour cloning on its own clean successes cannot teach it to. A recovery episode starts at
a state where the policy has just failed: the gripper closed on nothing, or no success by STALL_STEP. These are
taken at chunk boundaries, where the policy predicts its next chunk, so they are states it queries at test time.
Only the take's part is recorded, so the policy learns the retry, not the miss. The take is re-planned from the
current gripper and bowl (replay.scene_of, plan.make_plan) and must pass the replay checks.

Per task, init states 10, 11, ... (0-9 stay reserved for eval) are each rolled out `attempts` times. The first
success on an init state goes to the `own` dataset (as selfplay.py does); each failure gives up to MAX_SNAPS
takeover states, each tried with up to TAKES_PER_STATE takes until one passes. Task 4 gets no takeovers: its
retargeted replays leave the bowl hanging off-centre and trained worse than the policy's own runs (README).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

INITS = range(10, 50)
EMPTY_DZ = 0.06  # m: a closed gripper this far above the bowl centre holds nothing (holding: within ~2 cm)
CLOSED_STEPS = 10  # the gripper takes ~0.5 s to close
STALL_STEP = 150  # successes after this step are rare (1% for step3_40k), so take over regardless
SNAP_SPACING = 30  # steps between takeover states of one rollout
MAX_SNAPS = 2
TAKES_PER_STATE = 4
NO_TAKEOVER = {4}


def rollout(env, policy, init: int, seed: int):
    """One policy rollout. Returns (first success step or None, frames, actions, takeover snapshots)."""
    import torch

    from . import libero as L
    from .selfplay import frame_of

    L.reset(env, init, cameras=True)
    obs = env._format_raw_obs(L.raw(env)._get_observations(force_update=True))
    policy.reset()
    torch.manual_seed(seed)
    target = L.target_object(env)
    frames, actions, snaps, closed, first = [], [], [], 0, None
    while len(actions) < env._max_episode_steps:
        for a in policy.chunk(obs, env.task_description):
            frames.append(frame_of(obs))
            actions.append(np.asarray(a, dtype=np.float32))
            obs, _, _, _, info = env.step(np.asarray(a))
            closed = closed + 1 if a[6] > 0 else 0
            if first is None and info["is_success"]:
                first = len(actions) - 1
            if first is not None and len(actions) >= first + 3:
                return first, frames, actions, []
            if len(actions) >= env._max_episode_steps:
                break
        n = len(actions)
        empty = closed >= CLOSED_STEPS and L.gripper_pos(env)[2] > L.mesh_centre(env, target)[2] + EMPTY_DZ
        if (empty or n >= STALL_STEP) and len(snaps) < MAX_SNAPS and (not snaps or n - snaps[-1][0] >= SNAP_SPACING):
            snaps.append((n, L.checkpoint(env)))
    return None, frames, actions, snaps


def takeover(env, task: int, snap, takes: list, start: int, estimator=None):
    """Finish from a snapshot with the first take (from takes[start]) whose replay passes. Returns (Replay, take,
    side) or None. With an estimator (scene.load_model), the take is planned from the camera image at the snapshot
    instead of from sim state."""
    from . import libero as L
    from .plan import make_plan
    from .replay import ENCLOSED, run, scene_of
    from .scene import estimate, scene_from, snapshot_frame

    sides = (1, -1) if task == 9 else (None,)  # task 9: the passing rim side depends on the scene (grid.py)
    if estimator is not None:
        L.restore(env, snap)
        frame = snapshot_frame(env)
        y, eef0 = estimate(estimator, frame, task), frame["state"][:3]
    for j in range(TAKES_PER_STATE):
        take = takes[(start + j) % len(takes)]
        for side in sides:
            L.restore(env, snap)
            scene = scene_of(env, task, side) if estimator is None else scene_from(y, eef0, task, side)
            rep = run(env, make_plan(take, scene), record=True, track_bowl=task in ENCLOSED)
            if rep.ok:
                return rep, take.name, side
    return None


def main(
    policy: str,
    own: Path,
    recovered: Path,
    tasks: tuple[int, ...] = tuple(range(10)),
    n_action_steps: int | None = None,
    per_task: int = 20,
    recoveries: int = 20,
    attempts: int = 3,
    overwrite: bool = False,
    estimator: Path | None = None,
) -> None:
    """Record a policy's own successes and phone-take recoveries from its failures.

    Args:
        policy: pretrained_model directory or Hugging Face model repo.
        own: dataset root for the policy's own successes.
        recovered: dataset root for the recovery episodes.
        tasks: libero_spatial task ids.
        n_action_steps: actions executed per predicted chunk; default: the model's own.
        per_task: own successes to keep per task (at most one per init state).
        recoveries: recovery episodes to keep per task.
        attempts: rollouts per init state.
        estimator: scene estimator weights (scene.py); plan takeovers from the camera image, not sim state.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    from .dataset import add_episode, create, finish
    from .libero import make_env
    from .policy import Policy
    from .takes import load_takes

    pol = Policy(policy, n_action_steps=n_action_steps)
    est = None
    if estimator is not None:
        import torch

        from .scene import load_model

        torch.set_num_threads(2)  # several collectors share the CPU
        est = load_model(estimator, "cpu")
    takes = load_takes()
    ds_own, ds_rec = create(own, overwrite), create(recovered, overwrite)
    manifest = {"policy": policy, "n_action_steps": pol.n_action_steps, "attempts": attempts, "tasks": {}}
    for task in tasks:
        env = make_env(task)
        log = {"rollouts": 0, "successes": 0, "own": [], "takeover_states": 0, "recovered": []}
        for init in INITS:
            if len(log["own"]) >= per_task and (task in NO_TAKEOVER or len(log["recovered"]) >= recoveries):
                break
            kept = False
            for k in range(attempts):
                first, frames, actions, snaps = rollout(env, pol, init, seed=100000 * task + 100 * init + k)
                log["rollouts"] += 1
                if first is not None:
                    log["successes"] += 1
                    if not kept and len(log["own"]) < per_task:
                        n = add_episode(ds_own, frames, actions, first, env.task_description)
                        log["own"].append({"init": init, "attempt": k, "n_frames": n})
                        kept = True
                    continue
                if task in NO_TAKEOVER:
                    continue
                for step, snap in snaps:
                    if len(log["recovered"]) >= recoveries:
                        break
                    log["takeover_states"] += 1
                    got = takeover(env, task, snap, takes, log["takeover_states"] * TAKES_PER_STATE, est)
                    if got is None:
                        continue
                    rep, take, side = got
                    n = add_episode(ds_rec, rep.frames, rep.actions, rep.first_success, env.task_description)
                    log["recovered"].append(
                        {"init": init, "attempt": k, "from_step": step, "take": take, "side": side, "n_frames": n}
                    )
        manifest["tasks"][str(task)] = log
        print(
            f"task {task}: {log['successes']}/{log['rollouts']} rollouts succeeded, kept {len(log['own'])}; "
            f"{len(log['recovered'])} recoveries from {log['takeover_states']} takeover states",
            flush=True,
        )
        env.close()
    for ds, root in ((ds_own, own), (ds_rec, recovered)):
        finish(ds, root)
        (root / "meta" / "recover_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"-> {own}, {recovered}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)

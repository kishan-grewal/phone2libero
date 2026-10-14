"""Self-improvement (filtered behaviour cloning): run a trained policy on init states 10-49 and keep its own
successful episodes as a LeRobot dataset in the 5-shot format.

  python -m phone2libero.selfplay --policy <pretrained_model or HF repo> --out data/selfplay

Per task, init states 10, 11, ... are tried in order (0-9 stay reserved for eval), up to `attempts` policy
samples each, until `per_task` successes are collected. An init state contributes at most one episode, so the
episodes stay varied. Actions run chunk by chunk (n_action_steps = 50), as in lerobot-eval.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

INITS = range(10, 50)


def frame_of(obs: dict) -> dict:
    """lerobot LiberoEnv observation -> one 5-shot-format frame."""
    from .libero import frame

    s = obs["robot_state"]
    return frame(
        obs["pixels"]["image"], obs["pixels"]["image2"], s["eef"]["pos"], s["eef"]["quat"], s["gripper"]["qpos"]
    )


def episode(env, policy, init: int, seed: int) -> tuple[int | None, list[dict], list[np.ndarray]]:
    """One rollout from init state `init`. Returns (first success step or None, frames, actions)."""
    import torch

    from .libero import raw, reset

    reset(env, init, cameras=True)
    obs = env._format_raw_obs(raw(env)._get_observations(force_update=True))
    policy.reset()
    torch.manual_seed(seed)
    frames, actions, first = [], [], None
    while len(actions) < env._max_episode_steps:
        for a in policy.chunk(obs, env.task_description):
            frames.append(frame_of(obs))
            actions.append(np.asarray(a, dtype=np.float32))
            obs, _, _, _, info = env.step(np.asarray(a))
            if first is None and info["is_success"]:
                first = len(actions) - 1
            if first is not None and len(actions) >= first + 3:
                return first, frames, actions
            if len(actions) >= env._max_episode_steps:
                break
    return None, frames, actions


def main(
    policy: str,
    out: Path,
    tasks: tuple[int, ...] = tuple(range(10)),
    per_task: int = 20,
    attempts: int = 3,
    overwrite: bool = False,
) -> None:
    """Record a policy's own successes.

    Args:
        policy: pretrained_model directory or Hugging Face model repo.
        out: dataset root to create.
        tasks: libero_spatial task ids.
        per_task: successful episodes to collect per task.
        attempts: policy samples per init state before moving on.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    from .dataset import add_episode, create, finish
    from .libero import make_env
    from .policy import Policy

    pol = Policy(policy)
    ds = create(out, overwrite)
    manifest = {"policy": policy, "per_task": per_task, "attempts": attempts, "tasks": {}}
    for task in tasks:
        env = make_env(task)
        got, tries, used = 0, 0, []
        for init in INITS:
            if got >= per_task:
                break
            for k in range(attempts):
                tries += 1
                first, frames, actions = episode(env, pol, init, seed=100000 * task + 100 * init + k)
                if first is None:
                    continue
                n = add_episode(ds, frames, actions, first, env.task_description)
                got += 1
                used.append({"init": init, "attempt": k, "n_frames": n})
                break
        manifest["tasks"][str(task)] = {"episodes": got, "attempts": tries, "used": used}
        print(f"task {task}: {got} episodes from {tries} attempts", flush=True)
        env.close()
    finish(ds, out)
    (out / "meta" / "selfplay_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)

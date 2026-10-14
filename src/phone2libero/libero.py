"""LIBERO helpers: build the exact env lerobot-eval uses, reset to an init state, save/restore the full sim
state, and read object positions.

All positions are MuJoCo world frame, metres: +x away from the robot base, +y to the robot's left, +z up.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.libero import _make_env_fns
from libero.libero import benchmark
from robosuite.utils.buffers import Buffer

SUITE = "libero_spatial"
OBS_SIZE = 256  # the 5-shot data and every eval use 256 x 256 images (lerobot-eval's LIBERO default is 360)
TABLE_TASKS = [0, 1, 2, 6, 8]  # the bowl starts on the table
RAISED_TASKS = [3, 4, 5, 7, 9]  # on a cookie box, in a drawer, on a ramekin, on a stove, on a cabinet


def env_config() -> LiberoEnvConfig:
    return LiberoEnvConfig(task=SUITE, init_states=True, observation_height=OBS_SIZE, observation_width=OBS_SIZE)


def make_env(task: int):
    """The LiberoEnv lerobot-eval builds for `task` (its first env of a batch of 10), reset once."""
    cfg = env_config()
    gym_kwargs = dict(cfg.gym_kwargs)
    gym_kwargs.pop("task_ids", None)
    fns = _make_env_fns(
        suite=benchmark.get_benchmark_dict()[SUITE](),
        suite_name=SUITE,
        task_id=task,
        n_envs=10,
        camera_names=cfg.camera_name.split(","),
        episode_length=cfg.episode_length,
        init_states=cfg.init_states,
        gym_kwargs=gym_kwargs,
        control_mode=cfg.control_mode,
        camera_name_mapping=cfg.camera_name_mapping,
    )
    env = fns[0]()
    env.reset(seed=0)
    return env


def raw(env) -> Any:
    """lerobot's LiberoEnv -> the robosuite env inside it (sim data, robots, observables)."""
    return env._env.env


def reset(env, init_state: int, cameras: bool) -> dict:
    """Standard reset to one of the task's init states (load it, settle), as lerobot-eval does. Returns the
    robosuite observation. Init states 0-9 are the eval episodes; data is only generated from 10-49."""
    set_cameras_enabled(env, True)  # lerobot's reset formats camera observations, so they must be on
    env.init_state_id = init_state
    env.reset(seed=init_state)
    set_cameras_enabled(env, cameras)
    return raw(env)._get_observations(force_update=True)


def set_cameras_enabled(env, enabled: bool) -> None:
    """Toggle camera rendering; replays that record no frames run several times faster without it.

    Sets `_enabled` directly: robosuite's `set_enabled()` also resets the observable's sampling timer, which
    shifts the substep an image is captured on and makes later frames differ.
    """
    for name, obs in raw(env)._observables.items():
        if name.endswith("_image"):
            obs._enabled = enabled


def target_object(env) -> str:
    """The object the task moves, from its BDDL goal ('on', bowl, plate): akita_black_bowl_1 in every task."""
    goal = raw(env).parsed_problem["goal_state"]
    (on,) = [g for g in goal if g[0].lower() == "on"]
    return on[1]


def gripper_pos(env) -> np.ndarray:
    """Grip site: the point between the fingertips."""
    r = raw(env)
    return r.sim.data.site_xpos[r.robots[0].eef_site_id].copy()


def mesh_centre(env, obj: str) -> np.ndarray:
    """Centre of an object's main mesh (mid-height); the body origin sits at the object's base."""
    r = raw(env)
    model, data = r.sim.model._model, r.sim.data._data
    return data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{obj}_g0")].copy()


def mesh_box(env, geom: str) -> tuple[np.ndarray, np.ndarray]:
    """World-frame axis-aligned bounding box (lo, hi) of a mesh geom's vertices."""
    r = raw(env)
    model, data = r.sim.model._model, r.sim.data._data
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    m = model.geom_dataid[g]
    v = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]]
    w = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    return w.min(0), w.max(0)


def axis_angle(q: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) -> axis-angle, the same conversion lerobot's LIBERO processor applies."""
    w = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if den < 1e-10:
        return np.zeros(3)
    return q[:3] / den * 2.0 * np.arccos(w)


def frame(agentview: np.ndarray, wrist: np.ndarray, eef_pos, eef_quat, gripper_qpos) -> dict:
    """One frame in the 5-shot dataset's format: both images rotated 180 deg from the raw MuJoCo render (upright,
    as lerobot-eval shows them), 8-D state."""
    return {
        "observation.images.image": np.ascontiguousarray(agentview[::-1, ::-1]),
        "observation.images.image2": np.ascontiguousarray(wrist[::-1, ::-1]),
        "observation.state": np.concatenate([eef_pos, axis_angle(eef_quat), gripper_qpos]).astype(np.float32),
    }


# Full sim state save/restore. robosuite's `sim.get_state()` keeps only (time, qpos, qvel), which misses the
# solver warm start and controller state, so restored rollouts drift. MuJoCo's integration state plus the
# Python-side state on the env, robot and controller objects restores bit-exactly.
_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
_COPY_TYPES = (np.ndarray, np.generic, int, float, bool, Buffer, deque)
_SKIP_ATTRS = {"sim", "model", "viewer", "robot_model", "_enabled", "_active"}


@dataclass
class Snapshot:
    physics: np.ndarray
    py_state: list[dict[str, Any]]
    obs_cache: dict[str, Any]


def _py_objects(env) -> list[Any]:
    r = raw(env)
    objs = [r]
    for robot in r.robots:
        objs += [robot, robot.controller]
        if getattr(robot, "gripper", None) is not None:
            objs.append(robot.gripper)
    return objs + list(r._observables.values())


def restore(env, snap: Snapshot) -> None:
    r = raw(env)
    model, data = r.sim.model._model, r.sim.data._data
    mujoco.mj_setState(model, data, snap.physics, _SPEC)
    mujoco.mj_forward(model, data)
    for obj, attrs in zip(_py_objects(env), snap.py_state, strict=True):
        for k, v in attrs.items():
            setattr(obj, k, copy.deepcopy(v))
    r._obs_cache = copy.deepcopy(snap.obs_cache)


def checkpoint(env) -> Snapshot:
    """Save the state, then restore it at once, so the live state is exactly what every later restore gives
    (after `mj_step`, derived quantities such as `body_xpos` lag one substep; restore runs `mj_forward`)."""
    r = raw(env)
    model, data = r.sim.model._model, r.sim.data._data
    physics = np.empty(mujoco.mj_stateSize(model, _SPEC))
    mujoco.mj_getState(model, data, physics, _SPEC)
    py_state = [
        {k: copy.deepcopy(v) for k, v in vars(o).items() if k not in _SKIP_ATTRS and isinstance(v, _COPY_TYPES)}
        for o in _py_objects(env)
    ]
    snap = Snapshot(physics, py_state, copy.deepcopy(r._obs_cache))
    restore(env, snap)
    return snap

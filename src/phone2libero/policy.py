"""Run a SmolVLA checkpoint in one LIBERO env exactly as lerobot-eval does: the same policy factory, saved
pre/post processors with the same overrides, env processors and observation path (lerobot_eval.rollout)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import make_env_pre_post_processors, preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION

from .libero import env_config


def _batched(obs):
    """One LiberoEnv observation -> the shape a 1-env VectorEnv returns."""
    if isinstance(obs, dict):
        return {k: _batched(v) for k, v in obs.items()}
    return np.asarray(obs)[None]


class Policy:
    def __init__(self, path: str | Path, device: str = "cuda", n_action_steps: int | None = None):
        """n_action_steps: actions executed per predicted chunk, as lerobot-eval's --policy.n_action_steps; default:
        the model's own."""
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = Path(path)
        cfg.device = device
        if n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
        env_cfg = env_config()
        self.policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map={})
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=cfg.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": str(self.policy.config.device)},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
        self.n_action_steps = self.policy.config.n_action_steps

    def reset(self) -> None:
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    @torch.no_grad()
    def chunk(self, obs: dict, task: str) -> np.ndarray:
        """One action chunk (n_action_steps, 7) for a LiberoEnv observation, unnormalised, ready for env.step."""
        batch = preprocess_observation(_batched(obs))
        batch["task"] = [task]
        batch = self.pre(self.env_pre(batch))
        chunk = self.policy.predict_action_chunk(batch)[:, : self.n_action_steps]
        # lerobot-eval post-processes one step at a time; doing the same keeps the actions identical
        steps = [self.env_post({ACTION: self.post(chunk[:, t])})[ACTION].cpu().numpy() for t in range(chunk.shape[1])]
        return np.stack(steps, axis=1)[0]

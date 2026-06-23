"""Isaac Lab vector environment wrapper for the PufferLib-style runner."""

from __future__ import annotations

from math import prod
from typing import Any

import gymnasium as gym
import torch


def _obs_to_dict(obs: Any) -> dict[str, torch.Tensor]:
    if hasattr(obs, "to_dict") and not isinstance(obs, torch.Tensor):
        obs = obs.to_dict()
    if isinstance(obs, dict):
        return obs
    if isinstance(obs, torch.Tensor):
        return {"policy": obs}
    raise TypeError(f"Expected observation dict or torch.Tensor, got {type(obs)!r}")


def _resolve_groups(obs: dict[str, torch.Tensor], groups: list[str] | tuple[str, ...], set_name: str) -> list[str]:
    if groups:
        for group in groups:
            if group not in obs:
                raise KeyError(f"Observation group '{group}' for '{set_name}' is missing. Available: {list(obs.keys())}")
        return list(groups)
    if set_name in obs:
        return [set_name]
    if "policy" in obs:
        return ["policy"]
    raise KeyError(f"Could not resolve observation set '{set_name}'. Available groups: {list(obs.keys())}")


def _flatten_groups(obs: dict[str, torch.Tensor], groups: list[str]) -> torch.Tensor:
    parts = []
    for group in groups:
        tensor = obs[group]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected tensor for observation group '{group}', got {type(tensor)!r}")
        parts.append(tensor.reshape(tensor.shape[0], -1).float())
    return torch.cat(parts, dim=-1)


def _group_layout(obs: dict[str, torch.Tensor], groups: list[str]) -> dict[str, Any]:
    """Build flat observation slices for each named group."""

    layout: dict[str, Any] = {"groups": list(groups), "slices": {}, "shapes": {}, "flat_dims": {}}
    offset = 0
    for group in groups:
        tensor = obs[group]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected tensor for observation group '{group}', got {type(tensor)!r}")
        shape = tuple(int(dim) for dim in tensor.shape[1:])
        flat_dim = int(prod(shape))
        layout["shapes"][group] = shape
        layout["flat_dims"][group] = flat_dim
        layout["slices"][group] = (offset, offset + flat_dim)
        offset += flat_dim
    layout["flat_dim"] = offset
    return layout


def _preserve_single_group(obs: dict[str, torch.Tensor], groups: list[str], set_name: str) -> torch.Tensor:
    if len(groups) != 1:
        raise ValueError(f"Preserving observation shape for '{set_name}' requires exactly one group, got {groups}")
    tensor = obs[groups[0]]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor for observation group '{groups[0]}', got {type(tensor)!r}")
    return tensor.float()


def _flatten_obs(obs: Any, groups: list[str], set_name: str) -> torch.Tensor:
    obs_dict = _obs_to_dict(obs)
    groups = _resolve_groups(obs_dict, groups, set_name)
    return _flatten_groups(obs_dict, groups)


def _action_dim(space: gym.Space) -> int:
    if not isinstance(space, gym.spaces.Box):
        raise NotImplementedError(f"Only continuous Box action spaces are supported, got {space}")
    if len(space.shape) != 1:
        return gym.spaces.flatdim(space)
    return int(space.shape[0])


def _maybe_float_tensor(value: Any, device: torch.device) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.to(device).float()
    return None


def _mean_log_value(value: Any) -> float | None:
    tensor = _maybe_float_tensor(value, torch.device("cpu"))
    if tensor is None or tensor.numel() == 0:
        return None
    return float(tensor.mean().item())


def collect_scalar_logs(extras: dict[str, Any]) -> dict[str, float]:
    """Extract scalar-ish values from IsaacLab extras without assuming a fixed schema."""

    logs: dict[str, float] = {}
    for prefix in ("log", "episode"):
        payload = extras.get(prefix)
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            mean_value = _mean_log_value(value)
            if mean_value is not None:
                logs[f"{prefix}/{key}"] = mean_value
    return logs


def summarize_done_episodes(extras: dict[str, Any]) -> dict[str, float]:
    """Best-effort episodic metric extraction from common IsaacLab/RL schemas."""

    logs = collect_scalar_logs(extras)
    summary: dict[str, float] = {}
    for key, value in logs.items():
        lower = key.lower()
        if "return" in lower or "reward" in lower or "episode_length" in lower or "episode length" in lower:
            summary[key] = value
    return summary


class PufferVecEnvWrapper:
    """Thin tensor wrapper over an Isaac Lab Gymnasium vector environment."""

    def __init__(
        self,
        env: gym.Env,
        actor_obs_groups: list[str] | tuple[str, ...] | None = None,
        critic_obs_groups: list[str] | tuple[str, ...] | None = None,
        clip_actions: float | None = None,
        preserve_observation_shape: bool = False,
    ):
        from isaaclab.envs import DirectRLEnv, ManagerBasedEnv, ManagerBasedRLEnv

        try:
            from isaaclab_experimental.envs import DirectRLEnvWarp, ManagerBasedEnvWarp, ManagerBasedRLEnvWarp
        except ImportError:
            DirectRLEnvWarp = None
            ManagerBasedEnvWarp = None
            ManagerBasedRLEnvWarp = None

        allowed_types = (ManagerBasedRLEnv, ManagerBasedEnv, DirectRLEnv)
        if DirectRLEnvWarp is not None:
            allowed_types += (DirectRLEnvWarp,)
        if ManagerBasedEnvWarp is not None:
            allowed_types += (ManagerBasedEnvWarp,)
        if ManagerBasedRLEnvWarp is not None:
            allowed_types += (ManagerBasedRLEnvWarp,)

        if not isinstance(env.unwrapped, allowed_types):
            raise ValueError(
                "The environment must inherit from an Isaac Lab RL environment type. "
                f"Got {type(env.unwrapped)!r}."
            )

        self.env = env
        self.actor_obs_groups = list(actor_obs_groups or [])
        self.critic_obs_groups = list(critic_obs_groups or [])
        self.clip_actions = clip_actions
        self.preserve_observation_shape = preserve_observation_shape
        self.device = torch.device(env.unwrapped.device)
        self.num_envs = int(env.unwrapped.num_envs)
        self.total_agents = self.num_envs
        self.max_episode_length = int(getattr(env.unwrapped, "max_episode_length", 0))
        self.last_extras: dict[str, Any] = {}
        self.actor_obs_layout: dict[str, Any] = {}
        self.critic_obs_layout: dict[str, Any] = {}

        single_action_space = env.unwrapped.single_action_space
        self.action_dim = _action_dim(single_action_space)

        obs = self.reset()
        self.actor_obs_shape = tuple(obs["actor"].shape[1:])
        self.critic_obs_shape = tuple(obs["critic"].shape[1:])
        self.actor_obs_dim = int(prod(self.actor_obs_shape))
        self.critic_obs_dim = int(prod(self.critic_obs_shape))

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor | None:
        return getattr(self.env.unwrapped, "episode_length_buf", None)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        if hasattr(self.env.unwrapped, "episode_length_buf"):
            self.env.unwrapped.episode_length_buf = value

    def _format_obs(self, obs: Any) -> dict[str, torch.Tensor]:
        obs_dict = _obs_to_dict(obs)
        self.actor_obs_groups = _resolve_groups(obs_dict, self.actor_obs_groups, "actor")
        self.critic_obs_groups = _resolve_groups(obs_dict, self.critic_obs_groups, "critic")
        self.actor_obs_layout = _group_layout(obs_dict, self.actor_obs_groups)
        self.critic_obs_layout = _group_layout(obs_dict, self.critic_obs_groups)
        if self.preserve_observation_shape:
            return {
                "actor": _preserve_single_group(obs_dict, self.actor_obs_groups, "actor").to(self.device),
                "critic": _preserve_single_group(obs_dict, self.critic_obs_groups, "critic").to(self.device),
            }
        return {
            "actor": _flatten_groups(obs_dict, self.actor_obs_groups).to(self.device),
            "critic": _flatten_groups(obs_dict, self.critic_obs_groups).to(self.device),
        }

    def reset(self) -> dict[str, torch.Tensor]:
        obs_dict, extras = self.env.reset()
        self.last_extras = extras
        return self._format_obs(obs_dict)

    def get_observations(self) -> dict[str, torch.Tensor]:
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return self._format_obs(obs_dict)

    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
        actions = actions.reshape(self.num_envs, self.action_dim).to(self.device)
        obs_dict, reward, terminated, truncated, extras = self.env.step(actions)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        self.last_extras = extras
        done = terminated | truncated
        obs = self._format_obs(obs_dict)
        return obs, reward.to(self.device).float(), done.to(self.device).float(), extras

    def close(self) -> None:
        self.env.close()

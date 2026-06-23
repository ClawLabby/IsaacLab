"""Prototype Isaac Lab Warp/PufferLib Ocean buffer bridge.

This is the Isaac Lab side of PufferLib's external-vector ABI. Isaac Lab owns
the environment and CUDA observation/reward/done buffers; PufferLib owns the
policy/training buffers and passes a CUDA action pointer back through
``gpu_step`` after each native rollout step.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Any

import gymnasium as gym
import torch


class _CudaPtr:
    """Wrap a raw CUDA pointer with ``__cuda_array_interface__``."""

    def __init__(self, ptr: int, shape: tuple[int, ...], dtype: torch.dtype):
        typestr = {
            torch.float32: "<f4",
            torch.uint8: "|u1",
        }[dtype]
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": shape,
            "typestr": typestr,
            "version": 2,
        }


def _obs_from_group(obs: Any, group: str) -> torch.Tensor:
    if isinstance(obs, dict):
        obs = obs[group]
    if hasattr(obs, "to_dict") and not isinstance(obs, torch.Tensor):
        return _obs_from_group(obs.to_dict(), group)
    if not isinstance(obs, torch.Tensor):
        raise TypeError(f"Expected tensor observation for group '{group}', got {type(obs)!r}")
    return obs.reshape(obs.shape[0], -1).float()


def _flatten_float_view(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.reshape(tensor.shape[0], -1)
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor


def _persistent_obs_source(unwrapped: Any, group: str) -> torch.Tensor | None:
    """Return a persistent observation tensor when Warp managers expose one."""

    observation_manager = getattr(unwrapped, "observation_manager", None)
    group_outputs = getattr(observation_manager, "_group_out_torch", None)
    if isinstance(group_outputs, dict):
        tensor = group_outputs.get(group)
        if isinstance(tensor, torch.Tensor):
            return _flatten_float_view(tensor)

    tensor = getattr(unwrapped, "torch_obs_buf", None)
    if isinstance(tensor, torch.Tensor):
        return _flatten_float_view(tensor)

    return None


def _persistent_reward_source(unwrapped: Any) -> torch.Tensor | None:
    reward_manager = getattr(unwrapped, "reward_manager", None)
    tensor = getattr(reward_manager, "_reward_tensor_view", None)
    if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
        return tensor.reshape(-1)

    tensor = getattr(unwrapped, "torch_reward_buf", None)
    if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
        return tensor.reshape(-1)

    return None


def _persistent_done_source(unwrapped: Any) -> torch.Tensor | None:
    termination_manager = getattr(unwrapped, "termination_manager", None)
    for name in ("_dones_tensor_view", "_terminated_tensor_view"):
        tensor = getattr(termination_manager, name, None)
        if isinstance(tensor, torch.Tensor):
            return tensor.reshape(-1)

    terminated = getattr(unwrapped, "torch_reset_terminated", None)
    truncated = getattr(unwrapped, "torch_reset_time_outs", None)
    if isinstance(terminated, torch.Tensor) and isinstance(truncated, torch.Tensor):
        return terminated.reshape(-1) | truncated.reshape(-1)

    return None


def _mean_scalar(value: Any) -> float | None:
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return float(item())
        except (TypeError, ValueError):
            return None
    return None


@dataclass
class PufferWarpBridge:
    """Expose an Isaac Lab GPU env with PufferLib/Ocean-like buffers.

    Expected native-facing surface:

    - ``obs_size``, ``num_atns``, ``act_sizes``, ``total_agents``
    - ``gpu``, ``gpu_obs_ptr``, ``gpu_rewards_ptr``, ``gpu_terminals_ptr``
    - ``reset()``, ``gpu_step(actions_ptr)``, ``log()``, ``close()``

    The environment may be a Warp env or a normal GPU Isaac Lab env. The bridge
    deliberately stores contiguous torch buffers so a compiled shim can hand
    their pointers to PufferLib without copying through CPU memory.
    """

    env: gym.Env
    obs_group: str = "policy"
    clip_actions: float | None = 1.0

    def __post_init__(self) -> None:
        unwrapped = self.env.unwrapped
        self.device = torch.device(unwrapped.device)
        if self.device.type != "cuda":
            raise NotImplementedError("PufferWarpBridge only targets CUDA Isaac Lab environments.")

        self.gpu = True
        self.total_agents = int(unwrapped.num_envs)
        self.num_atns = gym.spaces.flatdim(unwrapped.single_action_space)
        self.act_sizes = [1] * self.num_atns
        self.obs_dtype = "FloatTensor"
        self.obs_elem_size = 4
        self.action_mask_size = 0

        obs_dict, extras = self.env.reset()
        self._obs_source = _persistent_obs_source(unwrapped, self.obs_group)
        obs = self._obs_source
        if obs is None:
            obs = _obs_from_group(obs_dict, self.obs_group).to(device=self.device, dtype=torch.float32)
        if not obs.is_contiguous():
            obs = obs.contiguous()
            self._obs_source = None
        self.obs_size = obs.shape[1]
        self._obs = obs if self._obs_source is not None else torch.empty(
            (self.total_agents, self.obs_size), dtype=torch.float32, device=self.device
        )
        self._reward_source = _persistent_reward_source(unwrapped)
        self._rewards = self._reward_source if self._reward_source is not None else torch.empty(
            self.total_agents, dtype=torch.float32, device=self.device
        )
        self._done_source = _persistent_done_source(unwrapped)
        self._terminals = torch.empty(self.total_agents, dtype=torch.float32, device=self.device)
        self._actions = torch.empty((self.total_agents, self.num_atns), dtype=torch.float32, device=self.device)
        self._last_extras: dict[str, Any] = extras

        if self._obs_source is None:
            self._obs.copy_(obs)
        if self._reward_source is None:
            self._rewards.zero_()
        self._terminals.zero_()
        self._actions.zero_()

        self.gpu_obs_ptr = self._obs.data_ptr()
        self.gpu_rewards_ptr = self._rewards.data_ptr()
        self.gpu_terminals_ptr = self._terminals.data_ptr()
        self.gpu_actions_ptr = self._actions.data_ptr()
        self.gpu_action_mask_ptr = 0
        self.zero_copy_obs = self._obs_source is not None
        self.zero_copy_rewards = self._reward_source is not None

    def reset(self) -> None:
        obs_dict, extras = self.env.reset()
        self._refresh_buffers(obs_dict, None, None, None)
        self._last_extras = extras

    def gpu_step(self, actions_ptr: int) -> None:
        actions = torch.as_tensor(
            _CudaPtr(actions_ptr, (self.total_agents, self.num_atns), torch.float32),
            device=self.device,
        )
        if self.clip_actions is not None:
            torch.clamp(actions, -self.clip_actions, self.clip_actions, out=self._actions)
        elif actions.data_ptr() != self._actions.data_ptr():
            self._actions.copy_(actions)

        obs_dict, reward, terminated, truncated, extras = self.env.step(self._actions)
        self._refresh_buffers(obs_dict, reward, terminated, truncated)
        self._last_extras = extras

    def log(self) -> dict[str, float]:
        logs: dict[str, float] = {}
        for prefix in ("log", "episode"):
            payload = self._last_extras.get(prefix, {})
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                mean_value = _mean_scalar(value)
                if mean_value is not None:
                    logs[f"{prefix}/{key}"] = mean_value
        return logs

    def close(self) -> None:
        self.env.close()

    def render(self, env_id: int = 0) -> None:
        if hasattr(self.env, "render"):
            self.env.render()

    def buffer_report(self) -> dict[str, bool | int | str]:
        """Return pointer-source diagnostics for probe scripts and notes."""

        return {
            "device": str(self.device),
            "total_agents": self.total_agents,
            "obs_size": self.obs_size,
            "num_atns": self.num_atns,
            "zero_copy_obs": self.zero_copy_obs,
            "zero_copy_rewards": self.zero_copy_rewards,
            "has_stable_action_buffer": True,
        }

    def _refresh_buffers(
        self,
        obs_dict: dict[str, torch.Tensor],
        reward: torch.Tensor | None,
        terminated: torch.Tensor | None,
        truncated: torch.Tensor | None,
    ) -> None:
        if self._obs_source is None:
            self._copy_obs(obs_dict)
        elif self._obs_source.shape != self._obs.shape:
            raise RuntimeError(f"Observation shape changed from {tuple(self._obs.shape)} to {tuple(self._obs_source.shape)}")

        if self._reward_source is None:
            if reward is None:
                self._rewards.zero_()
            else:
                self._rewards.copy_(reward.to(device=self.device, dtype=torch.float32))

        done_source = _persistent_done_source(self.env.unwrapped)
        if done_source is not None:
            self._terminals.copy_(done_source.to(device=self.device, dtype=torch.float32))
        elif terminated is None or truncated is None:
            self._terminals.zero_()
        else:
            self._terminals.copy_((terminated | truncated).to(device=self.device, dtype=torch.float32))

    def _copy_obs(self, obs_dict: dict[str, torch.Tensor]) -> None:
        obs = _obs_from_group(obs_dict, self.obs_group).to(device=self.device, dtype=torch.float32)
        if obs.shape != self._obs.shape:
            raise RuntimeError(f"Observation shape changed from {tuple(self._obs.shape)} to {tuple(obs.shape)}")
        self._obs.copy_(obs)

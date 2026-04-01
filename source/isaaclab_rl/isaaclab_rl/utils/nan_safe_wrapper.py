# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gymnasium wrapper that intercepts NaN before any RL library sees the data.

Usage:
    Apply this wrapper to the environment *before* passing it to any RL library wrapper::

        env = gym.make(task, cfg=env_cfg)
        env = NaNSafeWrapper(env)  # intercepts NaN in step()
        env = SkrlVecEnvWrapper(env)  # skrl never sees NaN

    Enabled by default. Disable via ``NAN_WATCHDOG=0``.
"""
from __future__ import annotations

import os

import gymnasium as gym
import torch


class NaNSafeWrapper(gym.Wrapper):
    """Gymnasium wrapper that detects NaN in observations/rewards and recovers.

    This wrapper sits between the Isaac Lab environment and any RL library wrapper.
    On each ``step()``, it checks for NaN in observations and rewards. If found, it:

    1. Dumps physics state + RL tensors for offline debugging
    2. Replaces NaN with zeros in observations and rewards
    3. Marks affected envs as terminated (triggers reset)

    The overhead is negligible (~650μs per step, <0.01% of typical training iterations).

    This is the universal solution that works with any RL library (rsl_rl, sb3, rl_games,
    skrl, or custom) since it operates at the gymnasium level.

    Configuration via environment variables:

    - ``NAN_WATCHDOG``: Set to ``0`` to disable. Default: ``1`` (enabled).
    - ``NAN_WATCHDOG_DIR``: Custom dump directory. Default: ``<log_dir>/nan_dumps``.
    - ``NAN_WATCHDOG_MAX_DUMPS``: Maximum dumps to save. Default: ``3``.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._nan_watchdog = None
        self._nan_step_counter = 0
        self._device = getattr(env.unwrapped, "device", "cpu")

        if os.environ.get("NAN_WATCHDOG", "1") == "0":
            return

        try:
            from isaaclab.utils.nan_watchdog import NaNWatchdog

            log_dir = getattr(env.unwrapped, "log_dir", None)
            dump_dir_base = log_dir or os.environ.get("NAN_WATCHDOG_DIR", ".")
            dump_dir = os.path.join(str(dump_dir_base), "nan_dumps")
            self._nan_watchdog = NaNWatchdog(
                dump_dir=dump_dir,
                max_dumps=int(os.environ.get("NAN_WATCHDOG_MAX_DUMPS", "3")),
            )
        except ImportError:
            pass

    @property
    def has_nan_watchdog(self) -> bool:
        """Whether the NaN watchdog is active."""
        return self._nan_watchdog is not None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._nan_watchdog is not None:
            self._check_and_recover(obs, reward, terminated, truncated, action)
        return obs, reward, terminated, truncated, info

    def _check_and_recover(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        """Check for NaN and recover in-place."""
        # Normalize obs to dict for the watchdog
        if isinstance(obs, torch.Tensor):
            obs_dict = {"obs": obs}
        elif isinstance(obs, dict):
            obs_dict = obs
        else:
            # TensorDict or similar — try to convert
            obs_dict = dict(obs) if hasattr(obs, "items") else {"obs": obs}

        dones = (terminated | truncated).to(dtype=torch.long) if terminated is not None else torch.zeros(1)

        nan_env_ids = self._nan_watchdog.check(
            obs_dict, reward, dones, action, step=self._nan_step_counter
        )
        self._nan_step_counter += 1

        if nan_env_ids:
            nan_idx = torch.tensor(nan_env_ids, device=self._device)
            # Replace NaN in observations
            if isinstance(obs, dict):
                for key in obs:
                    if isinstance(obs[key], torch.Tensor) and obs[key].is_floating_point():
                        obs[key][nan_idx] = torch.nan_to_num(obs[key][nan_idx], nan=0.0)
            elif isinstance(obs, torch.Tensor):
                obs[nan_idx] = torch.nan_to_num(obs[nan_idx], nan=0.0)
            # Zero rewards and mark as terminated
            reward[nan_idx] = 0.0
            terminated[nan_idx] = True

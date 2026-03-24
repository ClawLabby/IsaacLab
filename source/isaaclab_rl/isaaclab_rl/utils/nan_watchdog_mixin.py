# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""NaN watchdog mixin for RL environment wrappers.

Provides automatic NaN detection and recovery in the env.step() path.
Enable via environment variable: NAN_WATCHDOG=1
"""
from __future__ import annotations

import os

import torch


class NaNWatchdogMixin:
    """Mixin that adds NaN detection and recovery to an environment wrapper.

    When enabled (``NAN_WATCHDOG=1``), this mixin:
    1. Checks observations and rewards for NaN after each ``env.step()``
    2. Dumps Newton physics state + RL tensors for offline debugging
    3. Replaces NaN with zeros and marks affected envs as done (triggers reset)
    4. Training continues without interruption

    Configuration via environment variables:

    - ``NAN_WATCHDOG``: Set to ``1`` to enable. Default: ``0`` (disabled).
    - ``NAN_WATCHDOG_DIR``: Custom dump directory. Default: ``<log_dir>/nan_dumps``.
    - ``NAN_WATCHDOG_MAX_DUMPS``: Maximum dumps to save. Default: ``3``.

    Usage:
        Add this mixin to your wrapper class and call :meth:`_init_nan_watchdog` in
        ``__init__`` and :meth:`_check_nan` after each ``env.step()``.
    """

    def _init_nan_watchdog(self, device: str, log_dir: str | None = None) -> None:
        """Initialize the NaN watchdog if enabled via environment variable.

        Args:
            device: The device tensors are on (e.g., "cuda:0").
            log_dir: Log directory for dumps. Falls back to NAN_WATCHDOG_DIR env var.
        """
        self._nan_watchdog = None
        self._nan_step_counter = 0
        self._nan_device = device

        if os.environ.get("NAN_WATCHDOG", "0") != "1":
            return

        try:
            from isaaclab.utils.nan_watchdog import NaNWatchdog

            dump_dir_base = log_dir or os.environ.get("NAN_WATCHDOG_DIR", ".")
            dump_dir = os.path.join(str(dump_dir_base), "nan_dumps")
            self._nan_watchdog = NaNWatchdog(
                dump_dir=dump_dir,
                max_dumps=int(os.environ.get("NAN_WATCHDOG_MAX_DUMPS", "3")),
            )
        except ImportError:
            pass

    def _check_nan(
        self,
        obs_dict: dict[str, torch.Tensor],
        rewards: torch.Tensor,
        dones: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> None:
        """Check for NaN and recover if detected. Modifies tensors in-place.

        Args:
            obs_dict: Observation dictionary from env.step(). Modified in-place if NaN found.
            rewards: Reward tensor. Modified in-place if NaN found.
            dones: Done tensor. Modified in-place if NaN found (affected envs marked done).
            actions: Actions tensor for dump context. Not modified.
        """
        if self._nan_watchdog is None:
            return

        nan_env_ids = self._nan_watchdog.check(
            obs_dict, rewards, dones, actions, step=self._nan_step_counter
        )
        self._nan_step_counter += 1

        if nan_env_ids:
            nan_idx = torch.tensor(nan_env_ids, device=self._nan_device)
            # Replace NaN with zeros in observations
            for key in obs_dict:
                if isinstance(obs_dict[key], torch.Tensor) and obs_dict[key].is_floating_point():
                    obs_dict[key][nan_idx] = torch.nan_to_num(obs_dict[key][nan_idx], nan=0.0)
            # Zero rewards and mark as done so envs reset
            rewards[nan_idx] = 0.0
            dones[nan_idx] = 1

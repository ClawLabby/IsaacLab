"""NaN watchdog for Isaac Lab environments.

Always-on NaN detection (~70μs/step overhead). On NaN detection, dumps
Newton physics state (current + previous from Newton's own double buffer)
and RL tensors for offline reproduction.

For deeper state history, Newton's ViewerFile with RingBuffer can be
used directly — this watchdog focuses on detection and minimal dump.

Usage:
    watchdog = NaNWatchdog(dump_dir="/path/to/dumps")

    # In training loop after env.step():
    nan_env_ids = watchdog.check(obs, rewards, dones, actions, step=N)
    if nan_env_ids:
        # Reset affected envs and continue
        obs[nan_env_ids] = torch.nan_to_num(obs[nan_env_ids])
        dones[nan_env_ids] = True
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class NaNWatchdog:
    """Detects NaN in environment outputs and dumps Newton state for debugging."""

    def __init__(self, dump_dir: str = "/tmp/nan_dumps", max_dumps: int = 3):
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.max_dumps = max_dumps
        self.dump_count = 0
        self.step_count = 0

    def check(
        self,
        obs: dict[str, torch.Tensor],
        rewards: torch.Tensor,
        dones: torch.Tensor,
        actions: torch.Tensor | None = None,
        step: int | None = None,
    ) -> list[int]:
        """Check for NaN. Returns list of affected env indices (empty if clean).

        On NaN: dumps Newton state (current + previous step) and RL tensors.
        """
        self.step_count = step if step is not None else self.step_count + 1

        nan_info = self._detect_nan(obs, rewards, dones)
        if not nan_info:
            return []

        # Dump state
        if self.dump_count < self.max_dumps:
            self._dump(nan_info, obs, rewards, dones, actions)

        # Return affected env indices
        affected = set()
        for info in nan_info.values():
            if isinstance(info, dict) and "env_indices" in info:
                affected.update(info["env_indices"])
        return sorted(affected)

    def _detect_nan(self, obs, rewards, dones) -> dict | None:
        """Check all tensors for NaN."""
        nan_locations = {}

        for key, tensor in obs.items():
            nan_mask = torch.isnan(tensor)
            if nan_mask.any():
                if tensor.dim() >= 2:
                    nan_envs = nan_mask.flatten(1).any(dim=-1).nonzero(as_tuple=True)[0].unique().tolist()
                else:
                    nan_envs = nan_mask.nonzero(as_tuple=True)[0].unique().tolist()
                nan_locations[f"obs.{key}"] = {
                    "count": nan_mask.sum().item(),
                    "total": tensor.numel(),
                    "env_indices": nan_envs[:50],
                    "shape": list(tensor.shape),
                }

        if torch.isnan(rewards).any():
            nan_locations["rewards"] = {
                "count": torch.isnan(rewards).sum().item(),
                "env_indices": torch.isnan(rewards.flatten()).nonzero(as_tuple=True)[0].tolist()[:50],
            }

        return nan_locations if nan_locations else None

    def _dump(self, nan_info, obs, rewards, dones, actions):
        """Dump Newton state + RL tensors to disk."""
        self.dump_count += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dump_path = self.dump_dir / f"nan_step{self.step_count}_{timestamp}"
        dump_path.mkdir(parents=True, exist_ok=True)

        logger.error(f"NaN detected at step {self.step_count}! Dumping to {dump_path}")

        # 1. Dump Newton state (current + previous from Newton's double buffer)
        self._dump_newton_state(dump_path)

        # 2. NaN metadata
        with open(dump_path / "nan_info.json", "w") as f:
            json.dump({
                "step": self.step_count,
                "timestamp": timestamp,
                "nan_locations": nan_info,
                "dump_number": self.dump_count,
            }, f, indent=2)

        # 3. RL tensors
        torch.save({
            "observations": {k: v.cpu() for k, v in obs.items()},
            "rewards": rewards.cpu(),
            "dones": dones.cpu(),
            "actions": actions.cpu() if actions is not None else None,
        }, dump_path / "rl_tensors.pt")

        envs = set()
        for info in nan_info.values():
            if isinstance(info, dict) and "env_indices" in info:
                envs.update(info["env_indices"])
        logger.error(
            f"NaN dump #{self.dump_count} → {dump_path} "
            f"({len(envs)} envs affected: {sorted(envs)[:10]}{'...' if len(envs) > 10 else ''})"
        )

    def _dump_newton_state(self, dump_path: Path):
        """Dump Newton's current and previous state, plus control, from its own buffers."""
        try:
            from isaaclab_newton.physics.newton_manager import NewtonManager

            state_data = {}

            # Dump current and previous physics state
            for name, state in [("current", NewtonManager._state_0), ("previous", NewtonManager._state_1)]:
                if state is None:
                    continue
                for attr in ['body_q', 'body_qd', 'body_f', 'joint_q', 'joint_qd', 'body_q_prev']:
                    val = getattr(state, attr, None)
                    if val is not None:
                        try:
                            state_data[f"{name}.{attr}"] = torch.from_numpy(val.numpy()).cpu()
                        except Exception:
                            pass

            # Dump control (joint actions/forces — needed for replay)
            control = getattr(NewtonManager, '_control', None)
            if control is not None:
                for attr in ['joint_act', 'joint_f', 'joint_target_pos', 'joint_target_vel']:
                    val = getattr(control, attr, None)
                    if val is not None:
                        try:
                            state_data[f"control.{attr}"] = torch.from_numpy(val.numpy()).cpu()
                        except Exception:
                            pass

            # Dump solver config for replay
            solver = getattr(NewtonManager, '_solver', None)
            if solver is not None:
                solver_cfg = {}
                for attr in ['_solver_dt', '_num_substeps']:
                    val = getattr(NewtonManager, attr, None)
                    if val is not None:
                        solver_cfg[attr] = val
                if hasattr(solver, 'mjw_model') and solver.mjw_model is not None:
                    opt = solver.mjw_model.opt
                    for attr in ['iterations', 'ls_iterations', 'ccd_iterations', 'ccd_tolerance']:
                        val = getattr(opt, attr, None)
                        if val is not None:
                            try:
                                solver_cfg[f"opt.{attr}"] = val.numpy().item() if hasattr(val, 'numpy') else val
                            except Exception:
                                pass
                state_data["solver_config"] = solver_cfg

            # Model info for replay context
            model = NewtonManager._model
            if model is not None:
                for attr in ['body_count', 'joint_count', 'shape_count', 'joint_type', 'joint_axis']:
                    val = getattr(model, attr, None)
                    if val is not None:
                        try:
                            if hasattr(val, 'numpy'):
                                state_data[f"model.{attr}"] = torch.from_numpy(val.numpy()).cpu()
                            else:
                                state_data[f"model.{attr}"] = val
                        except Exception:
                            pass

            if state_data:
                torch.save(state_data, dump_path / "newton_state.pt")
                logger.info(f"Saved Newton state: {list(state_data.keys())}")

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to dump Newton state: {e}")

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Utility to compute and apply joint remapping between backends.

When a policy is trained on one physics backend (e.g., Newton) and evaluated
on another (e.g., PhysX), the joint ordering may differ. This module:

1. Detects the mismatch by comparing joint name lists
2. Computes a permutation from training order → eval order
3. Wraps the policy to remap actions and observations on the fly

Usage:
    # After creating env and loading policy:
    from joint_remapper import JointRemapper

    remapper = JointRemapper(
        train_joint_names=["iiwa7_joint_1", "index_joint_0", "index_joint_1", ...],
        eval_joint_names=env.unwrapped.scene['robot'].joint_names,
    )

    if remapper.needs_remap:
        print(f"Remapping {remapper.num_mismatched} joints")
        # In the step loop:
        actions = policy(obs)
        actions = remapper.remap_actions(actions)
        # Or remap obs before feeding to policy:
        obs = remapper.remap_obs(obs, obs_layout)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch


@dataclass
class ObsSlice:
    """Describes a slice of the observation vector that needs joint/body remapping."""

    start: int
    end: int
    dim_per_item: int = 1  # 1 for joint_pos/vel, 13 for body_state


class JointRemapper:
    """Computes and applies joint order permutation between training and eval environments.

    The remapper compares the joint name lists from the training and evaluation
    environments. If they contain the same joints but in different order, it computes
    a permutation that maps training indices → eval indices.

    For actions: train_action[i] → eval_action[perm[i]]
    For observations: eval_obs[..., perm[i]] → train_obs[..., i]  (inverse permutation)
    """

    def __init__(self, train_joint_names: list[str], eval_joint_names: list[str]):
        """Initialize the remapper.

        Args:
            train_joint_names: Joint names in the order the policy was trained with.
            eval_joint_names: Joint names in the order the eval environment uses.

        Raises:
            ValueError: If the joint sets don't match (different joints, not just reordered).
        """
        self.train_joint_names = list(train_joint_names)
        self.eval_joint_names = list(eval_joint_names)

        if set(train_joint_names) != set(eval_joint_names):
            train_only = set(train_joint_names) - set(eval_joint_names)
            eval_only = set(eval_joint_names) - set(train_joint_names)
            raise ValueError(
                f"Joint sets don't match!\n"
                f"  Training only: {train_only}\n"
                f"  Eval only: {eval_only}"
            )

        # Build permutation: train_to_eval[i] = j means train joint i maps to eval joint j
        eval_name_to_idx = {name: idx for idx, name in enumerate(eval_joint_names)}
        self.train_to_eval = [eval_name_to_idx[name] for name in train_joint_names]

        # Inverse permutation: eval_to_train[j] = i means eval joint j maps to train joint i
        self.eval_to_train = [0] * len(self.train_to_eval)
        for i, j in enumerate(self.train_to_eval):
            self.eval_to_train[j] = i

        # Check if remapping is actually needed
        self.needs_remap = self.train_to_eval != list(range(len(train_joint_names)))
        self.num_joints = len(train_joint_names)
        self.num_mismatched = sum(
            1 for i, j in enumerate(self.train_to_eval) if i != j
        )

        if self.needs_remap:
            # Pre-compute as tensors for GPU efficiency
            self._train_to_eval_idx = None  # Lazily created on first use
            self._eval_to_train_idx = None

    def _ensure_tensors(self, device: torch.device):
        """Create index tensors on the correct device."""
        if self._train_to_eval_idx is None or self._train_to_eval_idx.device != device:
            self._train_to_eval_idx = torch.tensor(self.train_to_eval, dtype=torch.long, device=device)
            self._eval_to_train_idx = torch.tensor(self.eval_to_train, dtype=torch.long, device=device)

    def remap_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Remap actions from training joint order to eval joint order.

        The policy outputs actions in training order. We need to permute them
        so each action component goes to the correct joint in the eval environment.

        Args:
            actions: Tensor of shape (num_envs, num_joints)

        Returns:
            Remapped actions tensor.
        """
        if not self.needs_remap:
            return actions
        self._ensure_tensors(actions.device)
        # actions[:, train_to_eval[i]] = original_actions[:, i]
        # Equivalent: result[:, j] = actions[:, eval_to_train[j]]  (wrong)
        # Actually: for each train index i, the action should go to eval index train_to_eval[i]
        # So result[:, train_to_eval] = actions  →  scatter
        result = torch.empty_like(actions)
        result[:, self._train_to_eval_idx] = actions
        return result

    def remap_joint_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Remap joint-ordered observations from eval order to training order.

        The eval environment produces observations in eval joint order.
        The policy expects them in training joint order.

        Args:
            obs: Tensor of shape (num_envs, num_joints) in eval order.

        Returns:
            Tensor reordered to training joint order.
        """
        if not self.needs_remap:
            return obs
        self._ensure_tensors(obs.device)
        # obs in eval order, policy wants training order
        # result[:, i] = obs[:, train_to_eval[i]]
        return obs[:, self._train_to_eval_idx]

    def remap_body_obs(self, obs: torch.Tensor, dim_per_body: int = 13) -> torch.Tensor:
        """Remap body-ordered observations (e.g., hand_tips_state_b) from eval to training order.

        Body ordering typically follows joint ordering. Each body contributes
        `dim_per_body` values (default 13: pos(3) + quat(4) + lin_vel(3) + ang_vel(3)).

        Args:
            obs: Tensor of shape (num_envs, num_bodies * dim_per_body) in eval order.
            dim_per_body: Number of features per body. Default 13.

        Returns:
            Tensor reordered to training body order.
        """
        if not self.needs_remap:
            return obs
        self._ensure_tensors(obs.device)
        num_envs = obs.shape[0]
        num_bodies = obs.shape[1] // dim_per_body
        # Reshape to (num_envs, num_bodies, dim_per_body), permute, reshape back
        reshaped = obs.view(num_envs, num_bodies, dim_per_body)
        # Use the same permutation (body order follows joint order in this robot)
        # But only if num_bodies matches num_joints
        if num_bodies == self.num_joints:
            permuted = reshaped[:, self._train_to_eval_idx, :]
            return permuted.reshape(num_envs, -1)
        else:
            # Body count differs from joint count — can't automatically remap
            return obs

    def print_mapping(self):
        """Print the joint mapping for debugging."""
        print(f"\nJoint Remapping ({self.num_mismatched}/{self.num_joints} joints differ):")
        print(f"{'Idx':>4s} | {'Training':>20s} → {'Eval':>20s} | {'Match':>5s}")
        print("-" * 60)
        for i in range(self.num_joints):
            j = self.train_to_eval[i]
            match = "  ✓" if i == j else "  ✗"
            print(f"{i:4d} | {self.train_joint_names[i]:>20s} → {self.eval_joint_names[j]:>20s} |{match}")

    @staticmethod
    def save_joint_names(joint_names: list[str], checkpoint_dir: str, body_names: list[str] | None = None):
        """Save joint and body names alongside a checkpoint for future cross-backend use.

        Args:
            joint_names: The joint names from the training environment.
            checkpoint_dir: Directory where the checkpoint is saved.
            body_names: Optional body names from the training environment.
        """
        path = os.path.join(checkpoint_dir, "joint_names.json")
        data = {"joint_names": joint_names, "num_joints": len(joint_names)}
        if body_names is not None:
            data["body_names"] = body_names
            data["num_bodies"] = len(body_names)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO]: Saved training joint names to {path}")

    @staticmethod
    def load_joint_names(checkpoint_dir: str) -> list[str] | None:
        """Load joint names saved alongside a checkpoint.

        Args:
            checkpoint_dir: Directory where the checkpoint is saved.

        Returns:
            List of joint names, or None if not found.
        """
        path = os.path.join(checkpoint_dir, "joint_names.json")
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
            return data["joint_names"]
        return None

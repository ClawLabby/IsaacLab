"""Wrapper that remaps joint-ordered observations and actions for cross-backend play."""

from __future__ import annotations

import torch
from tensordict import TensorDict


class RemappedPolicy:
    """Wraps a policy to transparently remap joint ordering.

    Handles history-stacked observations (e.g., joint_pos with history_length=5
    stored as flat [23*5] = [frame0_j0..j22, frame1_j0..j22, ...]).
    """

    def __init__(self, policy, joint_remapper, obs_manager, obs_groups: list[str]):
        """
        Args:
            policy: The original inference policy.
            joint_remapper: JointRemapper instance with computed permutation.
            obs_manager: The env's observation_manager.
            obs_groups: List of obs group names in the order the policy concatenates them.
        """
        self.policy = policy
        self.remapper = joint_remapper
        self.num_joints = joint_remapper.num_joints

        # Find which (group, offset, size) tuples need remapping
        # These are joint-ordered terms: joint_pos, joint_vel, actions (last_action)
        self.remap_info = []  # list of (group_name, offset_in_group, total_size, block_size)

        joint_terms = {"joint_pos", "joint_vel", "actions"}

        for group_name in obs_groups:
            if group_name not in obs_manager.active_terms:
                continue
            term_names = obs_manager.active_terms[group_name]
            term_dims = obs_manager.group_obs_term_dim[group_name]

            offset = 0
            for tname, tdim in zip(term_names, term_dims):
                tsize = 1
                for d in tdim:
                    tsize *= d
                if tname in joint_terms and tsize % self.num_joints == 0:
                    self.remap_info.append((group_name, offset, tsize, self.num_joints))
                    print(f"  [REMAP] group='{group_name}' term='{tname}' "
                          f"offset={offset} size={tsize} ({tsize // self.num_joints} blocks of {self.num_joints})")
                offset += tsize

        print(f"\n[INFO]: RemappedPolicy wrapping {len(self.remap_info)} observation terms for joint remapping.\n")

    def __call__(self, obs: TensorDict) -> torch.Tensor:
        """Remap obs, run policy, remap actions."""
        # Remap observations: eval joint order → training joint order
        for group_name, offset, total_size, block_size in self.remap_info:
            group_tensor = obs[group_name]  # (num_envs, group_dim)
            chunk = group_tensor[:, offset:offset + total_size]  # (num_envs, N*block)
            num_blocks = total_size // block_size
            # Reshape to (num_envs, num_blocks, block_size), permute, reshape back
            reshaped = chunk.view(chunk.shape[0], num_blocks, block_size)
            remapped = self.remapper.remap_joint_obs(reshaped.reshape(-1, block_size))
            remapped = remapped.view(chunk.shape[0], num_blocks, block_size).reshape(chunk.shape[0], -1)
            group_tensor[:, offset:offset + total_size] = remapped

        # Run policy
        actions = self.policy(obs)

        # Remap actions: training joint order → eval joint order
        actions = self.remapper.remap_actions(actions)

        return actions

    def reset(self, dones: torch.Tensor | None = None):
        """Forward reset to underlying policy."""
        if hasattr(self.policy, 'reset'):
            self.policy.reset(dones)

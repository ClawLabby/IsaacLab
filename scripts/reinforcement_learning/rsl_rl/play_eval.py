# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RL policy with metrics and optional cross-backend joint remapping.

Usage:
    python play_eval.py \
        --task Isaac-Dexsuite-Kuka-Allegro-Lift-v0 \
        --num_envs 4 --headless \
        --checkpoint path/to/model.pt \
        --num_episodes 50 \
        --remap-joints \
        presets=physx
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, os.path.dirname(__file__))
import cli_args  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate a trained RL policy with metrics.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--remap-joints", action="store_true", default=False,
    help="Remap joint ordering for cross-backend evaluation. Reads joint_names.json from checkpoint dir.",
)

cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate the checkpoint with metrics."""
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        if args_cli.checkpoint:
            resume_path = args_cli.checkpoint
        else:
            log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
            resume_path = get_checkpoint_path(os.path.abspath(log_root_path), agent_cfg.load_run, agent_cfg.load_checkpoint)

        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"\n[INFO]: Loading checkpoint from: {resume_path}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # -- Joint remapping --
        remap_info = []  # list of (group_name, offset_in_group, total_size)
        joint_remapper = None

        if args_cli.remap_joints:
            from joint_remapper import JointRemapper
            checkpoint_dir = os.path.dirname(resume_path)
            train_joint_names = JointRemapper.load_joint_names(checkpoint_dir)
            if train_joint_names is not None:
                eval_joint_names = env.unwrapped.scene["robot"].joint_names
                try:
                    joint_remapper = JointRemapper(train_joint_names, eval_joint_names)
                    if joint_remapper.needs_remap:
                        joint_remapper.print_mapping()
                        num_j = joint_remapper.num_joints
                        obs_mgr = env.unwrapped.observation_manager
                        for gname in obs_mgr.active_terms:
                            tdims = obs_mgr.group_obs_term_dim[gname]
                            tnames = obs_mgr.active_terms[gname]
                            goff = 0
                            for tn, td in zip(tnames, tdims):
                                tsz = 1
                                for d in td:
                                    tsz *= d
                                if tn in ("joint_pos", "joint_vel", "actions") and tsz % num_j == 0:
                                    remap_info.append((gname, goff, tsz, "joint"))
                                    print(f"  [REMAP] '{gname}/{tn}' offset={goff} size={tsz} ({tsz//num_j}x{num_j}) [joint]")
                                goff += tsz

                        # Build body remapper for hand_tips_state_b
                        body_remapper = None
                        import json as _json
                        bpath = os.path.join(checkpoint_dir, "joint_names.json")
                        if os.path.isfile(bpath):
                            bdata = _json.load(open(bpath))
                            if "body_names" in bdata:
                                try:
                                    eval_bnames = list(env.unwrapped.scene["robot"].body_names)
                                    body_remapper = JointRemapper(bdata["body_names"], eval_bnames)
                                    if not body_remapper.needs_remap:
                                        body_remapper = None
                                    else:
                                        print(f"  [REMAP] Body ordering differs: {body_remapper.num_mismatched}/{len(eval_bnames)} bodies")
                                        # Find body-ordered obs terms
                                        for gname in obs_mgr.active_terms:
                                            tdims = obs_mgr.group_obs_term_dim[gname]
                                            tnames = obs_mgr.active_terms[gname]
                                            goff = 0
                                            for tn, td in zip(tnames, tdims):
                                                tsz = 1
                                                for d in td:
                                                    tsz *= d
                                                if "state_b" in tn and tsz % 13 == 0:
                                                    nb = tsz // 13
                                                    remap_info.append((gname, goff, tsz, "body"))
                                                    print(f"  [REMAP] '{gname}/{tn}' offset={goff} size={tsz} ({nb}x13) [body]")
                                                goff += tsz
                                except ValueError as ve:
                                    print(f"  [WARN] Body sets differ: {ve}")

                        print(f"\n[INFO]: Remapping {len(remap_info)} obs terms, {joint_remapper.num_mismatched}/{num_j} joints differ.\n")
                    else:
                        print("[INFO]: Joint ordering matches. No remapping needed.")
                        joint_remapper = None
                except ValueError as e:
                    print(f"[ERROR]: Cannot remap: {e}")

        # -- Metrics --
        num_envs = args_cli.num_envs
        episode_rewards, episode_lengths, reward_term_totals = [], [], {}
        current_rewards = torch.zeros(num_envs, device=env.unwrapped.device)
        current_lengths = torch.zeros(num_envs, dtype=torch.long, device=env.unwrapped.device)

        try:
            reward_manager = env.unwrapped.reward_manager
            rw_term_names = list(reward_manager.active_terms)
            for n in rw_term_names:
                reward_term_totals[n] = []
        except Exception:
            rw_term_names = []

        obs = env.get_observations()
        total_episodes = 0
        timestep = 0
        t0 = time.time()

        print(f"[INFO]: Evaluating {args_cli.num_episodes} episodes with {num_envs} envs...")

        try:
            while total_episodes < args_cli.num_episodes:
                with torch.inference_mode():
                    # Remap obs: eval joint order → training joint order
                    if joint_remapper is not None:
                        for gname, goff, gsz, rtype in remap_info:
                            chunk = obs[gname][:, goff:goff + gsz]
                            if rtype == "joint":
                                flat = chunk.reshape(-1, joint_remapper.num_joints)
                                remapped = joint_remapper.remap_joint_obs(flat)
                            elif rtype == "body":
                                # Body remapping requires matching the exact subset of bodies
                                # used by the obs term. For now, skip with a note.
                                continue
                            else:
                                continue
                            obs[gname][:, goff:goff + gsz] = remapped.reshape(chunk.shape)

                    actions = policy(obs)

                    # Remap actions: training joint order → eval joint order
                    if joint_remapper is not None:
                        actions = joint_remapper.remap_actions(actions)

                    obs, rewards, dones, extras = env.step(actions)

                current_rewards += rewards.squeeze()
                current_lengths += 1
                timestep += 1

                done_mask = dones.squeeze().bool()
                if done_mask.any():
                    done_indices = done_mask.nonzero(as_tuple=False).squeeze(-1)
                    log_info = extras.get("log", {})

                    for idx in done_indices:
                        i = idx.item()
                        episode_rewards.append(current_rewards[i].item())
                        episode_lengths.append(current_lengths[i].item())
                        current_rewards[i] = 0
                        current_lengths[i] = 0
                        total_episodes += 1

                    for key, val in log_info.items():
                        if key.startswith("Episode_Reward/"):
                            tname = key.replace("Episode_Reward/", "")
                            if tname not in reward_term_totals:
                                reward_term_totals[tname] = []
                            v = val.item() if torch.is_tensor(val) else val
                            for _ in done_indices:
                                reward_term_totals[tname].append(v)

                if total_episodes > 0 and total_episodes % max(10, num_envs) == 0:
                    elapsed = time.time() - t0
                    mr = np.mean(episode_rewards[-100:])
                    ss = ""
                    if "success" in reward_term_totals and reward_term_totals["success"]:
                        ss = f" | Success: {np.mean(reward_term_totals['success'][-100:]):.2f}"
                    print(f"  Episodes: {total_episodes}/{args_cli.num_episodes} | Mean reward: {mr:.2f}{ss} | {elapsed:.1f}s")

        except KeyboardInterrupt:
            print("\n[INFO]: Interrupted.")

        elapsed = time.time() - t0
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"\nCheckpoint: {resume_path}")
        print(f"Task: {args_cli.task}")
        print(f"Num envs: {num_envs} | Episodes: {len(episode_rewards)} | Time: {elapsed:.1f}s")

        if episode_rewards:
            print(f"\n--- Episode Rewards ---")
            print(f"  Mean: {np.mean(episode_rewards):>8.3f} ± {np.std(episode_rewards):.3f}")
            print(f"  Min:  {np.min(episode_rewards):>8.3f}  Max: {np.max(episode_rewards):.3f}")

        if episode_lengths:
            print(f"\n--- Episode Lengths ---")
            print(f"  Mean: {np.mean(episode_lengths):>8.1f} ± {np.std(episode_lengths):.1f}")

        if reward_term_totals:
            print(f"\n--- Reward Terms (per-episode mean) ---")
            for name in sorted(reward_term_totals.keys()):
                vals = reward_term_totals[name]
                if vals:
                    print(f"  {name:>30s}: {np.mean(vals):>8.3f} ± {np.std(vals):.3f}")

        print("\n" + "=" * 70)
        env.close()


if __name__ == "__main__":
    main()

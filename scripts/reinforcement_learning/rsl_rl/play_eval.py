# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a trained RL policy with proper metrics logging.

Based on play.py but adds:
- Success rate tracking (from reward terms)
- Episode reward statistics  
- Position error tracking
- Episode count and timing
- Optional joint remapping for cross-backend evaluation

Usage:
    python play_eval.py \
        --task Isaac-Dexsuite-Kuka-Allegro-Lift-v0 \
        --num_envs 64 --headless \
        --checkpoint path/to/model.pt \
        --num_episodes 100 \
        presets=newton
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
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
sys.path.insert(0, os.path.dirname(__file__))
import cli_args  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate a trained RL policy with metrics.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to evaluate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point."
)
parser.add_argument(
    "--num_episodes", type=int, default=100,
    help="Minimum number of episodes to complete before reporting metrics."
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
parser.add_argument(
    "--remap-joints",
    action="store_true",
    default=False,
    help="Remap joint ordering when evaluating a checkpoint trained on a different backend. "
    "Reads training joint order from joint_names.json in the checkpoint directory.",
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
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # resolve checkpoint
        if args_cli.checkpoint:
            resume_path = args_cli.checkpoint
        else:
            log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
            log_root_path = os.path.abspath(log_root_path)
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent
            env = multi_agent_to_single_agent(env)

        # wrap for video
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "eval"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"\n[INFO]: Loading checkpoint from: {resume_path}")

        # create runner and load
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)

        # get policy
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        if version.parse(installed_version) >= version.parse("4.0.0"):
            policy_nn = None
        else:
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

        dt = env.unwrapped.step_dt

        # -- Joint remapping for cross-backend evaluation --
        joint_remapper = None
        obs_remap_slices = []
        if args_cli.remap_joints:
            from joint_remapper import JointRemapper

            checkpoint_dir = os.path.dirname(resume_path)
            train_joint_names = JointRemapper.load_joint_names(checkpoint_dir)
            if train_joint_names is None:
                print("[WARN]: --remap-joints specified but no joint_names.json found.")
                print(f"        Expected at: {os.path.join(checkpoint_dir, 'joint_names.json')}")
            else:
                eval_joint_names = env.unwrapped.scene["robot"].joint_names
                try:
                    joint_remapper = JointRemapper(train_joint_names, eval_joint_names)
                    if joint_remapper.needs_remap:
                        joint_remapper.print_mapping()

                        # Compute observation slices for joint-ordered terms
                        obs_manager = env.unwrapped.observation_manager
                        flat_offset = 0
                        num_joints = len(eval_joint_names)
                        for group_name in obs_manager.active_terms:
                            term_dims = obs_manager.group_obs_term_dim[group_name]
                            term_names = obs_manager.active_terms[group_name]
                            for term_name, term_dim in zip(term_names, term_dims):
                                term_size = 1
                                for d in term_dim:
                                    term_size *= d
                                if term_name in ("joint_pos", "joint_vel", "actions") and term_size == num_joints:
                                    obs_remap_slices.append((flat_offset, flat_offset + term_size))
                                    print(f"  [REMAP] Obs '{term_name}' at [{flat_offset}:{flat_offset + term_size}]")
                                flat_offset += term_size

                        print(f"\n[INFO]: Joint remapping active. {joint_remapper.num_mismatched}/{num_joints} joints remapped.\n")
                    else:
                        print("[INFO]: Joint ordering matches. No remapping needed.")
                        joint_remapper = None
                except ValueError as e:
                    print(f"[ERROR]: Cannot remap joints: {e}")

        # ---- Metrics tracking ----
        episode_rewards = []
        episode_lengths = []
        reward_term_totals = {}  # per-term reward accumulation

        # Per-env tracking
        num_envs = args_cli.num_envs
        current_rewards = torch.zeros(num_envs, device=env.unwrapped.device)
        current_lengths = torch.zeros(num_envs, dtype=torch.long, device=env.unwrapped.device)

        # Get reward term names
        try:
            reward_manager = env.unwrapped.reward_manager
            term_names = list(reward_manager.active_terms)
            for name in term_names:
                reward_term_totals[name] = []
            print(f"[INFO]: Tracking {len(term_names)} reward terms: {term_names}")
        except Exception as e:
            term_names = []
            print(f"[WARN]: Could not get reward terms: {e}")

        # Reset environment
        obs = env.get_observations()
        total_episodes = 0
        timestep = 0
        start_time = time.time()

        print(f"\n[INFO]: Evaluating for at least {args_cli.num_episodes} episodes...")
        print(f"[INFO]: Using {num_envs} parallel environments\n")

        try:
            while total_episodes < args_cli.num_episodes:
                with torch.inference_mode():
                    # Remap observations from eval order to training order
                    if joint_remapper is not None:
                        for start, end in obs_remap_slices:
                            obs[:, start:end] = joint_remapper.remap_joint_obs(obs[:, start:end])

                    actions = policy(obs)

                    # Remap actions from training order to eval order
                    if joint_remapper is not None:
                        actions = joint_remapper.remap_actions(actions)

                    obs, rewards, dones, extras = env.step(actions)

                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    elif policy_nn is not None:
                        policy_nn.reset(dones)

                # Accumulate per-env rewards
                current_rewards += rewards.squeeze()
                current_lengths += 1
                timestep += 1

                # Process completed episodes
                done_mask = dones.squeeze().bool()
                if done_mask.any():
                    done_indices = done_mask.nonzero(as_tuple=False).squeeze(-1)
                    
                    # Grab per-term episode info from extras["log"]
                    # The env logs Episode_Reward/<term> as averages over reset envs
                    log_info = extras.get("log", {})
                    
                    for idx in done_indices:
                        i = idx.item()
                        episode_rewards.append(current_rewards[i].item())
                        episode_lengths.append(current_lengths[i].item())
                        current_rewards[i] = 0
                        current_lengths[i] = 0
                        total_episodes += 1
                    
                    # Log extras contain averaged reward terms across reset envs
                    # Store them once per batch of resets
                    for key, val in log_info.items():
                        if key.startswith("Episode_Reward/"):
                            term_name = key.replace("Episode_Reward/", "")
                            if term_name not in reward_term_totals:
                                reward_term_totals[term_name] = []
                            v = val.item() if torch.is_tensor(val) else val
                            # This is already the mean over reset envs, store for each reset env
                            for _ in done_indices:
                                reward_term_totals[term_name].append(v)

                # Progress update
                if total_episodes > 0 and total_episodes % max(10, num_envs) == 0:
                    elapsed = time.time() - start_time
                    mean_r = np.mean(episode_rewards[-100:])
                    # Success term if exists
                    success_str = ""
                    if 'success' in reward_term_totals and reward_term_totals['success']:
                        recent = reward_term_totals['success'][-100:]
                        success_str = f" | Success(mean): {np.mean(recent):.2f}"
                    print(f"  Episodes: {total_episodes}/{args_cli.num_episodes} | "
                          f"Mean reward: {mean_r:.2f}{success_str} | "
                          f"Time: {elapsed:.1f}s")

        except KeyboardInterrupt:
            print("\n[INFO]: Evaluation interrupted.")

        # ---- Final Report ----
        elapsed_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"\nCheckpoint: {resume_path}")
        print(f"Task: {args_cli.task}")
        print(f"Num envs: {num_envs}")
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Total timesteps: {timestep}")
        print(f"Evaluation time: {elapsed_time:.1f}s")
        if elapsed_time > 0:
            print(f"Episodes/second: {len(episode_rewards)/elapsed_time:.2f}")

        if episode_rewards:
            print(f"\n--- Episode Rewards ---")
            print(f"  Mean:   {np.mean(episode_rewards):>10.3f}")
            print(f"  Std:    {np.std(episode_rewards):>10.3f}")
            print(f"  Min:    {np.min(episode_rewards):>10.3f}")
            print(f"  Max:    {np.max(episode_rewards):>10.3f}")
            print(f"  Median: {np.median(episode_rewards):>10.3f}")

        if episode_lengths:
            print(f"\n--- Episode Lengths ---")
            print(f"  Mean:   {np.mean(episode_lengths):>10.1f}")
            print(f"  Std:    {np.std(episode_lengths):>10.1f}")

        if reward_term_totals:
            print(f"\n--- Reward Term Breakdown (per-episode mean) ---")
            for name in sorted(reward_term_totals.keys()):
                vals = reward_term_totals[name]
                if vals:
                    print(f"  {name:>30s}: {np.mean(vals):>10.3f} ± {np.std(vals):.3f}")

        print("\n" + "=" * 70)

        env.close()


if __name__ == "__main__":
    main()

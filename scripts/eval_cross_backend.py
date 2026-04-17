#!/usr/bin/env python3
"""Cross-backend sim2sim evaluation script.

Evaluates a trained policy across physics backends and collects per-episode
cumulative rewards. Outputs mean/std/min/max stats and saves JSON results.

Usage:
    PYTHONDONTWRITEBYTECODE=1 DISPLAY=:99 __GLX_VENDOR_LIBRARY_NAME=nvidia OMNI_KIT_ACCEPT_EULA=yes \
    python scripts/eval_cross_backend.py \
        --task Isaac-Dexsuite-Kuka-Allegro-Lift-v0 \
        --checkpoint logs/rsl_rl/dexsuite_kuka_allegro/2026-03-31_21-53-28/model_14999.pt \
        --num_envs 16 --num_episodes 50 --label test_a \
        --headless \
        'presets=cube' 'env.sim.physics=newton'
"""

import argparse
import contextlib
import importlib.metadata as metadata
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from packaging import version

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# Add rsl_rl scripts dir to path for joint_remapper and remapped_policy
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RSL_RL_DIR = os.path.join(SCRIPT_DIR, "reinforcement_learning", "rsl_rl")
if RSL_RL_DIR not in sys.path:
    sys.path.insert(0, RSL_RL_DIR)

from joint_remapper import JointRemapper
from remapped_policy import RemappedPolicy

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Cross-backend sim2sim evaluation")
parser.add_argument("--task", type=str, default="Isaac-Dexsuite-Kuka-Allegro-Lift-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=50, help="Number of completed episodes to collect")
parser.add_argument("--max_steps", type=int, default=20000, help="Safety limit on total steps")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--label", type=str, default="eval", help="Label for this evaluation run")
parser.add_argument("--output_dir", type=str, default="/tmp/sim2sim_results")
parser.add_argument("--remap-joints", action="store_true", default=False,
                    help="Enable joint remapping for cross-backend playback")
add_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run cross-backend evaluation."""
    with launch_simulation(env_cfg, args_cli):
        # Configure
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        if hasattr(args_cli, 'device') and args_cli.device:
            env_cfg.sim.device = args_cli.device

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # Create environment
        env = gym.make(args_cli.task, cfg=env_cfg)

        # Wrap for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Load checkpoint
        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")

        agent_cfg.device = env.unwrapped.device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # Joint remapping setup
        if args_cli.remap_joints:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            train_joint_names = JointRemapper.load_joint_names(checkpoint_dir)
            if train_joint_names is None:
                print("[WARN] No joint_names.json found — skipping remapping")
            else:
                eval_joint_names = env.unwrapped.scene['robot'].joint_names
                print(f"[INFO] Training joints ({len(train_joint_names)}): {train_joint_names[:5]}...")
                print(f"[INFO] Eval joints ({len(eval_joint_names)}): {eval_joint_names[:5]}...")

                remapper = JointRemapper(train_joint_names, eval_joint_names)
                if remapper.needs_remap:
                    remapper.print_mapping()
                    obs_groups = list(env.unwrapped.observation_manager.active_terms.keys())
                    policy = RemappedPolicy(policy, remapper, env.unwrapped.observation_manager, obs_groups)
                    print("[INFO] Joint remapping ENABLED")
                else:
                    print("[INFO] Joint order matches — no remapping needed")

        # Episode tracking
        num_envs = args_cli.num_envs
        episode_rewards = torch.zeros(num_envs, device=env.unwrapped.device)
        episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=env.unwrapped.device)
        completed_rewards = []
        completed_lengths = []

        # Reset
        obs = env.get_observations()
        step_count = 0

        print(f"\n{'='*60}")
        print(f"Sim2Sim Evaluation: {args_cli.label}")
        print(f"Task: {args_cli.task}")
        print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"Num envs: {num_envs}, Target episodes: {args_cli.num_episodes}")
        print(f"{'='*60}\n")

        try:
            while len(completed_rewards) < args_cli.num_episodes and step_count < args_cli.max_steps:
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, rewards, dones, infos = env.step(actions)

                    # Reset recurrent states for completed episodes
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)

                step_count += 1
                episode_rewards += rewards.squeeze()
                episode_lengths += 1

                # Check for episode completions
                if dones.any():
                    done_mask = dones.squeeze().bool()
                    done_indices = done_mask.nonzero(as_tuple=True)[0]

                    for idx in done_indices:
                        completed_rewards.append(episode_rewards[idx].item())
                        completed_lengths.append(episode_lengths[idx].item())

                    # Reset accumulators for completed envs
                    episode_rewards[done_mask] = 0.0
                    episode_lengths[done_mask] = 0

                # Progress
                if step_count % 200 == 0:
                    n = len(completed_rewards)
                    if n > 0:
                        mean_r = np.mean(completed_rewards)
                        print(f"  Step {step_count}: {n}/{args_cli.num_episodes} episodes, "
                              f"mean_reward={mean_r:.3f}")
                    else:
                        print(f"  Step {step_count}: waiting for first episode completion...")

        except KeyboardInterrupt:
            print("\nInterrupted.")

        # Compute stats
        rewards_arr = np.array(completed_rewards) if completed_rewards else np.array([0.0])
        lengths_arr = np.array(completed_lengths) if completed_lengths else np.array([0])

        stats = {
            "label": args_cli.label,
            "task": args_cli.task,
            "checkpoint": str(checkpoint_path),
            "num_envs": num_envs,
            "num_episodes_completed": len(completed_rewards),
            "num_episodes_target": args_cli.num_episodes,
            "total_steps": step_count,
            "reward_mean": float(rewards_arr.mean()),
            "reward_std": float(rewards_arr.std()),
            "reward_min": float(rewards_arr.min()),
            "reward_max": float(rewards_arr.max()),
            "reward_median": float(np.median(rewards_arr)),
            "length_mean": float(lengths_arr.mean()),
            "length_std": float(lengths_arr.std()),
            "per_episode_rewards": [float(r) for r in completed_rewards],
            "per_episode_lengths": [int(l) for l in completed_lengths],
        }

        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS: {args_cli.label}")
        print(f"{'='*60}")
        print(f"Episodes completed: {stats['num_episodes_completed']}/{args_cli.num_episodes}")
        print(f"Total steps: {step_count}")
        print(f"Mean episode reward:   {stats['reward_mean']:.3f} ± {stats['reward_std']:.3f}")
        print(f"Median episode reward: {stats['reward_median']:.3f}")
        print(f"Min/Max reward:        {stats['reward_min']:.3f} / {stats['reward_max']:.3f}")
        print(f"Mean episode length:   {stats['length_mean']:.1f} ± {stats['length_std']:.1f}")
        print(f"{'='*60}\n")

        # Save results
        os.makedirs(args_cli.output_dir, exist_ok=True)
        output_file = os.path.join(args_cli.output_dir, f"{args_cli.label}.json")
        with open(output_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Results saved to: {output_file}")

        env.close()


if __name__ == "__main__":
    main()

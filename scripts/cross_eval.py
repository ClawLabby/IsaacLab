#!/usr/bin/env python3
"""Cross-domain evaluation script for DexSuite policies.

Loads a trained policy and evaluates it in a different physics/rendering backend
than it was trained on. Outputs metrics (success, position error, reward) and
optionally captures video.

Usage:
    python cross_eval.py \
        --task Isaac-Dexsuite-Kuka-Allegro-Lift-v0 \
        --checkpoint /path/to/model.pt \
        --num_envs 4 \
        --num_episodes 50 \
        --headless \
        presets=newton
"""

import argparse
import contextlib
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from collections import defaultdict

# Add the rsl_rl scripts dir to path for cli_args
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RSL_RL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "scripts/reinforcement_learning/rsl_rl")
if RSL_RL_DIR not in sys.path:
    sys.path.insert(0, RSL_RL_DIR)

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

with contextlib.suppress(ImportError):
    import cli_args  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Cross-domain policy evaluation")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=1000, help="Total steps to evaluate")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video", action="store_true", help="Record video")
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--output_dir", type=str, default="cross_eval_results")
parser.add_argument("--label", type=str, default="eval", help="Label for this evaluation")
add_launcher_args(parser)

# Parse known args, pass rest to hydra
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run cross-domain evaluation."""
    with launch_simulation(env_cfg, args_cli):
        # Configure
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        if hasattr(args_cli, 'device') and args_cli.device:
            env_cfg.sim.device = args_cli.device

        # Output directory
        os.makedirs(args_cli.output_dir, exist_ok=True)
        log_dir = args_cli.output_dir

        # Create environment
        env = gym.make(
            args_cli.task, cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None
        )

        # Video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # Wrap for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Load checkpoint
        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        
        agent_cfg.device = env.unwrapped.device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # Metrics tracking
        metrics = defaultdict(list)
        episode_metrics = defaultdict(list)
        step_count = 0
        episode_count = 0

        # Get environment internals for metrics
        unwrapped = env.unwrapped
        
        # Reset
        obs = env.get_observations()
        
        print(f"\n{'='*60}")
        print(f"Cross-Domain Evaluation: {args_cli.label}")
        print(f"Task: {args_cli.task}")
        print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"Num envs: {args_cli.num_envs}, Steps: {args_cli.num_steps}")
        print(f"{'='*60}\n")

        try:
            while step_count < args_cli.num_steps:
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, rewards, dones, infos = env.step(actions)

                step_count += 1
                
                # Collect per-step rewards
                if isinstance(rewards, torch.Tensor):
                    metrics["reward"].append(rewards.mean().item())

                # Try to extract episode info from the environment
                # DexSuite uses manager-based env with reward/termination managers
                try:
                    # Access reward components if available
                    if hasattr(unwrapped, 'reward_manager'):
                        rm = unwrapped.reward_manager
                        # Get individual reward terms
                        for name in rm.active_terms:
                            val = rm.get_term(name)
                            if isinstance(val, torch.Tensor):
                                metrics[f"reward/{name}"].append(val.mean().item())
                except Exception:
                    pass

                # Check for episode completions
                if isinstance(dones, torch.Tensor) and dones.any():
                    n_done = dones.sum().item()
                    episode_count += n_done

                # Print progress every 100 steps
                if step_count % 100 == 0:
                    avg_reward = np.mean(metrics["reward"][-100:]) if metrics["reward"] else 0
                    print(f"Step {step_count}/{args_cli.num_steps}: "
                          f"avg_reward={avg_reward:.2f}, "
                          f"episodes_completed={episode_count}")

        except KeyboardInterrupt:
            print("\nInterrupted.")

        # Final summary
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE: {args_cli.label}")
        print(f"{'='*60}")
        print(f"Total steps: {step_count}")
        print(f"Episodes completed: {episode_count}")
        
        if metrics["reward"]:
            rewards_arr = np.array(metrics["reward"])
            print(f"Mean reward: {rewards_arr.mean():.3f} ± {rewards_arr.std():.3f}")
        
        # Print reward components
        for key in sorted(metrics.keys()):
            if key.startswith("reward/") and metrics[key]:
                vals = np.array(metrics[key])
                print(f"  {key}: {vals.mean():.4f}")

        # Save metrics to file
        results_file = os.path.join(log_dir, f"{args_cli.label}_metrics.txt")
        with open(results_file, "w") as f:
            f.write(f"Label: {args_cli.label}\n")
            f.write(f"Task: {args_cli.task}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Steps: {step_count}\n")
            f.write(f"Episodes: {episode_count}\n")
            if metrics["reward"]:
                r = np.array(metrics["reward"])
                f.write(f"Mean reward: {r.mean():.3f} ± {r.std():.3f}\n")
            for key in sorted(metrics.keys()):
                if metrics[key]:
                    vals = np.array(metrics[key])
                    f.write(f"{key}: mean={vals.mean():.4f} std={vals.std():.4f}\n")
        
        print(f"\nMetrics saved to: {results_file}")
        if args_cli.video:
            print(f"Videos saved to: {os.path.join(log_dir, 'videos')}")
        
        env.close()


if __name__ == "__main__":
    main()

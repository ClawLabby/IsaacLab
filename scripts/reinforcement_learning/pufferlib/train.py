#!/usr/bin/env python3
"""Train an Isaac Lab task with the experimental PufferLib-style PyTorch backend."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata as metadata
import os
import sys
from dataclasses import MISSING, fields
from datetime import datetime

import gymnasium as gym

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.pufferlib import PufferTorchRunner, PufferTrainCfg, PufferVecEnvWrapper
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import fold_preset_tokens, setup_preset_cli
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got '{value}'")


def _int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.replace(",", " ").split()]


def _missing_to_default(value, default):
    if value is MISSING or isinstance(value, dataclasses._MISSING_TYPE):
        return default
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, conflict_handler="resolve")
    parser.add_argument("--task", type=str, required=True, help="Name of the Isaac Lab task.")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL agent cfg entry point.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Environment and runner seed.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Alias for --iterations.")
    parser.add_argument("--log_root", default=os.path.abspath("logs/pufferlib"), help="Root directory for logs.")
    parser.add_argument("--run_name", default="", help="Optional suffix for the run directory.")
    parser.add_argument(
        "--profile",
        choices=["matched_rsl", "camera_cnn", "mixed_camera_cnn", "puffer_mingru"],
        default=PufferTrainCfg.profile,
    )
    parser.add_argument("--match_rsl_cfg", type=_str_to_bool, default=True, help="Copy PPO settings from RSL-RL cfg.")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--update_epochs", type=int, default=None)
    parser.add_argument("--num_mini_batches", type=int, default=None)
    parser.add_argument("--minibatch_size", type=int, default=None)
    parser.add_argument("--actor_hidden_dims", type=_int_list, default=None)
    parser.add_argument("--critic_hidden_dims", type=_int_list, default=None)
    parser.add_argument("--activation", type=str, default=None)
    parser.add_argument("--cnn_output_channels", type=_int_list, default=None)
    parser.add_argument("--cnn_kernel_size", type=_int_list, default=None)
    parser.add_argument("--cnn_stride", type=_int_list, default=None)
    parser.add_argument("--cnn_activation", type=str, default=None)
    parser.add_argument("--share_cnn_encoders", type=_str_to_bool, default=None)
    parser.add_argument("--init_std", type=float, default=None)
    parser.add_argument("--std_type", choices=["scalar", "log"], default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "muon"], default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--schedule", choices=["adaptive", "fixed"], default=None)
    parser.add_argument("--desired_kl", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--clip_coef", type=float, default=None)
    parser.add_argument("--vf_coef", type=float, default=None)
    parser.add_argument("--ent_coef", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--use_clipped_value_loss", type=_str_to_bool, default=None)
    parser.add_argument("--normalize_advantage_per_minibatch", type=_str_to_bool, default=None)
    parser.add_argument("--check_for_nan", type=_str_to_bool, default=None)
    parser.add_argument("--finite_grad_guard", type=_str_to_bool, default=None)
    parser.add_argument("--init_at_random_ep_len", type=_str_to_bool, default=None)
    parser.add_argument("--clip_actions", type=float, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--pufferlib_path", type=str, default=None)
    add_launcher_args(parser)
    args, remaining_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + fold_preset_tokens(remaining_args)
    return args


def _cfg_from_agent(args: argparse.Namespace, agent_cfg=None) -> PufferTrainCfg:
    cfg = PufferTrainCfg(profile=args.profile)
    if args.match_rsl_cfg:
        if agent_cfg is None:
            agent_cfg = load_cfg_from_registry(args.task, args.agent)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        cfg.horizon = int(agent_cfg.num_steps_per_env)
        cfg.iterations = int(agent_cfg.max_iterations)
        cfg.actor_hidden_dims = list(agent_cfg.actor.hidden_dims)
        cfg.critic_hidden_dims = list(agent_cfg.critic.hidden_dims)
        cfg.activation = str(agent_cfg.actor.activation)
        actor_cnn_cfg = getattr(agent_cfg.actor, "cnn_cfg", None)
        if actor_cnn_cfg is not None:
            cfg.cnn_output_channels = list(actor_cnn_cfg.output_channels)
            cfg.cnn_kernel_size = list(actor_cnn_cfg.kernel_size)
            cfg.cnn_stride = list(actor_cnn_cfg.stride)
            cfg.cnn_activation = str(actor_cnn_cfg.activation)
        dist_cfg = agent_cfg.actor.distribution_cfg
        if dist_cfg is not None:
            cfg.init_std = float(dist_cfg.init_std)
            cfg.std_type = str(dist_cfg.std_type)
        alg = agent_cfg.algorithm
        cfg.update_epochs = int(alg.num_learning_epochs)
        cfg.num_mini_batches = int(alg.num_mini_batches)
        cfg.learning_rate = float(alg.learning_rate)
        cfg.schedule = str(alg.schedule)
        cfg.desired_kl = float(alg.desired_kl)
        cfg.gamma = float(alg.gamma)
        cfg.gae_lambda = float(alg.lam)
        cfg.clip_coef = float(alg.clip_param)
        cfg.vf_coef = float(alg.value_loss_coef)
        cfg.ent_coef = float(alg.entropy_coef)
        cfg.max_grad_norm = float(alg.max_grad_norm)
        cfg.optimizer = str(getattr(alg, "optimizer", "adam"))
        cfg.share_cnn_encoders = bool(getattr(alg, "share_cnn_encoders", cfg.share_cnn_encoders))
        cfg.use_clipped_value_loss = bool(alg.use_clipped_value_loss)
        cfg.normalize_advantage_per_minibatch = bool(getattr(alg, "normalize_advantage_per_mini_batch", False))
        cfg.clip_actions = _missing_to_default(agent_cfg.clip_actions, None)
        cfg.save_interval = int(_missing_to_default(agent_cfg.save_interval, cfg.save_interval))
        obs_groups = _missing_to_default(agent_cfg.obs_groups, {})
        cfg.actor_obs_groups = list(obs_groups.get("actor", []))
        cfg.critic_obs_groups = list(obs_groups.get("critic", []))

    for field in fields(PufferTrainCfg):
        if not hasattr(args, field.name):
            continue
        value = getattr(args, field.name)
        if value is not None:
            setattr(cfg, field.name, value)
    if args.max_iterations is not None:
        cfg.iterations = args.max_iterations
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.device = args.device
    return cfg


def main() -> None:
    args = _parse_args()
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent)
    cfg = _cfg_from_agent(args, agent_cfg)
    env_cfg.sim.device = args.device
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = cfg.seed

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        task_name = args.task.replace("/", "_")
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.run_name:
            stamp = f"{stamp}_{args.run_name}"
        log_dir = os.path.join(args.log_root, task_name, cfg.profile, stamp)
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "puffer.yaml"), cfg)

        wrapped_env = PufferVecEnvWrapper(
            env,
            actor_obs_groups=cfg.actor_obs_groups,
            critic_obs_groups=cfg.critic_obs_groups,
            clip_actions=cfg.clip_actions,
            preserve_observation_shape=cfg.profile == "camera_cnn",
        )
        try:
            PufferTorchRunner(wrapped_env, cfg, log_dir).learn()
        finally:
            wrapped_env.close()


if __name__ == "__main__":
    main()

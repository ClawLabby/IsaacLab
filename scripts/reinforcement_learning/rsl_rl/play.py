# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--remap-joints",
    action="store_true",
    default=False,
    help="Remap joint ordering when playing a checkpoint trained on a different backend. "
    "Reads training joint order from joint_names.json in the checkpoint directory.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# Check for installed RSL-RL version
installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # export the trained policy to JIT and ONNX formats
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        if version.parse(installed_version) >= version.parse("4.0.0"):
            # use the new export functions for rsl-rl >= 4.0.0
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None  # Not needed for rsl-rl >= 4.0.0
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            # export to JIT and ONNX
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt

        # -- Joint remapping for cross-backend play --
        joint_remapper = None
        if args_cli.remap_joints:
            from joint_remapper import JointRemapper

            checkpoint_dir = os.path.dirname(resume_path)
            train_joint_names = JointRemapper.load_joint_names(checkpoint_dir)
            if train_joint_names is None:
                print("[WARN]: --remap-joints specified but no joint_names.json found in checkpoint dir.")
                print(f"        Expected at: {os.path.join(checkpoint_dir, 'joint_names.json')}")
                print("        Continuing without remapping.")
            else:
                eval_joint_names = env.unwrapped.scene["robot"].joint_names
                try:
                    joint_remapper = JointRemapper(train_joint_names, eval_joint_names)
                    if joint_remapper.needs_remap:
                        joint_remapper.print_mapping()

                        # Compute per-group observation slices that need joint-order remapping.
                        obs_manager = env.unwrapped.observation_manager
                        obs_remap_slices = []  # list of (group_name, offset_in_group, size, type)

                        num_joints = len(eval_joint_names)
                        for group_name in obs_manager.active_terms:
                            term_dims = obs_manager.group_obs_term_dim[group_name]
                            term_names = obs_manager.active_terms[group_name]
                            group_offset = 0
                            for term_name, term_dim in zip(term_names, term_dims):
                                term_size = 1
                                for d in term_dim:
                                    term_size *= d
                                # Joint-ordered terms: joint_pos, joint_vel, last_action
                                if term_name in ("joint_pos", "joint_vel", "actions") and term_size % num_joints == 0:
                                    obs_remap_slices.append((group_name, group_offset, term_size, "joint"))
                                    n_hist = term_size // num_joints
                                    print(f"  [REMAP] '{group_name}/{term_name}' offset={group_offset} size={term_size} ({n_hist}x{num_joints})")
                                group_offset += term_size

                        joint_remapper._obs_remap_slices = obs_remap_slices
                        print(f"\n[INFO]: Joint remapping active. {joint_remapper.num_mismatched}/{num_joints} joints remapped.")
                        print(f"[INFO]: {len(obs_remap_slices)} observation terms will be remapped.\n")
                    else:
                        print("[INFO]: Joint ordering matches between training and eval. No remapping needed.")
                        joint_remapper = None
                except ValueError as e:
                    print(f"[ERROR]: Cannot remap joints: {e}")
                    joint_remapper = None

        # reset environment
        obs = env.get_observations()
        timestep = 0
        # simulate environment
        try:
            while True:
                start_time = time.time()
                # run everything in inference mode
                with torch.inference_mode():
                    # remap observations from eval joint order → training joint order
                    if joint_remapper is not None:
                        for gname, goff, gsz, rtype in joint_remapper._obs_remap_slices:
                            if rtype != "joint":
                                continue
                            group_tensor = obs[gname]
                            chunk = group_tensor[:, goff:goff + gsz]
                            nj = joint_remapper.num_joints
                            flat = chunk.reshape(-1, nj)
                            remapped = joint_remapper.remap_joint_obs(flat)
                            group_tensor[:, goff:goff + gsz] = remapped.reshape(chunk.shape)

                    # agent stepping
                    actions = policy(obs)

                    # remap actions from training joint order → eval joint order
                    if joint_remapper is not None:
                        actions = joint_remapper.remap_actions(actions)

                    # env stepping
                    obs, _, dones, _ = env.step(actions)
                    # reset recurrent states for episodes that have terminated
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)
                if args_cli.video:
                    timestep += 1
                    if timestep == args_cli.video_length:
                        break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            # close the simulator
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

"""Play a policy with penetration detection and Newton state ring buffer dump.

Monitors the target object Z position. When it drops below the table surface
by a configurable threshold, dumps the ring buffer of the last N physics states
for offline analysis.

Usage:
    python play_debug_penetration.py \
        --task Isaac-Dexsuite-Kuka-Allegro-Lift-Play-v0 \
        --num_envs 4 --viz newton \
        --checkpoint <path_to_model.pt> \
        --max_steps 2000 \
        --ring_size 60 \
        --dump_dir /tmp/penetration_dumps \
        presets=newton,cube
"""

import argparse
import contextlib
import importlib.metadata as metadata
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import gymnasium as gym
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
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

import cli_args  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play with penetration detection.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_steps", type=int, default=5000, help="Max steps to run")
parser.add_argument("--ring_size", type=int, default=60, help="Number of states in ring buffer")
parser.add_argument("--dump_dir", type=str, default="/tmp/penetration_dumps", help="Where to save dumps")
parser.add_argument("--max_dumps", type=int, default=5, help="Max number of dumps before stopping")
parser.add_argument("--threshold_z", type=float, default=0.22,
                    help="Object Z below this triggers dump (table surface ~0.255)")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

installed_version = metadata.version("rsl-rl-lib")


def capture_newton_state():
    """Capture current Newton state as CPU tensors."""
    try:
        from isaaclab_newton.physics.newton_manager import NewtonManager

        snapshot = {}
        for name, state in [("current", NewtonManager._state_0), ("previous", NewtonManager._state_1)]:
            if state is None:
                continue
            for attr in ['body_q', 'body_qd', 'body_f', 'joint_q', 'joint_qd']:
                val = getattr(state, attr, None)
                if val is not None:
                    try:
                        snapshot[f"{name}.{attr}"] = torch.from_numpy(val.numpy()).cpu().clone()
                    except Exception:
                        pass

        control = getattr(NewtonManager, '_control', None)
        if control is not None:
            for attr in ['joint_act', 'joint_target_pos', 'joint_target_vel']:
                val = getattr(control, attr, None)
                if val is not None:
                    try:
                        snapshot[f"control.{attr}"] = torch.from_numpy(val.numpy()).cpu().clone()
                    except Exception:
                        pass

        return snapshot
    except ImportError:
        return {}


def dump_ring_buffer(ring_buffer, dump_path, trigger_info):
    """Save the ring buffer and trigger metadata."""
    dump_path.mkdir(parents=True, exist_ok=True)

    # Save each frame
    for i, (step, state) in enumerate(ring_buffer):
        torch.save(state, dump_path / f"frame_{i:04d}_step{step}.pt")

    # Save trigger info
    with open(dump_path / "trigger_info.json", "w") as f:
        json.dump(trigger_info, f, indent=2, default=str)

    # Save model info for replay context
    try:
        from isaaclab_newton.physics.newton_manager import NewtonManager
        model_info = {}
        model = NewtonManager._model
        if model is not None:
            for attr in ['body_count', 'joint_count', 'shape_count']:
                val = getattr(model, attr, None)
                if val is not None:
                    model_info[attr] = int(val) if not hasattr(val, 'numpy') else val
            # Body names if available
            body_name = getattr(model, 'body_name', None)
            if body_name is not None:
                try:
                    model_info['body_names'] = list(body_name.numpy())
                except Exception:
                    pass
        solver = getattr(NewtonManager, '_solver', None)
        if solver is not None:
            model_info['solver_dt'] = getattr(NewtonManager, '_solver_dt', None)
            model_info['num_substeps'] = getattr(NewtonManager, '_num_substeps', None)
        torch.save(model_info, dump_path / "model_info.pt")
    except Exception:
        pass

    print(f"[DUMP] Saved {len(ring_buffer)} frames to {dump_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with penetration monitoring."""
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)

        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        # Create env with video recording if requested
        render_mode = "rgb_array" if args_cli.video else None
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "debug"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # Ring buffer: deque of (step, state_snapshot)
        ring_buffer = deque(maxlen=args_cli.ring_size)
        dump_dir = Path(args_cli.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_count = 0
        threshold_z = args_cli.threshold_z
        num_envs = env_cfg.scene.num_envs

        # Track per-env cooldown to avoid dumping same event multiple times
        cooldown = torch.zeros(num_envs, dtype=torch.int32)
        COOLDOWN_STEPS = 30  # Don't re-trigger for 30 steps after a dump

        print(f"\n[PENETRATION WATCHDOG] Active")
        print(f"  Threshold Z: {threshold_z} (table surface ~0.255)")
        print(f"  Ring buffer: {args_cli.ring_size} frames")
        print(f"  Max dumps: {args_cli.max_dumps}")
        print(f"  Dump dir: {dump_dir}\n")

        obs = env.get_observations()
        dt = env.unwrapped.step_dt

        for step in range(args_cli.max_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)

            # Capture state into ring buffer
            state_snapshot = capture_newton_state()
            if state_snapshot:
                ring_buffer.append((step, state_snapshot))

            # Check object positions
            try:
                obj_data = env.unwrapped.scene["object"].data
                pos_warp = obj_data.root_pos_w  # warp array of vec3f
                
                # Convert warp vec3f array to torch tensor (num_envs, 3)
                import warp as wp
                pos_torch = wp.to_torch(pos_warp)  # (num_envs, 3) float tensor
                obj_z = pos_torch[:, 2]  # Z positions

                # Decrement cooldowns
                cooldown = (cooldown - 1).clamp(min=0)

                # Check for penetration: Z below threshold AND not in cooldown
                penetrated = (obj_z.cpu() < threshold_z) & (cooldown == 0)

                if penetrated.any():
                    pen_indices = penetrated.nonzero(as_tuple=False).view(-1)
                    for env_idx in pen_indices:
                        env_idx = env_idx.item()
                        obj_xyz = pos_torch[env_idx].cpu().tolist()

                        print(f"\n[PENETRATION] Step {step}, env {env_idx}: "
                              f"object Z={obj_xyz[2]:.4f} < {threshold_z}")

                        if dump_count < args_cli.max_dumps:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            dump_path = dump_dir / f"penetration_step{step}_env{env_idx}_{timestamp}"
                            trigger_info = {
                                "step": step,
                                "env_idx": env_idx,
                                "object_pos": obj_xyz,
                                "threshold_z": threshold_z,
                                "table_surface_z": 0.255,
                                "timestamp": timestamp,
                                "checkpoint": str(resume_path),
                                "ring_buffer_frames": len(ring_buffer),
                            }
                            dump_ring_buffer(ring_buffer, dump_path, trigger_info)
                            dump_count += 1

                            if dump_count >= args_cli.max_dumps:
                                print(f"\n[PENETRATION WATCHDOG] Max dumps ({args_cli.max_dumps}) reached.")

                        cooldown[env_idx] = COOLDOWN_STEPS

            except Exception as e:
                if step < 5:
                    print(f"[WARN] Step {step}: Could not check object position: {e}")
                    import traceback; traceback.print_exc()

            # Progress update
            if (step + 1) % 500 == 0:
                print(f"  Step {step+1}/{args_cli.max_steps}, dumps: {dump_count}/{args_cli.max_dumps}")

            if args_cli.video and step >= args_cli.video_length - 1:
                break

        print(f"\n[DONE] {step+1} steps, {dump_count} penetration dumps saved to {dump_dir}")
        env.close()


if __name__ == "__main__":
    main()

"""Capture robot camera sensor observations as a tiled video grid.

Loads a ResNet vision checkpoint, runs the policy, and captures:
1. The wrist camera images from each environment (tiled in a grid)
2. Optionally a perspective viewport showing the full scene

Usage:
  cd /home/horde/claw/git/IsaacLab
  __GLX_VENDOR_LIBRARY_NAME=nvidia OMNI_KIT_ACCEPT_EULA=Y \
  env_isaaclab/bin/python scripts/capture_sensor_video.py \
    --checkpoint logs/.../model_3750.pt \
    --num_envs 4 --num_steps 300 --output /tmp/sensor_video.mp4 \
    presets=physx,rgb128
"""
import os, sys, argparse
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--output", default="/tmp/sensor_video.mp4")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--task", default="Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-ResNet-v0")
args, hydra_args = parser.parse_known_args()

os.chdir('/home/horde/claw/git/IsaacLab')
sys.path.insert(0, 'scripts/reinforcement_learning/rsl_rl')

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = launcher.app

import torch
import numpy as np
import gymnasium as gym
import imageio
import importlib.metadata
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa
import cli_args  # noqa

installed_version = importlib.metadata.version('rsl-rl-lib')


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"
    agent_cfg.device = "cuda:0"

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # Create environment
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load checkpoint
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Find camera sensor
    unwrapped = env.unwrapped
    camera = None
    for name, sensor in unwrapped.scene.sensors.items():
        print(f"Found sensor: {name} -> {type(sensor).__name__}")
        if hasattr(sensor, 'data') and hasattr(sensor.data, 'output'):
            camera = sensor
            cam_name = name
            print(f"Using camera: {cam_name}")
            break

    if camera is None:
        print("No camera sensor found! Available scene items:")
        for name in dir(unwrapped.scene):
            if not name.startswith('_'):
                print(f"  {name}")
        sim_app.close()
        return

    frames = []
    obs = env.get_observations()

    for step in range(args.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)

        # Get camera images
        try:
            cam_data = camera.data.output
            if "rgb" in cam_data:
                images = cam_data["rgb"]  # (num_envs, H, W, 4) typically
            elif "rgba" in cam_data:
                images = cam_data["rgba"]
            else:
                # Try first available key
                key = list(cam_data.keys())[0]
                images = cam_data[key]
                print(f"Using cam data key: {key}, shape: {images.shape}")

            if isinstance(images, torch.Tensor):
                images = images.cpu().numpy()

            # Take RGB channels only (drop alpha if present)
            if images.shape[-1] == 4:
                images = images[..., :3]

            # Create grid: 2x2 for 4 envs
            n = images.shape[0]
            h, w = images.shape[1], images.shape[2]
            cols = min(n, 2)
            rows = (n + cols - 1) // cols

            grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
            for i in range(n):
                r, c = i // cols, i % cols
                grid[r*h:(r+1)*h, c*w:(c+1)*w] = images[i]

            # Upscale if small
            if grid.shape[0] < 512:
                import cv2
                scale = 512 // grid.shape[0]
                grid = cv2.resize(grid, (grid.shape[1] * scale, grid.shape[0] * scale),
                                  interpolation=cv2.INTER_NEAREST)

            frames.append(grid)

            if step % 50 == 0:
                print(f"Step {step}/{args.num_steps}, cam shape: {images.shape}, "
                      f"mean: {images.mean():.1f}, grid: {grid.shape}")

        except Exception as e:
            print(f"Step {step}: Failed to get camera image: {e}")
            if step == 0:
                import traceback
                traceback.print_exc()

    # Write video
    if frames:
        print(f"Writing {len(frames)} frames to {args.output}")
        writer = imageio.get_writer(args.output, fps=args.fps)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"Done! Video saved: {args.output} ({os.path.getsize(args.output) / 1024:.0f}KB)")
    else:
        print("No frames captured!")

    env.close()
    sim_app.close()


# Override sys.argv for hydra
sys.argv = [sys.argv[0]] + hydra_args
main()

"""Record video of state-obs policy using Newton warp renderer or added camera.

For state-only checkpoints, we add a visualization camera to the scene
and capture its output while running the policy.

Usage:
  cd /home/horde/claw/git/IsaacLab
  . env_isaaclab/bin/activate
  DISPLAY=:99 python scripts/reinforcement_learning/rsl_rl/record_state_video.py \
    --checkpoint logs/rsl_rl/dexsuite_kuka_allegro/2026-03-31_21-53-28/model_9750.pt \
    --backend newton --remap-joints --output /tmp/physx_on_newton.mp4
"""
import os, sys, argparse, numpy as np

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--backend", default="newton", choices=["newton", "physx"])
parser.add_argument("--output", default="/tmp/policy_video.mp4")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--remap-joints", action="store_true")
parser.add_argument("--cam_width", type=int, default=256)
parser.add_argument("--cam_height", type=int, default=256)
args = parser.parse_args()

os.chdir('/home/horde/claw/git/IsaacLab')
sys.path.insert(0, 'scripts/reinforcement_learning/rsl_rl')

# AppLauncher with cameras enabled (forces Kit for rendering pipeline)
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = launcher.app

import torch
import gymnasium as gym
import imageio
import isaaclab_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils import launch_simulation
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata

installed_version = importlib.metadata.version('rsl-rl-lib')

if args.remap_joints:
    from joint_remapper import JointRemapper


@hydra_task_config('Isaac-Dexsuite-Kuka-Allegro-Lift-v0', 'rsl_rl_cfg_entry_point')
def main(env_cfg, agent_cfg):
    cli_ns = argparse.Namespace(
        headless=True, enable_cameras=True, video=False, num_envs=args.num_envs,
        device='cuda:0'
    )
    with launch_simulation(env_cfg, cli_ns):
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = 42

        # Add a visualization camera to the scene
        from isaaclab.sensors import TiledCameraCfg
        import isaaclab.sim as sim_utils
        env_cfg.scene.viz_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/VizCamera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.57, -0.8, 0.5),
                rot=(0.6124, 0.3536, 0.3536, 0.6124),
                convention="opengl",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 2.5)),
            width=args.cam_width,
            height=args.cam_height,
        )

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env = gym.make('Isaac-Dexsuite-Kuka-Allegro-Lift-v0', cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Load checkpoint
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # Set up joint remapping
        remapper = None
        remap_info = []
        if args.remap_joints:
            ckpt_dir = os.path.dirname(args.checkpoint)
            train_jn = JointRemapper.load_joint_names(ckpt_dir)
            eval_jn = env.unwrapped.scene['robot'].joint_names
            if train_jn:
                remapper = JointRemapper(train_jn, eval_jn)
                if remapper.needs_remap:
                    remapper.print_mapping()
                    nj = remapper.num_joints
                    obs_mgr = env.unwrapped.observation_manager
                    for gn in obs_mgr.active_terms:
                        tds = obs_mgr.group_obs_term_dim[gn]
                        tns = obs_mgr.active_terms[gn]
                        off = 0
                        for tn, td in zip(tns, tds):
                            sz = 1
                            for d in td: sz *= d
                            if tn in ('joint_pos', 'joint_vel', 'actions') and sz % nj == 0:
                                remap_info.append((gn, off, sz))
                                print(f'  [REMAP] {gn}/{tn} offset={off} size={sz}')
                            off += sz
                else:
                    remapper = None

        # Get the viz camera sensor
        viz_cam = env.unwrapped.scene.get('viz_camera')
        if viz_cam is None:
            print("ERROR: viz_camera not found in scene!")
            print(f"Available sensors: {list(env.unwrapped.scene._sensors.keys())}")
            # Try any camera
            for name, sensor in env.unwrapped.scene._sensors.items():
                if hasattr(sensor, 'data') and hasattr(sensor.data, 'output'):
                    if 'rgb' in sensor.data.output:
                        viz_cam = sensor
                        print(f"Using fallback camera: {name}")
                        break

        # Run and capture
        obs = env.get_observations()
        frames = []

        # Initial remap
        if remapper is not None:
            for gn, off, sz in remap_info:
                chunk = obs[gn][:, off:off+sz]
                nj = remapper.num_joints
                flat = chunk.reshape(-1, nj)
                remapped = remapper.remap_joint_obs(flat)
                obs[gn][:, off:off+sz] = remapped.reshape(chunk.shape)

        print(f"Recording {args.num_steps} steps...")
        for step in range(args.num_steps):
            with torch.inference_mode():
                actions = policy(obs)
                if remapper is not None:
                    actions = remapper.remap_actions(actions)
                obs, rew, dones, extras = env.step(actions)

                # Remap obs for next step
                if remapper is not None:
                    for gn, off, sz in remap_info:
                        chunk = obs[gn][:, off:off+sz]
                        nj = remapper.num_joints
                        flat = chunk.reshape(-1, nj)
                        remapped = remapper.remap_joint_obs(flat)
                        obs[gn][:, off:off+sz] = remapped.reshape(chunk.shape)

            # Capture frame
            if viz_cam is not None:
                rgb = viz_cam.data.output.get('rgb')
                if rgb is not None and len(rgb) > 0:
                    frame = rgb[0].cpu().numpy()[:, :, :3].astype(np.uint8)
                    frames.append(frame)

            if step % 50 == 0:
                print(f"  Step {step}/{args.num_steps}, frames={len(frames)}, rew={rew[0].item():.3f}")

        print(f"\nCaptured {len(frames)} frames")
        if frames:
            imageio.mimsave(args.output, frames, fps=args.fps)
            print(f"Video saved to {args.output}")
        else:
            print("No frames captured!")

        env.close()
        sim_app.close()


# Override sys.argv for hydra
sys.argv = ['record_state_video.py', f'presets={args.backend}']
main()

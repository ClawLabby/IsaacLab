"""Cross-sim eval with action clamping to simulate PhysX-like response.

The diagnostic shows the policy commands 2.3× larger actions on Newton than PhysX.
This suggests the feedback loop is unstable. Let's test clamping actions to PhysX ranges.
"""
import argparse, os, sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=40)
parser.add_argument("--task", default="Isaac-Dexsuite-Kuka-Allegro-Lift-v0")
parser.add_argument("--action_scale", type=float, default=1.0,
                    help="Scale actions by this factor before applying")
parser.add_argument("--action_clip", type=float, default=None,
                    help="Clip action magnitudes to this value")
args, hydra_args = parser.parse_known_args()

os.chdir('/home/horde/claw/git/IsaacLab')
sys.path.insert(0, 'scripts/reinforcement_learning/rsl_rl')

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=False)
sim_app = launcher.app

import torch, numpy as np, gymnasium as gym, importlib.metadata
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks, cli_args  # noqa
installed_version = importlib.metadata.version('rsl-rl-lib')


def build_remap_table(env):
    robot = env.unwrapped.scene["robot"]
    joint_names = robot.joint_names
    physx_order = [
        "iiwa7_joint_1", "iiwa7_joint_2", "iiwa7_joint_3",
        "iiwa7_joint_4", "iiwa7_joint_5", "iiwa7_joint_6", "iiwa7_joint_7",
        "index_joint_0", "middle_joint_0", "ring_joint_0", "thumb_joint_0",
        "index_joint_1", "middle_joint_1", "ring_joint_1", "thumb_joint_1",
        "index_joint_2", "middle_joint_2", "ring_joint_2", "thumb_joint_2",
        "index_joint_3", "middle_joint_3", "ring_joint_3", "thumb_joint_3",
    ]
    remap = [joint_names.index(p) if p in joint_names else 0 for p in physx_order]
    if remap == list(range(len(remap))):
        return None
    return torch.tensor(remap, dtype=torch.long, device=robot.device)


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"
    agent_cfg.device = "cuda:0"

    print(f"\n*** Action scale: {args.action_scale}, clip: {args.action_clip} ***\n")

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    remap_table = build_remap_table(env)
    print(f"Remap: {remap_table is not None}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    episode_count = 0
    step_count = 0
    total_success = 0.0
    total_pos_err = 0.0
    report_steps = 0

    while episode_count < args.num_episodes and step_count < args.num_episodes * 500:
        with torch.inference_mode():
            actions = policy(obs)

            # Apply remapping
            if remap_table is not None and actions.shape[-1] == len(remap_table):
                actions = actions[:, remap_table]

            # Apply action scaling/clamping
            if args.action_scale != 1.0:
                actions = actions * args.action_scale
            if args.action_clip is not None:
                actions = torch.clamp(actions, -args.action_clip, args.action_clip)

            obs, rewards, dones, infos = env.step(actions)
            policy.reset(dones)
            step_count += 1

            try:
                extras = env.unwrapped.extras.get("log", {})
                s = extras.get("Episode_Reward/success", 0.0)
                p = extras.get("Metrics/object_pose/position_error", 1.0)
                if isinstance(s, torch.Tensor): s = s.item()
                if isinstance(p, torch.Tensor): p = p.item()
                total_success += s
                total_pos_err += p
                report_steps += 1
            except:
                pass

            if dones.any():
                episode_count += int(dones.sum().item())

            if step_count % 200 == 0:
                avg_s = total_success / max(report_steps, 1)
                avg_p = total_pos_err / max(report_steps, 1)
                print(f"  Step {step_count}: ep={episode_count}/{args.num_episodes}, "
                      f"success={avg_s:.4f}, pos_err={avg_p:.3f}m")

    avg_success = total_success / max(report_steps, 1)
    avg_pos_err = total_pos_err / max(report_steps, 1)

    print(f"\n{'='*60}")
    print(f"RESULTS: scale={args.action_scale}, clip={args.action_clip}")
    print(f"  Episodes: {episode_count}")
    print(f"  Steps: {step_count}")
    print(f"  Avg success: {avg_success:.4f}")
    print(f"  Avg pos error: {avg_p:.3f}m")
    print(f"{'='*60}")

    env.close()
    sim_app.close()


sys.argv = [sys.argv[0]] + hydra_args
main()

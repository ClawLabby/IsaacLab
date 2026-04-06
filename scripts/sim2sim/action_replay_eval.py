"""Open-loop action replay: record actions from PhysX, replay on Newton.

This tests if Newton can achieve success with known-good actions.
If replay fails, the physics dynamics are fundamentally incompatible.
If replay succeeds, the closed-loop feedback is the problem.
"""
import argparse, os, sys, json

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--mode", default="record", choices=["record", "replay"])
parser.add_argument("--action_file", default="/tmp/recorded_actions.json")
parser.add_argument("--task", default="Isaac-Dexsuite-Kuka-Allegro-Lift-v0")
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

    # Fix seed for reproducibility
    env_cfg.seed = 42

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    remap_table = build_remap_table(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()

    if args.mode == "record":
        print(f"\n*** RECORDING actions on PhysX ***\n")
        recorded_actions = []
        recorded_rewards = []
        recorded_pos_errors = []

        for step in range(args.num_steps):
            with torch.inference_mode():
                actions = policy(obs)
                # Note: recording in PhysX native order (no remap yet)
                recorded_actions.append(actions[0].cpu().numpy().tolist())

                obs, rewards, dones, infos = env.step(actions)
                policy.reset(dones)

                rew = rewards[0].item()
                recorded_rewards.append(rew)

                try:
                    extras = env.unwrapped.extras.get("log", {})
                    pe = extras.get("Metrics/object_pose/position_error", -1.0)
                    if isinstance(pe, torch.Tensor): pe = pe.item()
                except:
                    pe = -1.0
                recorded_pos_errors.append(pe)

                if step % 50 == 0:
                    print(f"  Step {step}: reward={rew:.4f}, pos_err={pe:.3f}m")

                if dones[0]:
                    print(f"  Episode ended at step {step}")
                    break

        data = {
            "actions": recorded_actions,
            "rewards": recorded_rewards,
            "pos_errors": recorded_pos_errors,
            "backend": "physx",
        }
        with open(args.action_file, 'w') as f:
            json.dump(data, f)
        print(f"\nRecorded {len(recorded_actions)} steps to {args.action_file}")
        print(f"Mean reward: {np.mean(recorded_rewards):.4f}")
        if any(pe > 0 for pe in recorded_pos_errors):
            valid_pe = [pe for pe in recorded_pos_errors if pe > 0]
            print(f"Mean pos error: {np.mean(valid_pe):.3f}m")

    else:  # replay
        print(f"\n*** REPLAYING recorded actions on Newton ***\n")

        with open(args.action_file, 'r') as f:
            data = json.load(f)
        recorded_actions = data["actions"]
        print(f"Loaded {len(recorded_actions)} recorded actions from {data['backend']}")

        replay_rewards = []
        replay_pos_errors = []

        for step, action_list in enumerate(recorded_actions):
            actions = torch.tensor([action_list], dtype=torch.float32, device="cuda:0")

            # Apply PhysX->Newton remap
            if remap_table is not None and actions.shape[-1] == len(remap_table):
                actions = actions[:, remap_table]

            obs, rewards, dones, infos = env.step(actions)

            rew = rewards[0].item()
            replay_rewards.append(rew)

            try:
                extras = env.unwrapped.extras.get("log", {})
                pe = extras.get("Metrics/object_pose/position_error", -1.0)
                if isinstance(pe, torch.Tensor): pe = pe.item()
            except:
                pe = -1.0
            replay_pos_errors.append(pe)

            if step % 50 == 0:
                print(f"  Step {step}: reward={rew:.4f}, pos_err={pe:.3f}m")

            if dones[0]:
                print(f"  Episode ended at step {step}")
                break

        print(f"\n{'='*60}")
        print(f"REPLAY RESULTS ({len(replay_rewards)} steps)")
        print(f"  Mean reward: {np.mean(replay_rewards):.4f}")
        print(f"  vs PhysX:    {np.mean(data['rewards'][:len(replay_rewards)]):.4f}")
        if any(pe > 0 for pe in replay_pos_errors):
            valid_pe = [pe for pe in replay_pos_errors if pe > 0]
            print(f"  Mean pos error: {np.mean(valid_pe):.3f}m")
        print(f"{'='*60}")

    env.close()
    sim_app.close()


sys.argv = [sys.argv[0]] + hydra_args
main()

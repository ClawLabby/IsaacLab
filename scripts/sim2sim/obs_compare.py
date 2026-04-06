"""Record observations from both PhysX and Newton given identical actions.

Compares the actual observation vectors the policy would see from each backend
when running the exact same action sequence. This exposes any observation
remapping bugs or systematic observation differences.
"""
import argparse, os, sys, json

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--mode", required=True, choices=["record_physx", "replay_newton"])
parser.add_argument("--action_file", default="/tmp/obs_compare_actions.json")
parser.add_argument("--obs_file", default=None, help="Output obs file (auto-named if None)")
parser.add_argument("--task", default="Isaac-Dexsuite-Kuka-Allegro-Lift-v0")
args, hydra_args = parser.parse_known_args()

if args.obs_file is None:
    args.obs_file = f"/tmp/obs_{args.mode}.json"

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
        print("Joint ordering is IDENTICAL — no remapping needed")
        return None
    print(f"PhysX order: {physx_order[:10]}...")
    print(f"Newton order: {joint_names[:10]}...")
    print(f"Remap: {remap}")
    return torch.tensor(remap, dtype=torch.long, device=robot.device)


def get_joint_data(robot):
    """Get joint_pos and joint_vel as flat numpy arrays."""
    jp = robot.data.joint_pos[0]
    jv = robot.data.joint_vel[0]
    if hasattr(jp, 'cpu'):
        return jp.cpu().numpy(), jv.cpu().numpy()
    else:
        import warp as wp
        return (wp.to_torch(robot.data.joint_pos)[0].cpu().numpy(),
                wp.to_torch(robot.data.joint_vel)[0].cpu().numpy())


def obs_to_dict(obs):
    """Convert obs (TensorDict or dict of tensors) to JSON-serializable dict."""
    result = {}
    items = obs.items() if hasattr(obs, 'items') else [(str(i), v) for i, v in enumerate(obs)]
    for k, v in items:
        if isinstance(v, torch.Tensor):
            result[k] = v[0].cpu().numpy().tolist()
        elif isinstance(v, np.ndarray):
            result[k] = v[0].tolist()
    return result


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"
    agent_cfg.device = "cuda:0"
    env_cfg.seed = 42

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    remap_table = build_remap_table(env)

    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.joint_names)
    print(f"Joint names ({len(joint_names)}): {joint_names}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()

    all_data = {
        "joint_names": joint_names,
        "mode": args.mode,
        "steps": [],
    }

    if args.mode == "record_physx":
        print(f"\n*** RECORDING on PhysX ***\n")
        recorded_actions = []

        for step in range(args.num_steps):
            with torch.inference_mode():
                # Get observation breakdown
                obs_dict = obs_to_dict(obs)

                # Get joint state
                jp, jv = get_joint_data(robot)

                # Policy action (PhysX native order)
                actions = policy(obs)
                act = actions[0].cpu().numpy().tolist()
                recorded_actions.append(act)

                obs, rewards, dones, infos = env.step(actions)
                policy.reset(dones)

                rew = rewards[0].item()

                all_data["steps"].append({
                    "joint_pos": jp.tolist(),
                    "joint_vel": jv.tolist(),
                    "action": act,
                    "reward": rew,
                    "obs_policy": obs_dict.get("policy", []),
                    "obs_proprio": obs_dict.get("proprio", []),
                })

                if step % 50 == 0:
                    print(f"  Step {step}: rew={rew:.4f}, "
                          f"jp[0]={jp[0]:.3f}, jv[0]={jv[0]:.3f}, "
                          f"act_mean={np.abs(act).mean():.3f}")

                if dones[0]:
                    print(f"  Episode ended at step {step}")
                    break

        # Save actions for replay
        with open(args.action_file, 'w') as f:
            json.dump({"actions": recorded_actions}, f)
        print(f"Saved {len(recorded_actions)} actions to {args.action_file}")

    else:  # replay_newton
        print(f"\n*** REPLAYING on Newton ***\n")
        with open(args.action_file, 'r') as f:
            recorded = json.load(f)
        recorded_actions = recorded["actions"]
        print(f"Loaded {len(recorded_actions)} actions")

        for step, act_list in enumerate(recorded_actions):
            # Get observation breakdown BEFORE stepping
            obs_dict = obs_to_dict(obs)
            jp, jv = get_joint_data(robot)

            # Apply recorded action with remap
            actions = torch.tensor([act_list], dtype=torch.float32, device="cuda:0")
            if remap_table is not None and actions.shape[-1] == len(remap_table):
                actions = actions[:, remap_table]

            obs, rewards, dones, infos = env.step(actions)

            rew = rewards[0].item()

            all_data["steps"].append({
                "joint_pos": jp.tolist(),
                "joint_vel": jv.tolist(),
                "action_original": act_list,  # PhysX order
                "action_remapped": actions[0].cpu().numpy().tolist(),  # Newton order
                "reward": rew,
                "obs_policy": obs_dict.get("policy", []),
                "obs_proprio": obs_dict.get("proprio", []),
            })

            if step % 50 == 0:
                print(f"  Step {step}: rew={rew:.4f}, "
                      f"jp[0]={jp[0]:.3f}, jv[0]={jv[0]:.3f}")

            if dones[0]:
                print(f"  Episode ended at step {step}")
                break

    # Save all data
    with open(args.obs_file, 'w') as f:
        json.dump(all_data, f)
    print(f"\nSaved observation data to {args.obs_file}")

    env.close()
    sim_app.close()


sys.argv = [sys.argv[0]] + hydra_args
main()

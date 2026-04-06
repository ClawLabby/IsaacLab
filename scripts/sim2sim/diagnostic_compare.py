"""Diagnostic: compare actual robot behavior on PhysX vs Newton step-by-step.

Instead of just measuring final success, log the actual joint positions,
velocities, and actions at every step to see WHERE the divergence begins.

Also test: what if we run the same ACTIONS (recorded from PhysX) on Newton?
This separates "policy gives wrong actions on Newton obs" from
"Newton physics produces different outcomes from same actions".
"""
import argparse, os, sys, json

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--task", default="Isaac-Dexsuite-Kuka-Allegro-Lift-v0")
parser.add_argument("--output", default="/tmp/sim2sim_diag.json")
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


def record_episode(env, policy, remap_table, num_steps):
    """Record one episode of joint pos, vel, actions, rewards."""
    obs = env.get_observations()
    data = {"joint_pos": [], "joint_vel": [], "actions": [], "rewards": [],
            "obs_stats": [], "pos_error": []}

    for step in range(num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            if remap_table is not None and actions.shape[-1] == len(remap_table):
                actions = actions[:, remap_table]

            # Record pre-step state
            robot = env.unwrapped.scene["robot"]
            jp_raw = robot.data.joint_pos[0]
            jv_raw = robot.data.joint_vel[0]
            # Handle both torch tensors and warp arrays
            if hasattr(jp_raw, 'cpu'):
                jp = jp_raw.cpu().numpy().tolist()
                jv = jv_raw.cpu().numpy().tolist()
            else:
                import warp as wp
                jp = wp.to_torch(robot.data.joint_pos)[0].cpu().numpy().tolist()
                jv = wp.to_torch(robot.data.joint_vel)[0].cpu().numpy().tolist()
            act_raw = actions[0]
            act = act_raw.cpu().numpy().tolist() if hasattr(act_raw, 'cpu') else act_raw.numpy().tolist()

            obs, rewards, dones, infos = env.step(actions)
            policy.reset(dones)

            rew = rewards[0].item()

            # Get position error if available
            try:
                extras = env.unwrapped.extras.get("log", {})
                pe = extras.get("Metrics/object_pose/position_error", -1.0)
                if isinstance(pe, torch.Tensor): pe = pe.item()
            except:
                pe = -1.0

            data["joint_pos"].append(jp)
            data["joint_vel"].append(jv)
            data["actions"].append(act)
            data["rewards"].append(rew)
            data["pos_error"].append(pe)

            # Summary stats for obs
            try:
                if isinstance(obs, dict):
                    obs_flat = torch.cat([v[0].flatten() for v in obs.values()
                                          if isinstance(v, torch.Tensor)]).cpu().numpy()
                elif hasattr(obs, 'values'):  # TensorDict
                    obs_flat = torch.cat([v[0].flatten() for v in obs.values()
                                          if isinstance(v, torch.Tensor)]).cpu().numpy()
                else:
                    obs_flat = obs[0].cpu().numpy()
                data["obs_stats"].append({
                    "mean": float(obs_flat.mean()),
                    "std": float(obs_flat.std()),
                    "min": float(obs_flat.min()),
                    "max": float(obs_flat.max()),
                    "abs_mean": float(np.abs(obs_flat).mean()),
                })
            except Exception as e:
                data["obs_stats"].append({"error": str(e)})

            if dones[0]:
                print(f"  Episode ended at step {step}")
                break

    return data


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"
    agent_cfg.device = "cuda:0"

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    remap_table = build_remap_table(env)

    # Get joint names for the output
    robot = env.unwrapped.scene["robot"]
    joint_names = robot.joint_names

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    print(f"\nRecording {args.num_steps} steps on Newton...")
    data = record_episode(env, policy, remap_table, args.num_steps)
    data["joint_names"] = joint_names
    data["remap"] = remap_table.cpu().tolist() if remap_table is not None else None
    data["backend"] = "newton"

    # Compute summary
    jp = np.array(data["joint_pos"])
    jv = np.array(data["joint_vel"])
    acts = np.array(data["actions"])

    print(f"\nJoint position range per joint (first 7 = arm):")
    for i in range(min(7, jp.shape[1])):
        print(f"  {joint_names[i]:20s}: pos [{jp[:,i].min():+.3f}, {jp[:,i].max():+.3f}], "
              f"vel [{jv[:,i].min():+.3f}, {jv[:,i].max():+.3f}]")

    print(f"\nAction stats:")
    print(f"  Mean abs: {np.abs(acts).mean():.4f}")
    print(f"  Max abs:  {np.abs(acts).max():.4f}")
    print(f"  Std:      {acts.std():.4f}")

    print(f"\nReward stats:")
    rews = np.array(data["rewards"])
    print(f"  Mean: {rews.mean():.4f}, Min: {rews.min():.4f}, Max: {rews.max():.4f}")

    print(f"\nPosition error stats:")
    pe = [x for x in data["pos_error"] if x > 0]
    if pe:
        print(f"  Mean: {np.mean(pe):.3f}m, Min: {np.min(pe):.3f}m")

    # Check for NaN/inf in observations
    obs_abs_means = [x["abs_mean"] for x in data["obs_stats"]]
    obs_maxes = [x["max"] for x in data["obs_stats"]]
    print(f"\nObs abs_mean range: [{min(obs_abs_means):.3f}, {max(obs_abs_means):.3f}]")
    print(f"Obs max range: [{min(obs_maxes):.3f}, {max(obs_maxes):.3f}]")
    if any(np.isnan(x) or np.isinf(x) for x in obs_maxes):
        print("WARNING: NaN/Inf detected in observations!")

    # Save
    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nDiagnostic data saved to {args.output}")

    env.close()
    sim_app.close()


sys.argv = [sys.argv[0]] + hydra_args
main()

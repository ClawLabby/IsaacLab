"""Test: Zero j7 velocity on Newton to match PhysX velocity reporting bug.

PhysX underreports iiwa7_joint_7 velocity by ~100× (documented in MEMORY.md).
The policy was trained on this buggy data. On Newton, j7 velocity is reported
correctly, which might cause the policy to see "anomalous" velocities.

This test zeros out j7 velocity in observations to see if it helps.
"""
import argparse, os, sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--task", default="Isaac-Dexsuite-Kuka-Allegro-Lift-v0")
parser.add_argument("--zero_j7_vel", action="store_true", default=True)
parser.add_argument("--no_zero_j7", dest="zero_j7_vel", action="store_false")
args, hydra_args = parser.parse_known_args()

os.chdir('/home/horde/claw/git/IsaacLab')
sys.path.insert(0, 'scripts/reinforcement_learning/rsl_rl')

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=False)
sim_app = launcher.app

import torch, gymnasium as gym, importlib.metadata
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks, cli_args
installed_version = importlib.metadata.version('rsl-rl-lib')

# Full Newton→PhysX remap table
NEWTON_TO_PHYSX_JOINTS = [0,1,2,3,4,5,6, 7,11,15,19, 8,12,16,20, 9,13,17,21, 10,14,18,22]
NEWTON_TO_PHYSX_BODIES = [0, 3, 2, 1, 4]

CONTACT_SIZE = 12
JOINT_SIZE = 23
TIPS_SIZE = 65
FRAME_SIZE = CONTACT_SIZE + JOINT_SIZE + JOINT_SIZE + TIPS_SIZE
N_HISTORY = 5
POLICY_FRAME_SIZE = 34

J7_INDEX = 6  # iiwa7_joint_7


def build_action_remap(env):
    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.joint_names)
    physx_order = [
        "iiwa7_joint_1", "iiwa7_joint_2", "iiwa7_joint_3",
        "iiwa7_joint_4", "iiwa7_joint_5", "iiwa7_joint_6", "iiwa7_joint_7",
        "index_joint_0", "middle_joint_0", "ring_joint_0", "thumb_joint_0",
        "index_joint_1", "middle_joint_1", "ring_joint_1", "thumb_joint_1",
        "index_joint_2", "middle_joint_2", "ring_joint_2", "thumb_joint_2",
        "index_joint_3", "middle_joint_3", "ring_joint_3", "thumb_joint_3",
    ]
    if joint_names == physx_order:
        return None
    p2n = [joint_names.index(p) for p in physx_order]
    return torch.tensor(p2n, dtype=torch.long, device=robot.device)


def remap_proprio(proprio, n2p_joints, n2p_bodies, zero_j7=False):
    """Remap proprio obs and optionally zero j7 velocity."""
    if n2p_joints is None:
        return proprio
    
    batch = proprio.shape[0]
    out = proprio.clone()
    
    for h in range(N_HISTORY):
        off = h * FRAME_SIZE
        
        # Joint pos (12:35)
        jp_start = off + CONTACT_SIZE
        jp_end = jp_start + JOINT_SIZE
        out[:, jp_start:jp_end] = proprio[:, jp_start:jp_end][:, n2p_joints]
        
        # Joint vel (35:58)
        jv_start = off + CONTACT_SIZE + JOINT_SIZE
        jv_end = jv_start + JOINT_SIZE
        remapped_jv = proprio[:, jv_start:jv_end][:, n2p_joints]
        
        # Zero j7 velocity to match PhysX bug
        if zero_j7:
            remapped_jv[:, J7_INDEX] = 0.0
            
        out[:, jv_start:jv_end] = remapped_jv
        
        # Hand tips (58:123)
        tips_start = off + CONTACT_SIZE + JOINT_SIZE + JOINT_SIZE
        tips_end = tips_start + TIPS_SIZE
        tips = proprio[:, tips_start:tips_end].reshape(batch, 5, 13)
        tips_remapped = tips[:, n2p_bodies, :]
        out[:, tips_start:tips_end] = tips_remapped.reshape(batch, TIPS_SIZE)
    
    return out


def remap_policy_obs(policy_obs, n2p_joints):
    if n2p_joints is None:
        return policy_obs
    
    out = policy_obs.clone()
    for h in range(N_HISTORY):
        off = h * POLICY_FRAME_SIZE
        act_start = off + 11
        act_end = act_start + JOINT_SIZE
        out[:, act_start:act_end] = policy_obs[:, act_start:act_end][:, n2p_joints]
    return out


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"
    agent_cfg.device = "cuda:0"
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    action_remap = build_action_remap(env)
    device = env.unwrapped.device
    
    if action_remap is not None:
        n2p_joints = torch.tensor(NEWTON_TO_PHYSX_JOINTS, dtype=torch.long, device=device)
        n2p_bodies = NEWTON_TO_PHYSX_BODIES
        print(f"  Remapping enabled, zero_j7_vel={args.zero_j7_vel}")
    else:
        n2p_joints = None
        n2p_bodies = None
        print("  Native PhysX ordering")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)

    obs = env.get_observations()
    episode_count = 0
    step_count = 0
    total_success = 0.0
    total_pos_err = 0.0
    report_steps = 0

    while episode_count < args.num_episodes and step_count < args.num_episodes * 500:
        with torch.inference_mode():
            obs_remapped = {}
            for k, v in obs.items():
                if k == 'proprio' and isinstance(v, torch.Tensor):
                    obs_remapped[k] = remap_proprio(v, n2p_joints, n2p_bodies, args.zero_j7_vel)
                elif k == 'policy' and isinstance(v, torch.Tensor):
                    obs_remapped[k] = remap_policy_obs(v, n2p_joints)
                else:
                    obs_remapped[k] = v
            
            if hasattr(obs, '__class__') and obs.__class__.__name__ == 'TensorDict':
                from tensordict import TensorDict
                obs_remapped = TensorDict(obs_remapped, batch_size=obs.batch_size)

            actions = policy(obs_remapped)

            if action_remap is not None:
                actions = actions[:, action_remap]

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
                print(f"  Step {step_count}: ep={episode_count}/{args.num_episodes}, "
                      f"success={total_success/max(report_steps,1):.4f}")

    print(f"\n{'='*60}")
    print(f"zero_j7_vel={args.zero_j7_vel}")
    print(f"  Avg success: {total_success/max(report_steps,1):.4f}")
    print(f"  Avg pos error: {total_pos_err/max(report_steps,1):.3f}m")
    print(f"{'='*60}")

    env.close()
    sim_app.close()


sys.argv = [sys.argv[0]] + hydra_args
main()

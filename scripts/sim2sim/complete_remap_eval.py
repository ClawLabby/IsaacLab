"""Cross-sim eval with COMPLETE observation + action remapping.

Remaps ALL observation components from Newton ordering to PhysX ordering:
- joint_pos (23) in proprio
- joint_vel (23) in proprio  
- hand_tips_state_b (5 bodies × 13) in proprio — body permutation
- last_action (23) in policy obs
- Actions (23) — PhysX→Newton

Based on empirical measurement: joint remap is [0,1,2,3,4,5,6, 7,11,15,19, 8,12,16,20, 9,13,17,21, 10,14,18,22]
and hand_tips body permutation is (0, 3, 2, 1, 4) (Newton→PhysX).
"""
import argparse, os, sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=40)
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

# Newton joint index → PhysX joint index
# Newton: iiwa(7), index(4), middle(4), ring(4), thumb(4)  
# PhysX:  iiwa(7), idx0 mid0 rng0 thb0, idx1 mid1 rng1 thb1, ...
NEWTON_TO_PHYSX_JOINTS = [0,1,2,3,4,5,6, 7,11,15,19, 8,12,16,20, 9,13,17,21, 10,14,18,22]
PHYSX_TO_NEWTON_JOINTS = [0,1,2,3,4,5,6, 7,11,15,19, 8,12,16,20, 9,13,17,21, 10,14,18,22]

# Hand tips body permutation: Newton body[i] → PhysX body[perm[i]]
# From empirical measurement: (0, 3, 2, 1, 4)
# Meaning: Newton[0]→PhysX[0], Newton[1]→PhysX[3], Newton[2]→PhysX[2], Newton[3]→PhysX[1], Newton[4]→PhysX[4]
NEWTON_TO_PHYSX_BODIES = [0, 3, 2, 1, 4]

# Proprio obs layout per frame (×5 history):
# contact(12) + joint_pos(23) + joint_vel(23) + hand_tips(65) = 123
CONTACT_SIZE = 12  # 4 sensors × 3
JOINT_SIZE = 23
TIPS_SIZE = 65  # 5 bodies × 13
FRAME_SIZE = CONTACT_SIZE + JOINT_SIZE + JOINT_SIZE + TIPS_SIZE  # 123
N_HISTORY = 5

# Policy obs layout per frame (×5 history):
# object_quat(4) + target_pose(7) + last_action(23) = 34
POLICY_FRAME_SIZE = 34


def remap_proprio(proprio, n2p_joints, n2p_bodies, device):
    """Remap proprio obs from Newton ordering to PhysX ordering."""
    if n2p_joints is None:
        return proprio
    batch = proprio.shape[0]
    out = proprio.clone()
    
    for h in range(N_HISTORY):
        off = h * FRAME_SIZE
        
        # Contact sensors (0:12) — kept as-is (queried by name, same order)
        
        # Joint pos (12:35)
        jp_start = off + CONTACT_SIZE
        jp_end = jp_start + JOINT_SIZE
        out[:, jp_start:jp_end] = proprio[:, jp_start:jp_end][:, n2p_joints]
        
        # Joint vel (35:58)
        jv_start = off + CONTACT_SIZE + JOINT_SIZE
        jv_end = jv_start + JOINT_SIZE
        out[:, jv_start:jv_end] = proprio[:, jv_start:jv_end][:, n2p_joints]
        
        # Hand tips (58:123) — 5 bodies × 13
        tips_start = off + CONTACT_SIZE + JOINT_SIZE + JOINT_SIZE
        tips_end = tips_start + TIPS_SIZE
        tips = proprio[:, tips_start:tips_end].reshape(batch, 5, 13)
        tips_remapped = tips[:, n2p_bodies, :]
        out[:, tips_start:tips_end] = tips_remapped.reshape(batch, TIPS_SIZE)
    
    return out


def remap_policy_obs(policy_obs, n2p_joints):
    """Remap policy obs — the last_action component."""
    if n2p_joints is None:
        return policy_obs
    out = policy_obs.clone()
    
    for h in range(N_HISTORY):
        off = h * POLICY_FRAME_SIZE
        # last_action at offset + 4 + 7 = offset + 11
        act_start = off + 11
        act_end = act_start + JOINT_SIZE
        out[:, act_start:act_end] = policy_obs[:, act_start:act_end][:, n2p_joints]
    
    return out


def build_action_remap(env):
    """Build PhysX→Newton action remap."""
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
    
    # Only apply obs remap when action remap is needed (i.e., on Newton, not PhysX)
    if action_remap is not None:
        n2p_joints = torch.tensor(NEWTON_TO_PHYSX_JOINTS, dtype=torch.long, device=device)
        n2p_bodies = NEWTON_TO_PHYSX_BODIES
        print("  Obs remapping: ENABLED (Newton→PhysX)")
    else:
        n2p_joints = None
        n2p_bodies = None
        print("  Obs remapping: DISABLED (native PhysX ordering)")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)

    print(f"\n*** COMPLETE obs+action remap eval ***")
    print(f"  Action remap: {action_remap is not None}")
    print(f"  Joint remap: {NEWTON_TO_PHYSX_JOINTS}")
    print(f"  Body remap: {NEWTON_TO_PHYSX_BODIES}\n")

    obs = env.get_observations()
    episode_count = 0
    step_count = 0
    total_success = 0.0
    total_pos_err = 0.0
    report_steps = 0

    while episode_count < args.num_episodes and step_count < args.num_episodes * 500:
        with torch.inference_mode():
            # Full observation remapping
            if hasattr(obs, 'keys') or isinstance(obs, dict):
                obs_remapped = {}
                for k, v in obs.items():
                    if k == 'proprio' and isinstance(v, torch.Tensor):
                        obs_remapped[k] = remap_proprio(v, n2p_joints, n2p_bodies, device)
                    elif k == 'policy' and isinstance(v, torch.Tensor):
                        obs_remapped[k] = remap_policy_obs(v, n2p_joints)
                    else:
                        obs_remapped[k] = v
                # Reconstruct TensorDict if needed
                if hasattr(obs, '__class__') and obs.__class__.__name__ == 'TensorDict':
                    from tensordict import TensorDict
                    obs_remapped = TensorDict(obs_remapped, batch_size=obs.batch_size)
            else:
                obs_remapped = obs

            actions = policy(obs_remapped)

            if action_remap is not None and actions.shape[-1] == len(action_remap):
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

            if step_count % 100 == 0:
                avg_s = total_success / max(report_steps, 1)
                avg_p = total_pos_err / max(report_steps, 1)
                print(f"  Step {step_count}: ep={episode_count}/{args.num_episodes}, "
                      f"success={avg_s:.4f}, pos_err={avg_p:.3f}m")

    avg_success = total_success / max(report_steps, 1)
    avg_pos_err = total_pos_err / max(report_steps, 1)

    print(f"\n{'='*60}")
    print(f"RESULTS: Complete obs+action remap")
    print(f"  Episodes: {episode_count}, Steps: {step_count}")
    print(f"  Avg success: {avg_success:.4f}")
    print(f"  Avg pos error: {avg_pos_err:.3f}m")
    print(f"{'='*60}")

    env.close()
    sim_app.close()


sys.argv = [sys.argv[0]] + hydra_args
main()

# Add after remap_proprio function - patch to zero j7 velocity
def zero_j7_velocity(proprio, frame_size=123, n_history=5, joint_size=23, contact_size=12):
    """Zero out iiwa7_joint_7 velocity to match PhysX bug."""
    out = proprio.clone()
    j7_idx = 6  # iiwa7_joint_7 is at index 6
    for h in range(n_history):
        off = h * frame_size
        # joint_vel starts at offset + contact_size + joint_size
        jv_start = off + contact_size + joint_size + j7_idx
        out[:, jv_start] = 0.0
    return out

"""Cross-sim eval with BOTH action AND observation remapping.

The policy obs contain joint_pos and joint_vel in the env's native ordering.
When running a PhysX-trained policy on Newton, we need to:
1. Remap Newton observations → PhysX ordering before the policy sees them
2. Remap PhysX-ordered actions → Newton ordering before applying them

The observation remapping is the INVERSE of the action remapping.
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


def build_remaps(env):
    """Build bidirectional remap tables between PhysX and Newton joint ordering."""
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
        print("Joint ordering is IDENTICAL — no remapping needed")
        return None, None

    # physx_to_newton[i] = Newton index for PhysX joint i
    # Used for ACTION remapping: policy outputs in PhysX order, env expects Newton order
    physx_to_newton = []
    for pname in physx_order:
        physx_to_newton.append(joint_names.index(pname))

    # newton_to_physx[i] = PhysX index for Newton joint i  
    # Used for OBSERVATION remapping: env outputs in Newton order, policy expects PhysX order
    newton_to_physx = [0] * len(joint_names)
    for physx_idx, newton_idx in enumerate(physx_to_newton):
        newton_to_physx[newton_idx] = physx_idx

    device = robot.device
    print(f"PhysX order: {physx_order}")
    print(f"Newton order: {joint_names}")
    print(f"physx_to_newton (for actions): {physx_to_newton}")
    print(f"newton_to_physx (for obs):     {newton_to_physx}")

    return (torch.tensor(physx_to_newton, dtype=torch.long, device=device),
            torch.tensor(newton_to_physx, dtype=torch.long, device=device))


def remap_obs(obs, newton_to_physx, num_joints=23):
    """Remap observation tensor from Newton joint ordering to PhysX ordering.
    
    The observations contain joint_pos (23) and joint_vel (23) somewhere in the
    proprio group. We need to find and remap those segments.
    
    For a TensorDict with groups (policy, proprio, perception), we remap the
    proprio group which contains joint_pos and joint_vel.
    """
    if newton_to_physx is None:
        return obs
    
    # The obs is a TensorDict — we need to remap within the proprio group
    # proprio contains: contact(12), joint_pos(23), joint_vel(23), hand_tips(65)
    # with history_length=5, each is repeated 5 times
    # Total proprio = 5 * (12 + 23 + 23 + 65) = 5 * 123 = 615
    #
    # Actually the layout depends on obs config. Let's figure it out.
    # From the env: proprio has contact(12), joint_pos(23), joint_vel(23), hand_tips_state_b
    # history_length=5, flatten_history_dim=True
    # So each term appears 5 times concatenated: [t, t-1, t-2, t-3, t-4]
    
    if hasattr(obs, 'keys'):
        # TensorDict — remap the proprio group
        if 'proprio' in obs:
            proprio = obs['proprio']  # shape (num_envs, 615)
            # Layout: history stacks of [contact(12), joint_pos(23), joint_vel(23), hand_tips(N)]
            # With history_length=5 and flatten=True:
            # Each history frame = contact(12) + joint_pos(23) + joint_vel(23) + tips(N)
            # Total 615 = 5 * 123 → each frame is 123 elements
            # contact=12, joint_pos=23, joint_vel=23, hand_tips=65 (5 bodies * 13)
            frame_size = 123  # 12 + 23 + 23 + 65
            n_history = 5
            
            if proprio.shape[-1] == frame_size * n_history:
                remapped = proprio.clone()
                for h in range(n_history):
                    offset = h * frame_size
                    # joint_pos starts at offset+12, length 23
                    jp_start = offset + 12
                    jp_end = jp_start + num_joints
                    remapped[:, jp_start:jp_end] = proprio[:, jp_start:jp_end][:, newton_to_physx]
                    
                    # joint_vel starts at offset+12+23, length 23
                    jv_start = offset + 12 + num_joints
                    jv_end = jv_start + num_joints
                    remapped[:, jv_start:jv_end] = proprio[:, jv_start:jv_end][:, newton_to_physx]
                
                obs['proprio'] = remapped
            else:
                print(f"WARNING: proprio shape {proprio.shape[-1]} != expected {frame_size * n_history}")
                # Try to remap anyway — just remap all 23-element chunks
                
        # Also check policy group — it contains the last action (23 elements)
        if 'policy' in obs:
            policy_obs = obs['policy']  # shape (num_envs, 170)
            # Layout: object_quat(4) + target_pose(7) + last_action(23)
            # with history: 5 * (4 + 7 + 23) = 5 * 34 = 170
            frame_size_policy = 34
            n_history = 5
            
            if policy_obs.shape[-1] == frame_size_policy * n_history:
                remapped = policy_obs.clone()
                for h in range(n_history):
                    offset = h * frame_size_policy
                    # last_action starts at offset+4+7=offset+11, length 23
                    act_start = offset + 11
                    act_end = act_start + num_joints
                    # Last action was in PhysX order (policy output) — but the env
                    # records what was APPLIED, which is in Newton order after remapping.
                    # So we need to remap it back to PhysX order.
                    remapped[:, act_start:act_end] = policy_obs[:, act_start:act_end][:, newton_to_physx]
                
                obs['policy'] = remapped
    
    return obs


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"
    agent_cfg.device = "cuda:0"

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    physx_to_newton, newton_to_physx = build_remaps(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    print(f"\n*** Cross-sim eval with FULL remapping (obs + actions) ***\n")

    obs = env.get_observations()
    episode_count = 0
    step_count = 0
    total_success = 0.0
    total_pos_err = 0.0
    report_steps = 0

    while episode_count < args.num_episodes and step_count < args.num_episodes * 500:
        with torch.inference_mode():
            # Remap observations: Newton order → PhysX order
            obs_remapped = remap_obs(obs, newton_to_physx)

            # Policy sees PhysX-ordered observations
            actions = policy(obs_remapped)

            # Remap actions: PhysX order → Newton order
            if physx_to_newton is not None and actions.shape[-1] == len(physx_to_newton):
                actions = actions[:, physx_to_newton]

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
    print(f"RESULTS: Full obs+action remapping")
    print(f"  Episodes: {episode_count}")
    print(f"  Steps: {step_count}")
    print(f"  Avg success: {avg_success:.4f}")
    print(f"  Avg pos error: {avg_pos_err:.3f}m")
    print(f"{'='*60}")

    env.close()
    sim_app.close()


sys.argv = [sys.argv[0]] + hydra_args
main()

"""Record joint trajectories given recorded actions. Run once per backend.

Usage:
  # Record on PhysX (also records actions):
  python dynamics_record.py --checkpoint ... --mode record presets=physx
  # Replay on Newton:
  python dynamics_record.py --checkpoint ... --mode replay presets=newton
"""
import argparse, os, sys, json

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument("--mode", required=True, choices=["record", "replay"])
parser.add_argument("--action_file", default="/tmp/dynamics_actions.json")
parser.add_argument("--output", default=None)
parser.add_argument("--task", default="Isaac-Dexsuite-Kuka-Allegro-Lift-v0")
args, hydra_args = parser.parse_known_args()
if args.output is None:
    args.output = f"/tmp/dynamics_{args.mode}.json"

os.chdir('/home/horde/claw/git/IsaacLab')
sys.path.insert(0, 'scripts/reinforcement_learning/rsl_rl')

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=False)
sim_app = launcher.app

import torch, numpy as np, gymnasium as gym, importlib.metadata
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks, cli_args
installed_version = importlib.metadata.version('rsl-rl-lib')

PHYSX_ORDER = ["iiwa7_joint_1", "iiwa7_joint_2", "iiwa7_joint_3",
               "iiwa7_joint_4", "iiwa7_joint_5", "iiwa7_joint_6", "iiwa7_joint_7",
               "index_joint_0", "middle_joint_0", "ring_joint_0", "thumb_joint_0",
               "index_joint_1", "middle_joint_1", "ring_joint_1", "thumb_joint_1",
               "index_joint_2", "middle_joint_2", "ring_joint_2", "thumb_joint_2",
               "index_joint_3", "middle_joint_3", "ring_joint_3", "thumb_joint_3"]


def get_joint_data(robot):
    jp = robot.data.joint_pos[0]
    jv = robot.data.joint_vel[0]
    if hasattr(jp, 'cpu'):
        return jp.cpu().numpy(), jv.cpu().numpy()
    import warp as wp
    return (wp.to_torch(robot.data.joint_pos)[0].cpu().numpy(),
            wp.to_torch(robot.data.joint_vel)[0].cpu().numpy())


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = "cuda:0"
    agent_cfg.device = "cuda:0"
    env_cfg.seed = 42
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.joint_names)
    print(f"Joint names: {joint_names}")
    
    # Build remap if on Newton
    remap_n2p = None  # Newton→PhysX for position reporting
    remap_p2n = None  # PhysX→Newton for action application
    if joint_names != PHYSX_ORDER:
        remap_n2p = [joint_names.index(p) for p in PHYSX_ORDER]
        remap_p2n = [0]*len(joint_names)
        for pi, ni in enumerate(remap_n2p):
            remap_p2n[ni] = pi  # wrong direction, let me think...
        # Actually: remap_p2n[physx_idx] = newton_idx
        remap_p2n = []
        for pname in PHYSX_ORDER:
            remap_p2n.append(joint_names.index(pname))
        remap_p2n = torch.tensor(remap_p2n, dtype=torch.long, device="cuda:0")
        print(f"Newton→PhysX remap: {remap_n2p}")
    
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")
    
    obs = env.get_observations()
    
    positions = []  # Always stored in PhysX joint order for comparison
    velocities = []
    rewards = []
    actions_list = []
    
    if args.mode == "record":
        print(f"\nRECORDING on PhysX (closed-loop policy)")
        for step in range(args.num_steps):
            with torch.inference_mode():
                jp, jv = get_joint_data(robot)
                actions = policy(obs)
                act = actions[0].cpu().numpy()
                
                # PhysX is native order
                positions.append(jp.tolist())
                velocities.append(jv.tolist())
                actions_list.append(act.tolist())
                
                obs, rew, dones, infos = env.step(actions)
                policy.reset(dones)
                rewards.append(rew[0].item())
                
                if step % 20 == 0:
                    print(f"  Step {step}: j1={jp[0]:+.4f}, j7={jp[6]:+.4f}, rew={rew[0].item():.4f}")
                if dones[0]:
                    print(f"  Episode ended at step {step}")
                    break
        
        # Save actions
        with open(args.action_file, 'w') as f:
            json.dump({"actions": actions_list, "joint_names": joint_names}, f)
        print(f"Saved {len(actions_list)} actions to {args.action_file}")
    
    else:  # replay
        print(f"\nREPLAYING recorded actions on Newton")
        with open(args.action_file) as f:
            recorded = json.load(f)
        rec_actions = recorded["actions"]
        print(f"Loaded {len(rec_actions)} actions (PhysX order)")
        
        for step, act_physx in enumerate(rec_actions):
            jp, jv = get_joint_data(robot)
            
            # Convert positions to PhysX order for comparison
            if remap_n2p is not None:
                jp_physx = jp[remap_n2p]
                jv_physx = jv[remap_n2p]
            else:
                jp_physx = jp
                jv_physx = jv
            
            positions.append(jp_physx.tolist())
            velocities.append(jv_physx.tolist())
            actions_list.append(act_physx)
            
            # Apply action with PhysX→Newton remapping
            actions = torch.tensor([act_physx], dtype=torch.float32, device="cuda:0")
            if remap_p2n is not None:
                actions = actions[:, remap_p2n]
            
            obs, rew, dones, infos = env.step(actions)
            rewards.append(rew[0].item())
            
            if step % 20 == 0:
                print(f"  Step {step}: j1={jp_physx[0]:+.4f}, j7={jp_physx[6]:+.4f}, rew={rew[0].item():.4f}")
            if dones[0]:
                print(f"  Episode ended at step {step}")
                break
    
    data = {
        "mode": args.mode,
        "joint_names_physx_order": PHYSX_ORDER,
        "positions": positions,
        "velocities": velocities,
        "actions": actions_list,
        "rewards": rewards,
    }
    with open(args.output, 'w') as f:
        json.dump(data, f)
    print(f"Saved to {args.output}")
    
    env.close()
    sim_app.close()


sys.argv = [sys.argv[0]] + hydra_args
main()

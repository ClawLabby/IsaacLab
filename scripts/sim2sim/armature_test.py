"""Test: Increase armature on Newton to dampen joint response.

MuJoCo armature adds rotor inertia, which damps acceleration.
Higher armature → more damped response → closer to PhysX implicit PD behavior.

Current armature = 0.01 for all joints.
Test with armature = 0.1, 1.0, 10.0 for arm joints.
"""
import argparse, os, sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--task", default="Isaac-Dexsuite-Kuka-Allegro-Lift-v0")
parser.add_argument("--arm_armature", type=float, default=0.01)
parser.add_argument("--hand_armature", type=float, default=0.01)
parser.add_argument("--zero_j7_vel", action="store_true", default=True)
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
from isaaclab.actuators import ImplicitActuatorCfg

import isaaclab_tasks, cli_args
installed_version = importlib.metadata.version('rsl-rl-lib')

NEWTON_TO_PHYSX_JOINTS = [0,1,2,3,4,5,6, 7,11,15,19, 8,12,16,20, 9,13,17,21, 10,14,18,22]
NEWTON_TO_PHYSX_BODIES = [0, 3, 2, 1, 4]
CONTACT_SIZE, JOINT_SIZE, TIPS_SIZE = 12, 23, 65
FRAME_SIZE = CONTACT_SIZE + JOINT_SIZE + JOINT_SIZE + TIPS_SIZE
N_HISTORY = 5
POLICY_FRAME_SIZE = 34
J7_INDEX = 6


def build_action_remap(env):
    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.joint_names)
    physx_order = ["iiwa7_joint_1", "iiwa7_joint_2", "iiwa7_joint_3",
                   "iiwa7_joint_4", "iiwa7_joint_5", "iiwa7_joint_6", "iiwa7_joint_7",
                   "index_joint_0", "middle_joint_0", "ring_joint_0", "thumb_joint_0",
                   "index_joint_1", "middle_joint_1", "ring_joint_1", "thumb_joint_1",
                   "index_joint_2", "middle_joint_2", "ring_joint_2", "thumb_joint_2",
                   "index_joint_3", "middle_joint_3", "ring_joint_3", "thumb_joint_3"]
    if joint_names == physx_order:
        return None
    return torch.tensor([joint_names.index(p) for p in physx_order], dtype=torch.long, device=robot.device)


def remap_proprio(proprio, n2p_joints, n2p_bodies, zero_j7=True):
    if n2p_joints is None:
        return proprio
    batch = proprio.shape[0]
    out = proprio.clone()
    for h in range(N_HISTORY):
        off = h * FRAME_SIZE
        jp_start = off + CONTACT_SIZE
        out[:, jp_start:jp_start+JOINT_SIZE] = proprio[:, jp_start:jp_start+JOINT_SIZE][:, n2p_joints]
        jv_start = off + CONTACT_SIZE + JOINT_SIZE
        remapped_jv = proprio[:, jv_start:jv_start+JOINT_SIZE][:, n2p_joints]
        if zero_j7:
            remapped_jv[:, J7_INDEX] = 0.0
        out[:, jv_start:jv_start+JOINT_SIZE] = remapped_jv
        tips_start = off + CONTACT_SIZE + JOINT_SIZE + JOINT_SIZE
        tips = proprio[:, tips_start:tips_start+TIPS_SIZE].reshape(batch, 5, 13)
        out[:, tips_start:tips_start+TIPS_SIZE] = tips[:, n2p_bodies, :].reshape(batch, TIPS_SIZE)
    return out


def remap_policy_obs(policy_obs, n2p_joints):
    if n2p_joints is None:
        return policy_obs
    out = policy_obs.clone()
    for h in range(N_HISTORY):
        off = h * POLICY_FRAME_SIZE
        act_start = off + 11
        out[:, act_start:act_start+JOINT_SIZE] = policy_obs[:, act_start:act_start+JOINT_SIZE][:, n2p_joints]
    return out


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"
    agent_cfg.device = "cuda:0"

    # Set custom armature
    print(f"\n*** Armature test: arm={args.arm_armature}, hand={args.hand_armature} ***")
    env_cfg.scene.robot.actuators = {
        "kuka_allegro_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "iiwa7_joint_(1|2|3|4|5|6|7)",
                "index_joint_(0|1|2|3)",
                "middle_joint_(0|1|2|3)",
                "ring_joint_(0|1|2|3)",
                "thumb_joint_(0|1|2|3)",
            ],
            effort_limit_sim={
                "iiwa7_joint_(1|2|3|4|5|6|7)": 300.0,
                "index_joint_(0|1|2|3)": 0.5,
                "middle_joint_(0|1|2|3)": 0.5,
                "ring_joint_(0|1|2|3)": 0.5,
                "thumb_joint_(0|1|2|3)": 0.5,
            },
            stiffness={
                "iiwa7_joint_(1|2|3|4)": 300.0,
                "iiwa7_joint_5": 100.0,
                "iiwa7_joint_6": 50.0,
                "iiwa7_joint_7": 25.0,
                "index_joint_(0|1|2|3)": 3.0,
                "middle_joint_(0|1|2|3)": 3.0,
                "ring_joint_(0|1|2|3)": 3.0,
                "thumb_joint_(0|1|2|3)": 3.0,
            },
            damping={
                "iiwa7_joint_(1|2|3|4)": 45.0,
                "iiwa7_joint_5": 20.0,
                "iiwa7_joint_6": 15.0,
                "iiwa7_joint_7": 15.0,
                "index_joint_(0|1|2|3)": 0.1,
                "middle_joint_(0|1|2|3)": 0.1,
                "ring_joint_(0|1|2|3)": 0.1,
                "thumb_joint_(0|1|2|3)": 0.1,
            },
            friction={
                "iiwa7_joint_(1|2|3|4|5|6|7)": 1.0,
                "index_joint_(0|1|2|3)": 0.01,
                "middle_joint_(0|1|2|3)": 0.01,
                "ring_joint_(0|1|2|3)": 0.01,
                "thumb_joint_(0|1|2|3)": 0.01,
            },
            armature={
                "iiwa7_joint_(1|2|3|4|5|6|7)": args.arm_armature,
                "index_joint_(0|1|2|3)": args.hand_armature,
                "middle_joint_(0|1|2|3)": args.hand_armature,
                "ring_joint_(0|1|2|3)": args.hand_armature,
                "thumb_joint_(0|1|2|3)": args.hand_armature,
            },
        ),
    }

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    action_remap = build_action_remap(env)
    device = env.unwrapped.device
    n2p_joints = torch.tensor(NEWTON_TO_PHYSX_JOINTS, dtype=torch.long, device=device) if action_remap is not None else None
    n2p_bodies = NEWTON_TO_PHYSX_BODIES if action_remap is not None else None

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)

    obs = env.get_observations()
    episode_count, step_count = 0, 0
    total_success, total_pos_err, report_steps = 0.0, 0.0, 0

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

            if dones.any().item():
                episode_count += int(dones.sum().item())

            if step_count % 200 == 0:
                print(f"  Step {step_count}: ep={episode_count}/{args.num_episodes}, "
                      f"success={total_success/max(report_steps,1):.4f}")

    print(f"\n{'='*60}")
    print(f"arm_armature={args.arm_armature}, hand_armature={args.hand_armature}")
    print(f"  Avg success: {total_success/max(report_steps,1):.4f}")
    print(f"  Avg pos error: {total_pos_err/max(report_steps,1):.3f}m")
    print(f"{'='*60}")

    env.close()
    sim_app.close()


sys.argv = [sys.argv[0]] + hydra_args
main()

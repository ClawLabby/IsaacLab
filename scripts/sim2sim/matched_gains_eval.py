"""Hacky sim2sim test: tune Newton's actuator gains to match PhysX effective response.

From measurements:
- iiwa7_joint_1: PhysX displacement = 0.060, Newton = 0.991 → ratio 16.5×
- iiwa7_joint_6: ratio ~1.26×
- iiwa7_joint_7: ratio ~0.89×  
- finger joints: ratio ~0.98×

Strategy: Scale down Newton stiffness by the measured ratio to match PhysX effective response.
This is a HACK — it changes the actuator model to compensate for solver differences.

Usage:
  cd /home/horde/claw/git/IsaacLab
  PYTHONDONTWRITEBYTECODE=1 OMNI_KIT_ACCEPT_EULA=yes \
  env_isaaclab/bin/python scripts/sim2sim/matched_gains_eval.py \
    --checkpoint logs/rsl_rl/dexsuite_kuka_allegro/2026-03-31_21-53-28/model_9750.pt \
    --num_envs 4 --num_episodes 40 \
    presets=newton
"""
import argparse
import os
import sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=40)
parser.add_argument("--task", default="Isaac-Dexsuite-Kuka-Allegro-Lift-v0")
# Gain scale factors: Newton_effective_Kp = PhysX_Kp * scale
# Derived from step response ratio measurements
parser.add_argument("--kp_scale_arm_1234", type=float, default=None,
                    help="Kp multiplier for iiwa joints 1-4 (default: auto from measurements)")
parser.add_argument("--kp_scale_arm_567", type=float, default=None)
parser.add_argument("--kp_scale_hand", type=float, default=None)
parser.add_argument("--kd_scale", type=float, default=None,
                    help="Global Kd multiplier (default: same as Kp scale per group)")
parser.add_argument("--mode", default="scaled_gains",
                    choices=["scaled_gains", "high_damping", "substeps"])
parser.add_argument("--substeps", type=int, default=8, help="Number of substeps for substeps mode")
args, hydra_args = parser.parse_known_args()

os.chdir('/home/horde/claw/git/IsaacLab')
sys.path.insert(0, 'scripts/reinforcement_learning/rsl_rl')

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=False)
sim_app = launcher.app

import torch
import numpy as np
import gymnasium as gym
import importlib.metadata
from collections import defaultdict

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.actuators import ImplicitActuatorCfg

import isaaclab_tasks  # noqa
import cli_args  # noqa

installed_version = importlib.metadata.version('rsl-rl-lib')

# Measured displacement ratios (Newton/PhysX) from step response test
# Newton_displacement / PhysX_displacement at pulse end
MEASURED_RATIOS = {
    "iiwa7_joint_1": 16.5,  # Kp=300
    "iiwa7_joint_4": 1.05,  # Kp=120 (estimated from interpolation)
    "iiwa7_joint_5": 1.12,  # Kp=100
    "iiwa7_joint_6": 1.26,  # Kp=50
    "iiwa7_joint_7": 0.89,  # Kp=25
    "index_joint_0": 0.98,  # Kp=3
}

# Compute scale factors: to match PhysX, divide Newton Kp by the ratio
# i.e., Newton with Kp/ratio should give same displacement as PhysX with Kp
def get_scales(args):
    # High-stiffness arm joints need large correction
    s1234 = args.kp_scale_arm_1234 or (1.0 / 16.5)  # ~0.06
    s567 = args.kp_scale_arm_567 or (1.0 / 1.1)      # ~0.91
    shand = args.kp_scale_hand or 1.0                  # fingers already match

    return s1234, s567, shand


def build_remap_table(env):
    """Build PhysX->Newton joint remapping."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    joint_names = robot.joint_names

    physx_order = [
        "iiwa7_joint_1", "iiwa7_joint_2", "iiwa7_joint_3",
        "iiwa7_joint_4", "iiwa7_joint_5", "iiwa7_joint_6", "iiwa7_joint_7",
        "index_joint_0", "middle_joint_0", "ring_joint_0", "thumb_joint_0",
        "index_joint_1", "middle_joint_1", "ring_joint_1", "thumb_joint_1",
        "index_joint_2", "middle_joint_2", "ring_joint_2", "thumb_joint_2",
        "index_joint_3", "middle_joint_3", "ring_joint_3", "thumb_joint_3",
    ]

    remap = []
    for pname in physx_order:
        if pname in joint_names:
            remap.append(joint_names.index(pname))
        else:
            remap.append(0)

    if remap == list(range(len(remap))):
        return None
    return torch.tensor(remap, dtype=torch.long, device=robot.device)


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"
    agent_cfg.device = "cuda:0"

    s1234, s567, shand = get_scales(args)

    if args.mode == "scaled_gains":
        print(f"\n*** SCALED GAINS MODE ***")
        print(f"  Kp scale arm j1-4: {s1234:.3f} (Kp {300*s1234:.1f})")
        print(f"  Kp scale arm j5-7: {s567:.3f}")
        print(f"  Kp scale hand: {shand:.3f}")

        kd_s1234 = args.kd_scale or s1234
        kd_s567 = args.kd_scale or s567
        kd_shand = args.kd_scale or shand

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
                    "iiwa7_joint_(1|2|3|4)": 300.0 * s1234,
                    "iiwa7_joint_5": 100.0 * s567,
                    "iiwa7_joint_6": 50.0 * s567,
                    "iiwa7_joint_7": 25.0 * s567,
                    "index_joint_(0|1|2|3)": 3.0 * shand,
                    "middle_joint_(0|1|2|3)": 3.0 * shand,
                    "ring_joint_(0|1|2|3)": 3.0 * shand,
                    "thumb_joint_(0|1|2|3)": 3.0 * shand,
                },
                damping={
                    "iiwa7_joint_(1|2|3|4)": 45.0 * kd_s1234,
                    "iiwa7_joint_5": 20.0 * kd_s567,
                    "iiwa7_joint_6": 15.0 * kd_s567,
                    "iiwa7_joint_7": 15.0 * kd_s567,
                    "index_joint_(0|1|2|3)": 0.1 * kd_shand,
                    "middle_joint_(0|1|2|3)": 0.1 * kd_shand,
                    "ring_joint_(0|1|2|3)": 0.1 * kd_shand,
                    "thumb_joint_(0|1|2|3)": 0.1 * kd_shand,
                },
                friction={
                    "iiwa7_joint_(1|2|3|4|5|6|7)": 1.0,
                    "index_joint_(0|1|2|3)": 0.01,
                    "middle_joint_(0|1|2|3)": 0.01,
                    "ring_joint_(0|1|2|3)": 0.01,
                    "thumb_joint_(0|1|2|3)": 0.01,
                },
                armature={".*": 0.01},
            ),
        }

    elif args.mode == "high_damping":
        print(f"\n*** HIGH DAMPING MODE ***")
        print(f"  Keep original Kp, multiply Kd by 10× for arm joints")

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
                    "iiwa7_joint_(1|2|3|4)": 450.0,   # 10×
                    "iiwa7_joint_5": 200.0,            # 10×
                    "iiwa7_joint_6": 150.0,            # 10×
                    "iiwa7_joint_7": 150.0,            # 10×
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
                armature={".*": 0.01},
            ),
        }

    elif args.mode == "substeps":
        print(f"\n*** SUBSTEPS MODE: {args.substeps} substeps ***")
        env_cfg.sim.physics.num_substeps = args.substeps

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
            if remap_table is not None and actions.shape[-1] == len(remap_table):
                actions = actions[:, remap_table]
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
    print(f"RESULTS: {args.mode} on Newton")
    print(f"  Episodes: {episode_count}")
    print(f"  Steps: {step_count}")
    print(f"  Avg success: {avg_success:.4f}")
    print(f"  Avg pos error: {avg_pos_err:.3f}m")
    print(f"{'='*60}")

    env.close()
    sim_app.close()


sys.argv = [sys.argv[0]] + hydra_args
main()

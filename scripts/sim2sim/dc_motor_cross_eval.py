"""Cross-sim eval with DC Motor actuator model on Newton.

Loads a PhysX-trained policy and evaluates it on Newton using Isaac Lab's
DCMotor explicit actuator instead of the default ImplicitActuator. This tests
whether velocity-dependent torque saturation helps sim2sim transfer.

Usage:
  cd /home/horde/claw/git/IsaacLab
  PYTHONDONTWRITEBYTECODE=1 OMNI_KIT_ACCEPT_EULA=yes \
  env_isaaclab/bin/python scripts/sim2sim/dc_motor_cross_eval.py \
    --checkpoint logs/rsl_rl/dexsuite_kuka_allegro/2026-04-03_01-04-45_no_dr/model_4999.pt \
    --num_envs 4 --num_episodes 20 \
    presets=newton

Options:
  --actuator implicit|dc_motor|ideal_pd  Choose actuator model
  --velocity_limit FLOAT                  DC motor no-load speed (default: auto)
"""
import argparse
import os
import sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--actuator", default="dc_motor", choices=["implicit", "dc_motor", "ideal_pd"])
parser.add_argument("--velocity_limit_arm", type=float, default=2.3)
parser.add_argument("--velocity_limit_hand", type=float, default=8.7)
parser.add_argument("--task", default="Isaac-Dexsuite-Kuka-Allegro-Lift-v0")
parser.add_argument("--remap_joints", action="store_true", default=True)
parser.add_argument("--no_remap", dest="remap_joints", action="store_false")
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
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg, IdealPDActuatorCfg

import isaaclab_tasks  # noqa
import cli_args  # noqa

installed_version = importlib.metadata.version('rsl-rl-lib')


def make_dc_motor_actuators(args):
    """Create per-joint-group DCMotorCfg dicts for Kuka-Allegro.

    DCMotorCfg.saturation_effort is a scalar, so we need separate actuator
    groups for joints with different effort limits.
    """
    vel_arm = args.velocity_limit_arm
    vel_hand = args.velocity_limit_hand

    return {
        "kuka_arm_1234": DCMotorCfg(
            joint_names_expr=["iiwa7_joint_(1|2|3|4)"],
            effort_limit=300.0,
            stiffness=300.0,
            damping=45.0,
            friction=1.0,
            armature=0.01,
            saturation_effort=450.0,
            velocity_limit=vel_arm,
        ),
        "kuka_arm_5": DCMotorCfg(
            joint_names_expr=["iiwa7_joint_5"],
            effort_limit=100.0,
            stiffness=100.0,
            damping=20.0,
            friction=1.0,
            armature=0.01,
            saturation_effort=150.0,
            velocity_limit=vel_arm,
        ),
        "kuka_arm_6": DCMotorCfg(
            joint_names_expr=["iiwa7_joint_6"],
            effort_limit=50.0,
            stiffness=50.0,
            damping=15.0,
            friction=1.0,
            armature=0.01,
            saturation_effort=75.0,
            velocity_limit=vel_arm,
        ),
        "kuka_arm_7": DCMotorCfg(
            joint_names_expr=["iiwa7_joint_7"],
            effort_limit=25.0,
            stiffness=25.0,
            damping=15.0,
            friction=1.0,
            armature=0.01,
            saturation_effort=37.5,
            velocity_limit=vel_arm,
        ),
        "allegro_hand": DCMotorCfg(
            joint_names_expr=[
                "index_joint_(0|1|2|3)",
                "middle_joint_(0|1|2|3)",
                "ring_joint_(0|1|2|3)",
                "thumb_joint_(0|1|2|3)",
            ],
            effort_limit=0.7,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
            armature=0.01,
            saturation_effort=1.05,
            velocity_limit=vel_hand,
        ),
    }


def make_ideal_pd_actuators():
    """Create IdealPDActuatorCfg matching the Kuka-Allegro joint params."""
    return {
        "kuka_allegro_actuators": IdealPDActuatorCfg(
            joint_names_expr=[
                "iiwa7_joint_(1|2|3|4|5|6|7)",
                "index_joint_(0|1|2|3)",
                "middle_joint_(0|1|2|3)",
                "ring_joint_(0|1|2|3)",
                "thumb_joint_(0|1|2|3)",
            ],
            effort_limit={
                "iiwa7_joint_(1|2|3|4)": 300.0,
                "iiwa7_joint_5": 100.0,
                "iiwa7_joint_6": 50.0,
                "iiwa7_joint_7": 25.0,
                "index_joint_(0|1|2|3)": 0.7,
                "middle_joint_(0|1|2|3)": 0.7,
                "ring_joint_(0|1|2|3)": 0.7,
                "thumb_joint_(0|1|2|3)": 0.7,
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
            armature=0.01,
        ),
    }


def build_remap_table(env):
    """Build PhysX->Newton joint remapping from actual joint names."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    joint_names = robot.joint_names

    # PhysX canonical ordering for kuka_allegro
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
            print(f"WARNING: PhysX joint {pname} not found in Newton env")
            remap.append(0)

    print(f"Newton joint order: {joint_names}")
    print(f"PhysX->Newton remap: {remap}")

    # Check if remap is identity (no remapping needed)
    if remap == list(range(len(remap))):
        print("Joint ordering is IDENTICAL — no remapping needed")
        return None

    return torch.tensor(remap, dtype=torch.long, device=robot.device)


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"
    agent_cfg.device = "cuda:0"

    # Swap actuator model
    if args.actuator == "dc_motor":
        print(f"\n*** DC Motor actuator (vel_arm={args.velocity_limit_arm}, "
              f"vel_hand={args.velocity_limit_hand}) ***\n")
        env_cfg.scene.robot.actuators = make_dc_motor_actuators(args)
    elif args.actuator == "ideal_pd":
        print("\n*** Ideal PD actuator (explicit torque) ***\n")
        env_cfg.scene.robot.actuators = make_ideal_pd_actuators()
    else:
        print("\n*** Default Implicit actuator ***\n")

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # Create environment
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Build joint remap table
    remap_table = None
    if args.remap_joints:
        try:
            remap_table = build_remap_table(env)
        except Exception as e:
            print(f"Remap failed: {e}")

    # Load checkpoint
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    print(f"\nEvaluating {args.num_episodes} episodes...")
    print(f"Actuator: {args.actuator}, Remap: {remap_table is not None}")

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

            # Collect metrics from environment extras
            try:
                extras = env.unwrapped.extras.get("log", {})
                s = extras.get("Episode_Reward/success", 0.0)
                p = extras.get("Metrics/object_pose/position_error", 1.0)
                if isinstance(s, torch.Tensor):
                    s = s.item()
                if isinstance(p, torch.Tensor):
                    p = p.item()
                total_success += s
                total_pos_err += p
                report_steps += 1
            except:
                pass

            if dones.any():
                n_done = dones.sum().item()
                episode_count += int(n_done)

            if step_count % 100 == 0:
                avg_s = total_success / max(report_steps, 1)
                avg_p = total_pos_err / max(report_steps, 1)
                print(f"  Step {step_count}: ep={episode_count}/{args.num_episodes}, "
                      f"avg_success={avg_s:.4f}, avg_pos_err={avg_p:.3f}m")

    avg_success = total_success / max(report_steps, 1)
    avg_pos_err = total_pos_err / max(report_steps, 1)

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.actuator} actuator on Newton")
    print(f"  Episodes: {episode_count}")
    print(f"  Steps: {step_count}")
    print(f"  Avg success reward: {avg_success:.4f}")
    print(f"  Avg position error: {avg_pos_err:.3f}m")
    print(f"  Remap: {remap_table is not None}")
    print(f"{'='*60}")

    env.close()
    sim_app.close()


sys.argv = [sys.argv[0]] + hydra_args
main()

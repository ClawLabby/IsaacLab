#!/usr/bin/env python3
"""Sim2Sim Parameter Comparison: PhysX vs Newton for DexSuite Kuka Allegro.

This script:
1. Creates a single DexSuite env with PhysX (no DR) and dumps joint/physics params
2. Creates the same env with Newton and dumps the same params
3. Applies identical random actions and compares resulting states step-by-step

Usage:
    # PhysX
    python scripts/sim2sim/param_compare.py presets=physx --num_envs 1
    # Newton
    python scripts/sim2sim/param_compare.py presets=newton --num_envs 1
"""

import argparse
import json
import os
import sys
import time

import torch
import numpy as np


def main():
    # ── Parse our flags before Isaac Lab eats them ──────────────────────
    # Parse our flags (s2s_ prefix) before Isaac Lab sees them
    output_dir = "/tmp/sim2sim"
    num_steps = 200
    seed = 42
    isaac_args = [sys.argv[0]]
    for a in sys.argv[1:]:
        if a.startswith("s2s_output="):
            output_dir = a.split("=", 1)[1]
        elif a.startswith("s2s_steps="):
            num_steps = int(a.split("=", 1)[1])
        elif a.startswith("s2s_seed="):
            seed = int(a.split("=", 1)[1])
        else:
            isaac_args.append(a)
    sys.argv = isaac_args

    class Flags:
        pass
    flags = Flags()
    flags.output = output_dir
    flags.steps = num_steps
    flags.seed = seed

    os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"

    # ── Import Isaac Lab ───────────────────────────────────────────────
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, enable_cameras=False)
    simulation_app = launcher.app

    from isaaclab_tasks.utils.hydra import resolve_task_config
    env_cfg, agent_cfg = resolve_task_config(
        "Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
        "rsl_rl_cfg_entry_point",
    )

    # Force single env, no DR
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.0

    # Detect backend
    physics_type = type(env_cfg.sim.physics).__name__
    if "Newton" in physics_type:
        backend = "newton"
    else:
        backend = "physx"

    print(f"\n{'='*60}")
    print(f"  Backend: {backend} ({physics_type})")
    print(f"  Steps: {flags.steps}, Seed: {flags.seed}")
    print(f"{'='*60}\n")

    # Check if DR events exist
    has_startup_dr = hasattr(env_cfg.events, 'joint_stiffness_and_damping') and \
                     hasattr(getattr(env_cfg.events, 'joint_stiffness_and_damping', None), 'mode') and \
                     getattr(env_cfg.events.joint_stiffness_and_damping, 'mode', None) == 'startup'

    # Remove startup DR events if present (we want nominal params)
    if has_startup_dr:
        print("NOTE: Removing startup DR events for fair comparison")
        for attr_name in list(vars(env_cfg.events)):
            ev = getattr(env_cfg.events, attr_name)
            if hasattr(ev, 'mode') and getattr(ev, 'mode', None) == 'startup':
                print(f"  Removing: {attr_name}")
                delattr(env_cfg.events, attr_name)

    # ── Create environment ─────────────────────────────────────────────
    import gymnasium as gym
    import isaaclab_tasks  # noqa: register tasks
    from isaaclab.envs import ManagerBasedRLEnv

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # ── Extract joint parameters ───────────────────────────────────────
    robot = env.scene["robot"]

    joint_names = robot.joint_names
    num_joints = len(joint_names)

    print(f"\nJoint count: {num_joints}")
    print(f"Joint names: {joint_names}")

    # Get actuator parameters
    params = {}
    for act_name, actuator in robot.actuators.items():
        act_params = {
            "name": act_name,
            "type": type(actuator).__name__,
        }
        # Extract gains
        if hasattr(actuator, 'stiffness'):
            act_params["stiffness"] = actuator.stiffness.cpu().numpy().tolist()
        if hasattr(actuator, 'damping'):
            act_params["damping"] = actuator.damping.cpu().numpy().tolist()
        if hasattr(actuator, 'friction'):
            act_params["friction"] = actuator.friction.cpu().numpy().tolist()
        if hasattr(actuator, 'armature'):
            act_params["armature"] = actuator.armature.cpu().numpy().tolist()
        if hasattr(actuator, 'effort_limit'):
            act_params["effort_limit"] = actuator.effort_limit.cpu().numpy().tolist()

        params[act_name] = act_params

    # Get joint limits
    joint_limits = None
    if hasattr(robot, 'root_physx_view') and robot.root_physx_view is not None:
        try:
            lims = robot.root_physx_view.get_dof_limits()
            joint_limits = lims.cpu().numpy() if hasattr(lims, 'cpu') else np.array(lims)
        except Exception as e:
            print(f"Joint limits: {e}")

    # Get mass properties
    body_names = robot.body_names
    body_masses = None
    if hasattr(robot, 'root_physx_view') and robot.root_physx_view is not None:
        try:
            masses = robot.root_physx_view.get_body_masses()
            body_masses = (masses.cpu().numpy() if hasattr(masses, 'cpu') else np.array(masses)).tolist()
        except Exception as e:
            print(f"Body masses: {e}")

    # Try Newton-specific parameter access
    if backend == "newton":
        try:
            from isaaclab_newton.assets.articulation import Articulation as NewtonArticulation
            # Newton stores params differently
            if hasattr(robot, '_newton_model'):
                model = robot._newton_model
                print(f"Newton model joints: {model.njnt}")
        except Exception as e:
            print(f"Newton param access: {e}")

    # ── Print parameter table ──────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Joint Parameters ({backend})")
    print(f"{'='*80}")
    print(f"{'Joint':<20} {'Stiffness':>10} {'Damping':>10} {'Friction':>10} {'Armature':>10}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    stiffness = params.get("kuka_allegro_actuators", {}).get("stiffness", [[0]*num_joints])[0]
    damping_vals = params.get("kuka_allegro_actuators", {}).get("damping", [[0]*num_joints])[0]
    friction_vals = params.get("kuka_allegro_actuators", {}).get("friction", [[0]*num_joints])[0]
    armature_vals = params.get("kuka_allegro_actuators", {}).get("armature", [[0]*num_joints])[0]

    for i, name in enumerate(joint_names):
        s = stiffness[i] if i < len(stiffness) else "?"
        d = damping_vals[i] if i < len(damping_vals) else "?"
        f = friction_vals[i] if i < len(friction_vals) else "?"
        a = armature_vals[i] if i < len(armature_vals) else "?"
        print(f"{name:<20} {s:>10.4f} {d:>10.4f} {f:>10.4f} {a:>10.4f}")

    # ── Run identical actions and record trajectories ──────────────────
    print(f"\n{'='*60}")
    print(f"  Running {flags.steps} steps with fixed random actions")
    print(f"{'='*60}\n")

    torch.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    # Pre-generate action sequence
    action_dim = env.action_manager.total_action_dim
    actions = torch.randn(flags.steps, 1, action_dim, device=env.device) * 0.3

    def to_np(x):
        """Convert tensor, warp array, or numpy array to numpy."""
        if hasattr(x, 'cpu'):
            return x.cpu().numpy()
        if hasattr(x, 'numpy'):
            return x.numpy()
        # Warp array — convert via torch
        try:
            import warp as wp
            if isinstance(x, wp.array):
                return wp.to_torch(x).cpu().numpy()
        except:
            pass
        return np.array(x)

    # Reset env
    obs, info = env.reset()

    trajectory = []
    for step in range(flags.steps):
        action = actions[step]

        # Record pre-step state
        joint_pos = to_np(robot.data.joint_pos).copy()
        joint_vel = to_np(robot.data.joint_vel).copy()

        # Get object state if available
        obj_pos = None
        obj_quat = None
        try:
            obj = env.scene["object"]
            obj_pos = to_np(obj.data.root_pos_w).copy()
            obj_quat = to_np(obj.data.root_quat_w).copy()
        except:
            pass

        # Step
        obs, rew, terminated, truncated, info = env.step(action)

        # Record post-step state
        joint_pos_after = to_np(robot.data.joint_pos).copy()
        joint_vel_after = to_np(robot.data.joint_vel).copy()

        record = {
            "step": step,
            "action": to_np(action).tolist(),
            "joint_pos_before": joint_pos.tolist(),
            "joint_vel_before": joint_vel.tolist(),
            "joint_pos_after": joint_pos_after.tolist(),
            "joint_vel_after": joint_vel_after.tolist(),
            "reward": float(to_np(rew).item() if hasattr(rew, 'item') else rew),
        }
        if obj_pos is not None:
            record["obj_pos"] = obj_pos.tolist()
            record["obj_quat"] = obj_quat.tolist()
        # Get post-step object state too
        try:
            obj = env.scene["object"]
            record["obj_pos_after"] = to_np(obj.data.root_pos_w).tolist()
            record["obj_quat_after"] = to_np(obj.data.root_quat_w).tolist()
        except:
            pass

        trajectory.append(record)

        if step < 5 or step % 50 == 0:
            print(f"Step {step:4d}: reward={rew.item():.4f}, "
                  f"joint_pos_mean={joint_pos_after.mean():.4f}, "
                  f"joint_vel_rms={np.sqrt((joint_vel_after**2).mean()):.4f}")

    # ── Save results ───────────────────────────────────────────────────
    os.makedirs(flags.output, exist_ok=True)
    outfile = os.path.join(flags.output, f"{backend}_params.json")

    result = {
        "backend": backend,
        "physics_type": physics_type,
        "num_joints": num_joints,
        "joint_names": joint_names,
        "body_names": body_names,
        "actuator_params": params,
        "body_masses": body_masses,
        "seed": flags.seed,
        "num_steps": flags.steps,
        "trajectory": trajectory,
    }

    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {outfile}")

    # Also save just the actions for reproducibility
    action_file = os.path.join(flags.output, "actions.pt")
    if not os.path.exists(action_file):
        torch.save(actions.cpu(), action_file)
        print(f"Saved actions to {action_file}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

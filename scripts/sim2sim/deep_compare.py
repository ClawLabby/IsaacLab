#!/usr/bin/env python3
"""Sim2Sim deep physics comparison: PhysX vs Newton.

Applies zero actions to a single env and measures how the two solvers
differ in their response to identical PD control targets. This isolates
the solver dynamics from action noise.

Also tests: step response (sudden action pulse), gravity compensation,
and joint limit behavior.
"""

import json
import os
import sys
import numpy as np

# Parse our flags before Isaac Lab
output_dir = "/tmp/sim2sim"
isaac_args = [sys.argv[0]]
for a in sys.argv[1:]:
    if a.startswith("s2s_output="):
        output_dir = a.split("=", 1)[1]
    else:
        isaac_args.append(a)
sys.argv = isaac_args

os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = launcher.app

import torch
import warp as wp

from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab.envs import ManagerBasedRLEnv

env_cfg, agent_cfg = resolve_task_config(
    "Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
    "rsl_rl_cfg_entry_point",
)

env_cfg.scene.num_envs = 1
env_cfg.scene.env_spacing = 2.0

# Detect backend
physics_type = type(env_cfg.sim.physics).__name__
backend = "newton" if "Newton" in physics_type else "physx"

# Remove startup DR
for attr_name in list(vars(env_cfg.events)):
    ev = getattr(env_cfg.events, attr_name)
    if hasattr(ev, 'mode') and getattr(ev, 'mode', None) == 'startup':
        delattr(env_cfg.events, attr_name)

env = ManagerBasedRLEnv(cfg=env_cfg)
robot = env.scene["robot"]


def to_np(x):
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    if hasattr(x, 'numpy'):
        return x.numpy()
    try:
        if isinstance(x, wp.array):
            return wp.to_torch(x).cpu().numpy()
    except:
        pass
    return np.array(x)


joint_names = robot.joint_names
num_joints = len(joint_names)
action_dim = env.action_manager.total_action_dim

print(f"\nBackend: {backend}")
print(f"Joints: {num_joints}, Action dim: {action_dim}")
print(f"Joint names: {joint_names}")

# Get physics config
dt = env_cfg.sim.dt if hasattr(env_cfg.sim, 'dt') else None
decimation = env_cfg.decimation if hasattr(env_cfg, 'decimation') else None
print(f"dt: {dt}, decimation: {decimation}")

obs, info = env.reset()

results = {
    "backend": backend,
    "joint_names": joint_names,
    "tests": {}
}

# ── Test 1: Zero action response ──
# Apply zero actions for 100 steps and see how the robot behaves
print("\n=== Test 1: Zero Action Response (100 steps) ===")
zero_action = torch.zeros(1, action_dim, device=env.device)

# Reset first
obs, info = env.reset()

zero_traj = []
for step in range(100):
    joint_pos = to_np(robot.data.joint_pos).flatten().copy()
    joint_vel = to_np(robot.data.joint_vel).flatten().copy()
    
    # Get applied torques if available
    joint_torques = None
    if hasattr(robot.data, 'applied_torque'):
        joint_torques = to_np(robot.data.applied_torque).flatten().copy()
    elif hasattr(robot.data, 'computed_torque'):
        joint_torques = to_np(robot.data.computed_torque).flatten().copy()
    
    obs, rew, terminated, truncated, info = env.step(zero_action)
    
    joint_pos_after = to_np(robot.data.joint_pos).flatten().copy()
    joint_vel_after = to_np(robot.data.joint_vel).flatten().copy()
    
    record = {
        "step": step,
        "joint_pos": joint_pos.tolist(),
        "joint_vel": joint_vel.tolist(),
        "joint_pos_after": joint_pos_after.tolist(),
        "joint_vel_after": joint_vel_after.tolist(),
    }
    if joint_torques is not None:
        record["torques"] = joint_torques.tolist()
    
    zero_traj.append(record)
    
    if step < 5 or step % 20 == 0:
        print(f"  Step {step:3d}: pos_mean={joint_pos_after.mean():.4f}, "
              f"vel_rms={np.sqrt((joint_vel_after**2).mean()):.6f}")

results["tests"]["zero_action"] = zero_traj

# ── Test 2: Step response on joint 7 (wrist) ──
print("\n=== Test 2: Step Response (pulse on iiwa7_joint_7) ===")
obs, info = env.reset()

step_traj = []
for step in range(100):
    action = torch.zeros(1, action_dim, device=env.device)
    if 10 <= step < 20:  # Apply pulse for 10 steps
        action[0, 6] = 1.0  # iiwa7_joint_7 (index 6 in both backends for arm joints)
    
    joint_pos = to_np(robot.data.joint_pos).flatten().copy()
    joint_vel = to_np(robot.data.joint_vel).flatten().copy()
    obs, rew, terminated, truncated, info = env.step(action)
    joint_pos_after = to_np(robot.data.joint_pos).flatten().copy()
    joint_vel_after = to_np(robot.data.joint_vel).flatten().copy()
    
    step_traj.append({
        "step": step,
        "action_j7": float(action[0, 6].item()),
        "j7_pos": float(joint_pos_after[6]),
        "j7_vel": float(joint_vel_after[6]),
        "joint_pos": joint_pos_after.tolist(),
        "joint_vel": joint_vel_after.tolist(),
    })
    
    if step < 5 or (10 <= step <= 25) or step % 20 == 0:
        print(f"  Step {step:3d}: action_j7={action[0,6].item():.1f}, "
              f"j7_pos={joint_pos_after[6]:.4f}, j7_vel={joint_vel_after[6]:.4f}")

results["tests"]["step_response_j7"] = step_traj

# ── Test 3: Check initial joint positions ──
print("\n=== Test 3: Initial State Comparison ===")
obs, info = env.reset()
init_pos = to_np(robot.data.joint_pos).flatten()
init_vel = to_np(robot.data.joint_vel).flatten()

print("Joint initial positions:")
for i, name in enumerate(joint_names):
    print(f"  {name:<20}: pos={init_pos[i]:.6f}, vel={init_vel[i]:.6f}")

results["tests"]["initial_state"] = {
    "joint_pos": init_pos.tolist(),
    "joint_vel": init_vel.tolist(),
}

# ── Test 4: Object state ──
print("\n=== Test 4: Object State ===")
try:
    obj = env.scene["object"]
    obj_pos = to_np(obj.data.root_pos_w).flatten()
    obj_quat = to_np(obj.data.root_quat_w).flatten()
    print(f"  Object pos: {obj_pos}")
    print(f"  Object quat: {obj_quat}")
    
    # Check mass
    if hasattr(obj, 'root_physx_view') and obj.root_physx_view is not None:
        try:
            masses = obj.root_physx_view.get_body_masses()
            obj_mass = (masses.cpu().numpy() if hasattr(masses, 'cpu') else np.array(masses))
            print(f"  Object mass: {obj_mass}")
            results["tests"]["object"] = {
                "pos": obj_pos.tolist(),
                "quat": obj_quat.tolist(),
                "mass": obj_mass.tolist(),
            }
        except Exception as e:
            print(f"  Mass: {e}")
            results["tests"]["object"] = {"pos": obj_pos.tolist(), "quat": obj_quat.tolist()}
except Exception as e:
    print(f"  Object: {e}")

# ── Save ──
os.makedirs(output_dir, exist_ok=True)
outfile = os.path.join(output_dir, f"{backend}_deep.json")
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outfile}")

env.close()
simulation_app.close()

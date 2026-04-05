#!/usr/bin/env python3
"""Sim2Sim: Fixed initial state comparison.

Overrides the random reset to use IDENTICAL initial joint positions,
then applies identical actions. This isolates solver dynamics from reset randomization.
"""

import json, os, sys
import numpy as np

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

import torch, warp as wp
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab.envs import ManagerBasedRLEnv

env_cfg, agent_cfg = resolve_task_config(
    "Isaac-Dexsuite-Kuka-Allegro-Lift-v0", "rsl_rl_cfg_entry_point")

env_cfg.scene.num_envs = 1
physics_type = type(env_cfg.sim.physics).__name__
backend = "newton" if "Newton" in physics_type else "physx"

# Remove startup DR only (keep reset events - they're needed by curriculum)
for attr_name in list(vars(env_cfg.events)):
    ev = getattr(env_cfg.events, attr_name)
    if hasattr(ev, 'mode') and getattr(ev, 'mode', None) == 'startup':
        print(f"  Removing startup DR: {attr_name}")
        delattr(env_cfg.events, attr_name)

env = ManagerBasedRLEnv(cfg=env_cfg)
robot = env.scene["robot"]

def to_np(x):
    if hasattr(x, 'cpu'): return x.cpu().numpy()
    try:
        if isinstance(x, wp.array): return wp.to_torch(x).cpu().numpy()
    except: pass
    return np.array(x)

joint_names = robot.joint_names
num_joints = len(joint_names)
action_dim = env.action_manager.total_action_dim

# Build name→index map for both backends  
# Use the USD-defined initial positions (same for both)
INIT_POS = {
    "iiwa7_joint_1": 0.0, "iiwa7_joint_2": 0.0, "iiwa7_joint_3": 0.7854,
    "iiwa7_joint_4": 1.5708, "iiwa7_joint_5": -1.5708, "iiwa7_joint_6": -1.5708,
    "iiwa7_joint_7": 0.0,
    "index_joint_0": 0.0, "index_joint_1": 0.3, "index_joint_2": 0.3, "index_joint_3": 0.3,
    "middle_joint_0": 0.0, "middle_joint_1": 0.3, "middle_joint_2": 0.3, "middle_joint_3": 0.3,
    "ring_joint_0": 0.0, "ring_joint_1": 0.3, "ring_joint_2": 0.3, "ring_joint_3": 0.3,
    "thumb_joint_0": 1.5, "thumb_joint_1": 0.60147215, "thumb_joint_2": 0.33795027,
    "thumb_joint_3": 0.60845138,
}

print(f"\nBackend: {backend}")
print(f"Joints: {num_joints}, Action dim: {action_dim}")

# Reset env 
obs, info = env.reset()

# Force identical initial state
init_pos_tensor = torch.zeros(1, num_joints, device=env.device)
for i, name in enumerate(joint_names):
    init_pos_tensor[0, i] = INIT_POS.get(name, 0.0)

robot.write_joint_state_to_sim(init_pos_tensor, torch.zeros_like(init_pos_tensor))

# Step once with zero action to let physics settle
obs, _, _, _, _ = env.step(torch.zeros(1, action_dim, device=env.device))

actual_pos = to_np(robot.data.joint_pos).flatten()
print("\nForced initial positions (after 1 settle step):")
for i, name in enumerate(joint_names):
    target = INIT_POS.get(name, 0.0)
    print(f"  {name:<20}: target={target:.4f}, actual={actual_pos[i]:.6f}, diff={actual_pos[i]-target:.6f}")

results = {"backend": backend, "joint_names": joint_names, "tests": {}}

# ── Test 1: Zero action from fixed state ──
print("\n=== Zero action from fixed state (50 steps) ===")
zero_traj = []
for step in range(50):
    pos = to_np(robot.data.joint_pos).flatten().copy()
    vel = to_np(robot.data.joint_vel).flatten().copy()
    obs, rew, _, _, _ = env.step(torch.zeros(1, action_dim, device=env.device))
    pos_after = to_np(robot.data.joint_pos).flatten().copy()
    vel_after = to_np(robot.data.joint_vel).flatten().copy()
    zero_traj.append({
        "step": step, "pos": pos.tolist(), "vel": vel.tolist(),
        "pos_after": pos_after.tolist(), "vel_after": vel_after.tolist()
    })
    if step < 5 or step % 10 == 0:
        drift = np.abs(pos_after - actual_pos).max()
        print(f"  Step {step:3d}: max_drift={drift:.6f}, vel_rms={np.sqrt((vel_after**2).mean()):.6f}")

results["tests"]["zero_action_fixed"] = zero_traj

# ── Test 2: Step response from fixed state ──
print("\n=== Step response: each arm joint pulsed separately ===")
# Reset to initial state
robot.write_joint_state_to_sim(init_pos_tensor, torch.zeros_like(init_pos_tensor))
obs, _, _, _, _ = env.step(torch.zeros(1, action_dim, device=env.device))

step_responses = {}
for j_idx in range(7):  # Arm joints only (iiwa7)
    j_name = joint_names[j_idx]
    print(f"\n  Joint {j_idx} ({j_name}):")
    
    # Reset
    robot.write_joint_state_to_sim(init_pos_tensor, torch.zeros_like(init_pos_tensor))
    obs, _, _, _, _ = env.step(torch.zeros(1, action_dim, device=env.device))
    
    pre_pos = to_np(robot.data.joint_pos).flatten()[j_idx]
    
    traj = []
    for step in range(40):
        action = torch.zeros(1, action_dim, device=env.device)
        if 5 <= step < 15:
            action[0, j_idx] = 1.0  # Full positive action
        
        obs, rew, _, _, _ = env.step(action)
        pos = to_np(robot.data.joint_pos).flatten()[j_idx]
        vel = to_np(robot.data.joint_vel).flatten()[j_idx]
        traj.append({"step": step, "pos": float(pos), "vel": float(vel)})
        
        if step < 5 or (5 <= step <= 20) or step == 39:
            act_val = 1.0 if 5 <= step < 15 else 0.0
            print(f"    Step {step:2d}: act={act_val:.1f}, pos={pos:.6f}, vel={vel:.6f}, delta={pos-pre_pos:.6f}")
    
    step_responses[j_name] = traj

results["tests"]["step_responses"] = step_responses

# ── Test 3: Finger pulse ──
print("\n=== Step response: thumb_joint_0 ===")
robot.write_joint_state_to_sim(init_pos_tensor, torch.zeros_like(init_pos_tensor))
obs, _, _, _, _ = env.step(torch.zeros(1, action_dim, device=env.device))

# Find thumb_joint_0 action index
thumb0_idx = joint_names.index("thumb_joint_0")
pre_pos = to_np(robot.data.joint_pos).flatten()[thumb0_idx]
print(f"  thumb_joint_0 at action index {thumb0_idx}, init pos: {pre_pos:.6f}")

thumb_traj = []
for step in range(40):
    action = torch.zeros(1, action_dim, device=env.device)
    if 5 <= step < 15:
        action[0, thumb0_idx] = 1.0
    obs, rew, _, _, _ = env.step(action)
    pos = to_np(robot.data.joint_pos).flatten()[thumb0_idx]
    vel = to_np(robot.data.joint_vel).flatten()[thumb0_idx]
    thumb_traj.append({"step": step, "pos": float(pos), "vel": float(vel)})
    if step < 5 or (5 <= step <= 20) or step == 39:
        act_val = 1.0 if 5 <= step < 15 else 0.0
        print(f"  Step {step:2d}: act={act_val:.1f}, pos={pos:.6f}, vel={vel:.6f}, delta={pos-pre_pos:.6f}")

results["tests"]["thumb_response"] = thumb_traj

# Save
os.makedirs(output_dir, exist_ok=True)
outfile = os.path.join(output_dir, f"{backend}_fixed.json")
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outfile}")

env.close()
simulation_app.close()

#!/usr/bin/env python3
"""Sim2Sim: Solver iteration sweep.

Tests PhysX step response at different solver iteration counts (4, 8, 16, 32)
vs Newton, for the same fixed initial state + same action pulse.

Goal: determine if high iteration count causes the over-damping on low-stiffness joints.
"""

import json, os, sys
import numpy as np

output_dir = "/tmp/sim2sim"
solver_iters = 32  # default, overridden by s2s_iters=N
backend_override = None

isaac_args = [sys.argv[0]]
for a in sys.argv[1:]:
    if a.startswith("s2s_output="):
        output_dir = a.split("=", 1)[1]
    elif a.startswith("s2s_iters="):
        solver_iters = int(a.split("=", 1)[1])
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

# Remove startup DR
for attr_name in list(vars(env_cfg.events)):
    ev = getattr(env_cfg.events, attr_name)
    if hasattr(ev, 'mode') and getattr(ev, 'mode', None) == 'startup':
        delattr(env_cfg.events, attr_name)

# Override solver iterations for PhysX
if backend == "physx":
    # Override robot articulation solver iterations
    robot_cfg = env_cfg.scene.robot
    if hasattr(robot_cfg, 'spawn') and hasattr(robot_cfg.spawn, 'articulation_props'):
        robot_cfg.spawn.articulation_props.solver_position_iteration_count = solver_iters
        robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = max(1, solver_iters // 32)
        print(f"\n*** PhysX solver iterations: position={solver_iters}, velocity={max(1, solver_iters // 32)} ***")

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

# Fixed initial state
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

init_pos_tensor = torch.zeros(1, num_joints, device=env.device)
for i, name in enumerate(joint_names):
    init_pos_tensor[0, i] = INIT_POS.get(name, 0.0)

print(f"\nBackend: {backend}, Solver iters: {solver_iters}")
print(f"Joints: {num_joints}, Action dim: {action_dim}")

# Test joints across stiffness range
test_joints = {
    "iiwa7_joint_1": {"stiffness": 300, "damping": 45},
    "iiwa7_joint_4": {"stiffness": 120, "damping": 30},
    "iiwa7_joint_5": {"stiffness": 100, "damping": 20},
    "iiwa7_joint_6": {"stiffness": 50, "damping": 15},
    "iiwa7_joint_7": {"stiffness": 25, "damping": 15},
    "index_joint_0": {"stiffness": 3, "damping": 0.1},
    "thumb_joint_0": {"stiffness": 3, "damping": 0.1},
}

results = {"backend": backend, "solver_iters": solver_iters, "joint_names": joint_names, "tests": {}}

obs, info = env.reset()

for jname, params in test_joints.items():
    j_idx = joint_names.index(jname)
    print(f"\n--- {jname} (stiffness={params['stiffness']}, damping={params['damping']}) ---")
    
    # Reset to fixed state
    robot.write_joint_state_to_sim(init_pos_tensor, torch.zeros_like(init_pos_tensor))
    obs, _, _, _, _ = env.step(torch.zeros(1, action_dim, device=env.device))
    
    traj = []
    for step in range(40):
        action = torch.zeros(1, action_dim, device=env.device)
        if 5 <= step < 15:
            action[0, j_idx] = 1.0
        
        obs, rew, _, _, _ = env.step(action)
        pos = float(to_np(robot.data.joint_pos).flatten()[j_idx])
        vel = float(to_np(robot.data.joint_vel).flatten()[j_idx])
        traj.append({"step": step, "pos": pos, "vel": vel})
        
        if step == 5:  # First action step
            print(f"  First response: pos={pos:.6f}, vel={vel:.6f}")
        elif step == 14:  # Last action step
            print(f"  After 10 pulses: pos={pos:.6f}, vel={vel:.6f}")
        elif step == 39:
            print(f"  Settled: pos={pos:.6f}, vel={vel:.6f}")
    
    results["tests"][jname] = traj

# Save
os.makedirs(output_dir, exist_ok=True)
outfile = os.path.join(output_dir, f"{backend}_iters{solver_iters}.json")
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outfile}")

env.close()
simulation_app.close()

#!/usr/bin/env python3
"""Compare PhysX and Newton trajectories from param_compare.py output."""

import json
import numpy as np
import sys

def main():
    physx_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sim2sim/physx_params.json"
    newton_file = sys.argv[2] if len(sys.argv) > 2 else "/tmp/sim2sim/newton_params.json"

    with open(physx_file) as f:
        physx = json.load(f)
    with open(newton_file) as f:
        newton = json.load(f)

    print("=" * 70)
    print("  Sim2Sim Trajectory Comparison: PhysX vs Newton")
    print("=" * 70)

    # Joint ordering comparison
    print("\n1. JOINT ORDERING")
    print("-" * 70)
    print(f"PhysX joints: {physx['joint_names']}")
    print(f"Newton joints: {newton['joint_names']}")

    if physx['joint_names'] == newton['joint_names']:
        print("✓ Joint ordering MATCHES")
    else:
        print("✗ Joint ordering DIFFERS!")
        # Find the mapping
        physx_to_newton = []
        for i, pname in enumerate(physx['joint_names']):
            try:
                newton_idx = newton['joint_names'].index(pname)
                physx_to_newton.append(newton_idx)
            except ValueError:
                physx_to_newton.append(-1)
        print(f"  PhysX→Newton index mapping: {physx_to_newton}")

    # Actuator parameters comparison
    print("\n2. ACTUATOR PARAMETERS (from ImplicitActuatorCfg)")
    print("-" * 70)
    physx_act = physx['actuator_params']['kuka_allegro_actuators']
    newton_act = newton['actuator_params']['kuka_allegro_actuators']

    for param in ['stiffness', 'damping', 'friction', 'armature']:
        p_vals = np.array(physx_act.get(param, [[]])[0])
        n_vals = np.array(newton_act.get(param, [[]])[0])

        # Reorder Newton to PhysX joint order for comparison
        if physx['joint_names'] != newton['joint_names']:
            n_vals_reordered = np.array([n_vals[physx_to_newton[i]] for i in range(len(p_vals))])
        else:
            n_vals_reordered = n_vals

        diff = np.abs(p_vals - n_vals_reordered)
        max_diff = diff.max()
        if max_diff < 1e-6:
            print(f"  {param:12s}: ✓ MATCH")
        else:
            print(f"  {param:12s}: ✗ DIFFERS (max diff: {max_diff:.6f})")
            for i, (p, n) in enumerate(zip(p_vals, n_vals_reordered)):
                if abs(p - n) > 1e-6:
                    print(f"    [{i}] {physx['joint_names'][i]}: PhysX={p:.4f}, Newton={n:.4f}")

    # Trajectory comparison
    print("\n3. TRAJECTORY COMPARISON (same actions)")
    print("-" * 70)

    physx_traj = physx['trajectory']
    newton_traj = newton['trajectory']

    if len(physx_traj) != len(newton_traj):
        print(f"  WARNING: Different trajectory lengths! PhysX={len(physx_traj)}, Newton={len(newton_traj)}")

    num_steps = min(len(physx_traj), len(newton_traj))

    # Compute per-step differences
    pos_diffs = []
    vel_diffs = []
    reward_diffs = []

    for i in range(num_steps):
        p_pos = np.array(physx_traj[i]['joint_pos_after'][0])
        n_pos = np.array(newton_traj[i]['joint_pos_after'][0])

        p_vel = np.array(physx_traj[i]['joint_vel_after'][0])
        n_vel = np.array(newton_traj[i]['joint_vel_after'][0])

        # Reorder Newton to PhysX joint order
        if physx['joint_names'] != newton['joint_names']:
            n_pos = np.array([n_pos[physx_to_newton[j]] for j in range(len(p_pos))])
            n_vel = np.array([n_vel[physx_to_newton[j]] for j in range(len(p_vel))])

        pos_diffs.append(np.abs(p_pos - n_pos))
        vel_diffs.append(np.abs(p_vel - n_vel))
        reward_diffs.append(abs(physx_traj[i]['reward'] - newton_traj[i]['reward']))

    pos_diffs = np.array(pos_diffs)  # [steps, joints]
    vel_diffs = np.array(vel_diffs)

    print(f"  Steps analyzed: {num_steps}")
    print(f"\n  Position differences (rad):")
    print(f"    Mean: {pos_diffs.mean():.6f}")
    print(f"    Max:  {pos_diffs.max():.6f}")
    print(f"    Per-joint mean: {pos_diffs.mean(axis=0)}")

    print(f"\n  Velocity differences (rad/s):")
    print(f"    Mean: {vel_diffs.mean():.6f}")
    print(f"    Max:  {vel_diffs.max():.6f}")
    print(f"    Per-joint mean: {vel_diffs.mean(axis=0)}")

    print(f"\n  Reward differences:")
    print(f"    Mean: {np.mean(reward_diffs):.6f}")
    print(f"    Max:  {np.max(reward_diffs):.6f}")

    # Per-joint breakdown for arm vs hand
    arm_joints = [i for i, n in enumerate(physx['joint_names']) if 'iiwa' in n]
    hand_joints = [i for i, n in enumerate(physx['joint_names']) if 'iiwa' not in n]

    print(f"\n  Arm joints (iiwa7) position error:")
    print(f"    Mean: {pos_diffs[:, arm_joints].mean():.6f}")
    print(f"    Max:  {pos_diffs[:, arm_joints].max():.6f}")

    print(f"\n  Hand joints (Allegro) position error:")
    print(f"    Mean: {pos_diffs[:, hand_joints].mean():.6f}")
    print(f"    Max:  {pos_diffs[:, hand_joints].max():.6f}")

    # Time evolution of error
    print("\n4. ERROR EVOLUTION OVER TIME")
    print("-" * 70)
    checkpoints = [0, 10, 50, 100, 150, 199]
    for step in checkpoints:
        if step < num_steps:
            print(f"  Step {step:3d}: pos_err={pos_diffs[step].mean():.6f}, vel_err={vel_diffs[step].mean():.6f}")

    # Check if errors grow (diverging trajectories)
    early_err = pos_diffs[:20].mean()
    late_err = pos_diffs[-20:].mean()
    print(f"\n  Early (0-20) avg pos error: {early_err:.6f}")
    print(f"  Late (180-200) avg pos error: {late_err:.6f}")
    if late_err > early_err * 2:
        print(f"  ⚠ Trajectories are DIVERGING (late error {late_err/early_err:.1f}x early)")
    else:
        print(f"  ✓ Trajectories stay relatively aligned")

    # Joint-level analysis
    print("\n5. PER-JOINT ANALYSIS (sorted by mean position error)")
    print("-" * 70)
    joint_errors = [(physx['joint_names'][i], pos_diffs[:, i].mean(), vel_diffs[:, i].mean())
                    for i in range(len(physx['joint_names']))]
    joint_errors.sort(key=lambda x: -x[1])  # Sort by position error descending

    print(f"{'Joint':<20} {'Pos Error':>12} {'Vel Error':>12}")
    for name, pos_err, vel_err in joint_errors:
        print(f"{name:<20} {pos_err:>12.6f} {vel_err:>12.6f}")


if __name__ == "__main__":
    main()

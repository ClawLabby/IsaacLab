"""Render trajectory comparison using matplotlib 3D skeleton visualization.

No Isaac Sim rendering needed — just plots joint positions as stick figures.
Much more reliable than viewport capture.
"""
import json, numpy as np, os, sys

# DexSuite Kuka-Allegro forward kinematics (simplified)
# We'll use the recorded joint positions to compute approximate end-effector positions
# via a simplified DH model, then plot stick figures

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

PHYSX_JOINT_ORDER = [
    "iiwa7_joint_1", "iiwa7_joint_2", "iiwa7_joint_3",
    "iiwa7_joint_4", "iiwa7_joint_5", "iiwa7_joint_6", "iiwa7_joint_7",
    "index_joint_0", "middle_joint_0", "ring_joint_0", "thumb_joint_0",
    "index_joint_1", "middle_joint_1", "ring_joint_1", "thumb_joint_1",
    "index_joint_2", "middle_joint_2", "ring_joint_2", "thumb_joint_2",
    "index_joint_3", "middle_joint_3", "ring_joint_3", "thumb_joint_3",
]

def plot_joint_comparison(data_files, labels, output, max_steps=100):
    """Create a multi-panel time-series plot of joint positions."""
    
    all_data = []
    for fpath in data_files:
        with open(fpath) as f:
            data = json.load(f)
        positions = np.array([s['joint_pos'] for s in data['steps']])
        all_data.append(positions)
    
    n_steps = min(max_steps, min(d.shape[0] for d in all_data))
    n_joints = all_data[0].shape[1]
    
    # Create figure with subplots for each joint group
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
    fig.suptitle('Sim2Sim Joint Position Comparison\n(Identical Initial State + Joint Targets)', fontsize=14)
    
    time = np.arange(n_steps) / 60.0  # DexSuite runs at ~60Hz env steps
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Panel 1: Arm joints (0-6)
    ax = axes[0]
    ax.set_title('Arm Joints (iiwa7)', fontsize=12)
    arm_names_short = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']
    for ji in range(7):
        for di, (data, label) in enumerate(zip(all_data, labels)):
            ls = ['-', '--', ':'][di]
            ax.plot(time[:n_steps], data[:n_steps, ji], ls=ls, color=f'C{ji}', 
                    alpha=0.8, label=f'{arm_names_short[ji]} ({label})' if di == 0 else f'_{arm_names_short[ji]} ({label})')
    ax.set_ylabel('Position (rad)')
    ax.legend(ncol=7, fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Arm joint differences (PhysX standalone vs Newton standalone)
    ax = axes[1]
    ax.set_title('Arm Joint Differences (Standalone PhysX - Newton)', fontsize=12)
    if len(all_data) >= 3:
        # Index 1 = standalone PhysX, Index 2 = standalone Newton
        for ji in range(7):
            diff = all_data[1][:n_steps, ji] - all_data[2][:n_steps, ji]
            ax.plot(time[:n_steps], np.degrees(diff), label=arm_names_short[ji])
    ax.set_ylabel('Difference (degrees)')
    ax.legend(ncol=7, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    
    # Panel 3: Finger joints (worst diverging)
    ax = axes[2]
    ax.set_title('Finger Joints (Selected — Largest PhysX↔Newton Gaps)', fontsize=12)
    # Find worst finger joints by max PhysX-Newton diff
    if len(all_data) >= 3:
        finger_diffs = np.abs(all_data[1][:n_steps, 7:] - all_data[2][:n_steps, 7:]).max(axis=0)
        worst_fingers = np.argsort(finger_diffs)[-6:][::-1]  # top 6
        for rank, fi in enumerate(worst_fingers):
            ji = fi + 7  # global joint index
            for di, (data, label) in enumerate(zip(all_data, labels)):
                ls = ['-', '--', ':'][di]
                lbl = f'{PHYSX_JOINT_ORDER[ji]} ({label})' if di == 0 else f'_{PHYSX_JOINT_ORDER[ji]} ({label})'
                ax.plot(time[:n_steps], data[:n_steps, ji], ls=ls, color=f'C{rank}', alpha=0.8, label=lbl)
    ax.set_ylabel('Position (rad)')
    ax.legend(ncol=3, fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Finger joint differences
    ax = axes[3]
    ax.set_title('Finger Joint Differences (Standalone PhysX - Newton)', fontsize=12)
    if len(all_data) >= 3:
        for rank, fi in enumerate(worst_fingers):
            ji = fi + 7
            diff = all_data[1][:n_steps, ji] - all_data[2][:n_steps, ji]
            ax.plot(time[:n_steps], np.degrees(diff), label=PHYSX_JOINT_ORDER[ji], color=f'C{rank}')
    ax.set_ylabel('Difference (degrees)')
    ax.set_xlabel('Time (seconds)')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output}")
    
    # Also save a focused view of just the PhysX vs Newton difference
    fig2, axes2 = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig2.suptitle('Physics Gap: Standalone PhysX vs Newton\n(Same initial state, same joint targets, same table)', fontsize=14)
    
    if len(all_data) >= 3:
        ax = axes2[0]
        ax.set_title('All 23 Joint Position Differences', fontsize=12)
        for ji in range(23):
            diff = np.degrees(all_data[1][:n_steps, ji] - all_data[2][:n_steps, ji])
            name = PHYSX_JOINT_ORDER[ji]
            color = 'C0' if ji < 7 else 'C1'  # blue=arm, orange=hand
            alpha = 0.9 if ji < 7 else 0.4
            lw = 1.5 if ji < 7 else 0.8
            ax.plot(time[:n_steps], diff, color=color, alpha=alpha, lw=lw,
                    label=name if ji in [0,1,6] or ji == 7 else None)
        ax.set_ylabel('PhysX - Newton (degrees)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', lw=0.5)
        
        ax = axes2[1]
        ax.set_title('Absolute Difference Per Joint Over Time', fontsize=12)
        abs_diff = np.degrees(np.abs(all_data[1][:n_steps] - all_data[2][:n_steps]))
        # Heatmap
        im = ax.imshow(abs_diff.T, aspect='auto', cmap='hot_r',
                       extent=[0, n_steps/60, 22.5, -0.5],
                       vmin=0, vmax=30)
        ax.set_yticks(range(23))
        ax.set_yticklabels([n.replace('_joint_', '') for n in PHYSX_JOINT_ORDER], fontsize=7)
        ax.set_xlabel('Time (seconds)')
        plt.colorbar(im, ax=ax, label='|PhysX - Newton| (degrees)')
    
    plt.tight_layout()
    output2 = output.replace('.png', '_heatmap.png')
    plt.savefig(output2, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output2}")


if __name__ == "__main__":
    plot_joint_comparison(
        data_files=[
            "/tmp/controlled_dexsuite_physx.json",
            "/tmp/controlled_standalone_physx_table.json",
            "/tmp/controlled_standalone_newton_table.json",
        ],
        labels=["DexSuite PhysX", "Standalone PhysX", "Standalone Newton"],
        output="/home/horde/.openclaw/workspace/sim2sim_joints.png",
    )

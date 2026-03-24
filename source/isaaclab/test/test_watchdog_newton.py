#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Test NaN watchdog with Newton physics state injection.

Creates a Newton simulation, runs clean steps, then injects NaN directly
into physics state buffers. Verifies:
1. The watchdog detects NaN in observations derived from corrupted state
2. The dump contains Newton state data sufficient to identify the NaN source
3. The dump can be loaded and inspected to trace which body/joint went NaN

This tests the full detection→dump→debug pipeline without relying on
the solver producing NaN naturally (which is rare with MuJoCo Warp).
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import warp as wp
import newton

DUMP_DIR = tempfile.mkdtemp(prefix="watchdog_newton_inject_")


def test_nan_injection_body_state():
    """Inject NaN into body positions and verify the dump traces back to the source."""
    from isaaclab.utils.nan_watchdog import NaNWatchdog

    # Build a simple scene using Newton API
    builder = newton.ModelBuilder()
    # Free-floating box (body 0)
    b0 = builder.add_body(mass=1.0, xform=wp.transform((0, 1.0, 0), wp.quat_identity()))
    builder.add_shape_box(b0, hx=0.1, hy=0.1, hz=0.1)
    # Arm link 1 (body 1) — revolute joint
    b1 = builder.add_body(mass=0.5, xform=wp.transform((0.5, 0, 0), wp.quat_identity()))
    builder.add_shape_box(b1, hx=0.05, hy=0.2, hz=0.05)
    builder.add_joint_revolute(parent=-1, child=b1, axis=(0, 0, 1))
    # Arm link 2 (body 2)
    b2 = builder.add_body(mass=0.3, xform=wp.transform((1.0, 0, 0), wp.quat_identity()))
    builder.add_shape_box(b2, hx=0.05, hy=0.15, hz=0.05)
    builder.add_joint_revolute(parent=b1, child=b2, axis=(0, 0, 1))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = builder.finalize(device=device)
    state = model.state()

    print(f"Model: {model.body_count} bodies, {model.joint_count} joints, device={device}")

    dump_dir = os.path.join(DUMP_DIR, "body_inject")
    watchdog = NaNWatchdog(dump_dir=dump_dir, max_dumps=3)

    # Run 5 clean steps — build obs from flattened body_q
    for step in range(5):
        body_q_np = state.body_q.numpy()
        body_q_flat = torch.from_numpy(body_q_np).reshape(1, -1).float()
        joint_q_flat = torch.from_numpy(state.joint_q.numpy()).reshape(1, -1).float()
        obs = {"policy": body_q_flat, "joints": joint_q_flat}
        nan_envs = watchdog.check(obs, torch.zeros(1), torch.zeros(1, dtype=torch.bool), step=step)
        assert len(nan_envs) == 0, f"False positive at step {step}"

    # Inject NaN into body_q — corrupt body 1's position
    body_q_np = state.body_q.numpy()  # shape: (num_bodies, 7) — [px,py,pz,qx,qy,qz,qw]
    body_q_np[1, 0] = float("nan")   # Body 1 x position
    body_q_np[1, 1] = float("nan")   # Body 1 y position
    state.body_q.assign(wp.array(body_q_np, dtype=state.body_q.dtype, device=device))

    # Build observations from corrupted state
    body_q_flat = torch.from_numpy(state.body_q.numpy()).reshape(1, -1).float()
    joint_q_flat = torch.from_numpy(state.joint_q.numpy()).reshape(1, -1).float()
    obs = {"policy": body_q_flat, "joints": joint_q_flat}

    nan_envs = watchdog.check(obs, torch.zeros(1), torch.zeros(1, dtype=torch.bool), step=10)
    assert len(nan_envs) > 0, "Should detect NaN from corrupted body state"
    print(f"✓ NaN detected from body state injection (envs: {nan_envs})")

    # Verify dump
    dump_dirs = sorted([d for d in os.listdir(dump_dir) if d.startswith("nan_step")])
    assert len(dump_dirs) >= 1, f"Expected dump, found: {dump_dirs}"
    dump_path = os.path.join(dump_dir, dump_dirs[0])
    print(f"  Dump contents: {os.listdir(dump_path)}")

    with open(os.path.join(dump_path, "nan_info.json")) as f:
        nan_info = json.load(f)
    assert nan_info["step"] == 10
    assert "obs.policy" in nan_info["nan_locations"]
    print(f"  ✓ nan_info.json: step={nan_info['step']}, locations={list(nan_info['nan_locations'].keys())}")

    # Load RL tensors and trace NaN to specific features
    rl_data = torch.load(os.path.join(dump_path, "rl_tensors.pt"), weights_only=False)
    policy_obs = rl_data["observations"]["policy"]
    nan_features = policy_obs.isnan().nonzero(as_tuple=True)[1].tolist()
    print(f"  ✓ NaN at policy feature indices: {nan_features}")

    # Features 7,8 = body 1's x,y position in the flattened body_q (3 bodies × 7 = 21 total)
    assert any(f in [7, 8] for f in nan_features), (
        f"NaN should include body 1 position (indices 7,8 in flat body_q), got: {nan_features}"
    )
    print(f"  ✓ NaN traces to body 1 position")
    print(f"\n✓ Body state NaN injection test PASSED")


def test_nan_injection_joint_state():
    """Inject NaN into joint velocities and verify detection."""
    from isaaclab.utils.nan_watchdog import NaNWatchdog

    builder = newton.ModelBuilder()
    b0 = builder.add_body(mass=1.0, xform=wp.transform((0, 0, 0), wp.quat_identity()))
    builder.add_shape_box(b0, hx=0.1, hy=0.1, hz=0.1)
    b1 = builder.add_body(mass=0.5, xform=wp.transform((0.3, 0, 0), wp.quat_identity()))
    builder.add_shape_box(b1, hx=0.05, hy=0.15, hz=0.05)
    builder.add_joint_revolute(parent=-1, child=b1, axis=(0, 0, 1))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = builder.finalize(device=device)
    state = model.state()

    dump_dir = os.path.join(DUMP_DIR, "joint_inject")
    watchdog = NaNWatchdog(dump_dir=dump_dir, max_dumps=1)

    # Inject NaN into joint velocity
    if state.joint_qd is not None and state.joint_qd.shape[0] > 0:
        joint_qd_np = state.joint_qd.numpy()
        joint_qd_np.reshape(-1)[0] = float("nan")
        state.joint_qd.assign(wp.array(joint_qd_np, dtype=state.joint_qd.dtype, device=device))

        joint_qd_flat = torch.from_numpy(state.joint_qd.numpy()).reshape(1, -1).float()
        obs = {"joint_vel": joint_qd_flat}
        nan_envs = watchdog.check(obs, torch.zeros(1), torch.zeros(1, dtype=torch.bool), step=0)
        assert len(nan_envs) > 0, "Should detect NaN from joint velocity injection"
        print(f"✓ Joint velocity NaN injection detected (envs: {nan_envs})")

        dump_dirs = [d for d in os.listdir(dump_dir) if d.startswith("nan_step")]
        assert len(dump_dirs) >= 1
        dump_path = os.path.join(dump_dir, dump_dirs[0])
        with open(os.path.join(dump_path, "nan_info.json")) as f:
            nan_info = json.load(f)
        assert "obs.joint_vel" in nan_info["nan_locations"]
        print(f"  ✓ NaN location traced to obs.joint_vel")
    else:
        print("⚠️  No joint velocities available, skipping joint injection test")

    print(f"✓ Joint state NaN injection test PASSED")


def test_nan_in_rewards():
    """Inject NaN into rewards tensor and verify detection + dump."""
    from isaaclab.utils.nan_watchdog import NaNWatchdog

    dump_dir = os.path.join(DUMP_DIR, "reward_inject")
    watchdog = NaNWatchdog(dump_dir=dump_dir, max_dumps=1)

    obs = {"policy": torch.randn(8, 20)}
    rewards = torch.randn(8)
    rewards[3] = float("nan")  # Corrupt env 3's reward
    dones = torch.zeros(8, dtype=torch.bool)

    nan_envs = watchdog.check(obs, rewards, dones, step=42)
    assert len(nan_envs) > 0, "Should detect NaN in rewards"
    assert 3 in nan_envs, f"Env 3 should be affected: {nan_envs}"
    print(f"✓ Reward NaN detected for env 3")

    # Verify dump traces to rewards
    dump_dirs = [d for d in os.listdir(dump_dir) if d.startswith("nan_step")]
    assert len(dump_dirs) >= 1
    dump_path = os.path.join(dump_dir, dump_dirs[0])
    with open(os.path.join(dump_path, "nan_info.json")) as f:
        nan_info = json.load(f)
    assert "rewards" in nan_info["nan_locations"]
    assert 3 in nan_info["nan_locations"]["rewards"]["env_indices"]
    print(f"  ✓ Dump traces NaN to rewards, env 3")
    print(f"✓ Reward NaN injection test PASSED")


if __name__ == "__main__":
    try:
        test_nan_injection_body_state()
        print()
        test_nan_injection_joint_state()
        print()
        test_nan_in_rewards()
        print()
        print("=" * 60)
        print("✓ All NaN injection tests PASSED")
        print(f"  Dumps at: {DUMP_DIR}")
        print("=" * 60)
    finally:
        if "--keep" not in sys.argv:
            shutil.rmtree(DUMP_DIR, ignore_errors=True)
        else:
            print(f"\nDumps preserved at: {DUMP_DIR}")

#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Test NaN watchdog dump traceability — can we identify the source of NaN?

This test validates that the NaN watchdog dumps contain enough information
to debug the root cause of a NaN event:

1. Which observation groups contain NaN
2. Which specific environment indices are affected
3. Which features within the observation are NaN (to trace back to physics state)
4. The RL tensors (obs, rewards, actions) are fully captured for replay

The key question: if you get a NaN dump, can you figure out what went wrong?
"""

import json
import os
import shutil
import sys
import tempfile

import torch

DUMP_DIR = tempfile.mkdtemp(prefix="watchdog_trace_test_")


def test_multi_group_nan_tracing():
    """NaN in one obs group should NOT appear in other clean groups in the dump."""
    from isaaclab.utils.nan_watchdog import NaNWatchdog

    dump_dir = os.path.join(DUMP_DIR, "multi_group")
    watchdog = NaNWatchdog(dump_dir=dump_dir, max_dumps=1)

    num_envs = 16
    obs = {
        "policy": torch.randn(num_envs, 34),       # Object pose, target, actions
        "proprio": torch.randn(num_envs, 123),     # Joint pos/vel, fingertips, contacts
        "perception": torch.randn(num_envs, 192),  # Point cloud
    }

    # Corrupt only the proprio group for envs 5 and 11
    obs["proprio"][5, 23:46] = float("nan")    # Joint velocities go NaN
    obs["proprio"][11, 100:110] = float("nan") # Fingertip state goes NaN

    rewards = torch.randn(num_envs)
    dones = torch.zeros(num_envs, dtype=torch.bool)
    actions = torch.randn(num_envs, 23)

    nan_envs = watchdog.check(obs, rewards, dones, actions, step=100)

    # Should detect exactly envs 5 and 11
    assert sorted(nan_envs) == [5, 11], f"Expected [5, 11], got {nan_envs}"
    print(f"✓ Detected NaN in exactly envs {nan_envs}")

    # Load the dump and verify traceability
    dump_dirs = sorted([d for d in os.listdir(dump_dir) if d.startswith("nan_step")])
    assert len(dump_dirs) == 1
    dump_path = os.path.join(dump_dir, dump_dirs[0])

    # 1. Check nan_info.json — should identify proprio, not policy or perception
    with open(os.path.join(dump_path, "nan_info.json")) as f:
        info = json.load(f)

    assert "obs.proprio" in info["nan_locations"], f"Should trace to proprio: {info['nan_locations']}"
    assert "obs.policy" not in info["nan_locations"], "Policy should be clean"
    assert "obs.perception" not in info["nan_locations"], "Perception should be clean"

    proprio_info = info["nan_locations"]["obs.proprio"]
    assert sorted(proprio_info["env_indices"]) == [5, 11], (
        f"Proprio NaN should be in envs 5,11: {proprio_info['env_indices']}"
    )
    print(f"  ✓ NaN correctly isolated to obs.proprio, envs {proprio_info['env_indices']}")

    # 2. Load RL tensors and verify we can reconstruct the NaN pattern
    rl_data = torch.load(os.path.join(dump_path, "rl_tensors.pt"), weights_only=False)

    # Proprio should have NaN
    proprio_dump = rl_data["observations"]["proprio"]
    assert proprio_dump[5].isnan().any(), "Env 5 proprio should have NaN in dump"
    assert proprio_dump[11].isnan().any(), "Env 11 proprio should have NaN in dump"

    # Policy and perception should be clean
    assert not rl_data["observations"]["policy"].isnan().any(), "Policy should be clean in dump"
    assert not rl_data["observations"]["perception"].isnan().any(), "Perception should be clean in dump"

    # 3. Verify we can identify WHICH features are NaN (for physics debugging)
    env5_nan_features = proprio_dump[5].isnan().nonzero(as_tuple=True)[0].tolist()
    env11_nan_features = proprio_dump[11].isnan().nonzero(as_tuple=True)[0].tolist()
    print(f"  ✓ Env 5: NaN at proprio features {env5_nan_features}")
    print(f"  ✓ Env 11: NaN at proprio features {env11_nan_features}")

    # Features 23-45 = joint velocities, 100-109 = fingertip state
    assert all(23 <= f <= 45 for f in env5_nan_features), (
        f"Env 5 NaN should be in joint vel range [23,45]: {env5_nan_features}"
    )
    assert all(100 <= f <= 109 for f in env11_nan_features), (
        f"Env 11 NaN should be in fingertip range [100,109]: {env11_nan_features}"
    )
    print(f"  ✓ NaN feature ranges match injection: joint_vel=[23,45], fingertip=[100,109]")

    # 4. Actions should be fully captured (needed for replay)
    assert rl_data["actions"] is not None, "Actions should be in dump"
    assert rl_data["actions"].shape == (num_envs, 23), f"Wrong actions shape: {rl_data['actions'].shape}"
    assert not rl_data["actions"].isnan().any(), "Actions should be clean (NaN was in obs, not actions)"
    print(f"  ✓ Actions captured: shape {rl_data['actions'].shape}, clean")

    print(f"\n✓ Multi-group NaN tracing test PASSED")


def test_cascading_nan_detection():
    """If NaN propagates from obs to rewards in the same step, both should be in the dump."""
    from isaaclab.utils.nan_watchdog import NaNWatchdog

    dump_dir = os.path.join(DUMP_DIR, "cascade")
    watchdog = NaNWatchdog(dump_dir=dump_dir, max_dumps=1)

    num_envs = 8
    obs = {"policy": torch.randn(num_envs, 10)}
    rewards = torch.randn(num_envs)

    # Simulate cascade: env 2 has NaN in both obs AND rewards
    # (this happens when a reward function uses the NaN observation)
    obs["policy"][2, 3] = float("nan")
    rewards[2] = float("nan")

    nan_envs = watchdog.check(obs, rewards, torch.zeros(num_envs, dtype=torch.bool), step=50)
    assert 2 in nan_envs

    dump_dirs = sorted([d for d in os.listdir(dump_dir) if d.startswith("nan_step")])
    dump_path = os.path.join(dump_dir, dump_dirs[0])

    with open(os.path.join(dump_path, "nan_info.json")) as f:
        info = json.load(f)

    # Both obs and rewards should be flagged
    assert "obs.policy" in info["nan_locations"], "Obs NaN should be recorded"
    assert "rewards" in info["nan_locations"], "Reward NaN should be recorded"
    assert 2 in info["nan_locations"]["obs.policy"]["env_indices"]
    assert 2 in info["nan_locations"]["rewards"]["env_indices"]
    print(f"✓ Cascading NaN: both obs and rewards flagged for env 2")
    print(f"  obs.policy NaN count: {info['nan_locations']['obs.policy']['count']}")
    print(f"  rewards NaN count: {info['nan_locations']['rewards']['count']}")
    print(f"\n✓ Cascading NaN detection test PASSED")


def test_dump_sufficient_for_replay():
    """Verify the dump contains everything needed to replay the failure."""
    from isaaclab.utils.nan_watchdog import NaNWatchdog

    dump_dir = os.path.join(DUMP_DIR, "replay")
    watchdog = NaNWatchdog(dump_dir=dump_dir, max_dumps=1)

    num_envs = 4
    obs = {
        "policy": torch.randn(num_envs, 34),
        "proprio": torch.randn(num_envs, 123),
    }
    rewards = torch.randn(num_envs)
    dones = torch.zeros(num_envs, dtype=torch.bool)
    dones[1] = True  # Env 1 was already done
    actions = torch.randn(num_envs, 23)

    # Inject NaN
    obs["policy"][0, 5] = float("nan")

    watchdog.check(obs, rewards, dones, actions, step=999)

    dump_dirs = sorted([d for d in os.listdir(dump_dir) if d.startswith("nan_step")])
    dump_path = os.path.join(dump_dir, dump_dirs[0])

    # Load everything
    with open(os.path.join(dump_path, "nan_info.json")) as f:
        info = json.load(f)
    rl_data = torch.load(os.path.join(dump_path, "rl_tensors.pt"), weights_only=False)

    # Verify completeness for replay
    assert info["step"] == 999, f"Step should be 999: {info['step']}"
    assert "timestamp" in info, "Timestamp needed for log correlation"
    assert rl_data["observations"]["policy"].shape == (num_envs, 34)
    assert rl_data["observations"]["proprio"].shape == (num_envs, 123)
    assert rl_data["rewards"].shape == (num_envs,)
    assert rl_data["dones"].shape == (num_envs,)
    assert rl_data["dones"][1] == True, "Done state should be preserved"
    assert rl_data["actions"].shape == (num_envs, 23)
    print(f"✓ Dump at step {info['step']} has complete replay data:")
    print(f"  observations: {list(rl_data['observations'].keys())}")
    print(f"  rewards: {rl_data['rewards'].shape}")
    print(f"  dones: {rl_data['dones'].shape} (env 1 done={rl_data['dones'][1].item()})")
    print(f"  actions: {rl_data['actions'].shape}")
    print(f"  timestamp: {info['timestamp']}")
    print(f"\n✓ Dump replay completeness test PASSED")


if __name__ == "__main__":
    try:
        test_multi_group_nan_tracing()
        print()
        test_cascading_nan_detection()
        print()
        test_dump_sufficient_for_replay()
        print()
        print("=" * 60)
        print("✓ All NaN traceability tests PASSED")
        print(f"  Dumps at: {DUMP_DIR}")
        print("=" * 60)
    finally:
        if "--keep" not in sys.argv:
            shutil.rmtree(DUMP_DIR, ignore_errors=True)
        else:
            print(f"\nDumps preserved at: {DUMP_DIR}")

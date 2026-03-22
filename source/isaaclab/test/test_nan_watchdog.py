"""Tests for NaN watchdog.

Run: python source/isaaclab/test/test_nan_watchdog.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import torch


def test_nan_watchdog_detects_and_dumps():
    """NaN watchdog should detect injected NaN and produce a valid dump."""
    from isaaclab.utils.nan_watchdog import NaNWatchdog

    dump_dir = tempfile.mkdtemp(prefix="nan_watchdog_test_")
    try:
        watchdog = NaNWatchdog(dump_dir=dump_dir, max_dumps=2)

        # 10 clean steps
        for step in range(10):
            obs = {"policy": torch.randn(4, 10)}
            nan_envs = watchdog.check(obs, torch.randn(4), torch.zeros(4, dtype=torch.bool), step=step)
            assert len(nan_envs) == 0, f"False positive at step {step}"

        # Inject NaN
        obs_nan = {"policy": torch.randn(4, 10)}
        obs_nan["policy"][2, 5] = float('nan')
        nan_envs = watchdog.check(obs_nan, torch.randn(4), torch.zeros(4, dtype=torch.bool), step=10)
        assert len(nan_envs) > 0, "Should detect NaN"
        assert 2 in nan_envs, f"Env 2 should be affected: {nan_envs}"

        # Verify dump
        dump_dirs = [d for d in os.listdir(dump_dir) if d.startswith("nan_step")]
        assert len(dump_dirs) == 1
        dump_path = os.path.join(dump_dir, dump_dirs[0])

        with open(os.path.join(dump_path, "nan_info.json")) as f:
            info = json.load(f)
        assert info["step"] == 10
        assert 2 in info["nan_locations"]["obs.policy"]["env_indices"]

        tensors = torch.load(os.path.join(dump_path, "rl_tensors.pt"), weights_only=False)
        assert tensors["observations"]["policy"].isnan().any()

        print(f"✓ Detection + dump test passed. Contents: {os.listdir(dump_path)}")
    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


def test_nan_watchdog_max_dumps():
    """Should stop dumping after max_dumps but still detect."""
    from isaaclab.utils.nan_watchdog import NaNWatchdog

    dump_dir = tempfile.mkdtemp(prefix="nan_watchdog_max_")
    try:
        watchdog = NaNWatchdog(dump_dir=dump_dir, max_dumps=2)
        obs_nan = {"policy": torch.tensor([[float('nan')]])}
        rewards = torch.zeros(1)
        dones = torch.zeros(1, dtype=torch.bool)

        assert len(watchdog.check(obs_nan, rewards, dones, step=1)) > 0
        assert len(watchdog.check(obs_nan, rewards, dones, step=2)) > 0
        assert len(watchdog.check(obs_nan, rewards, dones, step=3)) > 0  # Still detects

        dumps = [d for d in os.listdir(dump_dir) if d.startswith("nan_step")]
        assert len(dumps) == 2, f"Should have 2 dumps, got {len(dumps)}"
        print("✓ max_dumps test passed")
    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


def test_nan_watchdog_identifies_envs():
    """Should correctly identify affected environments."""
    from isaaclab.utils.nan_watchdog import NaNWatchdog

    dump_dir = tempfile.mkdtemp(prefix="nan_watchdog_envs_")
    try:
        watchdog = NaNWatchdog(dump_dir=dump_dir, max_dumps=1)

        obs = {"policy": torch.randn(10, 20)}
        obs["policy"][0, 5] = float('nan')
        obs["policy"][3, 10] = float('nan')
        obs["policy"][7, :] = float('nan')

        nan_envs = watchdog.check(obs, torch.zeros(10), torch.zeros(10, dtype=torch.bool), step=42)
        assert 0 in nan_envs
        assert 3 in nan_envs
        assert 7 in nan_envs
        print(f"✓ Env identification test passed. Affected: {nan_envs}")
    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


def test_nan_watchdog_returns_empty_on_clean():
    """Should return empty list when no NaN present."""
    from isaaclab.utils.nan_watchdog import NaNWatchdog

    watchdog = NaNWatchdog(dump_dir=tempfile.mkdtemp(), max_dumps=1)
    for _ in range(100):
        result = watchdog.check(
            {"policy": torch.randn(64, 157)},
            torch.randn(64),
            torch.zeros(64, dtype=torch.bool),
        )
        assert result == [], f"False positive: {result}"
    print("✓ No false positives over 100 steps")


if __name__ == "__main__":
    test_nan_watchdog_detects_and_dumps()
    test_nan_watchdog_max_dumps()
    test_nan_watchdog_identifies_envs()
    test_nan_watchdog_returns_empty_on_clean()
    print("\n✓ All NaN watchdog tests passed!")

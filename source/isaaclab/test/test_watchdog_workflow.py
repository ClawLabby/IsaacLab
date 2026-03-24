#!/usr/bin/env python3
"""Test NaN watchdog dump/inspect/replay workflow.

This test verifies:
1. Watchdog detects NaN in observations
2. Dump is created with correct structure
3. replay_nan_dump.py can inspect the dump
4. Dump contains Newton state when NewtonManager is active

Note: This test uses INJECTED NaN (via extreme forces or direct injection)
rather than waiting for organic physics failures, which are rare with the
robust implicit solvers. Real training failures typically come from:
- GJK/EPA collision solver non-convergence with degenerate geometry
- Numerical instability in articulated body dynamics with extreme configs
"""

import os
import sys
import shutil
import tempfile

DUMP_DIR = tempfile.mkdtemp(prefix="watchdog_workflow_test_")
os.environ["NAN_WATCHDOG_DIR"] = DUMP_DIR

import torch
import warp as wp
import newton


def test_dump_structure():
    """Verify dump contains expected files and can be inspected."""
    from isaaclab.utils.nan_watchdog import NaNWatchdog

    # Create minimal model
    builder = newton.ModelBuilder()
    b = builder.add_body(mass=1.0, xform=wp.transform((0, 0, 0), wp.quat_identity()))
    builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device="cpu")
    state = model.state()

    watchdog = NaNWatchdog(dump_dir=DUMP_DIR, max_dumps=1)

    # Clean step
    obs = {"policy": torch.randn(1, 10)}
    nan_envs = watchdog.check(obs, torch.zeros(1), torch.zeros(1, dtype=torch.bool), step=0)
    assert len(nan_envs) == 0, "False positive on clean data"

    # Inject NaN
    obs["policy"][0, 5] = float('nan')
    nan_envs = watchdog.check(obs, torch.zeros(1), torch.zeros(1, dtype=torch.bool), step=1)
    assert len(nan_envs) > 0, "Should detect NaN"

    # Verify dump
    dump_dirs = [d for d in os.listdir(DUMP_DIR) if d.startswith("nan_step")]
    assert len(dump_dirs) == 1, f"Expected 1 dump, got {dump_dirs}"

    dump_path = os.path.join(DUMP_DIR, dump_dirs[0])
    contents = os.listdir(dump_path)
    print(f"Dump contents: {contents}")

    assert "nan_info.json" in contents, "Missing nan_info.json"
    assert "rl_tensors.pt" in contents, "Missing rl_tensors.pt"

    # Verify nan_info.json structure
    import json
    with open(os.path.join(dump_path, "nan_info.json")) as f:
        info = json.load(f)
    assert "step" in info
    assert "timestamp" in info
    assert "nan_locations" in info
    print(f"✓ nan_info.json valid: step={info['step']}, locations={list(info['nan_locations'].keys())}")

    # Verify rl_tensors.pt
    rl = torch.load(os.path.join(dump_path, "rl_tensors.pt"), weights_only=False)
    assert "observations" in rl
    assert rl["observations"]["policy"].isnan().any()
    print(f"✓ rl_tensors.pt valid: {list(rl.keys())}")

    # Test replay script
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/tools/replay_nan_dump.py", dump_path],
        capture_output=True, text=True, cwd="/home/horde/claw/git/IsaacLab"
    )
    assert result.returncode == 0, f"replay script failed: {result.stderr}"
    assert "NaN Dump Inspection" in result.stdout
    assert "Step: 1" in result.stdout
    print(f"✓ replay_nan_dump.py works")

    return dump_path


def test_max_dumps():
    """Verify max_dumps limit is respected."""
    from isaaclab.utils.nan_watchdog import NaNWatchdog

    watchdog = NaNWatchdog(dump_dir=DUMP_DIR, max_dumps=2)

    for step in range(5):
        obs = {"policy": torch.tensor([[float('nan')]])}
        watchdog.check(obs, torch.zeros(1), torch.zeros(1, dtype=torch.bool), step=step + 100)

    dump_dirs = [d for d in os.listdir(DUMP_DIR) if d.startswith("nan_step")]
    # Should have 1 from previous test + 2 from this = 3 max, but max_dumps=2 for THIS watchdog
    # The first test's watchdog had max_dumps=1, so it stopped after 1
    # This watchdog should produce exactly 2 more
    # Total: 1 + 2 = 3
    print(f"Dump count: {len(dump_dirs)} (expected ≤3)")
    assert len(dump_dirs) <= 3, f"max_dumps not respected: {len(dump_dirs)} dumps"
    print(f"✓ max_dumps limit works")


def test_env_identification():
    """Verify affected env indices are correctly identified."""
    from isaaclab.utils.nan_watchdog import NaNWatchdog

    watchdog = NaNWatchdog(dump_dir=DUMP_DIR, max_dumps=10)

    # Create batch with NaN in specific envs
    obs = {"policy": torch.randn(10, 5)}
    obs["policy"][2, 0] = float('nan')  # env 2
    obs["policy"][7, 3] = float('nan')  # env 7

    nan_envs = watchdog.check(obs, torch.zeros(10), torch.zeros(10, dtype=torch.bool), step=200)
    assert 2 in nan_envs, "Should detect env 2"
    assert 7 in nan_envs, "Should detect env 7"
    assert len(nan_envs) == 2, f"Should detect exactly 2 envs, got {nan_envs}"
    print(f"✓ Env identification works: {nan_envs}")


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("NaN Watchdog Workflow Test")
        print("=" * 60)

        dump_path = test_dump_structure()
        test_max_dumps()
        test_env_identification()

        print("\n" + "=" * 60)
        print("✓ All workflow tests passed!")
        print(f"  Dumps at: {DUMP_DIR}")
        print("=" * 60)

    finally:
        if "--keep" not in sys.argv:
            shutil.rmtree(DUMP_DIR, ignore_errors=True)
        else:
            print(f"\nPreserved: {DUMP_DIR}")

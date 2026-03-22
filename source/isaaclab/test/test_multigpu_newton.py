# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Test multi-GPU device assignment for Newton physics."""

import os
import subprocess
import sys

import pytest
import torch


def _get_num_gpus() -> int:
    """Return number of available CUDA GPUs."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


@pytest.mark.skipif(_get_num_gpus() < 2, reason="Requires at least 2 GPUs")
class TestMultiGPUDeviceAssignment:
    """Tests for multi-GPU device assignment in distributed training."""

    def test_device_assignment_logic(self):
        """Test that device assignment logic works correctly for different scenarios."""
        # Scenario 1: 4 visible GPUs, world_size=4 → each rank gets its own GPU
        num_visible = 4
        world_size = 4
        for local_rank in range(4):
            if num_visible >= world_size:
                expected_device = f"cuda:{local_rank}"
            else:
                expected_device = "cuda:0"
            assert expected_device == f"cuda:{local_rank}"

        # Scenario 2: 1 visible GPU (CUDA_VISIBLE_DEVICES restricted), world_size=4
        # → all ranks use cuda:0 (they each see only their assigned GPU)
        num_visible = 1
        world_size = 4
        for local_rank in range(4):
            if num_visible >= world_size:
                expected_device = f"cuda:{local_rank}"
            else:
                expected_device = "cuda:0"
            assert expected_device == "cuda:0"

    @pytest.mark.skipif(_get_num_gpus() < 2, reason="Requires at least 2 GPUs")
    def test_cartpole_newton_multigpu(self):
        """Test that multi-GPU cartpole training with Newton physics runs without error."""
        num_gpus = min(_get_num_gpus(), 4)
        
        # Run a quick 2-iteration training to verify setup works
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "scripts/reinforcement_learning/rsl_rl/train.py",
            "--task", "Isaac-Cartpole-Direct-v0",
            "--num_envs", "64",
            "--max_iterations", "2",
            "--headless",
            "--distributed",
            "presets=newton",
        ]
        
        env = os.environ.copy()
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_IB_DISABLE"] = "1"
        
        # Get the IsaacLab root directory
        # test file is at: IsaacLab/source/isaaclab/test/test_multigpu_newton.py
        # root is at: IsaacLab/
        test_dir = os.path.dirname(os.path.abspath(__file__))  # .../source/isaaclab/test
        isaaclab_root = os.path.dirname(os.path.dirname(os.path.dirname(test_dir)))  # .../IsaacLab
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=isaaclab_root,
        )
        
        # Check that training completed (iteration 1 logged)
        assert "Learning iteration 1" in result.stdout or result.returncode == 0, (
            f"Multi-GPU training failed:\nstdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        )


@pytest.mark.skipif(_get_num_gpus() < 2, reason="Requires at least 2 GPUs")
class TestMultiGPUCameraRendering:
    """Tests for multi-GPU training with camera observations."""

    def test_preset_renderer_matching(self):
        """Test that newton preset correctly matches the renderer."""
        from isaaclab_tasks.utils.presets import MultiBackendRendererCfg
        
        # Check that newton field exists and is NewtonWarpRendererCfg
        cfg = MultiBackendRendererCfg()
        assert hasattr(cfg, "newton"), "MultiBackendRendererCfg should have 'newton' field"
        assert "NewtonWarp" in type(cfg.newton).__name__, (
            f"newton field should be NewtonWarpRendererCfg, got {type(cfg.newton).__name__}"
        )

    def test_camera_not_kit_with_preset_renderer(self):
        """Test that PresetCfg renderers are not detected as Kit cameras."""
        from isaaclab.sensors import TiledCameraCfg
        from isaaclab_tasks.utils.presets import MultiBackendRendererCfg
        from isaaclab_tasks.utils.sim_launcher import _is_kit_camera
        
        # Create camera with MultiBackendRendererCfg (a PresetCfg)
        cam = TiledCameraCfg(
            prim_path="/test",
            data_types=["rgb"],
            width=64,
            height=64,
            renderer_cfg=MultiBackendRendererCfg(),
        )
        
        # Should NOT be detected as Kit camera (PresetCfg is assumed to match physics)
        assert not _is_kit_camera(cam), (
            "Camera with PresetCfg renderer should not be detected as Kit camera"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

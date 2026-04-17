# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for batched usd_replicate (no Kit/AppLauncher required).

Verifies that usd_replicate with per-env ChangeBlock batching produces the
same prim structure as expected. Uses raw pxr (OpenUSD) without Kit.
"""

from __future__ import annotations

import pytest
import torch

from pxr import Sdf, Usd

from isaaclab.cloner.cloner_utils import _replicate_single_env, usd_replicate


@pytest.fixture
def stage():
    """Create a fresh in-memory USD stage with a source prim."""
    s = Usd.Stage.CreateInMemory()
    # Create source prims via Sdf (low-level, no Kit needed)
    rl = s.GetRootLayer()
    Sdf.CreatePrimInLayer(rl, "/World")
    Sdf.CreatePrimInLayer(rl, "/World/template")
    Sdf.CreatePrimInLayer(rl, "/World/template/child")
    # Create env namespace
    Sdf.CreatePrimInLayer(rl, "/World/envs")
    for i in range(4):
        Sdf.CreatePrimInLayer(rl, f"/World/envs/env_{i}")
    return s


class TestUsdReplicateBatched:
    """Test that usd_replicate creates correct prim structure."""

    def test_basic_replication(self, stage):
        """All env prims should exist after replication."""
        env_ids = torch.arange(4, dtype=torch.long)
        usd_replicate(
            stage,
            sources=["/World/template"],
            destinations=["/World/envs/env_{}/Object"],
            env_ids=env_ids,
        )
        rl = stage.GetRootLayer()
        for i in range(4):
            assert rl.GetPrimAtPath(f"/World/envs/env_{i}/Object") is not None

    def test_replication_with_mask(self, stage):
        """Only masked envs should receive the prim."""
        env_ids = torch.arange(4, dtype=torch.long)
        mask = torch.zeros((1, 4), dtype=torch.bool)
        mask[0, [0, 2]] = True

        usd_replicate(
            stage,
            sources=["/World/template"],
            destinations=["/World/envs/env_{}/Object"],
            env_ids=env_ids,
            mask=mask,
        )
        rl = stage.GetRootLayer()
        assert rl.GetPrimAtPath("/World/envs/env_0/Object") is not None
        assert rl.GetPrimAtPath("/World/envs/env_1/Object") is None
        assert rl.GetPrimAtPath("/World/envs/env_2/Object") is not None
        assert rl.GetPrimAtPath("/World/envs/env_3/Object") is None

    def test_positions_authored(self, stage):
        """xformOp:translate should be authored from positions tensor."""
        env_ids = torch.arange(4, dtype=torch.long)
        positions = torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        usd_replicate(
            stage,
            sources=["/World/template"],
            destinations=["/World/envs/env_{}/Object"],
            env_ids=env_ids,
            positions=positions,
        )
        rl = stage.GetRootLayer()
        for i in range(4):
            ps = rl.GetPrimAtPath(f"/World/envs/env_{i}/Object")
            assert ps is not None
            t_attr = ps.GetAttributeAtPath(f"/World/envs/env_{i}/Object.xformOp:translate")
            assert t_attr is not None
            val = t_attr.default
            assert abs(val[0] - float(i + 1)) < 1e-6

    def test_quaternions_authored(self, stage):
        """xformOp:orient should be authored from quaternions tensor (xyzw)."""
        env_ids = torch.arange(2, dtype=torch.long)
        # Identity quaternion in xyzw = (0, 0, 0, 1)
        quaternions = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        usd_replicate(
            stage,
            sources=["/World/template"],
            destinations=["/World/envs/env_{}/Object"],
            env_ids=env_ids,
            quaternions=quaternions,
        )
        rl = stage.GetRootLayer()
        for i in range(2):
            ps = rl.GetPrimAtPath(f"/World/envs/env_{i}/Object")
            o_attr = ps.GetAttributeAtPath(f"/World/envs/env_{i}/Object.xformOp:orient")
            assert o_attr is not None
            q = o_attr.default
            assert abs(q.GetReal() - 1.0) < 1e-6  # w component

    def test_depth_ordering(self, stage):
        """Parent prims should be created before children even if sources are out of order."""
        env_ids = torch.arange(2, dtype=torch.long)
        # Provide child first, parent second — replication should handle order
        usd_replicate(
            stage,
            sources=["/World/template/child", "/World/template"],
            destinations=["/World/envs/env_{}/Object/child", "/World/envs/env_{}/Object"],
            env_ids=env_ids,
        )
        rl = stage.GetRootLayer()
        for i in range(2):
            assert rl.GetPrimAtPath(f"/World/envs/env_{i}/Object") is not None
            assert rl.GetPrimAtPath(f"/World/envs/env_{i}/Object/child") is not None


class TestReplicateSingleEnv:
    """Test the _replicate_single_env helper directly."""

    def test_creates_prim(self, stage):
        """Helper should create the destination prim."""
        rl = stage.GetRootLayer()
        _replicate_single_env(rl, "/World/template", "/World/envs/env_0/Test", None, None, 0)
        assert rl.GetPrimAtPath("/World/envs/env_0/Test") is not None

    def test_self_copy_safe(self, stage):
        """Self-copy (src == dp) should not crash."""
        rl = stage.GetRootLayer()
        _replicate_single_env(rl, "/World/template", "/World/template", None, None, 0)
        assert rl.GetPrimAtPath("/World/template") is not None

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test that collide() is called once per substep, not once per step.

This is a unit test using mocks — no physics runtime or Kit required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from isaaclab_newton.physics.newton_manager import NewtonManager


class TestPerSubstepCollision:
    """Verify collide() is called per-substep inside _simulate_physics_only."""

    @staticmethod
    def _setup_manager(num_substeps: int, use_single_state: bool = True) -> None:
        """Configure NewtonManager class-level attrs for the test."""
        NewtonManager._num_substeps = num_substeps
        NewtonManager._use_single_state = use_single_state
        NewtonManager._needs_collision_pipeline = True
        NewtonManager._collision_pipeline = MagicMock()
        NewtonManager._contacts = MagicMock(name="contacts")
        NewtonManager._state_0 = MagicMock(name="state_0")
        NewtonManager._state_1 = MagicMock(name="state_1")
        NewtonManager._control = MagicMock(name="control")
        NewtonManager._solver = MagicMock(name="solver")
        NewtonManager._solver_dt = 1.0 / 60.0
        NewtonManager._report_contacts = False
        NewtonManager._newton_frame_transform_sensors = []
        NewtonManager._newton_imu_sensors = []
        NewtonManager._newton_contact_sensors = {}

    @pytest.mark.parametrize("num_substeps", [1, 2, 4, 8])
    def test_collide_called_per_substep_single_state(self, num_substeps: int):
        """collide() must be called exactly num_substeps times (single-state path)."""
        self._setup_manager(num_substeps, use_single_state=True)
        NewtonManager._simulate_physics_only()

        assert NewtonManager._collision_pipeline.collide.call_count == num_substeps, (
            f"Expected collide() to be called {num_substeps} times, "
            f"got {NewtonManager._collision_pipeline.collide.call_count}"
        )

    @pytest.mark.parametrize("num_substeps", [1, 2, 4])
    def test_collide_called_per_substep_dual_state(self, num_substeps: int):
        """collide() must be called exactly num_substeps times (dual-state path)."""
        self._setup_manager(num_substeps, use_single_state=False)
        # PhysicsManager._cfg is accessed for use_cuda_graph — mock it
        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager._cfg", None):
            NewtonManager._simulate_physics_only()

        assert NewtonManager._collision_pipeline.collide.call_count == num_substeps, (
            f"Expected collide() to be called {num_substeps} times, "
            f"got {NewtonManager._collision_pipeline.collide.call_count}"
        )

    def test_collide_not_called_without_pipeline(self):
        """When collision pipeline is disabled, collide() must not be called."""
        self._setup_manager(num_substeps=4, use_single_state=True)
        NewtonManager._needs_collision_pipeline = False
        NewtonManager._simulate_physics_only()

        NewtonManager._collision_pipeline.collide.assert_not_called()

    def test_collide_receives_current_state(self):
        """collide() must receive the current state_0 (not a stale reference)."""
        self._setup_manager(num_substeps=2, use_single_state=True)
        states_seen = []

        def capture_collide(state, contacts):
            states_seen.append(state)

        NewtonManager._collision_pipeline.collide.side_effect = capture_collide
        NewtonManager._simulate_physics_only()

        # Both calls should receive state_0 (same object in single-state mode)
        assert len(states_seen) == 2
        for s in states_seen:
            assert s is NewtonManager._state_0

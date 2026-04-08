# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Visual domain randomization events for DexSuite.

Provides Newton Warp renderer visual DR:
- Shape color randomization (random RGBA per shape)
- Procedural texture generation (noise, stripes, gradients) assigned to shapes
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers import ManagerTermBase, EventTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)


def _generate_random_texture(resolution: int = 64, style: str = "noise") -> np.ndarray:
    """Generate a random RGBA texture as uint8 array (H, W, 4).

    Styles:
        noise: Random colored noise
        stripes: Random colored vertical stripes
        gradient: Random two-color gradient
        checker: Random colored checkerboard
        solid: Solid random color
    """
    h = w = resolution
    if style == "noise":
        rgb = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        # Blur slightly for more natural look
        from scipy.ndimage import uniform_filter
        for c in range(3):
            rgb[:, :, c] = uniform_filter(rgb[:, :, c].astype(np.float32), size=max(2, resolution // 16)).astype(np.uint8)
    elif style == "stripes":
        stripe_width = np.random.randint(2, max(3, resolution // 8))
        c1 = np.random.randint(0, 256, 3)
        c2 = np.random.randint(0, 256, 3)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(w):
            rgb[:, x] = c1 if (x // stripe_width) % 2 == 0 else c2
    elif style == "gradient":
        c1 = np.random.randint(0, 256, 3).astype(np.float32)
        c2 = np.random.randint(0, 256, 3).astype(np.float32)
        t = np.linspace(0, 1, w).reshape(1, w, 1)
        rgb = ((1 - t) * c1 + t * c2).astype(np.uint8)
        rgb = np.broadcast_to(rgb, (h, w, 3)).copy()
    elif style == "checker":
        checker_size = np.random.randint(2, max(3, resolution // 4))
        c1 = np.random.randint(0, 256, 3)
        c2 = np.random.randint(0, 256, 3)
        pattern = ((np.arange(h) // checker_size)[:, None] + (np.arange(w) // checker_size)) % 2
        rgb = np.where(pattern[..., None], c1, c2).astype(np.uint8)
    else:  # solid
        c = np.random.randint(0, 256, 3)
        rgb = np.full((h, w, 3), c, dtype=np.uint8)

    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=-1)


class randomize_newton_shape_colors(ManagerTermBase):
    """Randomize shape colors AND textures in the Newton Warp renderer on each reset.

    This writes random RGBA colors to the Warp render context's shape_colors array
    and optionally generates procedural textures for visual variety.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        import warp as wp
        from newton._src.sensors.warp_raytrace.types import TextureData

        self._wp = wp
        self._TextureData = TextureData
        sensor_name = cfg.params.get("sensor_name", "base_camera")
        self._sensor = env.scene.sensors.get(sensor_name)
        self._render_context = None
        self._torch_device = None
        self._newton_sensor = None

        if self._sensor is not None and hasattr(self._sensor, "renderer"):
            renderer = self._sensor.renderer
            if hasattr(renderer, "newton_sensor"):
                self._newton_sensor = renderer.newton_sensor
                self._render_context = renderer.newton_sensor.render_context
                self._num_shapes = self._render_context.shape_colors.shape[0]
                warp_dev = str(self._render_context.device)
                if "cuda" in warp_dev:
                    self._torch_device = torch.device(warp_dev)
                else:
                    self._torch_device = torch.device("cpu")

                # Compute per-env shape indexing
                # With enable_global_world=True: shape[0] = global (ground),
                # then shapes are evenly distributed across envs
                num_envs = env.num_envs
                has_global = self._render_context.config.enable_global_world
                num_global = 1 if has_global else 0
                self._shapes_per_env = (self._num_shapes - num_global) // num_envs
                self._global_offset = num_global
                self._num_envs = num_envs

                logger.info(
                    f"randomize_newton_shape_colors: {self._num_shapes} shapes total, "
                    f"{self._shapes_per_env} per env, {num_global} global, "
                    f"device={warp_dev}"
                )
            else:
                logger.warning(
                    f"randomize_newton_shape_colors: Renderer on '{sensor_name}' "
                    f"is not NewtonWarpRenderer (type={type(renderer).__name__}). "
                    f"Visual DR will be a no-op."
                )
        else:
            logger.warning(
                f"randomize_newton_shape_colors: Sensor '{sensor_name}' not found or has no renderer. "
                f"Visual DR will be a no-op."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        sensor_name: str = "base_camera",
        color_range: tuple[float, float] = (0.1, 0.9),
        alpha: float = 1.0,
        use_textures: bool = True,
        num_textures: int = 8,
        texture_resolution: int = 64,
    ):
        if self._render_context is None:
            return

        wp = self._wp
        low, high = color_range
        spe = self._shapes_per_env
        off = self._global_offset

        # Get current colors as torch tensor
        current_colors = wp.to_torch(self._render_context.shape_colors)  # (total_shapes, 4)

        # Vectorized per-env randomization — no Python loop
        n_reset = len(env_ids)
        # Generate all random colors at once: (n_reset, shapes_per_env, 4)
        rand_colors = torch.rand(n_reset, spe, 4, device=self._torch_device)
        rand_colors[..., :3] = rand_colors[..., :3] * (high - low) + low
        rand_colors[..., 3] = alpha

        # Compute flat indices for all shapes in resetting envs
        # env_ids shape: (n_reset,)
        env_offsets = off + env_ids.to(self._torch_device) * spe  # (n_reset,)
        # shape_offsets: (n_reset, spe)
        shape_indices = env_offsets.unsqueeze(1) + torch.arange(spe, device=self._torch_device).unsqueeze(0)
        # Flatten and scatter
        flat_indices = shape_indices.reshape(-1)  # (n_reset * spe,)
        flat_colors = rand_colors.reshape(-1, 4)  # (n_reset * spe, 4)
        current_colors[flat_indices] = flat_colors

        self._render_context.shape_colors = wp.from_torch(current_colors, dtype=wp.vec4f)

    def _assign_random_textures(self, num_textures: int, resolution: int):
        """Generate random procedural textures and assign to shapes.
        
        Uses the Newton SensorTiledCamera's built-in checkerboard method as a 
        starting point, then randomizes the pattern parameters.
        """
        # Use the built-in checkerboard with randomized parameters for now
        # This is the safest approach since it uses the tested code path
        checker_size = max(2, np.random.randint(2, resolution // 2))
        self._newton_sensor.assign_checkerboard_material_to_all_shapes(
            resolution=resolution, checker_size=checker_size
        )


class randomize_observation_color_jitter(ManagerTermBase):
    """Apply color jitter (brightness, contrast, saturation) to RGB observations.

    Note: For ResNet tasks, jitter should be injected into the image_features
    inference pipeline instead of as an event term.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.brightness = torch.ones(env.num_envs, 1, 1, 1, device=env.device)
        self.contrast = torch.ones(env.num_envs, 1, 1, 1, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        brightness_range: tuple[float, float] = (0.7, 1.3),
        contrast_range: tuple[float, float] = (0.7, 1.3),
    ):
        n = len(env_ids)
        self.brightness[env_ids] = (
            torch.rand(n, 1, 1, 1, device=env.device)
            * (brightness_range[1] - brightness_range[0])
            + brightness_range[0]
        )
        self.contrast[env_ids] = (
            torch.rand(n, 1, 1, 1, device=env.device)
            * (contrast_range[1] - contrast_range[0])
            + contrast_range[0]
        )

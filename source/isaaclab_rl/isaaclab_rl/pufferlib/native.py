"""Experimental external-vector bridge for PufferLib native-style backends.

This module intentionally keeps the Isaac Lab side small. Isaac Lab owns the
environment, while :class:`PufferWarpBridge` exposes the pointer/callback surface
consumed by PufferLib's compiled external-vector hook when that hook is present
in the local ``pufferlib._C`` build.
"""

from __future__ import annotations

import sys
from typing import Any

from .puffer_cfg import PufferTrainCfg
from .warp_bridge import PufferWarpBridge


def _add_pufferlib_to_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def build_pufferlib_args(
    bridge: PufferWarpBridge,
    cfg: PufferTrainCfg,
    *,
    env_name: str = "isaaclab_external",
    gpu_id: int = 0,
) -> dict[str, Any]:
    """Build the PufferLib args dictionary needed by ``torch_pufferl.PuffeRL``.

    PufferLib's CLI normally loads this structure from ``config/*.ini``. For an
    Isaac Lab-owned environment there is no Ocean config file, so we synthesize
    the minimum equivalent from the live bridge and the matched PPO config.
    """

    batch_size = int(cfg.horizon) * int(bridge.total_agents)
    minibatch_size = int(cfg.minibatch_size) if cfg.minibatch_size else batch_size // int(cfg.num_mini_batches)
    minibatch_size = max(1, min(minibatch_size, batch_size))

    return {
        "env_name": env_name,
        "rank": 0,
        "world_size": 1,
        "gpu_id": gpu_id,
        "nccl_id": b"",
        "profile": False,
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
        "checkpoint_interval": int(cfg.save_interval),
        "reset_state": True,
        "cudagraphs": -1,
        "seed": int(cfg.seed),
        "wandb": False,
        "load_id": None,
        "load_model_path": None,
        "vec": {
            "total_agents": int(bridge.total_agents),
            "num_buffers": 1,
            "num_threads": 1,
        },
        "env": {},
        "policy": {
            "hidden_size": int(cfg.hidden_size),
            "num_layers": int(cfg.num_layers),
            "expansion_factor": 1,
        },
        "torch": {
            "network": "MinGRU",
            "encoder": "DefaultEncoder",
            "decoder": "DefaultDecoder",
        },
        "train": {
            "gpus": 1,
            "seed": int(cfg.seed),
            "total_timesteps": int(cfg.iterations) * batch_size,
            "learning_rate": float(cfg.learning_rate),
            "anneal_lr": 0,
            "min_lr_ratio": 0.0,
            "gamma": float(cfg.gamma),
            "gae_lambda": float(cfg.gae_lambda),
            "replay_ratio": float(cfg.update_epochs),
            "clip_coef": float(cfg.clip_coef),
            "vf_coef": float(cfg.vf_coef),
            "vf_clip_coef": float(cfg.clip_coef),
            "max_grad_norm": float(cfg.max_grad_norm),
            "ent_coef": float(cfg.ent_coef),
            "anneal_ent_coef": 0,
            "min_ent_coef_ratio": 0.1,
            "beta1": 0.95,
            "beta2": 0.999,
            "eps": 1e-12,
            "minibatch_size": minibatch_size,
            "horizon": int(cfg.horizon),
            "vtrace_rho_clip": 1.0,
            "vtrace_c_clip": 1.0,
            "prio_alpha": 0.8,
            "prio_beta0": 0.2,
        },
    }


def probe_native_backend(pufferlib_path: str) -> dict[str, Any]:
    """Return import/hook availability for the local PufferLib backend."""

    _add_pufferlib_to_path(pufferlib_path)
    result: dict[str, Any] = {
        "can_import_c": False,
        "can_import_torch_pufferl": False,
        "c_env_name": None,
        "c_gpu": None,
        "c_precision_bytes": None,
        "has_create_vec": False,
        "has_create_pufferl": False,
        "has_external_vec_hook": False,
        "external_vec_hook": None,
        "has_external_gpu_vec": False,
        "native_train_available": False,
        "error": None,
    }
    try:
        from pufferlib import _C  # type: ignore

        result["can_import_c"] = True
        result["c_env_name"] = getattr(_C, "env_name", None)
        result["c_gpu"] = getattr(_C, "gpu", None)
        result["c_precision_bytes"] = getattr(_C, "precision_bytes", None)
        result["has_create_vec"] = hasattr(_C, "create_vec")
        result["has_create_pufferl"] = hasattr(_C, "create_pufferl")
        for hook_name in ("create_external_pufferl", "create_pufferl_from_vec"):
            if hasattr(_C, hook_name):
                result["has_external_vec_hook"] = True
                result["external_vec_hook"] = hook_name
                break
        result["native_train_available"] = bool(result["has_external_vec_hook"] and result["c_gpu"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    try:
        from pufferlib.external import ExternalGPUVec  # noqa: F401

        result["has_external_gpu_vec"] = True
    except Exception as exc:
        if result["error"] is None:
            result["error"] = f"{type(exc).__name__}: {exc}"

    try:
        import pufferlib.torch_pufferl  # noqa: F401

        result["can_import_torch_pufferl"] = True
    except Exception as exc:
        if result["error"] is None:
            result["error"] = f"{type(exc).__name__}: {exc}"

    return result


class NativeExternalPuffeRL:
    """Small lifetime wrapper for PufferLib's compiled external-vector trainer."""

    def __init__(self, backend: Any, pufferl: Any, bridge: PufferWarpBridge) -> None:
        self.backend = backend
        self.pufferl = pufferl
        self.bridge = bridge

    @property
    def global_step(self) -> int:
        return int(self.pufferl.global_step)

    def num_params(self) -> int:
        return int(self.pufferl.num_params())

    def rollouts(self) -> None:
        self.backend.external_rollouts(self.pufferl, self.bridge.gpu_step)

    def train(self) -> Any:
        return self.backend.train(self.pufferl)

    def log(self) -> dict[str, Any]:
        logs = dict(self.backend.log(self.pufferl))
        env_logs = dict(logs.get("env", {}))
        env_logs.update(self.bridge.log())
        logs["env"] = env_logs
        return logs

    def close(self) -> None:
        try:
            self.backend.close(self.pufferl)
        finally:
            self.bridge.close()


def create_torch_pufferl_from_isaaclab(
    env: Any,
    cfg: PufferTrainCfg,
    *,
    env_name: str = "isaaclab_external",
    obs_group: str = "policy",
    gpu_id: int = 0,
):
    """Create PufferLib's Python ``PuffeRL`` runner from an Isaac Lab env.

    This is the currently executable low-change bridge when PufferLib's compiled
    extension is available. It still uses PufferLib's PyTorch policy/training
    wrapper, but environment interaction goes through the same pointer protocol
    as Ocean vecenvs.
    """

    _add_pufferlib_to_path(cfg.pufferlib_path)
    try:
        from pufferlib.torch_pufferl import PuffeRL, load_policy
    except Exception as exc:
        raise RuntimeError(
            "PufferLib torch_pufferl is unavailable. Build a float32 PufferLib extension first, "
            "for example: PATH=/path/to/venv/bin:$PATH ./build.sh squared_continuous --float"
        ) from exc

    bridge = PufferWarpBridge(env, obs_group=obs_group, clip_actions=cfg.clip_actions)
    args = build_pufferlib_args(bridge, cfg, env_name=env_name, gpu_id=gpu_id)
    policy = load_policy(args, bridge)
    return PuffeRL(args, bridge, policy)


def create_native_pufferl_from_isaaclab(
    env: Any,
    cfg: PufferTrainCfg,
    *,
    env_name: str = "isaaclab_external",
    obs_group: str = "policy",
    gpu_id: int = 0,
):
    """Create a compiled PufferLib native trainer from an Isaac Lab-owned env."""

    _add_pufferlib_to_path(cfg.pufferlib_path)
    from pufferlib import _C  # type: ignore

    bridge = PufferWarpBridge(env, obs_group=obs_group, clip_actions=cfg.clip_actions)
    args = build_pufferlib_args(bridge, cfg, env_name=env_name, gpu_id=gpu_id)
    for hook_name in ("create_external_pufferl", "create_pufferl_from_vec"):
        hook = getattr(_C, hook_name, None)
        if hook is not None:
            return NativeExternalPuffeRL(_C, hook(args, bridge), bridge)
    raise NotImplementedError(
        "The imported PufferLib _C extension does not expose create_external_pufferl. "
        "Rebuild the local PufferLib checkout after applying the external-vector hook."
    )

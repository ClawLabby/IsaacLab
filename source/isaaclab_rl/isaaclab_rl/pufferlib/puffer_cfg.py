"""Configuration for the experimental PufferLib-style PyTorch runner."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PufferTrainCfg:
    """PPO config used by :class:`PufferTorchRunner`.

    ``profile="matched_rsl"`` is the apples-to-apples state-observation path:
    it uses separate Gaussian actor and critic MLPs plus PPO hyperparameters
    copied from the task's RSL-RL config. ``profile="camera_cnn"`` keeps one
    raw image observation channel-first and uses a CNN actor/critic matching the
    Cartpole camera RSL-RL config shape. ``profile="mixed_camera_cnn"`` handles
    DexSuite-style observation sets with vector groups plus one or more camera
    groups. ``profile="puffer_mingru"`` keeps the earlier PufferLib model/
    optimizer experiment explicit so it is not confused with a matched RSL-RL
    comparison.
    """

    profile: str = "matched_rsl"
    iterations: int = 100
    horizon: int = 24
    update_epochs: int = 5
    num_mini_batches: int = 4
    minibatch_size: int = 0
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    cnn_output_channels: list[int] = field(default_factory=lambda: [32, 64, 64])
    cnn_kernel_size: list[int] = field(default_factory=lambda: [8, 4, 3])
    cnn_stride: list[int] = field(default_factory=lambda: [4, 2, 1])
    cnn_activation: str = "relu"
    share_cnn_encoders: bool = True
    init_std: float = 1.0
    std_type: str = "scalar"
    hidden_size: int = 128
    num_layers: int = 2
    optimizer: str = "adam"
    learning_rate: float = 0.001
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 1.0
    ent_coef: float = 0.005
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    normalize_advantage_per_minibatch: bool = False
    check_for_nan: bool = True
    finite_grad_guard: bool = True
    init_at_random_ep_len: bool = True
    seed: int = 42
    device: str = "cuda:0"
    actor_obs_groups: list[str] = field(default_factory=lambda: ["policy"])
    critic_obs_groups: list[str] = field(default_factory=lambda: ["policy"])
    clip_actions: float | None = None
    save_interval: int = 50
    pufferlib_path: str = "/home/horde/claw/research/PufferLib"

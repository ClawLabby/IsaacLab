"""Experimental pure-PyTorch PufferLib-style PPO runner."""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal

from .puffer_cfg import PufferTrainCfg
from .vecenv_wrapper import PufferVecEnvWrapper, collect_scalar_logs


Normal.set_default_validate_args(False)


def _add_pufferlib_to_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "elu":
        return nn.ELU()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation '{name}'")


def _mlp(input_dim: int, output_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential:
    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(in_dim, out_dim), _activation(activation)]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def _conv_activation(name: str) -> nn.Module:
    return _activation(name)


def _cnn_encoder(input_shape: tuple[int, ...], cfg: PufferTrainCfg) -> tuple[nn.Sequential, int]:
    if len(input_shape) != 3:
        raise ValueError(f"CNN profile expects channel-first image observations [C,H,W], got {input_shape}")
    channels = int(input_shape[0])
    layers: list[nn.Module] = []
    for out_channels, kernel_size, stride in zip(cfg.cnn_output_channels, cfg.cnn_kernel_size, cfg.cnn_stride):
        layers += [
            nn.Conv2d(channels, int(out_channels), kernel_size=int(kernel_size), stride=int(stride)),
            _conv_activation(cfg.cnn_activation),
        ]
        channels = int(out_channels)
    layers.append(nn.Flatten())
    encoder = nn.Sequential(*layers)
    with torch.no_grad():
        sample = torch.zeros(1, *input_shape)
        output_dim = int(encoder(sample).shape[-1])
    return encoder, output_dim


def _std_parameter(action_dim: int, init_std: float, std_type: str) -> nn.Parameter:
    if std_type == "scalar":
        return nn.Parameter(torch.log(torch.exp(init_std * torch.ones(action_dim)) - 1.0))
    if std_type == "log":
        return nn.Parameter(torch.log(init_std * torch.ones(action_dim)))
    raise ValueError(f"Unsupported std_type '{std_type}'")


def _std_from_parameter(param: torch.Tensor, std_type: str, like: torch.Tensor) -> torch.Tensor:
    if std_type == "scalar":
        std = torch.nn.functional.softplus(param).clamp(min=1e-6)
    else:
        std = torch.exp(param).clamp(min=1e-6)
    return std.expand_as(like)


def _gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    cfg: PufferTrainCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(cfg.horizon)):
        next_nonterminal = 1.0 - dones[t]
        next_value = last_value if t == cfg.horizon - 1 else values[t + 1]
        delta = rewards[t] + cfg.gamma * next_value * next_nonterminal - values[t]
        last_advantage = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_advantage
        advantages[t] = last_advantage
    returns = advantages + values
    if not cfg.normalize_advantage_per_minibatch:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def _check_finite(name: str, *tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite values detected in {name}")


def _distribution_kl(old_mean: torch.Tensor, old_std: torch.Tensor, new_mean: torch.Tensor, new_std: torch.Tensor) -> torch.Tensor:
    old_dist = Normal(old_mean, old_std)
    new_dist = Normal(new_mean, new_std)
    return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)


class MatchedMLPActorCritic(nn.Module):
    """Separate Gaussian actor and critic MLPs matching the task RSL-RL config."""

    def __init__(self, env: PufferVecEnvWrapper, cfg: PufferTrainCfg):
        super().__init__()
        self.cfg = cfg
        self.actor = _mlp(env.actor_obs_dim, env.action_dim, cfg.actor_hidden_dims, cfg.activation)
        self.critic = _mlp(env.critic_obs_dim, 1, cfg.critic_hidden_dims, cfg.activation)
        self.std_param = _std_parameter(env.action_dim, cfg.init_std, cfg.std_type)

    @property
    def output_std(self) -> torch.Tensor:
        if self.cfg.std_type == "scalar":
            return torch.nn.functional.softplus(self.std_param).clamp(min=1e-6)
        return torch.exp(self.std_param).clamp(min=1e-6)

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)

    def distribution(self, actor_obs: torch.Tensor) -> Normal:
        mean = self.actor(actor_obs)
        std = _std_from_parameter(self.std_param, self.cfg.std_type, mean)
        return Normal(mean, std)

    def act(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs["actor"])
        actions = dist.sample()
        logprob = dist.log_prob(actions).sum(-1)
        value = self.value(obs["critic"])
        return actions, logprob, value, dist.mean, dist.stddev

    def evaluate_actions(
        self, actor_obs: torch.Tensor, critic_obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(actor_obs)
        values = self.value(critic_obs)
        logprob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprob, entropy, values, dist.mean, dist.stddev


class CameraCNNActorCritic(nn.Module):
    """CNN actor/critic for raw Cartpole camera tensors."""

    def __init__(self, env: PufferVecEnvWrapper, cfg: PufferTrainCfg):
        super().__init__()
        self.cfg = cfg
        self.share_cnn_encoders = cfg.share_cnn_encoders
        self.actor_encoder, actor_features = _cnn_encoder(env.actor_obs_shape, cfg)
        self.reuse_shared_features = (
            self.share_cnn_encoders
            and env.actor_obs_shape == env.critic_obs_shape
            and env.actor_obs_groups == env.critic_obs_groups
        )
        if self.share_cnn_encoders:
            self.critic_encoder = self.actor_encoder
            critic_features = actor_features
        else:
            self.critic_encoder, critic_features = _cnn_encoder(env.critic_obs_shape, cfg)
        self.actor_head = _mlp(actor_features, env.action_dim, cfg.actor_hidden_dims, cfg.activation)
        self.critic_head = _mlp(critic_features, 1, cfg.critic_hidden_dims, cfg.activation)
        self.std_param = _std_parameter(env.action_dim, cfg.init_std, cfg.std_type)

    @property
    def output_std(self) -> torch.Tensor:
        if self.cfg.std_type == "scalar":
            return torch.nn.functional.softplus(self.std_param).clamp(min=1e-6)
        return torch.exp(self.std_param).clamp(min=1e-6)

    def _actor_features(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor_encoder(actor_obs.float())

    def _critic_features(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic_encoder(critic_obs.float())

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self._critic_features(critic_obs)).squeeze(-1)

    def distribution(self, actor_obs: torch.Tensor) -> Normal:
        mean = self.actor_head(self._actor_features(actor_obs))
        std = _std_from_parameter(self.std_param, self.cfg.std_type, mean)
        return Normal(mean, std)

    def act(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.reuse_shared_features:
            features = self._actor_features(obs["actor"])
            mean = self.actor_head(features)
            value = self.critic_head(features).squeeze(-1)
            std = _std_from_parameter(self.std_param, self.cfg.std_type, mean)
            dist = Normal(mean, std)
        else:
            dist = self.distribution(obs["actor"])
            value = self.value(obs["critic"])
        actions = dist.sample()
        logprob = dist.log_prob(actions).sum(-1)
        return actions, logprob, value, dist.mean, dist.stddev

    def evaluate_actions(
        self, actor_obs: torch.Tensor, critic_obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.reuse_shared_features:
            features = self._actor_features(actor_obs)
            mean = self.actor_head(features)
            values = self.critic_head(features).squeeze(-1)
            std = _std_from_parameter(self.std_param, self.cfg.std_type, mean)
            dist = Normal(mean, std)
        else:
            dist = self.distribution(actor_obs)
            values = self.value(critic_obs)
        logprob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprob, entropy, values, dist.mean, dist.stddev


class MixedCameraCNNActorCritic(nn.Module):
    """CNN/vector actor-critic for DexSuite-style camera observations."""

    def __init__(self, env: PufferVecEnvWrapper, cfg: PufferTrainCfg):
        super().__init__()
        self.cfg = cfg
        self.actor_layout = env.actor_obs_layout
        self.critic_layout = env.critic_obs_layout
        self.actor_image_groups = self._image_groups(self.actor_layout)
        self.critic_image_groups = self._image_groups(self.critic_layout)
        self.actor_vector_groups = [g for g in self.actor_layout["groups"] if g not in self.actor_image_groups]
        self.critic_vector_groups = [g for g in self.critic_layout["groups"] if g not in self.critic_image_groups]
        if not self.actor_image_groups:
            raise ValueError("mixed_camera_cnn requires at least one actor image observation group.")

        self.share_cnn_encoders = cfg.share_cnn_encoders
        self.actor_encoders, actor_features = self._build_encoders(self.actor_layout, self.actor_image_groups)
        if (
            self.share_cnn_encoders
            and self.actor_image_groups == self.critic_image_groups
            and all(
                self.actor_layout["shapes"][group] == self.critic_layout["shapes"].get(group)
                for group in self.actor_image_groups
            )
        ):
            self.critic_encoders = self.actor_encoders
            critic_features = actor_features
        else:
            self.critic_encoders, critic_features = self._build_encoders(self.critic_layout, self.critic_image_groups)

        actor_input_dim = actor_features + self._vector_dim(self.actor_layout, self.actor_vector_groups)
        critic_input_dim = critic_features + self._vector_dim(self.critic_layout, self.critic_vector_groups)
        self.actor_head = _mlp(actor_input_dim, env.action_dim, cfg.actor_hidden_dims, cfg.activation)
        self.critic_head = _mlp(critic_input_dim, 1, cfg.critic_hidden_dims, cfg.activation)
        self.std_param = _std_parameter(env.action_dim, cfg.init_std, cfg.std_type)

    @staticmethod
    def _image_groups(layout: dict[str, Any]) -> list[str]:
        return [group for group in layout["groups"] if len(layout["shapes"][group]) == 3]

    @staticmethod
    def _vector_dim(layout: dict[str, Any], groups: list[str]) -> int:
        return sum(int(layout["flat_dims"][group]) for group in groups)

    def _build_encoders(self, layout: dict[str, Any], image_groups: list[str]) -> tuple[nn.ModuleDict, int]:
        encoders = nn.ModuleDict()
        feature_dim = 0
        for group in image_groups:
            encoder, group_features = _cnn_encoder(layout["shapes"][group], self.cfg)
            encoders[group] = encoder
            feature_dim += group_features
        return encoders, feature_dim

    @property
    def output_std(self) -> torch.Tensor:
        if self.cfg.std_type == "scalar":
            return torch.nn.functional.softplus(self.std_param).clamp(min=1e-6)
        return torch.exp(self.std_param).clamp(min=1e-6)

    @staticmethod
    def _slice_group(obs: torch.Tensor, layout: dict[str, Any], group: str) -> torch.Tensor:
        start, end = layout["slices"][group]
        return obs[:, start:end]

    def _features(
        self,
        obs: torch.Tensor,
        layout: dict[str, Any],
        image_groups: list[str],
        vector_groups: list[str],
        encoders: nn.ModuleDict,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for group in vector_groups:
            parts.append(self._slice_group(obs, layout, group).float())
        for group in image_groups:
            shape = layout["shapes"][group]
            image = self._slice_group(obs, layout, group).reshape(obs.shape[0], *shape).float()
            parts.append(encoders[group](image))
        if not parts:
            raise ValueError("mixed_camera_cnn produced an empty feature list.")
        return torch.cat(parts, dim=-1)

    def _actor_features(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self._features(
            actor_obs, self.actor_layout, self.actor_image_groups, self.actor_vector_groups, self.actor_encoders
        )

    def _critic_features(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self._features(
            critic_obs, self.critic_layout, self.critic_image_groups, self.critic_vector_groups, self.critic_encoders
        )

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self._critic_features(critic_obs)).squeeze(-1)

    def distribution(self, actor_obs: torch.Tensor) -> Normal:
        mean = self.actor_head(self._actor_features(actor_obs))
        std = _std_from_parameter(self.std_param, self.cfg.std_type, mean)
        return Normal(mean, std)

    def act(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs["actor"])
        actions = dist.sample()
        logprob = dist.log_prob(actions).sum(-1)
        value = self.value(obs["critic"])
        return actions, logprob, value, dist.mean, dist.stddev

    def evaluate_actions(
        self, actor_obs: torch.Tensor, critic_obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(actor_obs)
        values = self.value(critic_obs)
        logprob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprob, entropy, values, dist.mean, dist.stddev


class PufferMinGRUActorCritic(nn.Module):
    """PufferLib MinGRU policy adapter kept as an experimental, non-matched profile."""

    def __init__(self, env: PufferVecEnvWrapper, cfg: PufferTrainCfg):
        super().__init__()
        self.env = env
        self.cfg = cfg
        _add_pufferlib_to_path(cfg.pufferlib_path)
        from pufferlib.models import DefaultDecoder, DefaultEncoder, MinGRU, Policy

        encoder = DefaultEncoder(env.actor_obs_dim, cfg.hidden_size)
        network = MinGRU(cfg.hidden_size, cfg.num_layers)
        decoder = DefaultDecoder([1] * env.action_dim, cfg.hidden_size)
        self.policy = Policy(encoder, decoder, network)
        self.state = self.policy.initial_state(env.total_agents, torch.device(cfg.device))

    @property
    def output_std(self) -> torch.Tensor:
        return self.policy.decoder.decoder_logstd.detach().exp().flatten()

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        dist, values = self.policy(critic_obs.unsqueeze(1))
        del dist
        return values.squeeze(1)

    def act(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value, self.state = self.policy.forward_eval(obs["actor"], self.state)
        actions = dist.sample()
        logprob = dist.log_prob(actions).sum(-1)
        return actions, logprob, value.flatten(), dist.mean, dist.stddev

    def evaluate_actions(
        self, actor_obs: torch.Tensor, critic_obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del critic_obs
        dist, values = self.policy(actor_obs.unsqueeze(1))
        values = values.squeeze(1)
        logprob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprob, entropy, values, dist.mean, dist.stddev


class PufferTorchRunner:
    """PPO runner using tensor-only storage plus optional PufferLib model/optimizer pieces."""

    def __init__(self, env: PufferVecEnvWrapper, cfg: PufferTrainCfg, log_dir: str | Path):
        self.env = env
        self.cfg = cfg
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device)
        self.model = self._build_model().to(self.device)
        self.optimizer = self._build_optimizer()
        self.log_path = self.log_dir / "metrics.jsonl"
        self.tb_writer = self._build_tb_writer()

    def _build_model(self) -> nn.Module:
        if self.cfg.profile == "matched_rsl":
            return MatchedMLPActorCritic(self.env, self.cfg)
        if self.cfg.profile == "camera_cnn":
            return CameraCNNActorCritic(self.env, self.cfg)
        if self.cfg.profile == "mixed_camera_cnn":
            return MixedCameraCNNActorCritic(self.env, self.cfg)
        if self.cfg.profile == "puffer_mingru":
            return PufferMinGRUActorCritic(self.env, self.cfg)
        raise ValueError(f"Unsupported Puffer profile '{self.cfg.profile}'")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.cfg.optimizer == "muon":
            _add_pufferlib_to_path(self.cfg.pufferlib_path)
            from pufferlib.muon import Muon

            return Muon(self.model.parameters(), lr=self.cfg.learning_rate, momentum=0.95, eps=1e-12)
        if self.cfg.optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        if self.cfg.optimizer == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        raise ValueError(f"Unsupported optimizer '{self.cfg.optimizer}'")

    def _build_tb_writer(self):
        try:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(log_dir=str(self.log_dir))
        except Exception:
            return None

    def _set_learning_rate(self, learning_rate: float) -> None:
        self.cfg.learning_rate = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

    def _maybe_update_learning_rate(self, kl_mean: float) -> None:
        if self.cfg.schedule != "adaptive" or self.cfg.desired_kl is None or not math.isfinite(kl_mean):
            return
        lr = self.cfg.learning_rate
        if kl_mean > self.cfg.desired_kl * 2.0:
            lr = max(1e-5, lr / 1.5)
        elif kl_mean > 0.0 and kl_mean < self.cfg.desired_kl / 2.0:
            lr = min(1e-2, lr * 1.5)
        if lr != self.cfg.learning_rate:
            self._set_learning_rate(lr)

    def _write_logs(self, row: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        if self.tb_writer is not None:
            step = int(row["steps"])
            for key, value in row.items():
                if isinstance(value, (float, int)) and key not in {"iteration", "steps"}:
                    self.tb_writer.add_scalar(key, value, step)
            self.tb_writer.flush()
        print(json.dumps(row, sort_keys=True), flush=True)

    def _save(self, name: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cfg": asdict(self.cfg),
            },
            self.log_dir / name,
        )

    def learn(self) -> list[dict[str, Any]]:
        torch.manual_seed(self.cfg.seed)
        if self.cfg.init_at_random_ep_len and self.env.max_episode_length and self.env.episode_length_buf is not None:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        batch_size = self.cfg.horizon * self.env.total_agents
        minibatch_size = self.cfg.minibatch_size or batch_size // self.cfg.num_mini_batches
        minibatch_size = max(1, min(minibatch_size, batch_size))
        rows: list[dict[str, Any]] = []
        start_time = time.time()

        for iteration in range(self.cfg.iterations):
            actor_obs = torch.zeros(
                self.cfg.horizon, self.env.total_agents, *self.env.actor_obs_shape, device=self.device
            )
            critic_obs = torch.zeros(
                self.cfg.horizon, self.env.total_agents, *self.env.critic_obs_shape, device=self.device
            )
            actions = torch.zeros(self.cfg.horizon, self.env.total_agents, self.env.action_dim, device=self.device)
            old_logprobs = torch.zeros(self.cfg.horizon, self.env.total_agents, device=self.device)
            rewards = torch.zeros(self.cfg.horizon, self.env.total_agents, device=self.device)
            dones = torch.zeros(self.cfg.horizon, self.env.total_agents, device=self.device)
            values = torch.zeros(self.cfg.horizon, self.env.total_agents, device=self.device)
            old_means = torch.zeros(self.cfg.horizon, self.env.total_agents, self.env.action_dim, device=self.device)
            old_stds = torch.zeros_like(old_means)
            extra_accum: dict[str, list[float]] = defaultdict(list)

            _sync(self.device)
            rollout_t0 = time.time()
            for t in range(self.cfg.horizon):
                actor_obs[t] = obs["actor"]
                critic_obs[t] = obs["critic"]
                with torch.no_grad():
                    action, logprob, value, mean, std = self.model.act(obs)
                next_obs, reward, done, extras = self.env.step(action)
                if self.cfg.check_for_nan:
                    _check_finite("environment step", next_obs["actor"], next_obs["critic"], reward, done)
                timeout = extras.get("time_outs")
                if isinstance(timeout, torch.Tensor):
                    reward = reward + self.cfg.gamma * value * timeout.to(self.device).float()

                actions[t] = action
                old_logprobs[t] = logprob
                rewards[t] = reward
                dones[t] = done
                values[t] = value
                old_means[t] = mean
                old_stds[t] = std
                for key, value_log in collect_scalar_logs(extras).items():
                    extra_accum[key].append(value_log)
                obs = next_obs
            _sync(self.device)
            rollout_s = time.time() - rollout_t0

            with torch.no_grad():
                last_value = self.model.value(obs["critic"])
            advantages, returns = _gae(rewards, dones, values, last_value.flatten(), self.cfg)

            _sync(self.device)
            train_t0 = time.time()
            flat_actor_obs = actor_obs.reshape(batch_size, *self.env.actor_obs_shape)
            flat_critic_obs = critic_obs.reshape(batch_size, *self.env.critic_obs_shape)
            flat_actions = actions.reshape(batch_size, self.env.action_dim)
            flat_old_logprobs = old_logprobs.reshape(batch_size)
            flat_advantages = advantages.reshape(batch_size)
            flat_returns = returns.reshape(batch_size)
            flat_values = values.reshape(batch_size)
            flat_old_means = old_means.reshape(batch_size, self.env.action_dim)
            flat_old_stds = old_stds.reshape(batch_size, self.env.action_dim)

            indices = torch.randperm((batch_size // minibatch_size) * minibatch_size, device=self.device)
            loss_sums = defaultdict(float)
            num_updates = 0
            last_kl_mean = 0.0
            for _epoch in range(self.cfg.update_epochs):
                for start in range(0, len(indices), minibatch_size):
                    mb_idx = indices[start : start + minibatch_size]
                    mb_adv = flat_advantages[mb_idx]
                    if self.cfg.normalize_advantage_per_minibatch:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    new_logprob, entropy, new_values, new_mean, new_std = self.model.evaluate_actions(
                        flat_actor_obs[mb_idx], flat_critic_obs[mb_idx], flat_actions[mb_idx]
                    )
                    with torch.no_grad():
                        kl = _distribution_kl(flat_old_means[mb_idx], flat_old_stds[mb_idx], new_mean, new_std)
                        last_kl_mean = float(kl.mean().detach().cpu())
                        self._maybe_update_learning_rate(last_kl_mean)

                    ratio = (new_logprob - flat_old_logprobs[mb_idx]).exp()
                    surrogate = -mb_adv * ratio
                    surrogate_clipped = -mb_adv * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
                    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                    if self.cfg.use_clipped_value_loss:
                        value_clipped = flat_values[mb_idx] + (new_values - flat_values[mb_idx]).clamp(
                            -self.cfg.clip_coef, self.cfg.clip_coef
                        )
                        value_loss = torch.max(
                            (new_values - flat_returns[mb_idx]).square(),
                            (value_clipped - flat_returns[mb_idx]).square(),
                        ).mean()
                    else:
                        value_loss = (new_values - flat_returns[mb_idx]).square().mean()

                    entropy_mean = entropy.mean()
                    loss = surrogate_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy_mean
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    if not self.cfg.finite_grad_guard or torch.isfinite(grad_norm):
                        self.optimizer.step()
                    else:
                        self.optimizer.zero_grad(set_to_none=True)

                    loss_sums["loss"] += float(loss.detach().cpu())
                    loss_sums["value_loss"] += float(value_loss.detach().cpu())
                    loss_sums["surrogate_loss"] += float(surrogate_loss.detach().cpu())
                    loss_sums["entropy"] += float(entropy_mean.detach().cpu())
                    loss_sums["grad_norm"] += float(grad_norm.detach().cpu()) if torch.isfinite(grad_norm) else float("nan")
                    num_updates += 1
            _sync(self.device)
            train_s = time.time() - train_t0

            steps = (iteration + 1) * batch_size
            elapsed = time.time() - start_time
            row: dict[str, Any] = {
                "iteration": iteration + 1,
                "steps": steps,
                "sps": steps / max(elapsed, 1e-9),
                "iteration_sps": batch_size / max(rollout_s + train_s, 1e-9),
                "mean_reward": float(rewards.mean().cpu()),
                "mean_done": float(dones.mean().cpu()),
                "rollout_s": rollout_s,
                "train_s": train_s,
                "learning_rate": self.cfg.learning_rate,
                "kl": last_kl_mean,
                "action_std": float(self.model.output_std.mean().detach().cpu()),
            }
            for key, value in loss_sums.items():
                row[key] = value / max(num_updates, 1)
            for key, values_list in extra_accum.items():
                if values_list:
                    row[key] = sum(values_list) / len(values_list)
            rows.append(row)
            self._write_logs(row)

            if self.cfg.save_interval > 0 and (iteration + 1) % self.cfg.save_interval == 0:
                self._save(f"model_{iteration + 1}.pt")

        with (self.log_dir / "puffer_cfg.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        self._save("model_final.pt")
        if self.tb_writer is not None:
            self.tb_writer.close()
        print(f"[INFO] wrote PufferLib-style logs to {self.log_dir}", flush=True)
        return rows

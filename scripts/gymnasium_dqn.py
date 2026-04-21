"""Minimal SARL DQN training entrypoint.

Examples:

```bash
uv run -m scripts.gymnasium_dqn
uv run -m scripts.gymnasium_dqn env.id=Acrobot-v1
uv run -m scripts.gymnasium_dqn collector.total_frames=10000
```
"""

from __future__ import annotations

import random
from collections import deque

import hydra
import numpy as np
import torch
from loguru import logger as pylogger
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchrl.record.loggers import get_logger
from torchrl.record.loggers.common import Logger


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


def make_env(cfg: DictConfig):
    try:
        import gymnasium as gym
    except ImportError as exc:
        msg = "gymnasium is required to run scripts.gymnasium_dqn"
        raise RuntimeError(msg) from exc

    pylogger.info("Preparing SARL env '{}'", cfg.env.id)
    return gym.make(cfg.env.id)


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    progress = min(step / decay_steps, 1.0)
    return start + progress * (end - start)


def make_exp_logger(cfg: DictConfig) -> Logger | None:
    backend = cfg.logger.get("backend", "stdout")
    if backend in ("stdout", "", None):
        return None
    return get_logger(
        logger_type=backend,
        logger_name=cfg.logger.log_dir,
        experiment_name=cfg.logger.experiment_name,
        wandb_kwargs={"project": cfg.logger.wandb_project},
    )


def train(cfg: DictConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_env(cfg)

    reset_out = env.reset(seed=cfg.seed)
    obs, _info = reset_out
    env.action_space.seed(cfg.seed)

    if len(obs.shape) != 1:
        msg = f"Only flat observation spaces are supported, got shape={obs.shape}."
        raise RuntimeError(msg)

    if not hasattr(env.action_space, "n"):
        msg = "Only discrete action spaces are supported for this minimal DQN example."
        raise RuntimeError(msg)

    obs_dim = int(obs.shape[0])
    action_dim = int(env.action_space.n)
    device = torch.device(cfg.train.device)

    q_net = QNetwork(obs_dim, action_dim, cfg.model.hidden_dim, cfg.model.depth).to(device)
    target_q_net = QNetwork(obs_dim, action_dim, cfg.model.hidden_dim, cfg.model.depth).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    target_q_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.optim.lr)
    replay_buffer = deque(maxlen=cfg.train.replay_buffer_size)

    pylogger.info(
        "Starting minimal SARL DQN: env={} total_frames={} batch_size={} device={}",
        cfg.env.id,
        cfg.collector.total_frames,
        cfg.train.batch_size,
        device,
    )

    exp_logger = make_exp_logger(cfg)
    if exp_logger is not None:
        exp_logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))

    try:
        _run_training_loop(
            cfg,
            env,
            obs,
            device,
            q_net,
            target_q_net,
            optimizer,
            replay_buffer,
            exp_logger,
        )
    finally:
        env.close()
        if cfg.logger.get("backend") == "wandb":
            import wandb

            if wandb.run is not None:
                wandb.finish()


def _run_training_loop(
    cfg: DictConfig,
    env,
    obs,
    device: torch.device,
    q_net: QNetwork,
    target_q_net: QNetwork,
    optimizer: torch.optim.Adam,
    replay_buffer: deque,
    exp_logger: Logger | None,
) -> None:
    episode_reward = 0.0
    episode_length = 0
    episode_idx = 0
    recent_rewards = deque(maxlen=cfg.train.reward_window)
    last_loss: float | None = None

    for step in range(1, cfg.collector.total_frames + 1):
        epsilon = linear_epsilon(
            step,
            cfg.train.epsilon_start,
            cfg.train.epsilon_end,
            cfg.train.epsilon_decay_steps,
        )

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(q_net(obs_tensor).argmax(dim=-1).item())

        step_out = env.step(action)
        next_obs, reward, terminated, truncated, _info = step_out
        done = bool(terminated or truncated)

        replay_buffer.append((obs, action, reward, next_obs, float(terminated)))

        obs = next_obs
        episode_reward += float(reward)
        episode_length += 1

        if done:
            episode_idx += 1
            recent_rewards.append(episode_reward)
            if bool(cfg.train.get("log_episodes", False)):
                pylogger.info(
                    "Episode {} finished: reward={:.2f} length={} epsilon={:.3f}",
                    episode_idx,
                    episode_reward,
                    episode_length,
                    epsilon,
                )
            obs, _info = env.reset()
            episode_reward = 0.0
            episode_length = 0

        should_learn = step >= cfg.train.learning_starts and len(replay_buffer) >= cfg.train.batch_size
        if should_learn:
            transitions = random.sample(replay_buffer, cfg.train.batch_size)
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_terminated = zip(*transitions, strict=True)

            obs_batch = torch.as_tensor(np.array(batch_obs), dtype=torch.float32, device=device)
            actions_batch = torch.as_tensor(batch_actions, dtype=torch.long, device=device).unsqueeze(-1)
            rewards_batch = torch.as_tensor(batch_rewards, dtype=torch.float32, device=device)
            next_obs_batch = torch.as_tensor(np.array(batch_next_obs), dtype=torch.float32, device=device)
            terminated_batch = torch.as_tensor(batch_terminated, dtype=torch.float32, device=device)

            q_values = q_net(obs_batch).gather(1, actions_batch).squeeze(-1)
            with torch.no_grad():
                next_q_values = target_q_net(next_obs_batch).max(dim=1).values
                targets = rewards_batch + cfg.loss.gamma * (1.0 - terminated_batch) * next_q_values

            loss = torch.nn.functional.mse_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            if bool(cfg.optim.get("clip_grad_norm", False)):
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), float(cfg.optim.get("max_grad_norm", 10.0)))
            optimizer.step()
            last_loss = float(loss.item())

            if step % cfg.train.target_update_interval == 0:
                target_q_net.load_state_dict(q_net.state_dict())

        if step % cfg.train.log_interval == 0:
            mean_reward = float(np.mean(recent_rewards)) if recent_rewards else float("nan")
            pylogger.info(
                "step={} epsilon={:.3f} replay_size={} mean_reward({})={:.2f} loss={}",
                step,
                epsilon,
                len(replay_buffer),
                cfg.train.reward_window,
                mean_reward,
                "n/a" if last_loss is None else f"{last_loss:.4f}",
            )
            if exp_logger is not None:
                exp_logger.log_scalar("train/epsilon", epsilon, step=step)
                exp_logger.log_scalar("train/replay_size", float(len(replay_buffer)), step=step)
                exp_logger.log_scalar("train/mean_reward", mean_reward, step=step)
                if last_loss is not None:
                    exp_logger.log_scalar("train/loss", last_loss, step=step)

    pylogger.info(
        "Training finished: steps={} episodes={} replay_size={}",
        cfg.collector.total_frames,
        episode_idx,
        len(replay_buffer),
    )


@hydra.main(config_path="../configs", config_name="gymnasium_dqn", version_base=None)
def main(cfg: DictConfig) -> None:
    pylogger.info("Starting SARL DQN for env {}", cfg.env.id)
    train(cfg)


if __name__ == "__main__":
    main()

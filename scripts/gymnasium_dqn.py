"""Train DQN on Gymnasium environments with TorchRL trainers.

Examples:

```bash
uv run -m scripts.gymnasium_dqn
uv run -m scripts.gymnasium_dqn env.id=Acrobot-v1
uv run -m scripts.gymnasium_dqn collector.total_frames=10000
```
"""

from __future__ import annotations

import hydra
import torch
from loguru import logger as pylogger
from omegaconf import DictConfig
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn

from torchrl.collectors import Collector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import EnvCreator, GymEnv, RewardSum, SerialEnv, TransformedEnv
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate, ValueEstimators
from torchrl.trainers.algorithms.dqn import DQNTrainer

from scripts.build import close_experiment_logger, make_experiment_logger
from xdrl.trainer_hooks import LoggingEvaluationHookSet, LoggingHookSet


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


def _single_env(cfg: DictConfig, *, render_mode: str | None = None, max_steps: int | None = None) -> TransformedEnv:
    kwargs = {} if render_mode is None else {"render_mode": render_mode}
    if max_steps is not None:
        kwargs["max_episode_steps"] = max_steps
    base_env = GymEnv(cfg.env.id, device=cfg.env.device, **kwargs)
    return TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env.reward_key], out_keys=["episode_reward"]),
    )


def make_env(cfg: DictConfig) -> TransformedEnv | SerialEnv:
    def _build() -> TransformedEnv:
        return _single_env(cfg, max_steps=cfg.env.max_steps)

    if cfg.env.num_envs == 1:
        return _build()
    return SerialEnv(cfg.env.num_envs, EnvCreator(_build))


def make_eval_env(cfg: DictConfig) -> TransformedEnv | SerialEnv:
    def _build() -> TransformedEnv:
        render_mode = "rgb_array" if cfg.eval.render else None
        return _single_env(cfg, render_mode=render_mode, max_steps=cfg.eval.max_steps)

    if cfg.eval.episodes == 1:
        return _build()
    return SerialEnv(cfg.eval.episodes, EnvCreator(_build))


def make_modules(
    env: TransformedEnv | SerialEnv, cfg: DictConfig
) -> tuple[SafeSequential, TensorDictSequential, EGreedyModule]:
    obs_shape = env.observation_spec["observation"].shape
    if len(obs_shape) != 1:
        msg = f"Only flat observation spaces are supported, got shape={obs_shape}."
        raise RuntimeError(msg)

    action_spec = env.full_action_spec_unbatched["action"]
    if not hasattr(action_spec.space, "n"):
        msg = "Only discrete action spaces are supported for this DQN trainer."
        raise RuntimeError(msg)

    q_net = QNetwork(
        obs_dim=int(obs_shape[0]),
        action_dim=int(action_spec.space.n),
        hidden_dim=cfg.model.hidden_dim,
        depth=cfg.model.depth,
    )
    actor_module = TensorDictModule(
        q_net,
        in_keys=["observation"],
        out_keys=["action_value"],
    )
    value_module = QValueModule(
        action_value_key="action_value",
        out_keys=["action", "action_value", "chosen_action_value"],
        spec=env.full_action_spec_unbatched,
    )
    qnet = SafeSequential(actor_module, value_module)

    greedy_module = EGreedyModule(
        eps_init=cfg.exploration.eps_init,
        eps_end=cfg.exploration.eps_end,
        annealing_num_steps=cfg.exploration.anneal_frames,
        action_key="action",
        spec=env.full_action_spec_unbatched,
    )
    qnet_explore = TensorDictSequential(qnet, greedy_module)
    return qnet, qnet_explore, greedy_module


def make_trainer(cfg: DictConfig, env: TransformedEnv | SerialEnv) -> tuple[DQNTrainer, LoggingHookSet]:
    qnet, qnet_explore, greedy_module = make_modules(env, cfg)

    collector = Collector(
        env,
        qnet_explore,
        policy_device=cfg.train.device,
        env_device=cfg.env.device,
        storing_device=cfg.train.storing_device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        no_cuda_sync=cfg.collector.no_cuda_sync,
        init_random_frames=cfg.buffer.init_random_frames,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.storing_device),
        batch_size=cfg.train.minibatch_size,
    )

    loss_module = DQNLoss(
        qnet,
        delay_value=cfg.loss.delay_value,
        double_dqn=cfg.loss.double_dqn,
        loss_function=cfg.loss.loss_function,
    )
    loss_module.set_keys(
        reward="reward",
        done="done",
        terminated="terminated",
        action_value="action_value",
        action="action",
        value="chosen_action_value",
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=cfg.loss.gamma)

    if cfg.loss.soft_target_update:
        target_net_updater = SoftUpdate(loss_module, tau=cfg.loss.tau)
    else:
        target_net_updater = HardUpdate(
            loss_module,
            value_network_update_interval=cfg.loss.hard_target_update_interval,
        )

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)
    trainer_logger = make_experiment_logger(cfg)

    trainer = DQNTrainer(
        collector=collector,
        total_frames=cfg.collector.total_frames,
        frame_skip=cfg.train.frame_skip,
        optim_steps_per_batch=cfg.train.optim_steps_per_batch,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=trainer_logger,
        clip_grad_norm=cfg.optim.clip_grad_norm,
        clip_norm=cfg.optim.max_grad_norm,
        progress_bar=cfg.train.progress_bar,
        seed=cfg.seed,
        save_trainer_interval=cfg.train.save_trainer_interval,
        log_interval=cfg.logger.log_interval,
        replay_buffer=replay_buffer,
        enable_logging=False,
        target_net_updater=target_net_updater,
        greedy_module=greedy_module,
    )

    eval_hook_set = None
    if cfg.eval.enabled:
        pylogger.info(
            "Evaluation enabled: pre_eval={} interval_frames={} episodes={} render={} deterministic={} non_deterministic={}",
            cfg.eval.pre_eval,
            cfg.eval.interval_frames,
            cfg.eval.episodes,
            cfg.eval.render,
            cfg.eval.deterministic,
            cfg.eval.non_deterministic,
        )
        eval_env = make_eval_env(cfg)
        eval_hook_set = LoggingEvaluationHookSet(
            policy=qnet_explore,
            environment=eval_env,
            group=cfg.logger.metric_group,
            reward_key=("next", "reward"),
            interval_frames=cfg.eval.interval_frames,
            max_steps=cfg.eval.max_steps,
            deterministic=cfg.eval.deterministic,
            non_deterministic=cfg.eval.non_deterministic,
            render=cfg.eval.render,
            video_fps=cfg.eval.video_fps,
        )

    logging_hooks = LoggingHookSet(
        group=cfg.logger.metric_group,
        frame_skip=cfg.train.frame_skip,
        reward_key=("next", "reward"),
        done_key=("next", "done"),
        episode_reward_key=("next", "episode_reward"),
        eval_hook_set=eval_hook_set,
    )
    logging_hooks.register(trainer)

    return trainer, logging_hooks


@hydra.main(config_path="../configs", config_name="gymnasium_dqn", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    pylogger.info(
        "Starting DQN on {} with {} envs and {} total frames",
        cfg.env.id,
        cfg.env.num_envs,
        cfg.collector.total_frames,
    )

    env = make_env(cfg)
    trainer, logging_hooks = make_trainer(cfg, env)
    try:
        if cfg.eval.enabled and cfg.eval.pre_eval:
            pylogger.info("Running pre-training evaluation")
            logging_hooks.run_pre_eval()
        trainer.train()
    finally:
        logging_hooks.close()
        trainer.collector.shutdown()
        if hasattr(env, "is_closed") and not env.is_closed:
            env.close()
        close_experiment_logger(cfg)


if __name__ == "__main__":
    main()

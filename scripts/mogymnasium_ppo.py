"""Train PPO on MO-Gymnasium environments with weighted-sum scalarization.

Examples:

```bash
uv run -m scripts.mogymnasium_ppo
uv run -m scripts.mogymnasium_ppo env.id=mo-mountaincarcontinuous-v0
```
"""

from __future__ import annotations

import hydra
import mo_gymnasium  # noqa: F401
import torch
from loguru import logger as pylogger
from omegaconf import DictConfig
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch.distributions import Categorical

from torchrl.collectors import Collector
from torchrl.envs import EnvCreator, GymEnv, RewardSum, SerialEnv, TransformedEnv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.trainers.algorithms.ppo import PPOTrainer
from torchrl.trainers.trainers import TrainerHookBase

from scripts.build import close_experiment_logger, make_experiment_logger
from xdrl.trainer_hooks import (
    LoggingEvaluationHookSet,
    LoggingHookSet,
    WeightedSumRewardHook,
)


def _to_key(value: str | list[str] | tuple[str, ...] | None) -> str | tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return tuple(value)


def _strip_next_prefix(key: str | tuple[str, ...]) -> str | tuple[str, ...]:
    if isinstance(key, tuple) and len(key) > 0 and key[0] == "next":
        if len(key) == 2:
            return key[1]
        return key[1:]
    return key


def _with_next_prefix(key: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(key, str):
        return ("next", key)
    return ("next",) + key


def _make_mlp(input_dim: int, output_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class ScalarizedGAEHook(TrainerHookBase):
    """Pre-epoch hook that scalarizes rewards and computes GAE."""

    def __init__(
        self,
        *,
        loss_module: ClipPPOLoss,
        scalarization_hook: WeightedSumRewardHook,
        reward_key: str | tuple[str, ...],
        gamma: float,
        lmbda: float,
    ) -> None:
        self.loss_module = loss_module
        self.scalarization_hook = scalarization_hook
        self.gae = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=self.loss_module.critic_network,
            average_gae=True,
        )
        self.gae.set_keys(
            reward=reward_key,
            done="done",
            terminated="terminated",
            advantage="advantage",
            value_target="value_target",
            value="state_value",
        )

    def __call__(self, batch: TensorDictBase) -> TensorDictBase:
        self.scalarization_hook(batch)
        with torch.no_grad():
            self.gae(
                batch,
                params=self.loss_module.critic_network_params,
                target_params=self.loss_module.target_critic_network_params,
            )
        return batch


def _single_env(cfg: DictConfig, *, render_mode: str | None = None) -> TransformedEnv:
    kwargs = {} if render_mode is None else {"render_mode": render_mode}
    base_env = GymEnv(cfg.env.id, device=cfg.env.device, **kwargs)
    return TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env.reward_key], out_keys=["episode_reward"]),
    )


def make_env(cfg: DictConfig) -> TransformedEnv | SerialEnv:
    def _build() -> TransformedEnv:
        return _single_env(cfg)

    if cfg.env.num_envs == 1:
        return _build()
    return SerialEnv(cfg.env.num_envs, EnvCreator(_build))


def make_eval_env(cfg: DictConfig) -> TransformedEnv | SerialEnv:
    def _build() -> TransformedEnv:
        return _single_env(cfg, render_mode="rgb_array" if cfg.eval.render else None)

    if cfg.eval.episodes == 1:
        return _build()
    return SerialEnv(cfg.eval.episodes, EnvCreator(_build))


def make_modules(env: GymEnv | SerialEnv, cfg: DictConfig) -> tuple[ProbabilisticActor, ValueOperator]:
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_spec = env.action_spec_unbatched

    if cfg.env.continuous_actions:
        action_dim = action_spec.shape[-1]
        actor_module = TensorDictModule(
            nn.Sequential(
                _make_mlp(obs_dim, 2 * action_dim, cfg.model.hidden_dim, cfg.model.depth),
                NormalParamExtractor(),
            ),
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )
        actor = ProbabilisticActor(
            module=actor_module,
            spec=action_spec,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low,
                "high": action_spec.space.high,
            },
            return_log_prob=True,
            log_prob_key="sample_log_prob",
        )
    else:
        n_actions = int(action_spec.space.n)
        actor_module = TensorDictModule(
            _make_mlp(obs_dim, n_actions, cfg.model.hidden_dim, cfg.model.depth),
            in_keys=["observation"],
            out_keys=["logits"],
        )
        actor = ProbabilisticActor(
            module=actor_module,
            spec=action_spec,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=Categorical,
            return_log_prob=True,
            log_prob_key="sample_log_prob",
        )

    critic = ValueOperator(
        module=_make_mlp(obs_dim, 1, cfg.model.hidden_dim, cfg.model.depth),
        in_keys=["observation"],
        out_keys=["state_value"],
    )
    return actor, critic


def make_trainer(cfg: DictConfig, env: TransformedEnv | SerialEnv) -> tuple[PPOTrainer, LoggingHookSet]:
    actor, critic = make_modules(env, cfg)

    pylogger.info(
        "Collector policy_device={} env_device={} storing_device={}",
        cfg.train.device,
        cfg.env.device,
        cfg.train.storing_device,
    )

    collector = Collector(
        env,
        actor,
        policy_device=cfg.train.device,
        env_device=cfg.env.device,
        storing_device=cfg.train.storing_device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        no_cuda_sync=cfg.collector.no_cuda_sync,
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coeff=cfg.loss.entropy_coef,
        normalize_advantage=False,
    )

    if cfg.scalarization.method != "weighted_sum":
        msg = f"Unsupported scalarization method '{cfg.scalarization.method}'. Use 'weighted_sum'."
        raise ValueError(msg)

    scalar_out_key = _to_key(cfg.scalarization.out_key)
    if scalar_out_key is None:
        msg = "scalarization.out_key must be configured."
        raise ValueError(msg)
    reward_key = _strip_next_prefix(scalar_out_key)

    loss_module.set_keys(
        reward=reward_key,
        action="action",
        done="done",
        terminated="terminated",
        advantage="advantage",
        value_target="value_target",
        value="state_value",
        sample_log_prob="sample_log_prob",
    )

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)

    trainer_logger = make_experiment_logger(cfg)

    trainer = PPOTrainer(
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
        num_epochs=cfg.train.num_epochs,
        replay_buffer=None,
        enable_logging=False,
        add_gae=False,
    )

    scalarization_hook = WeightedSumRewardHook(
        weights=cfg.scalarization.weights,
        in_key=_to_key(cfg.scalarization.in_key),
        out_key=scalar_out_key,
        preserve_input_key=_to_key(cfg.scalarization.get("preserve_input_key", None)),
    )
    trainer.register_op(
        dest="pre_epoch",
        op=ScalarizedGAEHook(
            loss_module=loss_module,
            scalarization_hook=scalarization_hook,
            reward_key=reward_key,
            gamma=cfg.loss.gamma,
            lmbda=cfg.loss.lmbda,
        ),
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
            policy=actor,
            environment=eval_env,
            group=cfg.logger.metric_group,
            reward_key=_with_next_prefix(reward_key),
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
        reward_key=_with_next_prefix(reward_key),
        done_key=("next", "done"),
        episode_reward_key=("next", "episode_reward"),
        eval_hook_set=eval_hook_set,
    )
    logging_hooks.register(trainer)

    return trainer, logging_hooks


@hydra.main(config_path="../configs", config_name="mogymnasium_ppo", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    pylogger.info(
        "Starting MORL PPO on {} with {} envs and {} total frames",
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

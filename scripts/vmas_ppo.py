"""Train IPPO or MAPPO on VMAS scenarios with TorchRL PPOTrainer hooks.

Examples:

```bash
uv run -m scripts.vmas_ppo
uv run -m scripts.vmas_ppo algo=mappo
uv run -m scripts.vmas_ppo env.scenario=navigation
```
"""

from __future__ import annotations

import hydra
import torch
from loguru import logger as pylogger
from omegaconf import DictConfig, OmegaConf
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch.distributions import Categorical

from torchrl.collectors import Collector
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.record.loggers import get_logger
from torchrl.trainers.algorithms.ppo import PPOTrainer
from torchrl.trainers.trainers import BatchSubSampler

from xdrl.trainer_hooks import (
    ExpandSharedNextKeysHook,
    LoggingEvaluationMetricsHook,
    LoggingHookSet,
    MultiAgentGAEHook,
    PolicyCheckpointHook,
)


def make_env(cfg: DictConfig) -> TransformedEnv:
    base_env = VmasEnv(
        scenario=cfg.env.scenario,
        num_envs=cfg.env.num_envs,
        continuous_actions=cfg.env.continuous_actions,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        **cfg.env.scenario_kwargs,
    )
    return TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env.reward_key], out_keys=[("agents", "episode_reward")]),
    )


def make_eval_env(cfg: DictConfig) -> TransformedEnv:
    base_env = VmasEnv(
        scenario=cfg.env.scenario,
        num_envs=cfg.eval.episodes,
        continuous_actions=cfg.env.continuous_actions,
        max_steps=cfg.eval.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        **cfg.env.scenario_kwargs,
    )
    return TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env.reward_key], out_keys=[("agents", "episode_reward")]),
    )


def make_modules(env: TransformedEnv, cfg: DictConfig) -> tuple[ProbabilisticActor, ValueOperator]:
    group = cfg.model.group
    action_spec = env.full_action_spec_unbatched[env.action_key]
    obs_dim = env.observation_spec[group, "observation"].shape[-1]

    if cfg.env.continuous_actions:
        action_dim = action_spec.shape[-1]
        actor_net = nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=obs_dim,
                n_agent_outputs=2 * action_dim,
                n_agents=env.n_agents,
                centralised=False,
                share_params=cfg.model.share_policy_params,
                device=cfg.train.device,
                depth=cfg.model.depth,
                num_cells=cfg.model.hidden_dim,
                activation_class=nn.Tanh,
            ),
            NormalParamExtractor(),
        )
        actor_module = TensorDictModule(
            actor_net,
            in_keys=[(group, "observation")],
            out_keys=[(group, "loc"), (group, "scale")],
        )
        actor = ProbabilisticActor(
            module=actor_module,
            spec=env.full_action_spec_unbatched,
            in_keys=[(group, "loc"), (group, "scale")],
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low,
                "high": action_spec.space.high,
            },
            return_log_prob=True,
            log_prob_key=(group, "log_prob"),
        )
    else:
        n_actions = action_spec.space.n
        actor_module = TensorDictModule(
            MultiAgentMLP(
                n_agent_inputs=obs_dim,
                n_agent_outputs=n_actions,
                n_agents=env.n_agents,
                centralised=False,
                share_params=cfg.model.share_policy_params,
                device=cfg.train.device,
                depth=cfg.model.depth,
                num_cells=cfg.model.hidden_dim,
                activation_class=nn.Tanh,
            ),
            in_keys=[(group, "observation")],
            out_keys=[(group, "logits")],
        )
        actor = ProbabilisticActor(
            module=actor_module,
            spec=env.full_action_spec_unbatched,
            in_keys=[(group, "logits")],
            out_keys=[env.action_key],
            distribution_class=Categorical,
            return_log_prob=True,
            log_prob_key=(group, "log_prob"),
        )

    critic_net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=cfg.algo == "mappo",
        share_params=cfg.model.share_critic_params,
        device=cfg.train.device,
        depth=cfg.model.depth,
        num_cells=cfg.model.hidden_dim,
        activation_class=nn.Tanh,
    )
    critic = ValueOperator(
        module=critic_net,
        in_keys=[(group, "observation")],
        out_keys=[(group, "state_value")],
    )
    return actor, critic


def make_trainer(cfg: DictConfig, env: TransformedEnv) -> tuple[PPOTrainer, LoggingHookSet]:
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
    group = cfg.model.group
    loss_module.set_keys(
        reward=(group, "reward"),
        action=(group, "action"),
        done=(group, "done"),
        terminated=(group, "terminated"),
        advantage=(group, "advantage"),
        value_target=(group, "value_target"),
        value=(group, "state_value"),
        sample_log_prob=(group, "log_prob"),
    )
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=cfg.loss.gamma, lmbda=cfg.loss.lmbda)
    loss_module.value_estimator.set_keys(
        reward=(group, "reward"),
        done=(group, "done"),
        terminated=(group, "terminated"),
        advantage=(group, "advantage"),
        value_target=(group, "value_target"),
        value=(group, "state_value"),
    )

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)

    trainer_logger = get_logger(
        logger_type=cfg.logger.backend,
        logger_name=cfg.logger.log_dir,
        experiment_name=cfg.logger.experiment_name,
        wandb_kwargs={"project": cfg.logger.wandb_project},
    )
    trainer_logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))

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
        log_interval=cfg.train.log_interval,
        num_epochs=cfg.train.num_epochs,
        replay_buffer=None,
        enable_logging=False,
        add_gae=False,
    )

    trainer.register_op(
        dest="pre_epoch",
        op=ExpandSharedNextKeysHook(
            group=group,
            key_names=("done", "terminated"),
        ),
    )
    trainer.register_op(
        dest="pre_epoch",
        op=MultiAgentGAEHook(
            loss_module=loss_module,
            gamma=cfg.loss.gamma,
            lmbda=cfg.loss.lmbda,
            group=group,
        ),
    )
    trainer.register_op(dest="process_optim_batch", op=BatchSubSampler(batch_size=cfg.train.minibatch_size))

    policy_checkpoint_interval = int(cfg.train.get("policy_checkpoint_interval", 0))
    if policy_checkpoint_interval > 0:
        checkpoint_dir = cfg.train.get("policy_checkpoint_dir", "checkpoints/policy")
        checkpoint_prefix = cfg.train.get("policy_checkpoint_prefix", "policy")
        trainer.register_op(
            dest="post_steps",
            op=PolicyCheckpointHook(
                policy=actor,
                directory=checkpoint_dir,
                interval=policy_checkpoint_interval,
                prefix=checkpoint_prefix,
                meta={"algo": cfg.algo, "group": group},
            ),
        )
        pylogger.info(
            "Policy checkpointing enabled interval={} dir='{}' prefix='{}'",
            policy_checkpoint_interval,
            checkpoint_dir,
            checkpoint_prefix,
        )

    eval_hook = None
    if cfg.eval.enabled:
        pylogger.info(
            "Evaluation enabled: pre_eval={} interval_frames={} episodes={} render={}",
            cfg.eval.pre_eval,
            cfg.eval.interval_frames,
            cfg.eval.episodes,
            cfg.eval.render,
        )
        eval_hook = LoggingEvaluationMetricsHook(
            policy=actor,
            environment=make_eval_env(cfg),
            group=group,
            interval_frames=cfg.eval.interval_frames,
            max_steps=cfg.eval.max_steps,
            deterministic=cfg.eval.deterministic,
            render=cfg.eval.render,
            video_fps=cfg.eval.video_fps,
        )

    logging_hooks = LoggingHookSet(
        group=group,
        frame_skip=cfg.train.frame_skip,
        eval_hook=eval_hook,
    )
    logging_hooks.register(trainer)

    return trainer, logging_hooks


@hydra.main(config_path="../configs", config_name="vmas_ppo", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    pylogger.info(
        "Starting {} on VMAS {} with {} envs and {} total frames",
        cfg.algo.upper(),
        cfg.env.scenario,
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
        if not env.is_closed:
            env.close()


if __name__ == "__main__":
    main()

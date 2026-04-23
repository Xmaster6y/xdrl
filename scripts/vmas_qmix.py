"""Train QMIX on VMAS scenarios with trainer and hooksets.

Examples:

```bash
uv run -m scripts.vmas_qmix
uv run -m scripts.vmas_qmix env.scenario=navigation
uv run -m scripts.vmas_qmix collector.total_frames=300000
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
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential
from torchrl.modules.models.multiagent import MultiAgentMLP, QMixer
from torchrl.objectives import HardUpdate, SoftUpdate, ValueEstimators
from torchrl.objectives.multiagent.qmixer import QMixerLoss

from scripts.build import close_experiment_logger, make_experiment_logger
from xdrl.trainer import QmixTrainer
from xdrl.trainer_hooks import LoggingEvaluationHookSet, LoggingHookSet


def make_env(cfg: DictConfig, *, num_envs: int, max_steps: int) -> TransformedEnv:
    base_env = VmasEnv(
        scenario=cfg.env.scenario,
        num_envs=num_envs,
        continuous_actions=False,
        max_steps=max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        **cfg.env.get("scenario_kwargs", {}),
    )
    return TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env.reward_key], out_keys=[("agents", "episode_reward")]),
    )


def make_modules(
    env: TransformedEnv,
    cfg: DictConfig,
) -> tuple[SafeSequential, TensorDictSequential, EGreedyModule, TensorDictModule]:
    group = cfg.model.group
    obs_dim = env.observation_spec[group, "observation"].shape[-1]
    n_actions = int(env.full_action_spec_unbatched[env.action_key].space.n)

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
        out_keys=[(group, "action_value")],
    )
    value_module = QValueModule(
        action_value_key=(group, "action_value"),
        out_keys=[
            env.action_key,
            (group, "action_value"),
            (group, "chosen_action_value"),
        ],
        spec=env.full_action_spec_unbatched,
    )
    qnet = SafeSequential(actor_module, value_module)

    greedy_module = EGreedyModule(
        eps_init=cfg.exploration.eps_init,
        eps_end=cfg.exploration.eps_end,
        annealing_num_steps=cfg.exploration.anneal_frames,
        action_key=env.action_key,
        spec=env.full_action_spec_unbatched,
    )
    qnet_explore = TensorDictSequential(qnet, greedy_module)

    mixer = TensorDictModule(
        module=QMixer(
            state_shape=env.observation_spec_unbatched[group, "observation"].shape,
            mixing_embed_dim=cfg.model.mixing_embed_dim,
            n_agents=env.n_agents,
            device=cfg.train.device,
        ),
        in_keys=[(group, "chosen_action_value"), (group, "observation")],
        out_keys=["chosen_action_value"],
    )
    return qnet, qnet_explore, greedy_module, mixer


def make_trainer(cfg: DictConfig, env: TransformedEnv) -> tuple[QmixTrainer, LoggingHookSet]:
    qnet, qnet_explore, greedy_module, mixer = make_modules(env, cfg)

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
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    loss_module = QMixerLoss(
        qnet,
        mixer,
        delay_value=cfg.loss.delay_value,
        loss_function=cfg.loss.loss_function,
    )
    loss_module.set_keys(
        reward="reward",
        done="done",
        terminated="terminated",
        action_value=(cfg.model.group, "action_value"),
        local_value=(cfg.model.group, "chosen_action_value"),
        global_value="chosen_action_value",
        action=env.action_key,
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

    trainer = QmixTrainer(
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
        target_net_updater=target_net_updater,
        greedy_module=greedy_module,
        group=cfg.model.group,
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
        eval_env = make_env(cfg, num_envs=cfg.eval.episodes, max_steps=cfg.eval.max_steps)
        eval_hook_set = LoggingEvaluationHookSet(
            policy=qnet_explore,
            environment=eval_env,
            group=cfg.logger.metric_group,
            reward_key=("next", cfg.model.group, "reward"),
            interval_frames=cfg.eval.interval_frames,
            max_steps=cfg.eval.max_steps,
            deterministic=cfg.eval.deterministic,
            non_deterministic=cfg.eval.non_deterministic,
            render=cfg.eval.render,
            render_kwargs={"mode": "rgb_array"},
            video_fps=cfg.eval.video_fps,
        )

    logging_hooks = LoggingHookSet(
        group=cfg.logger.metric_group,
        frame_skip=cfg.train.frame_skip,
        reward_key=("next", cfg.model.group, "reward"),
        done_key=("next", "done"),
        episode_reward_key=("next", cfg.model.group, "episode_reward"),
        eval_hook_set=eval_hook_set,
    )
    logging_hooks.register(trainer)

    return trainer, logging_hooks


@hydra.main(config_path="../configs", config_name="vmas_qmix", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.env.continuous_actions:
        msg = "QMIX only supports discrete actions. Set env.continuous_actions=false."
        raise ValueError(msg)

    torch.manual_seed(cfg.seed)

    pylogger.info(
        "Starting QMIX on VMAS {} with {} envs and {} total frames",
        cfg.env.scenario,
        cfg.env.num_envs,
        cfg.collector.total_frames,
    )

    env = make_env(cfg, num_envs=cfg.env.num_envs, max_steps=cfg.env.max_steps)
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
        close_experiment_logger(cfg)


if __name__ == "__main__":
    main()
